"""
Cross-cell-line evaluation: apply K562-trained RF to HepG2, Jurkat, RPE1.
For each cell line: extract BFS + GO features, load precomputed ESM-2,
merge into 649-dim feature matrix, predict with rf_model.pkl, report metrics.
"""

import os, time, gzip, pickle
import torch
import pandas as pd
import numpy as np
import networkx as nx
import joblib
from collections import defaultdict
from multiprocessing import Pool
from goatools.obo_parser import GODag
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR  = "/path/to/your/project"          # ← set this to your project root
DATA_DIR  = f"{BASE_DIR}/raw_data/data"
RAW_DIR   = f"{BASE_DIR}/raw_data"
FEAT_DIR  = f"{BASE_DIR}/features"
RES_DIR   = f"{BASE_DIR}/results"
N_WORKERS = 4
BFS_CUTOFF = 6

CELL_LINES = ["HepG2", "Jurkat", "RPE1"]


def process_pert_bfs(args):
    pert, targets, G = args
    rows = []
    if not G.has_node(pert):
        for tgt, label, split in targets:
            rows.append((pert, tgt, label, split, -1, 0, 0, 0, 0))
        return rows
    dist    = nx.single_source_shortest_path_length(G, pert, cutoff=BFS_CUTOFF)
    p_nbrs  = set(G.neighbors(pert))
    deg_p   = G.degree(pert)
    for tgt, label, split in targets:
        if not G.has_node(tgt):
            rows.append((pert, tgt, label, split, -1, 0, 0, deg_p, 0))
            continue
        rows.append((pert, tgt, label, split,
                     dist.get(tgt, -1),
                     int(tgt in p_nbrs),
                     len(p_nbrs & set(G.neighbors(tgt))),
                     deg_p, G.degree(tgt)))
    return rows


def build_gene2go(gaf_path, godag):
    ns_map = {'biological_process':'BP','molecular_function':'MF','cellular_component':'CC'}
    gene2go = defaultdict(lambda: {'BP':set(),'MF':set(),'CC':set()})
    opener = gzip.open if gaf_path.endswith('.gz') else open
    with opener(gaf_path, 'rt') as f:
        for line in f:
            if line.startswith('!'): continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or 'NOT' in cols[3]: continue
            go_id = cols[4]
            if go_id not in godag: continue
            ns = ns_map.get(godag[go_id].namespace)
            if ns: gene2go[cols[2]][ns].add(go_id)
    return dict(gene2go)


def process_pert_go(args):
    pert, targets, gene2go = args
    p_go = gene2go.get(pert, {'BP':set(),'MF':set(),'CC':set()})
    rows = []
    for tgt, label, split in targets:
        t_go = gene2go.get(tgt, {'BP':set(),'MF':set(),'CC':set()})
        bp = len(p_go['BP'] & t_go['BP'])
        mf = len(p_go['MF'] & t_go['MF'])
        cc = len(p_go['CC'] & t_go['CC'])
        rows.append((pert, tgt, label, split, bp, mf, cc, bp+mf+cc))
    return rows


def run_cellline(cl, G, godag, gene2go, esm, rf):
    print(f"\n{'='*55}")
    print(f"  Cell line: {cl}")
    t0 = time.time()

    df = pd.read_csv(f"{DATA_DIR}/PerturbQA/{cl}.csv")
    print(f"  Rows: {len(df)}")

    groups = {}
    for _, row in df.iterrows():
        p = row['pert']
        if p not in groups: groups[p] = []
        groups[p].append((row['gene'], row['label'], row['split']))

    print(f"  BFS features ({len(groups)} unique pert genes) ...")
    with Pool(N_WORKERS) as pool:
        bfs_res = pool.map(process_pert_bfs, [(p,t,G) for p,t in groups.items()])
    bfs_df = pd.DataFrame(
        [r for g in bfs_res for r in g],
        columns=['pert','gene','label','split',
                 'path_length','is_direct_neighbor',
                 'n_common_neighbors','degree_pert','degree_gene'])

    print("  GO features ...")
    with Pool(N_WORKERS) as pool:
        go_res = pool.map(process_pert_go, [(p,t,gene2go) for p,t in groups.items()])
    go_df = pd.DataFrame(
        [r for g in go_res for r in g],
        columns=['pert','gene','label','split',
                 'n_shared_go_bp','n_shared_go_mf',
                 'n_shared_go_cc','n_shared_go_all'])

    go_cols = ['pert','gene','n_shared_go_bp','n_shared_go_mf',
               'n_shared_go_cc','n_shared_go_all']
    merged = bfs_df.merge(go_df[go_cols], on=['pert','gene'], how='inner')

    zero_vec = np.zeros(320, dtype=np.float32)
    esm_pert = np.array([esm.get(g, zero_vec) for g in merged['pert']], dtype=np.float32)
    esm_gene = np.array([esm.get(g, zero_vec) for g in merged['gene']], dtype=np.float32)

    bfs_cols = ['path_length','is_direct_neighbor','n_common_neighbors','degree_pert','degree_gene']
    go_fcols = ['n_shared_go_bp','n_shared_go_mf','n_shared_go_cc','n_shared_go_all']
    tab = merged[bfs_cols + go_fcols].values.astype(np.float32)
    X   = np.hstack([tab, esm_pert, esm_gene])
    y   = merged['label'].values

    print(f"  Feature matrix: {X.shape}")
    label_dist = np.bincount(y)
    print(f"  Labels: Non-diff={label_dist[0]}, Down={label_dist[1]}, Up={label_dist[2]}")

    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average='macro')

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(classification_report(y, y_pred,
          target_names=['Non-diff','Down','Up'], digits=4))
    print(f"  Elapsed  : {time.time()-t0:.1f}s")

    return cl, acc, f1


def main():
    print("Loading shared resources ...")

    t_load = time.time()
    G = torch.load(f"{DATA_DIR}/Knowledge_Graph/gene_graph.pth",
                   map_location='cpu', weights_only=False)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    godag   = GODag(f"{RAW_DIR}/go/go-basic.obo", optional_attrs={'namespace'})
    gene2go = build_gene2go(f"{RAW_DIR}/go/goa_human.gaf.gz", godag)
    print(f"  GO: {len(gene2go)} genes annotated")

    esm = pickle.load(open(f"{FEAT_DIR}/esm2_embeddings.pkl", 'rb'))
    print(f"  ESM-2: {len(esm)} gene embeddings")

    rf = joblib.load(f"{RES_DIR}/rf_model.pkl")
    print(f"  RF model loaded  ({time.time()-t_load:.1f}s)")

    results = []
    for cl in CELL_LINES:
        name, acc, f1 = run_cellline(cl, G, godag, gene2go, esm, rf)
        results.append((name, acc, f1))

    print(f"\n{'='*55}")
    print("CROSS-CELL-LINE SUMMARY (K562-trained RF)")
    print(f"{'='*55}")
    print(f"  {'Cell Line':<12} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'K562 (train)':12} {'0.9141':>10} {'0.6021':>10}  <- reference")
    for cl, acc, f1 in results:
        print(f"  {cl:<12} {acc:>10.4f} {f1:>10.4f}")

    lines = ["CROSS-CELL-LINE EVALUATION (K562-trained RF)\n\n"]
    lines.append(f"  {'Cell Line':<12} {'Accuracy':>10} {'Macro F1':>10}\n")
    lines.append(f"  {'K562 (train)':12} {'0.9141':>10} {'0.6021':>10}  reference\n")
    for cl, acc, f1 in results:
        lines.append(f"  {cl:<12} {acc:>10.4f} {f1:>10.4f}\n")
    os.makedirs(RES_DIR, exist_ok=True)
    with open(f"{RES_DIR}/cross_cellline_summary.txt", 'w') as fh:
        fh.writelines(lines)
    print(f"Saved -> results/cross_cellline_summary.txt")


if __name__ == '__main__':
    main()
