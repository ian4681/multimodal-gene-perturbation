"""
BFS path feature extraction for cancer perturbation prediction.

For each (pert_gene, target_gene) pair in K562.csv, extracts:
  - path_length       : BFS shortest path (-1 if unreachable or not in graph)
  - is_direct_neighbor: 1 if directly connected
  - n_common_neighbors: |N(pert) ∩ N(target)|
  - degree_pert       : degree of pert gene node
  - degree_gene       : degree of target gene node

Strategy: group by unique pert gene, run single-source BFS once per pert gene.
Reduces BFS calls from 157k to ~number of unique pert genes.
"""

import os
import time
import pickle
import torch
import pandas as pd
import networkx as nx
from multiprocessing import Pool

BASE_DIR  = "/path/to/your/project"          # ← set this to your project root
DATA_DIR  = f"{BASE_DIR}/raw_data/data"
FEAT_DIR  = f"{BASE_DIR}/features"
LOG_DIR   = f"{BASE_DIR}/logs"
BFS_CUTOFF = 6
N_WORKERS  = 5

os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


def process_pert_group(args):
    pert_gene, targets, G = args

    rows = []
    if not G.has_node(pert_gene):
        for tgt_gene, label, split in targets:
            rows.append((pert_gene, tgt_gene, label, split, -1, 0, 0, 0, 0))
        return rows

    distances      = nx.single_source_shortest_path_length(G, pert_gene, cutoff=BFS_CUTOFF)
    pert_neighbors = set(G.neighbors(pert_gene))
    deg_pert       = G.degree(pert_gene)

    for tgt_gene, label, split in targets:
        if not G.has_node(tgt_gene):
            rows.append((pert_gene, tgt_gene, label, split, -1, 0, 0, deg_pert, 0))
            continue

        path_len  = distances.get(tgt_gene, -1)
        is_direct = int(tgt_gene in pert_neighbors)
        n_common  = len(pert_neighbors & set(G.neighbors(tgt_gene)))
        deg_gene  = G.degree(tgt_gene)

        rows.append((pert_gene, tgt_gene, label, split,
                     path_len, is_direct, n_common, deg_pert, deg_gene))
    return rows


def main():
    t0 = time.time()
    print("Loading graph ...")
    G = torch.load(f"{DATA_DIR}/Knowledge_Graph/gene_graph.pth",
                   map_location="cpu", weights_only=False)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Loading K562.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/PerturbQA/K562.csv")
    print(f"  {len(df)} rows")

    groups = {}
    for _, row in df.iterrows():
        p = row["pert"]
        if p not in groups:
            groups[p] = []
        groups[p].append((row["gene"], row["label"], row["split"]))
    print(f"  {len(groups)} unique pert genes")

    args_list = [(pert, targets, G) for pert, targets in groups.items()]

    print(f"Running BFS with {N_WORKERS} workers ...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_pert_group, args_list)

    all_rows = [r for group in results for r in group]
    out = pd.DataFrame(all_rows, columns=[
        "pert", "gene", "label", "split",
        "path_length", "is_direct_neighbor",
        "n_common_neighbors", "degree_pert", "degree_gene"
    ])

    out_path = f"{FEAT_DIR}/bfs_features_K562.pkl"
    out.to_pickle(out_path)

    elapsed      = time.time() - t0
    not_in_graph = (out["path_length"] == -1).sum()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Rows          : {len(out)}")
    print(f"  Saved to      : {out_path}")
    print(f"  Not-in-graph  : {not_in_graph} ({100*not_in_graph/len(out):.1f}%)")
    print(f"  Path len dist :\n{out['path_length'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
