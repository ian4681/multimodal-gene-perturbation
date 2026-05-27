"""
GO term shared feature extraction for cancer perturbation prediction.

For each (pert_gene, target_gene) pair, extracts:
  - n_shared_go_bp : shared GO Biological Process terms
  - n_shared_go_mf : shared GO Molecular Function terms
  - n_shared_go_cc : shared GO Cellular Component terms
  - n_shared_go_all: total shared GO terms (all namespaces)

Strategy: build gene→GO set mapping once, then lookup per pair.
Groups by pert gene for cache efficiency.
"""

import os
import time
import gzip
import pickle
import pandas as pd
from collections import defaultdict
from goatools.obo_parser import GODag
from multiprocessing import Pool

BASE_DIR  = "/path/to/your/project"          # ← set this to your project root
DATA_DIR  = f"{BASE_DIR}/raw_data"
FEAT_DIR  = f"{BASE_DIR}/features"
N_WORKERS = 5

os.makedirs(FEAT_DIR, exist_ok=True)


def build_gene2go(gaf_path, godag):
    """Parse GAF file → {gene_symbol: {'BP': set, 'MF': set, 'CC': set}}"""
    ns_map = {
        'biological_process': 'BP',
        'molecular_function': 'MF',
        'cellular_component': 'CC',
    }
    gene2go = defaultdict(lambda: {'BP': set(), 'MF': set(), 'CC': set()})

    opener = gzip.open if gaf_path.endswith('.gz') else open
    with opener(gaf_path, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9:
                continue
            gene_symbol = cols[2]
            go_id       = cols[4]
            qualifier   = cols[3]
            if 'NOT' in qualifier:
                continue
            if go_id not in godag:
                continue
            ns = ns_map.get(godag[go_id].namespace)
            if ns:
                gene2go[gene_symbol][ns].add(go_id)

    return dict(gene2go)


def process_group(args):
    pert_gene, targets, gene2go = args
    p_go = gene2go.get(pert_gene, {'BP': set(), 'MF': set(), 'CC': set()})

    rows = []
    for tgt_gene, label, split in targets:
        t_go = gene2go.get(tgt_gene, {'BP': set(), 'MF': set(), 'CC': set()})
        bp = len(p_go['BP'] & t_go['BP'])
        mf = len(p_go['MF'] & t_go['MF'])
        cc = len(p_go['CC'] & t_go['CC'])
        rows.append((pert_gene, tgt_gene, label, split, bp, mf, cc, bp+mf+cc))
    return rows


def main():
    t0 = time.time()

    print("Loading GO DAG ...")
    godag = GODag(f"{DATA_DIR}/go/go-basic.obo", optional_attrs={'namespace'})

    print("Parsing GAF annotations ...")
    gene2go = build_gene2go(f"{DATA_DIR}/go/goa_human.gaf.gz", godag)
    print(f"  {len(gene2go)} genes with GO annotations")

    print("Loading K562.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/data/PerturbQA/K562.csv")
    print(f"  {len(df)} rows")

    groups = {}
    for _, row in df.iterrows():
        p = row['pert']
        if p not in groups:
            groups[p] = []
        groups[p].append((row['gene'], row['label'], row['split']))
    print(f"  {len(groups)} unique pert genes")

    args_list = [(pert, targets, gene2go) for pert, targets in groups.items()]

    print(f"Extracting GO features with {N_WORKERS} workers ...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_group, args_list)

    all_rows = [r for group in results for r in group]
    out = pd.DataFrame(all_rows, columns=[
        'pert', 'gene', 'label', 'split',
        'n_shared_go_bp', 'n_shared_go_mf',
        'n_shared_go_cc', 'n_shared_go_all'
    ])

    out_path = f"{FEAT_DIR}/go_features_K562.pkl"
    out.to_pickle(out_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Rows    : {len(out)}")
    print(f"  Saved to: {out_path}")
    print(f"  n_shared_go_all stats:\n{out['n_shared_go_all'].describe()}")
    zero_pct = 100 * (out['n_shared_go_all'] == 0).sum() / len(out)
    print(f"  Zero shared GO: {zero_pct:.1f}%")


if __name__ == '__main__':
    main()
