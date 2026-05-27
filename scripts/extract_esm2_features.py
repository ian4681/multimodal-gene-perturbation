"""
ESM-2 8M batch inference for all genes in the knowledge graph.

Steps:
  1. Load gene list from gene_graph.pth (node names = gene symbols)
  2. Parse UniProt human FASTA, build gene_symbol → sequence mapping
  3. Batch inference with ESM-2 8M (frozen, CPU), mean-pool last_hidden_state
  4. Save {gene_symbol: embedding_vector} as esm2_embeddings.pkl

Output embedding dim: 320 (ESM-2 8M hidden size)
"""

import os
import gzip
import time
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

BASE_DIR  = "/path/to/your/project"          # ← set this to your project root
DATA_DIR  = f"{BASE_DIR}/raw_data"
FEAT_DIR  = f"{BASE_DIR}/features"
MODEL_DIR = f"{BASE_DIR}/models/esm2_8M"     # local copy of facebook/esm2_t6_8M_UR50D
FASTA_GZ  = f"{DATA_DIR}/uniprot_human.fasta.gz"
BATCH_SIZE = 32
MAX_LEN    = 512   # truncate very long sequences

os.makedirs(FEAT_DIR, exist_ok=True)


def parse_uniprot_fasta(fasta_gz):
    """Parse UniProt FASTA → {gene_symbol: sequence}. Keep longest per gene."""
    gene2seq = {}
    cur_genes, cur_seq = [], []

    opener = gzip.open if fasta_gz.endswith('.gz') else open
    with opener(fasta_gz, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur_genes and cur_seq:
                    seq = ''.join(cur_seq)
                    for g in cur_genes:
                        if g not in gene2seq or len(seq) > len(gene2seq[g]):
                            gene2seq[g] = seq
                cur_seq = []
                cur_genes = []
                if 'GN=' in line:
                    gn_part = line.split('GN=')[1].split()[0]
                    cur_genes = [gn_part]
            else:
                cur_seq.append(line)

    if cur_genes and cur_seq:
        seq = ''.join(cur_seq)
        for g in cur_genes:
            if g not in gene2seq or len(seq) > len(gene2seq[g]):
                gene2seq[g] = seq

    return gene2seq


def main():
    t0 = time.time()

    print("Loading gene list from graph ...")
    G = torch.load(f"{DATA_DIR}/data/Knowledge_Graph/gene_graph.pth",
                   map_location="cpu", weights_only=False)
    gene_list = list(G.nodes())
    print(f"  {len(gene_list)} genes in graph")

    print("Parsing UniProt FASTA ...")
    gene2seq = parse_uniprot_fasta(FASTA_GZ)
    print(f"  {len(gene2seq)} genes with sequences")

    matched = [(g, gene2seq[g]) for g in gene_list if g in gene2seq]
    missing = [g for g in gene_list if g not in gene2seq]
    print(f"  Matched: {len(matched)}, Missing: {len(missing)} ({100*len(missing)/len(gene_list):.1f}%)")

    print("Loading ESM-2 8M model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model     = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    print("  Model loaded.")

    print(f"Running batch inference (batch_size={BATCH_SIZE}) ...")
    embeddings = {}

    for i in range(0, len(matched), BATCH_SIZE):
        batch = matched[i:i+BATCH_SIZE]
        genes, seqs = zip(*batch)
        seqs_trunc = [s[:MAX_LEN] for s in seqs]

        inputs = tokenizer(list(seqs_trunc), return_tensors="pt",
                           padding=True, truncation=True, max_length=MAX_LEN+2)
        with torch.no_grad():
            out = model(**inputs)

        # Mean-pool over residues (exclude [CLS] and [EOS] tokens)
        hidden = out.last_hidden_state[:, 1:-1, :]
        mask   = inputs['attention_mask'][:, 1:-1].unsqueeze(-1).float()
        emb    = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        for gene, vec in zip(genes, emb):
            embeddings[gene] = vec.numpy()

        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] {i+len(batch)}/{len(matched)} sequences processed")

    zero = np.zeros(320, dtype=np.float32)
    for g in missing:
        embeddings[g] = zero

    out_path = f"{FEAT_DIR}/esm2_embeddings.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Saved to        : {out_path}")
    print(f"  Missing (zero)  : {len(missing)}")


if __name__ == '__main__':
    main()
