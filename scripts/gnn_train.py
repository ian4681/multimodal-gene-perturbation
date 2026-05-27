"""
GNN (2-layer GAT) for cancer perturbation prediction.
Architecture:
  PPI graph (18479 nodes, ESM-2 320-dim node features)
  -> GAT layer 1: 320 -> 128 (4 heads x 32)
  -> GAT layer 2: 128 -> 64 (1 head)
  -> For each (pert, gene) pair:
      concat(pert_emb[64], gene_emb[64], GO_feat[4]) -> MLP -> 3 classes

RF baseline for comparison: Acc=0.9141, Macro F1=0.6021
"""

import os, time, pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR  = "/path/to/your/project"          # <- set this to your project root
RES_DIR   = f"{BASE_DIR}/results"
FEAT_DIR  = f"{BASE_DIR}/features"
DATA_DIR  = f"{BASE_DIR}/raw_data/data"

EMBED_DIM  = 64
HIDDEN_DIM = 128
HEADS      = 4
DROPOUT    = 0.3
LR         = 0.001
WD         = 1e-4
EPOCHS     = 150
PATIENCE   = 20

print("Loading PPI graph ...", flush=True)
G = torch.load(f"{DATA_DIR}/Knowledge_Graph/gene_graph.pth",
               map_location='cpu', weights_only=False)
nodes    = list(G.nodes())
node2idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
print(f"  {N} nodes, {G.number_of_edges()} edges", flush=True)

src, dst = zip(*G.edges())
src_i = [node2idx[u] for u in src]
dst_i = [node2idx[v] for v in dst]
edge_index = torch.tensor([src_i + dst_i, dst_i + src_i], dtype=torch.long)

print("Building node feature matrix ...", flush=True)
esm      = pickle.load(open(f"{FEAT_DIR}/esm2_embeddings.pkl", 'rb'))
zero_vec = np.zeros(320, dtype=np.float32)
x_np     = np.array([esm.get(n, zero_vec) for n in nodes], dtype=np.float32)
x        = torch.from_numpy(x_np)
print(f"  Node features: {x.shape}", flush=True)

data = Data(x=x, edge_index=edge_index)

print("Loading K562 data + GO features ...", flush=True)
df     = pd.read_csv(f"{DATA_DIR}/PerturbQA/K562.csv")
go_df  = pickle.load(open(f"{FEAT_DIR}/go_features_K562.pkl", 'rb'))
GO_COLS = ['n_shared_go_bp','n_shared_go_mf','n_shared_go_cc','n_shared_go_all']
merged = df.merge(go_df[['pert','gene'] + GO_COLS], on=['pert','gene'], how='inner')
print(f"  Merged rows: {len(merged)}", flush=True)

pert_idx = torch.tensor([node2idx.get(p, 0) for p in merged['pert']], dtype=torch.long)
gene_idx = torch.tensor([node2idx.get(g, 0) for g in merged['gene']], dtype=torch.long)
go_feat  = torch.tensor(merged[GO_COLS].values, dtype=torch.float)
labels   = torch.tensor(merged['label'].values, dtype=torch.long)
split    = merged['split'].values

train_m = split == 'train'
test_m  = split == 'test'

tr_pi, tr_gi, tr_go, tr_y = pert_idx[train_m], gene_idx[train_m], go_feat[train_m], labels[train_m]
te_pi, te_gi, te_go, te_y = pert_idx[test_m],  gene_idx[test_m],  go_feat[test_m],  labels[test_m]
print(f"  Train: {tr_y.shape[0]}  Test: {te_y.shape[0]}", flush=True)

counts  = np.bincount(labels.numpy())
weights = torch.tensor(len(counts) / (counts * len(counts)), dtype=torch.float)
print(f"  Class weights: {weights.numpy().round(3)}", flush=True)


class GATClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(320, HIDDEN_DIM // HEADS, heads=HEADS, dropout=DROPOUT)
        self.gat2 = GATConv(HIDDEN_DIM, EMBED_DIM, heads=1, concat=False, dropout=DROPOUT)
        mlp_in = EMBED_DIM * 2 + len(GO_COLS)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def encode(self, data):
        h = F.elu(self.gat1(data.x, data.edge_index))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.gat2(h, data.edge_index)

    def forward(self, node_emb, p_idx, g_idx, go):
        h = torch.cat([node_emb[p_idx], node_emb[g_idx], go], dim=1)
        return self.mlp(h)


model     = GATClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
n_params  = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}", flush=True)

print(f"\n{'='*60}")
print(f"Training: {EPOCHS} epochs, patience={PATIENCE}")
print(f"{'='*60}\n")

best_f1, best_epoch, patience_cnt = 0.0, 0, 0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    node_emb = model.encode(data)
    logits   = model(node_emb, tr_pi, tr_gi, tr_go)
    loss     = criterion(logits, tr_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            ne    = model.encode(data)
            preds = model(ne, te_pi, te_gi, te_go).argmax(dim=1).numpy()
        acc = accuracy_score(te_y.numpy(), preds)
        f1  = f1_score(te_y.numpy(), preds, average='macro')
        print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | "
              f"Acc={acc:.4f} | MacroF1={f1:.4f} | "
              f"{time.time()-t_start:.0f}s elapsed", flush=True)

        if f1 > best_f1:
            best_f1, best_epoch, patience_cnt = f1, epoch, 0
            torch.save(model.state_dict(), f"{RES_DIR}/gnn_model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE // 10:
                print(f"Early stopping (best F1={best_f1:.4f} at epoch {best_epoch})", flush=True)
                break

print(f"\nLoading best model (epoch {best_epoch}, F1={best_f1:.4f}) ...", flush=True)
model.load_state_dict(torch.load(f"{RES_DIR}/gnn_model.pt"))
model.eval()
with torch.no_grad():
    ne    = model.encode(data)
    preds = model(ne, te_pi, te_gi, te_go).argmax(dim=1).numpy()

acc = accuracy_score(te_y.numpy(), preds)
f1  = f1_score(te_y.numpy(), preds, average='macro')
report = classification_report(te_y.numpy(), preds,
                                target_names=['Non-diff','Down','Up'], digits=4)

print(f"\n{'='*60}")
print("GNN (2-layer GAT) FINAL RESULTS -- K562 test set")
print(f"{'='*60}")
print(f"  Accuracy : {acc:.4f}   [RF: 0.9141]")
print(f"  Macro F1 : {f1:.4f}   [RF: 0.6021]")
print(report)
print(f"  Total time: {(time.time()-t_start)/60:.1f} min")

os.makedirs(RES_DIR, exist_ok=True)
with open(f"{RES_DIR}/gnn_summary.txt", 'w') as fh:
    fh.write("GNN (2-layer GAT) Results\n\n")
    fh.write(f"Accuracy : {acc:.4f}  [RF baseline: 0.9141]\n")
    fh.write(f"Macro F1 : {f1:.4f}  [RF baseline: 0.6021]\n")
    fh.write(f"Best epoch: {best_epoch}\n\n")
    fh.write(report)
print("Saved -> results/gnn_summary.txt", flush=True)
