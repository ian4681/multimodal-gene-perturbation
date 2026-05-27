"""
Feature fusion, model training, and evaluation.

Merges BFS + GO + ESM-2 features into a 649-dim vector per (pert, target) pair.
Trains Random Forest and Logistic Regression classifiers on K562 data.
Outputs confusion matrices, feature importance plot, and model files.

Feature layout (649 dims):
  0-4   : BFS  (path_length, is_direct_neighbor, n_common_neighbors, degree_pert, degree_gene)
  5-8   : GO   (n_shared_go_bp, n_shared_go_mf, n_shared_go_cc, n_shared_go_all)
  9-328 : ESM-2 perturbed gene (320)
  329-648: ESM-2 target gene  (320)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os

BASE_DIR = "/path/to/your/project"          # ← set this to your project root

print("=" * 50)
print("Loading feature files...")
t0 = time.time()

bfs = pickle.load(open(f'{BASE_DIR}/features/bfs_features_K562.pkl', 'rb'))
go  = pickle.load(open(f'{BASE_DIR}/features/go_features_K562.pkl', 'rb'))
esm = pickle.load(open(f'{BASE_DIR}/features/esm2_embeddings.pkl', 'rb'))

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  BFS shape: {bfs.shape}")
print(f"  GO  shape: {go.shape}")
print(f"  ESM genes: {len(esm)}")

print("\nMerging BFS + GO...")
go_cols = ['pert', 'gene', 'n_shared_go_bp', 'n_shared_go_mf', 'n_shared_go_cc', 'n_shared_go_all']
df = bfs.merge(go[go_cols], on=['pert', 'gene'], how='inner')
print(f"  Merged shape: {df.shape}")

print("\nBuilding ESM-2 feature matrix...")
zero_vec = np.zeros(320, dtype=np.float32)
esm_pert = np.array([esm.get(g, zero_vec) for g in df['pert']], dtype=np.float32)
esm_gene = np.array([esm.get(g, zero_vec) for g in df['gene']], dtype=np.float32)
missing_pert = sum(1 for g in df['pert'] if g not in esm)
missing_gene = sum(1 for g in df['gene'] if g not in esm)
print(f"  pert missing from ESM-2: {missing_pert}")
print(f"  gene missing from ESM-2: {missing_gene}")

bfs_feat_cols = ['path_length', 'is_direct_neighbor', 'n_common_neighbors', 'degree_pert', 'degree_gene']
go_feat_cols  = ['n_shared_go_bp', 'n_shared_go_mf', 'n_shared_go_cc', 'n_shared_go_all']
tab = df[bfs_feat_cols + go_feat_cols].values.astype(np.float32)

X = np.hstack([tab, esm_pert, esm_gene])
y = df['label'].values

feature_names = (bfs_feat_cols + go_feat_cols +
                 [f'esm_pert_{i}' for i in range(320)] +
                 [f'esm_gene_{i}' for i in range(320)])

print(f"\nFinal feature matrix: {X.shape}")
label_counts = np.bincount(y)
print(f"Label distribution: Non-diff={label_counts[0]}, Down={label_counts[1]}, Up={label_counts[2]}")

os.makedirs(f'{BASE_DIR}/features', exist_ok=True)
with open(f'{BASE_DIR}/features/X_K562.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y, 'feature_names': feature_names,
                 'split': df['split'].values, 'pert': df['pert'].values,
                 'gene': df['gene'].values}, f)
print(f"  Saved merged matrix -> features/X_K562.pkl")

train_mask = (df['split'] == 'train').values
test_mask  = (df['split'] == 'test').values
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

os.makedirs(f'{BASE_DIR}/results', exist_ok=True)


def evaluate_and_plot(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    cm  = confusion_matrix(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=['Non-diff', 'Down', 'Up'], digits=4))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-diff', 'Down', 'Up'],
                yticklabels=['Non-diff', 'Down', 'Up'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name}\nAcc={acc:.4f}  Macro F1={f1:.4f}')
    plt.tight_layout()
    safe_name = name.replace(' ', '_')
    plt.savefig(f'{BASE_DIR}/results/cm_{safe_name}.png', dpi=150)
    plt.close()
    return acc, f1


# ── Random Forest ──────────────────────────────────────────
print("\nTraining Random Forest (100 trees, n_jobs=4)...")
t1 = time.time()
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                             n_jobs=4, random_state=42, max_features='sqrt')
rf.fit(X_train, y_train)
print(f"  Done in {time.time()-t1:.1f}s")

rf_acc, rf_f1 = evaluate_and_plot('Random Forest', rf, X_test, y_test)

importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(20), importances[top_idx])
ax.set_xticks(range(20))
ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Importance')
ax.set_title('Random Forest — Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/results/feature_importance_RF.png', dpi=150)
plt.close()

print("\n  Top 10 features:")
for i in top_idx[:10]:
    print(f"    {feature_names[i]:<30s}  {importances[i]:.5f}")

joblib.dump(rf, f'{BASE_DIR}/results/rf_model.pkl')


# ── Logistic Regression ────────────────────────────────────
print("\nTraining Logistic Regression (saga, max_iter=500)...")
t2 = time.time()
lr = LogisticRegression(max_iter=500, class_weight='balanced',
                         solver='saga', n_jobs=4, random_state=42)
lr.fit(X_train, y_train)
print(f"  Done in {time.time()-t2:.1f}s")

lr_acc, lr_f1 = evaluate_and_plot('Logistic_Regression', lr, X_test, y_test)
joblib.dump(lr, f'{BASE_DIR}/results/lr_model.pkl')


summary = f"""
========== FINAL SUMMARY ==========
  Samples   : train={X_train.shape[0]}, test={X_test.shape[0]}
  Features  : {X.shape[1]} (BFS=5, GO=4, ESM-2=640)

  Random Forest     Acc={rf_acc:.4f}  Macro F1={rf_f1:.4f}
  Logistic Reg.     Acc={lr_acc:.4f}  Macro F1={lr_f1:.4f}
====================================
"""
print(summary)
with open(f'{BASE_DIR}/results/summary.txt', 'w') as f:
    f.write(summary)

print("Done.")
