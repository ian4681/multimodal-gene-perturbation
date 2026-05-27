"""
Ablation study: compare 7 feature combinations on K562 test set.
Feature matrix X_K562.pkl layout (649 cols):
  0-4   : BFS (5)
  5-8   : GO  (4)
  9-648 : ESM-2 pert (320) + ESM-2 gene (320)
"""

import pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = "/path/to/your/project"          # ← set this to your project root
FEAT_DIR = f"{BASE_DIR}/features"
RES_DIR  = f"{BASE_DIR}/results"

IDX = {
    "BFS":      list(range(0,   5)),
    "GO":       list(range(5,   9)),
    "ESM2":     list(range(9,  649)),
    "BFS+GO":   list(range(0,   9)),
    "BFS+ESM2": list(range(0,   5)) + list(range(9, 649)),
    "GO+ESM2":  list(range(5,  649)),
    "All":      list(range(0,  649)),
}


def train_eval(X_tr, y_tr, X_te, y_te, name):
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                n_jobs=4, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    mf1  = f1_score(y_te, y_pred, average='macro')
    rep  = classification_report(y_te, y_pred,
                                 target_names=['Non-diff','Down','Up'],
                                 output_dict=True)
    down_r = rep['Down']['recall']
    up_r   = rep['Up']['recall']
    print(f"  {name:<12} Acc={acc:.4f}  MacroF1={mf1:.4f}  "
          f"Down-R={down_r:.4f}  Up-R={up_r:.4f}  ({time.time()-t0:.0f}s)")
    return acc, mf1, down_r, up_r


def main():
    print("Loading X_K562.pkl ...")
    t0 = time.time()
    data  = pickle.load(open(f"{FEAT_DIR}/X_K562.pkl", 'rb'))
    X     = data['X']
    y     = data['y']
    split = data['split']
    print(f"  Loaded in {time.time()-t0:.1f}s  shape={X.shape}")

    train_mask = (split == 'train')
    test_mask  = (split == 'test')
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]
    print(f"  Train={train_mask.sum()}  Test={test_mask.sum()}")

    print("\nRunning ablation ...")
    rows = []
    for name, cols in IDX.items():
        acc, mf1, down_r, up_r = train_eval(
            X_tr[:, cols], y_tr, X_te[:, cols], y_te, name)
        rows.append(dict(Combination=name, Dims=len(cols),
                         Accuracy=acc, MacroF1=mf1,
                         Down_Recall=down_r, Up_Recall=up_r))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    df.to_csv(f"{RES_DIR}/ablation_results.csv", index=False)
    print(f"\nSaved -> results/ablation_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [('MacroF1', 'Macro F1'), ('Down_Recall', 'Down Recall'),
               ('Accuracy', 'Accuracy')]
    colors  = ['#4C72B0','#DD8452','#55A868','#C44E52',
               '#8172B2','#937860','#DA8BC3']
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].values
        bars = ax.bar(df['Combination'], vals, color=colors, edgecolor='white')
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, min(vals.max() * 1.2, 1.0))
        ax.tick_params(axis='x', rotation=30)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('Ablation Study — K562 RF (100 trees, balanced)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/ablation_study.png", dpi=150)
    print("Saved -> results/ablation_study.png")


if __name__ == '__main__':
    main()
