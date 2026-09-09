import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
from pathlib import Path

# train.py + preprocessing.py both live alongside this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import cross_validate, fit, make_loaders, predict
from preprocessing import load_n_clean, make_preprocessor

Xtr, Xte, ytr, test_ids = load_n_clean()
n_features = make_preprocessor().fit_transform(Xtr, ytr).shape[1]

def make_model():
    model = nn.Sequential(
        nn.BatchNorm1d(n_features),
        nn.Linear(n_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1)
    )
    return model

def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

loss_fn = nn.BCEWithLogitsLoss()

SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv = cross_validate(
    make_model, make_optimizer, Xtr, ytr,
    loss_fn=loss_fn,
    splitter=SPLITTER,
    X_test=Xte,
    make_preprocessor=make_preprocessor,
    epochs=50,
    patience=5,
    verbose=0,
)

# --- assess -----------------------------------------------------------------
y = ytr.to_numpy()
oof_logit = cv["oof"]                 # raw logits; AUC is invariant to sigmoid
oof_prob = expit(oof_logit)

# per-fold AUC on the same splits cross_validate used
fold_auc = [roc_auc_score(y[va], oof_logit[va])
            for _, va in SPLITTER.split(np.zeros(len(y)), y)]

print("\n" + "=" * 50)
print("CV assessment")
print(f"  OOF AUC        : {roc_auc_score(y, oof_logit):.5f}")
print(f"  OOF log loss   : {log_loss(y, oof_prob):.5f}")
print(f"  fold AUC       : {[round(a, 5) for a in fold_auc]}")
print(f"    min {min(fold_auc):.5f}   mean {np.mean(fold_auc):.5f}   "
      f"spread {max(fold_auc) - min(fold_auc):.5f}")
print(f"  fold BCE loss  : {[round(s, 4) for s in cv['scores']]}")

# sanity checks
print(f"\n  oof rows       : {len(oof_logit)}   NaN: {int(np.isnan(oof_logit).sum())}")
test_prob = expit(cv["test_preds"])
print(f"  test prob      : min {test_prob.min():.4f}  max {test_prob.max():.4f}  "
      f"mean {test_prob.mean():.4f}")
print(f"  train base rate: {y.mean():.4f}   (test mean should be near this)")
print("=" * 50)

# --- submission -----------------------------------------------------------
# test_prob = sigmoid(mean of the 5 fold models' logits); clip off the exact
# 0/1 tails (harmless for AUC, saves you on any log-loss metric)
sub = pd.DataFrame({
    "id": test_ids.to_numpy(),
    "Will_Buy_EV": np.clip(test_prob, 1e-4, 1 - 1e-4),
})
sub.to_csv("submission_nn.csv", index=False)
print(f"wrote submission_nn.csv   {len(sub)} rows   "
      f"mean {sub['Will_Buy_EV'].mean():.4f}")
print("=" * 50)

