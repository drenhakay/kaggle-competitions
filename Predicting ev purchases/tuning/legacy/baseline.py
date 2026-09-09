"""Baseline yardsticks for the EV-purchase NN.

LightGBM and logistic regression, scored on the SAME StratifiedKFold splits and
the SAME cleaned features as model.py, so the NN's OOF AUC has something honest
to be compared against. Also writes a LightGBM submission — worth having a
baseline on the leaderboard.

Run:  python baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import load_n_clean, make_preprocessor

Xtr, Xte, ytr, test_ids = load_n_clean()
y = ytr.to_numpy()

# identical to model.py so the folds line up exactly
SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def assess(name, oof_prob):
    fold_auc = [roc_auc_score(y[va], oof_prob[va])
                for _, va in SPLITTER.split(np.zeros(len(y)), y)]
    print(f"\n{name}")
    print(f"  OOF AUC       : {roc_auc_score(y, oof_prob):.5f}")
    print(f"  OOF log loss  : {log_loss(y, oof_prob):.5f}")
    print(f"  fold AUC      : {[round(a, 5) for a in fold_auc]}")
    print(f"    min {min(fold_auc):.5f}   mean {np.mean(fold_auc):.5f}   "
          f"spread {max(fold_auc) - min(fold_auc):.5f}")


# both models get the same preprocessing pipeline as the NN. Scaling / power
# transform is harmless for the trees and required for logistic regression, so
# the comparison stays apples-to-apples. cross_val_predict refits the whole
# pipeline per fold, so nothing leaks across the split.
lgbm_pipe = Pipeline([
    ("prep", make_preprocessor()),
    ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbose=-1)),
])
lr_pipe = Pipeline([
    ("prep", make_preprocessor()),
    ("clf", LogisticRegression(max_iter=1000)),
])

oof_lgbm = cross_val_predict(lgbm_pipe, Xtr, y, cv=SPLITTER,
                             method="predict_proba", n_jobs=1)[:, 1]
assess("LightGBM", oof_lgbm)

oof_lr = cross_val_predict(lr_pipe, Xtr, y, cv=SPLITTER,
                           method="predict_proba", n_jobs=1)[:, 1]
assess("LogisticRegression", oof_lr)

# --- leaderboard baseline from the stronger model, refit on all rows ---
lgbm_pipe.fit(Xtr, y)
test_prob = np.clip(lgbm_pipe.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
sub = pd.DataFrame({"id": test_ids.to_numpy(), "Will_Buy_EV": test_prob})  # check the
sub.to_csv("submission_baseline_lgbm.csv", index=False)                     # sample file
print(f"\nwrote submission_baseline_lgbm.csv   test mean {test_prob.mean():.4f}   "
      f"(train base rate {y.mean():.4f})")
