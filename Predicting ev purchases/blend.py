"""Blend the base-model OOF predictions and write the two submissions.

Reads every oof_*.csv / test_*.csv in ./oof (written by run.py), scored on the
saved folds (oof_utils.fold_splits). Reports each model alone and their rank
correlation, then tries three blends:

  1. equal      - equal-weight mean of the rank-transformed columns
  2. hillclimb  - non-negative weights, greedily added one at a time (Caruana)
  3. stacker    - logistic regression on the rank columns, scored OUT-OF-FOLD
                  so the weights never see the rows they're judged on

Every candidate (4 singles + 3 blends) is then ranked, and two submissions are
written from the pool:

  submission_private.csv  - highest MIN fold AUC   (safe pick for the private 80%)
  submission_public.csv   - highest pooled OOF AUC (aggressive pick for public 20%)

If the same candidate tops both, the public slot falls to the next-best distinct
one so the two submissions actually differ.

Run:  python blend.py   (after run.py has populated ./oof)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models import NAMES
from oof_utils import (EV_DIR, fold_splits, load_ids, load_matrix, min_fold_auc,
                       oof_auc, rank01)

CLIP = (1e-4, 1 - 1e-4)
HILLCLIMB_STEPS = 50

# restrict to the current pipeline's models so stale oof_*.csv from an older
# run don't get pulled into the blend
oof_df, test_df, y = load_matrix(NAMES)
names = list(oof_df.columns)
print(f"models: {names}   train rows: {len(oof_df)}   test rows: {len(test_df)}")

# rank-transform every column; test columns get ranked within the test set
R = np.column_stack([rank01(oof_df[n]) for n in names])
T = np.column_stack([rank01(test_df[n]) for n in names])

# --- each model on its own -------------------------------------------------
print("\nindividual   OOF AUC   min-fold AUC")
for i, n in enumerate(names):
    print(f"  {n:14s} {oof_auc(y, R[:, i]):.5f}   {min_fold_auc(y, R[:, i]):.5f}")

print("\nrank correlation (low pairs = the diversity worth blending)")
print(pd.DataFrame(R, columns=names).corr(method="spearman").round(3))

blends = {}   # name -> (oof_pred, test_pred)

# --- 1. equal-weight rank mean ------------------------------------------
blends["equal"] = (R.mean(axis=1), T.mean(axis=1))

# --- 2. greedy non-negative-weight hill-climb on the OOF matrix ---------
counts = np.zeros(len(names))
cur = np.zeros(len(y))                      # running mean of picked columns
for step in range(HILLCLIMB_STEPS):
    # score adding one more copy of each candidate, keep the best
    scored = [(min_fold_auc(y, (cur * step + R[:, j]) / (step + 1)), j)
              for j in range(len(names))]
    _, j = max(scored)
    counts[j] += 1
    cur = (cur * step + R[:, j]) / (step + 1)
w = counts / counts.sum()
print(f"\nhill-climb weights: {dict(zip(names, w.round(3)))}")
blends["hillclimb"] = (R @ w, T @ w)

# --- 3. logistic stacker, evaluated out-of-fold ------------------------
stack_oof = np.zeros(len(y))
for tr, va in fold_splits():
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(R[tr], y[tr])
    stack_oof[va] = lr.predict_proba(R[va])[:, 1]
lr_full = LogisticRegression(C=1.0, max_iter=1000).fit(R, y)   # for the test frame
blends["stacker"] = (stack_oof, lr_full.predict_proba(T)[:, 1])
print(f"stacker coefs: {dict(zip(names, lr_full.coef_[0].round(3)))}")

# --- pool the singles and the blends, score both ways -----------------
# candidate -> (oof_pred, test_pred)
candidates = {n: (R[:, i], T[:, i]) for i, n in enumerate(names)}
candidates.update(blends)

scored = {n: (min_fold_auc(y, oof_p), oof_auc(y, oof_p))
          for n, (oof_p, _) in candidates.items()}

print("\ncandidate      min-fold AUC   OOF AUC")
for n, (mf, oa) in sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True):
    print(f"  {n:12s} {mf:.5f}        {oa:.5f}")

# private = best floor; public = best pooled fit, forced distinct from private
private = max(scored, key=lambda n: scored[n][0])
public = max((n for n in scored if n != private), key=lambda n: scored[n][1])

# --- write both submissions ------------------------------------------
_, test_ids = load_ids()
base_rate = y.mean()
for slot, pick in (("private", private), ("public", public)):
    test_pred = np.clip(candidates[pick][1], *CLIP)
    out = EV_DIR / f"submission_{slot}.csv"
    pd.DataFrame({"id": test_ids, "Will_Buy_EV": test_pred}).to_csv(out, index=False)
    print(f"wrote {out}   pick={pick}  min-fold {scored[pick][0]:.5f}  "
          f"OOF {scored[pick][1]:.5f}  test mean {test_pred.mean():.4f} "
          f"(base rate {base_rate:.4f})")
