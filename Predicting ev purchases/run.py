"""Run the base models over the saved folds, write their OOF/test predictions,
and print the comparison.

    python run.py                 # every model in models.NAMES
    python run.py lgbm cat        # a subset, by NAME

For each model:
  1. prepare(X, X_test, cat_cols)          once, before the fold loop
  2. for each (train_idx, val_idx) from folds.iter_splits():
       - carve a stratified inner holdout out of train_idx for early stopping,
         so the stopping iteration never sees the rows the fold is scored on
       - val_pred, test_pred_k, best_iter = model.fit_predict(...)
       - oof[val_idx] = val_pred ;  test_pred += test_pred_k / n_folds
  3. oof_utils.save_oof(NAME, oof, test_pred)

Then: per-fold AUC, min-fold AUC (the selection rule), OOF AUC, fold spread,
median best_iter, and the Spearman correlation matrix across the models' OOF
vectors (low pairs = the diversity worth blending).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import folds as folds_mod
from dataset import build
from models import NAMES, load
from oof_utils import save_oof

SEED = 42
INNER_ES_FRAC = 0.1          # fraction of each fold's train rows held out for early stopping


def _fold_aucs(y, pred, fold_vec):
    """Per-fold AUC of an OOF vector, fold 0..N-1."""
    return [roc_auc_score(y[fold_vec == k], pred[fold_vec == k])
            for k in range(folds_mod.N_SPLITS)]


def run_model(mod, X, X_test, y, cat_cols):
    """Full OOF loop for one model module. Returns (oof, test_pred, best_iters)."""
    Xp, Xtp = mod.prepare(X, X_test, cat_cols)
    y = np.asarray(y)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xtp))
    best_iters = []
    n_folds = folds_mod.N_SPLITS

    for k, (train_idx, val_idx) in enumerate(folds_mod.iter_splits()):
        # inner holdout for early stopping, taken only from this fold's train rows
        fit_idx, es_idx = train_test_split(
            train_idx, test_size=INNER_ES_FRAC,
            stratify=y[train_idx], random_state=SEED)

        val_pred, test_pred_k, best_iter = mod.fit_predict(
            Xp.iloc[fit_idx], y[fit_idx],
            Xp.iloc[es_idx], y[es_idx],
            Xp.iloc[val_idx], Xtp)

        oof[val_idx] = val_pred
        test_pred += test_pred_k / n_folds
        best_iters.append(best_iter)
        print(f"    fold {k}: AUC {roc_auc_score(y[val_idx], val_pred):.5f}  "
              f"best_iter {best_iter}")

    return oof, test_pred, best_iters


def analyse(results, y, fold_vec):
    """results: {name: (oof, best_iters)}. Print per-model AUCs + the OOF
    correlation matrix."""
    print("\n=== per-model (selection rule = min fold AUC) ===")
    print(f"  {'model':6s}  {'min':>8s}  {'mean':>8s}  {'OOF':>8s}  "
          f"{'spread':>7s}  {'med_iter':>8s}")
    for name, (oof, best_iters) in results.items():
        fa = _fold_aucs(y, oof, fold_vec)
        print(f"  {name:6s}  {min(fa):8.5f}  {np.mean(fa):8.5f}  "
              f"{roc_auc_score(y, oof):8.5f}  {max(fa) - min(fa):7.5f}  "
              f"{int(np.median(best_iters)):8d}")

    if len(results) > 1:
        print("\n=== Spearman correlation of OOF predictions ===")
        corr = pd.DataFrame({n: o for n, (o, _) in results.items()}).corr(
            method="spearman")
        print(corr.round(3))


def main(argv):
    names = argv or NAMES
    unknown = [n for n in names if n not in NAMES]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}   known: {NAMES}")
    # nn last: the GBDT libs must load before torch does, or a multi-threaded
    # torch segfaults next to their OpenMP runtime (see models/nn.py)
    names = sorted(names, key=lambda n: n == "nn")

    X, X_test, y, cat_cols = build()
    fold_vec = folds_mod.load()
    assert len(fold_vec) == len(y), "folds.npy is stale - rerun `python folds.py`"
    y_np = y.to_numpy()

    results = {}
    for name in names:
        mod = load(name)                              # import the submodule now
        print(f"\n=== {name} ===")
        oof, test_pred, best_iters = run_model(mod, X, X_test, y, cat_cols)
        save_oof(mod.NAME, oof, test_pred)
        results[mod.NAME] = (oof, best_iters)

    analyse(results, y_np, fold_vec)
    print("\ndone - now run `python blend.py`")


if __name__ == "__main__":
    main(sys.argv[1:])
