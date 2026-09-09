"""Shared plumbing for the model blend: fold access, artifact IO, scorers.

Every base model writes its out-of-fold train predictions and its test
predictions into ./oof through save_oof(). Scoring here goes through
fold_splits(), which reads the same folds.npy run.py trained on, so blend.py
can score any combination of them honestly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

EV_DIR = Path(__file__).resolve().parent   # project root: dataset, preprocessing, features_ev, train, csvs
OOF_DIR = EV_DIR / "oof"                    # artifact store
OOF_DIR.mkdir(exist_ok=True)

# project root on the path so sibling modules import from any working dir
if str(EV_DIR) not in sys.path:
    sys.path.insert(0, str(EV_DIR))


def fold_splits():
    """Yield (train_idx, val_idx) per fold from folds.npy - the exact split
    run.py trained on, so blend scoring can never drift from training."""
    from folds import load

    fv = load()
    for k in range(int(fv.max()) + 1):
        yield np.flatnonzero(fv != k), np.flatnonzero(fv == k)


def load_ids():
    """(train_ids, test_ids) in each csv's natural row order."""
    tr = pd.read_csv(EV_DIR / "train.csv", usecols=["id"])["id"].to_numpy()
    te = pd.read_csv(EV_DIR / "test.csv", usecols=["id"])["id"].to_numpy()
    return tr, te


def save_oof(name, oof_pred, test_pred):
    """Persist one model's OOF (train-row) and test probabilities as csv."""
    tr_ids, te_ids = load_ids()
    oof_pred = np.asarray(oof_pred, dtype="float64")
    test_pred = np.asarray(test_pred, dtype="float64")
    # guard against a half-filled OOF vector or a misaligned frame
    assert len(oof_pred) == len(tr_ids) and not np.isnan(oof_pred).any()
    assert len(test_pred) == len(te_ids)
    pd.DataFrame({"id": tr_ids, "pred": oof_pred}).to_csv(
        OOF_DIR / f"oof_{name}.csv", index=False)
    pd.DataFrame({"id": te_ids, "pred": test_pred}).to_csv(
        OOF_DIR / f"test_{name}.csv", index=False)
    print(f"saved oof_{name}.csv / test_{name}.csv  "
          f"(oof mean {oof_pred.mean():.4f}, test mean {test_pred.mean():.4f})")


def load_matrix(names=None):
    """Assemble (oof_df, test_df, y). Columns are model names; rows are aligned
    on id to the training csv order (y's order). `names` restricts to a known
    model set (pass models.NAMES) so stale artifacts from an old run are ignored;
    None picks up every oof_*.csv present."""
    from preprocessing import load_n_clean

    _, _, ytr, _ = load_n_clean()
    y = ytr.to_numpy()
    tr_ids, te_ids = load_ids()

    if names is None:
        oof_files = sorted(OOF_DIR.glob("oof_*.csv"))
        if not oof_files:
            raise SystemExit("no oof_*.csv artifacts in ./oof - run `python run.py` first")
        names = [f.stem[len("oof_"):] for f in oof_files]
    else:
        missing = [n for n in names if not (OOF_DIR / f"oof_{n}.csv").exists()
                   or not (OOF_DIR / f"test_{n}.csv").exists()]
        if missing:
            raise SystemExit(f"missing oof artifacts for {missing} - run `python run.py`")

    def _col(prefix, n):
        return pd.read_csv(OOF_DIR / f"{prefix}_{n}.csv").set_index("id")["pred"].rename(n)

    oof = pd.concat([_col("oof", n) for n in names], axis=1).reindex(tr_ids)
    test = pd.concat([_col("test", n) for n in names], axis=1).reindex(te_ids)
    assert len(oof) == len(y) and not oof.isna().any().any(), "OOF matrix has gaps"
    assert not test.isna().any().any(), "test matrix has gaps"
    return oof, test, y


def rank01(a):
    """Rank-transform to (0, 1]. AUC depends only on order, and this puts models
    with different output scales (NN sigmoid vs GBDT) on a common footing before
    they're averaged."""
    a = np.asarray(a, dtype="float64")
    return rankdata(a) / len(a)


def min_fold_auc(y, p):
    """Lowest per-fold AUC on the saved folds - what we select models on."""
    return min(roc_auc_score(y[va], p[va]) for _, va in fold_splits())


def oof_auc(y, p):
    return roc_auc_score(y, p)
