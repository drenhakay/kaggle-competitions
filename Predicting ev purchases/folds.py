"""Generate the CV fold assignment ONCE and save it, so every model in
models/ scores on byte-identical splits.

Run:  python folds.py     ->  writes folds.npy (int8, length = n_train)

Splitter stays StratifiedKFold(5, shuffle=True, random_state=42) - the same
one baseline.py and the tuning scripts used, so old fold numbers still compare.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dataset import build

_DIR = Path(__file__).resolve().parent
FOLDS_PATH = _DIR / "folds.npy"
N_SPLITS = 5
SEED = 42


def generate():
    """Build the length-n_train fold-index vector from dataset.build()'s y and
    write it to FOLDS_PATH. Value at row i = the fold row i is the val set for."""
    _, _, y, _ = build()
    y = y.to_numpy()
    folds = np.full(len(y), -1, dtype="int8")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = k
    assert (folds >= 0).all()                        # every row assigned once
    np.save(FOLDS_PATH, folds)
    counts = np.bincount(folds)
    print(f"wrote {FOLDS_PATH.name}  n={len(folds)}  fold sizes {counts.tolist()}")


def load():
    """Return the saved fold-index vector. Raise if folds.npy is missing so a
    model can never silently run on freshly-drawn splits."""
    if not FOLDS_PATH.exists():
        raise FileNotFoundError("folds.npy missing - run `python folds.py` first")
    return np.load(FOLDS_PATH)


def iter_splits():
    """Yield (train_idx, val_idx) arrays from the saved vector, fold 0..N_SPLITS-1."""
    folds = load()
    for k in range(N_SPLITS):
        val = np.flatnonzero(folds == k)
        train = np.flatnonzero(folds != k)
        yield train, val


if __name__ == "__main__":
    generate()
