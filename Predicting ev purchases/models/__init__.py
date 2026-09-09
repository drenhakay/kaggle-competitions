"""Base models for the EV-purchase blend.

NOTE: LightGBM / XGBoost / CatBoost and PyTorch each vendor their own OpenMP
runtime; loading several into one process segfaults on macOS. Allowing the
duplicate load (here) + torch pinning to one thread when a GBDT lib is already
imported (nn.py) is what keeps the NN from crashing alongside the tree models.
Submodules are imported lazily via load() so `run.py nn` pulls in only torch -
no GBDT OpenMP - and the NN keeps every core.


Every module here exposes the same three names so run.py can treat them
uniformly:

    NAME : str
        short id used for oof_<NAME>.csv / test_<NAME>.csv

    prepare(X, X_test, cat_cols) -> (X, X_test)
        apply THIS model's encoding (category dtype / object / dense matrix).
        Called once, before the fold loop. Must not look at y.

    fit_predict(X_fit, y_fit, X_es, y_es, X_val, X_test) -> (val_pred, test_pred, best_iter)
        train on (X_fit, y_fit), early-stop on (X_es, y_es), then predict
        probabilities for X_val and X_test. No fold logic in here — run.py
        owns the split and the inner early-stopping holdout.
"""
import importlib
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # see module docstring

# public NAME -> submodule file stem
_MODULES = {"lgbm": "lgbm", "xgb": "xgboost", "cat": "catboost", "nn": "nn"}

# default set for `python run.py` with no args. CatBoost is left out - it tracks
# lgbm/xgb almost exactly, barely moves the blend, and roughly doubles the run
# time. Still runnable by name: `python run.py cat`.
NAMES = ["lgbm", "xgb", "nn"]


def load(name):
    """Import and return one model submodule by its NAME."""
    if name not in _MODULES:
        raise KeyError(f"unknown model {name!r}; known: {NAMES}")
    return importlib.import_module(f"{__name__}.{_MODULES[name]}")
