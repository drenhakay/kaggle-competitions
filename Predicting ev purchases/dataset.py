"""Model-agnostic data prep for the EV-purchase task.

One entry point, `build()`, that does everything every model shares:
cleaning + the Buy_Score reconstruction (preprocessing.load_n_clean) and the
digit-decomposition block (features_ev). It deliberately does NOT one-hot /
scale / cast categoricals - that encoding differs per model and lives in each
models/<name>.py `prepare()`.
"""
import pandas as pd

from features_ev import digit_features, load_raw, prune_constant_and_dup
from preprocessing import load_n_clean

# object-valued categoricals; each model's prepare() casts these its own way.
# (Number_of_Cars_Owned / Environmental_Concern_Level are numeric, not here.)
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Range_Anxiety_Level"]

# names of the generated digit-decomposition columns; filled in by build().
# The NN drops these and keeps only the base frame.
DIGIT_FEATURE_COLS: list[str] = []


def freeze_categoricals(X, X_test, cat_cols):
    """Cast cat_cols to pandas 'category' with categories frozen on the
    train+test union, so both frames encode to identical codes. Shared by the
    LightGBM and XGBoost models (CatBoost takes object columns instead)."""
    X, X_test = X.copy(), X_test.copy()
    for c in cat_cols:
        dtype = pd.CategoricalDtype(
            sorted(set(X[c]) | set(X_test[c])), ordered=False)
        X[c] = X[c].astype(dtype)
        X_test[c] = X_test[c].astype(dtype)
    return X, X_test


def build():
    """Return (X, X_test, y, cat_cols).

    X / X_test: DataFrames, RangeIndex reset, categoricals still object/string,
                digit block appended, identical column set on both frames.
    y:          int Series (1 = Will_Buy_EV), aligned to X's rows.
    cat_cols:   dataset.CAT_COLS (all four survive cleaning).
    """
    global DIGIT_FEATURE_COLS

    Xtr, Xte, ytr, _ = load_n_clean()                 # cleaning + Buy_Score
    raw_tr, raw_te = load_raw()                       # untouched csv values

    # per-digit decomposition of Age / Income / Commute; prune decides the
    # surviving columns from TRAIN only, then applies that set to both frames
    dig_tr, dig_te = prune_constant_and_dup(
        digit_features(raw_tr), digit_features(raw_te))
    DIGIT_FEATURE_COLS = list(dig_tr.columns)

    # load_raw and load_n_clean both read the csvs in natural row order, so a
    # plain column concat lines the digit block up with the cleaned frame
    X = pd.concat([Xtr.reset_index(drop=True),
                   dig_tr.reset_index(drop=True)], axis=1)
    X_test = pd.concat([Xte.reset_index(drop=True),
                        dig_te.reset_index(drop=True)], axis=1)
    y = ytr.reset_index(drop=True).astype(int)

    assert list(X.columns) == list(X_test.columns)
    assert len(X) == len(y)
    return X, X_test, y, list(CAT_COLS)
