"""Engineered features for the EV-purchase task.

These are all deterministic functions of the given inputs (no target signal).
The latent scores reconstruct the continuous quantities the two quantised
columns (Range_Anxiety_Level, and the buying rule) were derived from, so they
have to be built from the RAW csv values before preprocessing.py clips the
charging-station counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

_DIR = Path(__file__).resolve().parent
RAW_TRAIN = _DIR / "train.csv"
RAW_TEST = _DIR / "test.csv"

DIGIT_COLS = ("Age", "Annual_Income_USD", "Daily_Commute_km")
DIGIT_K = range(-4, 4)


def load_raw():
    """Raw train/test frames, id dropped, row order matches preprocessing.load_n_clean."""
    tr = pd.read_csv(RAW_TRAIN).drop(columns="id")
    te = pd.read_csv(RAW_TEST).drop(columns="id")
    return tr, te


def latent_features(df):
    """worry_score and buy_score reconstructions (see module docstring)."""
    home = (df["Home_Charging_Possible"] == "Yes").astype(float)
    sub = (df["Subsidy_Available"] == "Yes").astype(float)
    anx = df["Range_Anxiety_Level"]

    # anxiety latent: commute distance penalised by nearby charging and a home
    # charger; its 3 buckets are cut from this + noise
    worry = (df["Daily_Commute_km"]
             - 5 * df["Charging_Stations_Near_Home"]
             - 5 * df["Charging_Stations_Near_Work"]
             - 150 * home)

    # the generator's buying rule, using the quantised anxiety column
    buy = (1.2 * (df["Annual_Income_USD"] / 1e5)
           + 0.6 * df["Environmental_Concern_Level"]
           + 2.0 * sub
           - 1.0 * (anx == "Medium")
           - 3.0 * (anx == "High"))

    # same rule but the anxiety penalty is a smooth function of the reconstructed
    # latent rather than its 3 buckets. Bucket cuts sit near -25 / +75, noise
    # sigma ~50, so expit(.) approximates P(Medium)/P(High)
    smooth_pen = -(1.0 * expit((worry + 25) / 50.0)
                   + 2.0 * expit((worry - 75) / 50.0))
    buy_smooth = (1.2 * (df["Annual_Income_USD"] / 1e5)
                  + 0.6 * df["Environmental_Concern_Level"]
                  + 2.0 * sub + smooth_pen)

    return pd.DataFrame(
        {"worry_score": worry.astype("float32"),
         "buy_score": buy.astype("float32"),
         "buy_score_smooth": buy_smooth.astype("float32")},
        index=df.index)


def digit_features(df):
    """Per-digit decomposition of the numeric columns."""
    out = {}
    for c in DIGIT_COLS:
        x = df[c].to_numpy(dtype="float64")
        for k in DIGIT_K:
            out[f"{c}_d{k}"] = (np.floor(x / (10.0 ** k)) % 10).astype("int16")
    return pd.DataFrame(out, index=df.index)


def prune_constant_and_dup(tr_df, te_df):
    """Drop columns constant on train, then exact-duplicate columns, applying the
    same surviving column set to both frames."""
    keep = [c for c in tr_df.columns if tr_df[c].nunique() > 1]
    tr_df = tr_df[keep]
    seen, uniq = {}, []
    for c in tr_df.columns:
        # fingerprint the column's train values to spot exact duplicates
        key = hash(pd.util.hash_pandas_object(tr_df[c], index=False).values.tobytes())
        if key not in seen:
            seen[key] = c
            uniq.append(c)
    return tr_df[uniq], te_df[uniq]
