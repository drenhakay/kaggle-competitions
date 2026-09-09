"""Feature-engineering comparison for the EV-purchase task.

Fixed LightGBM config (no fresh Optuna). Adds engineered feature blocks one at a
time on top of the cleaned baseline frame and scores each on the same folds as
baseline.py, so we can see which block earns its place. A pure-noise column is
included as a reference for how much OOF AUC drifts from adding a useless column.

Selection is on minimum fold AUC. Run:  python3 tune_gbm_fe.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from preprocessing import load_n_clean  # noqa: E402

import lightgbm as lgb  # noqa: E402
from features_ev import (  # noqa: E402
    load_raw, latent_features, digit_features, prune_constant_and_dup)

SEED = 42
WALL_BUDGET_S = 1500
MAX_ROUNDS = 4000
ES_ROUNDS = 200
CLIP = (1e-4, 1 - 1e-4)
BEST_SO_FAR = 0.94323          # OOF AUC of the tuned-LGBM run to beat
START = time.time()

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Range_Anxiety_Level"]
SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# fixed, sensible config informed by the earlier search (shallow-ish trees,
# raised bin resolution, low learning rate)
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, max_bin=511, num_leaves=31,
    max_depth=6, min_child_samples=100, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=5.0,
)

# --- assemble the candidate feature frames --------------------------------
Xtr_base, Xte_base, ytr, test_ids = load_n_clean()
y = ytr.to_numpy()
raw_tr, raw_te = load_raw()

for c in CAT_COLS:
    cats = pd.CategoricalDtype(
        sorted(set(Xtr_base[c]) | set(Xte_base[c])), ordered=False)
    Xtr_base[c] = Xtr_base[c].astype(cats)
    Xte_base[c] = Xte_base[c].astype(cats)

lat_tr, lat_te = latent_features(raw_tr), latent_features(raw_te)
dig_tr, dig_te = prune_constant_and_dup(digit_features(raw_tr),
                                        digit_features(raw_te))
print(f"digit block: {dig_tr.shape[1]} columns after pruning")

rng = np.random.RandomState(SEED)
noise_tr = pd.DataFrame({"noise": rng.standard_normal(len(Xtr_base)).astype("float32")})
noise_te = pd.DataFrame({"noise": rng.standard_normal(len(Xte_base)).astype("float32")})

BLOCKS = {
    "base": (Xtr_base, Xte_base),
    "base+noise": (pd.concat([Xtr_base, noise_tr], axis=1),
                   pd.concat([Xte_base, noise_te], axis=1)),
    "base+latent": (pd.concat([Xtr_base, lat_tr], axis=1),
                    pd.concat([Xte_base, lat_te], axis=1)),
    "base+digits": (pd.concat([Xtr_base, dig_tr], axis=1),
                    pd.concat([Xte_base, dig_te], axis=1)),
    "base+latent+digits": (pd.concat([Xtr_base, lat_tr, dig_tr], axis=1),
                           pd.concat([Xte_base, lat_te, dig_te], axis=1)),
}


def cv_oof(X):
    oof = np.zeros(len(y))
    best_iters = []
    splits = list(SPLITTER.split(X, y))
    for tr, va in splits:
        X_tr, y_tr = X.iloc[tr], y[tr]
        # inner holdout so the stopping iteration is not tuned on the scored rows
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
        model = lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, random_state=SEED,
                                   n_jobs=-1, verbose=-1, **LGB_PARAMS)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(model.best_iteration_ or MAX_ROUNDS)
    fold_auc = [roc_auc_score(y[va], oof[va]) for _, va in splits]
    return oof, fold_auc, best_iters


results = {}
for name, (Xtr, _) in BLOCKS.items():
    if time.time() - START > WALL_BUDGET_S - 120:
        print(f"\nskipped {name} (wall budget)")
        continue
    oof, folds, iters = cv_oof(Xtr)
    results[name] = (oof, folds, iters)
    print(f"\n{name}  ({Xtr.shape[1]} cols)")
    print(f"  fold AUC : {[round(a, 5) for a in folds]}")
    print(f"  OOF {roc_auc_score(y, oof):.5f}  min {min(folds):.5f}  "
          f"mean {np.mean(folds):.5f}  ({time.time() - START:.0f}s)")

# --- rank on minimum fold AUC -------------------------------------------
print("\n=== ranking (by min fold AUC) ===")
noise_ref = results.get("base+noise")
base_ref = results.get("base")
if base_ref and noise_ref:
    print(f"  noise-column OOF shift vs base: "
          f"{roc_auc_score(y, noise_ref[0]) - roc_auc_score(y, base_ref[0]):+.5f}")

ranked = sorted(results.items(), key=lambda kv: min(kv[1][1]), reverse=True)
for name, (oof, folds, _) in ranked:
    print(f"  {name:22s} min {min(folds):.5f}  mean {np.mean(folds):.5f}  "
          f"OOF {roc_auc_score(y, oof):.5f}")

best_name, (best_oof, best_folds, best_iters) = ranked[0]
best_oof_auc = roc_auc_score(y, best_oof)
print(f"\nbest feature set: {best_name}  (min fold {min(best_folds):.5f}, "
      f"OOF {best_oof_auc:.5f})")

# --- refit + submission if it beats the standing best -------------------
if best_oof_auc > BEST_SO_FAR:
    n_est = int(np.median(best_iters))
    Xtr_b, Xte_b = BLOCKS[best_name]
    final = lgb.LGBMClassifier(n_estimators=n_est, random_state=SEED,
                               n_jobs=-1, verbose=-1, **LGB_PARAMS)
    final.fit(Xtr_b, y)
    prob = np.clip(final.predict_proba(Xte_b)[:, 1], *CLIP)
    out = HERE / "submission_lgbm_fe.csv"
    pd.DataFrame({"id": test_ids.to_numpy(), "Will_Buy_EV": prob}).to_csv(
        out, index=False)
    print(f"\nwrote {out}  test mean {prob.mean():.4f} "
          f"(train base rate {y.mean():.4f})  n_estimators={n_est}")
    meta = {
        "feature_set": best_name,
        "features": list(Xtr_b.columns),
        "lgb_params": LGB_PARAMS,
        "n_estimators": n_est,
        "cv": {"splitter": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "fold_auc": [round(a, 6) for a in best_folds],
               "min_fold": round(min(best_folds), 6),
               "oof_auc": round(best_oof_auc, 6)},
    }
    with open(HERE / "lgbm_fe_best.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {HERE / 'lgbm_fe_best.json'}")
else:
    print(f"\nbest {best_oof_auc:.5f} does not beat standing best "
          f"{BEST_SO_FAR} - no submission written")

print(f"\ntotal wall time {time.time() - START:.0f}s")
