"""Tuned GBDT on the base + digit-decomposition feature set.

Round-2 tuning. The digit block was the only engineered block that earned its
place (see lgbm_fe_best.json / tune_gbm_fe.py); the latent reconstruction is
dropped. Two-stage LightGBM search: a fast single-holdout Optuna sweep to rank
many configs, then a full 5-fold re-score of the top few with selection on the
minimum fold AUC. XGBoost / CatBoost / a fully-additive LightGBM are scored as
single comparison points if the wall budget allows.

Run:  python3 tune_gbm_digits.py
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
from xgboost import XGBClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
import optuna  # noqa: E402
from features_ev import load_raw, digit_features, prune_constant_and_dup  # noqa: E402

SEED = 42
WALL_BUDGET_S = 1750
STAGE1_TIME_S = 800
STAGE1_TRIALS = 60
STAGE2_TOPK = 3
LGB_MAX_ROUNDS = 3000
LGB_ES_ROUNDS = 150
ES_ROUNDS = 150               # xgb / catboost
CLIP = (1e-4, 1 - 1e-4)
BEST_SO_FAR = 0.94378         # OOF AUC of base+digits fixed config
START = time.time()

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Range_Anxiety_Level"]
SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --- feature frames ------------------------------------------------------
Xtr_obj, Xte_obj, ytr, test_ids = load_n_clean()
y = ytr.to_numpy()
raw_tr, raw_te = load_raw()
dig_tr, dig_te = prune_constant_and_dup(digit_features(raw_tr),
                                        digit_features(raw_te))
print(f"digit block: {dig_tr.shape[1]} columns")

Xtr_cat = Xtr_obj.copy()
Xte_cat = Xte_obj.copy()
for c in CAT_COLS:
    cats = pd.CategoricalDtype(
        sorted(set(Xtr_obj[c]) | set(Xte_obj[c])), ordered=False)
    Xtr_cat[c] = Xtr_cat[c].astype(cats)
    Xte_cat[c] = Xte_cat[c].astype(cats)
    # plain python str for the CatBoost frame (its cat_features want object)
    Xtr_obj[c] = Xtr_obj[c].astype("object")
    Xte_obj[c] = Xte_obj[c].astype("object")

# category dtype for lgb / xgb, object categoricals for catboost
Xtr_lx = pd.concat([Xtr_cat, dig_tr], axis=1)
Xte_lx = pd.concat([Xte_cat, dig_te], axis=1)
Xtr_cb = pd.concat([Xtr_obj, dig_tr], axis=1)
Xte_cb = pd.concat([Xte_obj, dig_te], axis=1)
N_FEATURES = Xtr_lx.shape[1]

# fixed fast holdout used only to rank Optuna configs
h_fit, h_val, yh_fit, yh_val = train_test_split(
    Xtr_lx, y, test_size=0.2, stratify=y, random_state=SEED)


def lgb_params(trial):
    return dict(
        objective="binary",
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.045, log=True),
        max_bin=trial.suggest_int("max_bin", 255, 511),
        num_leaves=trial.suggest_int("num_leaves", 16, 96),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 400),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 0.95),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        bagging_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.3, 12.0, log=True),
    )


def objective(trial):
    params = lgb_params(trial)
    model = lgb.LGBMClassifier(n_estimators=LGB_MAX_ROUNDS, random_state=SEED,
                               n_jobs=-1, verbose=-1, **params)
    model.fit(h_fit, yh_fit, eval_set=[(h_val, yh_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(LGB_ES_ROUNDS, verbose=False)])
    return roc_auc_score(yh_val, model.predict_proba(h_val)[:, 1])


print("=== stage 1: fast holdout Optuna sweep ===")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
s1_budget = min(STAGE1_TIME_S, WALL_BUDGET_S - (time.time() - START) - 700)
study.optimize(objective, n_trials=STAGE1_TRIALS, timeout=max(60, s1_budget))
print(f"completed {len(study.trials)} trials  best holdout AUC "
      f"{study.best_value:.5f}  ({time.time() - START:.0f}s)")


def cv_oof(fit_predict, X):
    oof = np.zeros(len(y))
    best_iters = []
    splits = list(SPLITTER.split(X, y))
    for tr, va in splits:
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
        p, bi = fit_predict(X_fit, y_fit, X_es, y_es, X.iloc[va])
        oof[va] = p
        best_iters.append(bi)
    fold_auc = [roc_auc_score(y[va], oof[va]) for _, va in splits]
    return oof, fold_auc, best_iters


def lgb_fp_factory(params):
    def _fp(X_fit, y_fit, X_es, y_es, X_va):
        m = lgb.LGBMClassifier(n_estimators=LGB_MAX_ROUNDS, random_state=SEED,
                               n_jobs=-1, verbose=-1, **params)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(LGB_ES_ROUNDS, verbose=False)])
        return m.predict_proba(X_va)[:, 1], m.best_iteration_ or LGB_MAX_ROUNDS
    return _fp


# --- stage 2: full 5-fold re-score of the top-k configs -----------------
print(f"\n=== stage 2: 5-fold re-score of top {STAGE2_TOPK} ===")
top = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:STAGE2_TOPK]
results = {}
for rank, t in enumerate(top):
    params = dict(objective="binary", bagging_freq=1,
                  **{k: v for k, v in t.params.items()})
    oof, folds, iters = cv_oof(lgb_fp_factory(params), Xtr_lx)
    name = f"lgbm_cfg{rank}"
    results[name] = (oof, folds, iters, params, "lx")
    print(f"  {name}: fold {[round(a, 5) for a in folds]}  "
          f"min {min(folds):.5f}  OOF {roc_auc_score(y, oof):.5f}  "
          f"({time.time() - START:.0f}s)")

best_lgb = max(results.items(), key=lambda kv: min(kv[1][1]))
best_params = best_lgb[1][3]

# fully-additive variant on the best config
if time.time() - START < WALL_BUDGET_S - 400:
    add_params = dict(best_params,
                      interaction_constraints=[[i] for i in range(N_FEATURES)])
    oof, folds, iters = cv_oof(lgb_fp_factory(add_params), Xtr_lx)
    results["lgbm_additive"] = (oof, folds, iters, add_params, "lx")
    print(f"  lgbm_additive: fold {[round(a, 5) for a in folds]}  "
          f"min {min(folds):.5f}  OOF {roc_auc_score(y, oof):.5f}")
else:
    print("  skipped additive variant (wall budget)")


def xgb_fp(X_fit, y_fit, X_es, y_es, X_va):
    m = XGBClassifier(
        tree_method="hist", enable_categorical=True, max_bin=384,
        n_estimators=4000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5.0, eval_metric="auc",
        early_stopping_rounds=ES_ROUNDS, random_state=SEED, n_jobs=-1)
    m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    bi = m.best_iteration
    return m.predict_proba(X_va)[:, 1], (bi + 1 if bi is not None else 4000)


if time.time() - START < WALL_BUDGET_S - 260:
    oof, folds, iters = cv_oof(xgb_fp, Xtr_lx)
    results["xgboost"] = (oof, folds, iters, None, "lx")
    print(f"  xgboost: fold {[round(a, 5) for a in folds]}  "
          f"min {min(folds):.5f}  OOF {roc_auc_score(y, oof):.5f}")
else:
    print("  skipped xgboost (wall budget)")


def cat_fp(X_fit, y_fit, X_es, y_es, X_va):
    m = CatBoostClassifier(
        iterations=4000, learning_rate=0.03, depth=6, eval_metric="AUC",
        cat_features=CAT_COLS, early_stopping_rounds=ES_ROUNDS,
        random_seed=SEED, verbose=False)
    m.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    return m.predict_proba(X_va)[:, 1], m.get_best_iteration() or 4000


if time.time() - START < WALL_BUDGET_S - 260:
    oof, folds, iters = cv_oof(cat_fp, Xtr_cb)
    results["catboost"] = (oof, folds, iters, None, "cb")
    print(f"  catboost: fold {[round(a, 5) for a in folds]}  "
          f"min {min(folds):.5f}  OOF {roc_auc_score(y, oof):.5f}")
else:
    print("  skipped catboost (wall budget)")

# --- rank on minimum fold AUC ------------------------------------------
print("\n=== ranking (by min fold AUC) ===")
ranked = sorted(results.items(), key=lambda kv: min(kv[1][1]), reverse=True)
for name, (oof, folds, _, _, _) in ranked:
    print(f"  {name:16s} min {min(folds):.5f}  mean {np.mean(folds):.5f}  "
          f"OOF {roc_auc_score(y, oof):.5f}")

win_name, (win_oof, win_folds, win_iters, win_params, win_kind) = ranked[0]
win_oof_auc = roc_auc_score(y, win_oof)
print(f"\nwinner: {win_name}  min fold {min(win_folds):.5f}  OOF {win_oof_auc:.5f}")

progression = {"baseline": 0.9416, "round1_tuned": 0.94323,
               "base_digits_fixed": 0.94378, "round2": round(win_oof_auc, 5)}
print(f"progression: {progression}")

# --- submission + JSON if it beats the standing best ------------------
if win_oof_auc > BEST_SO_FAR:
    n_est = int(np.median(win_iters))
    Xtr_w, Xte_w = {"lx": (Xtr_lx, Xte_lx), "cb": (Xtr_cb, Xte_cb)}[win_kind]
    if win_name.startswith("lgbm"):
        p = {k: v for k, v in win_params.items() if k != "n_estimators"}
        final = lgb.LGBMClassifier(n_estimators=n_est, random_state=SEED,
                                   n_jobs=-1, verbose=-1, **p)
    elif win_name == "xgboost":
        final = XGBClassifier(
            tree_method="hist", enable_categorical=True, max_bin=384,
            n_estimators=n_est, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, eval_metric="auc",
            random_state=SEED, n_jobs=-1)
    else:
        final = CatBoostClassifier(
            iterations=n_est, learning_rate=0.03, depth=6, eval_metric="AUC",
            cat_features=CAT_COLS, random_seed=SEED, verbose=False)
    final.fit(Xtr_w, y)
    prob = np.clip(final.predict_proba(Xte_w)[:, 1], *CLIP)
    out = HERE / "submission_lgbm_digits_tuned.csv"
    pd.DataFrame({"id": test_ids.to_numpy(), "Will_Buy_EV": prob}).to_csv(
        out, index=False)
    print(f"\nwrote {out}  test mean {prob.mean():.4f} "
          f"(base rate {y.mean():.4f})  n_estimators={n_est}")
    meta = {
        "model": win_name,
        "features": list(Xtr_w.columns),
        "params": (win_params if win_params else "see script"),
        "n_estimators": n_est,
        "cv": {"splitter": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "fold_auc": [round(a, 6) for a in win_folds],
               "min_fold": round(min(win_folds), 6),
               "oof_auc": round(win_oof_auc, 6)},
        "progression": progression,
    }
    with open(HERE / "lgbm_digits_tuned_best.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {HERE / 'lgbm_digits_tuned_best.json'}")
else:
    print(f"\n{win_oof_auc:.5f} does not beat standing best {BEST_SO_FAR} "
          f"- no submission written")

print(f"\ntotal wall time {time.time() - START:.0f}s")
