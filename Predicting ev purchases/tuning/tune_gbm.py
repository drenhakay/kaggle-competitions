"""Gradient-boosted-tree tuning for the EV-purchase task.

Compares a tuned LightGBM (Optuna) against single-config XGBoost and CatBoost on
the exact same folds as baseline.py, scored by ROC-AUC. Trees get the raw cleaned
frame with native categorical handling rather than the OHE/PowerTransform
pipeline the NN uses.

On this dataset the main lever is value resolution (high max_bin + low learning
rate) rather than tree shape, and the signal is close to additive, so the search
space is centred on shallow trees and a fully-additive variant is tried too.

Run:  python3 tune_gbm.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import load_n_clean  # noqa: E402

import lightgbm as lgb  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
import optuna  # noqa: E402

SEED = 42
WALL_BUDGET_S = 1800          # hard cap for the whole run
OPTUNA_TRIALS = 30
OPTUNA_TIME_S = 1150          # leave headroom for XGB + CatBoost + refits
LGB_MAX_ROUNDS = 5000         # high cap; real length set by early stopping
LGB_ES_ROUNDS = 200
ES_ROUNDS = 100               # xgb / catboost
CLIP = (1e-4, 1 - 1e-4)
START = time.time()

# object columns the trees should treat as categorical (the two Yes/No columns
# are already mapped to 0/1 by load_n_clean)
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Range_Anxiety_Level"]

# identical splitter to baseline.py so fold-level numbers line up exactly
SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

Xtr_raw, Xte_raw, ytr, test_ids = load_n_clean()
y = ytr.to_numpy()

# category dtype for LightGBM / XGBoost; CatBoost takes the object frame directly
Xtr_cat = Xtr_raw.copy()
Xte_cat = Xte_raw.copy()
for c in CAT_COLS:
    # freeze categories on the train/test union so the two frames encode alike
    cats = pd.CategoricalDtype(
        sorted(set(Xtr_raw[c]) | set(Xte_raw[c])), ordered=False)
    Xtr_cat[c] = Xtr_cat[c].astype(cats)
    Xte_cat[c] = Xte_cat[c].astype(cats)

N_FEATURES = Xtr_cat.shape[1]


def report(name, oof, fold_auc):
    fold_auc = [round(a, 5) for a in fold_auc]
    print(f"\n{name}")
    print(f"  OOF AUC  : {roc_auc_score(y, oof):.5f}")
    print(f"  fold AUC : {fold_auc}")
    print(f"    min {min(fold_auc):.5f}  mean {np.mean(fold_auc):.5f}  "
          f"spread {max(fold_auc) - min(fold_auc):.5f}")


def cv_oof(fit_predict, X):
    """Run the 5-fold OOF loop. fit_predict(X_tr, y_tr, X_es, y_es, X_va) trains
    on the fold's own data (with an inner holdout for early stopping) and returns
    (val_probs, best_iteration)."""
    oof = np.zeros(len(y))
    best_iters = []
    splits = list(SPLITTER.split(X, y))
    for k, (tr, va) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr], y[tr]
        # inner split so the early-stopping iteration is not chosen on the
        # rows we then score the fold on
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
        p, bi = fit_predict(X_fit, y_fit, X_es, y_es, X.iloc[va])
        oof[va] = p
        best_iters.append(bi)
        print(f"    fold {k}: AUC {roc_auc_score(y[va], p):.5f}  "
              f"best_iter {bi}  ({time.time() - START:.0f}s)")
    fold_auc = [roc_auc_score(y[va], oof[va]) for _, va in splits]
    return oof, fold_auc, best_iters


# --- LightGBM --------------------------------------------------------------
def lgb_fit_predict_factory(params):
    def _fp(X_fit, y_fit, X_es, y_es, X_va):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=LGB_MAX_ROUNDS,
            random_state=SEED, n_jobs=-1, verbose=-1, **params)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(LGB_ES_ROUNDS, verbose=False)])
        p = model.predict_proba(X_va)[:, 1]
        # fall back to the cap if early stopping never triggered
        return p, model.best_iteration_ or model.n_estimators
    return _fp


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.035,
                                          log=True),
        max_bin=trial.suggest_int("max_bin", 255, 1023),
        num_leaves=trial.suggest_int("num_leaves", 16, 64),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
        feature_fraction=trial.suggest_float("feature_fraction", 0.3, 0.95),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.7, 1.0),
        bagging_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    )
    oof, fold_auc, _ = cv_oof(lgb_fit_predict_factory(params), Xtr_cat)
    trial.set_user_attr("min_fold", float(min(fold_auc)))
    return float(np.mean(fold_auc))


print("=== LightGBM Optuna search ===")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
opt_budget = min(OPTUNA_TIME_S, WALL_BUDGET_S - (time.time() - START) - 400)
study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=max(60, opt_budget))
print(f"\ncompleted {len(study.trials)} trials  "
      f"best mean fold AUC {study.best_value:.5f}  "
      f"min-fold {study.best_trial.user_attrs.get('min_fold'):.5f}")
best_lgb_params = study.best_params

# --- final scored comparison ----------------------------------------------
results = {}

lgb_oof, lgb_folds, lgb_iters = cv_oof(
    lgb_fit_predict_factory(best_lgb_params), Xtr_cat)
report("LightGBM (tuned)", lgb_oof, lgb_folds)
results["lgbm_tuned"] = (lgb_oof, lgb_folds, lgb_iters, dict(best_lgb_params))

# fully-additive variant: every feature in its own interaction group. Public
# write-ups show the signal here is close to additive, so this can help.
if time.time() - START < WALL_BUDGET_S - 400:
    add_params = dict(best_lgb_params,
                      interaction_constraints=[[i] for i in range(N_FEATURES)])
    add_oof, add_folds, add_iters = cv_oof(
        lgb_fit_predict_factory(add_params), Xtr_cat)
    report("LightGBM (additive)", add_oof, add_folds)
    results["lgbm_additive"] = (add_oof, add_folds, add_iters, add_params)
else:
    print("\nskipped LightGBM additive variant (wall budget)")


def xgb_fit_predict(X_fit, y_fit, X_es, y_es, X_va):
    model = XGBClassifier(
        tree_method="hist", enable_categorical=True, max_bin=512,
        n_estimators=8000, learning_rate=0.02, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=4.0, eval_metric="auc",
        early_stopping_rounds=ES_ROUNDS, random_state=SEED, n_jobs=-1)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    p = model.predict_proba(X_va)[:, 1]
    bi = model.best_iteration
    return p, (bi + 1 if bi is not None else model.n_estimators)


if time.time() - START < WALL_BUDGET_S - 240:
    print("\n=== XGBoost (single config) ===")
    xgb_oof, xgb_folds, xgb_iters = cv_oof(xgb_fit_predict, Xtr_cat)
    report("XGBoost", xgb_oof, xgb_folds)
    results["xgboost"] = (xgb_oof, xgb_folds, xgb_iters, None)
else:
    print("\nskipped XGBoost (wall budget)")


def cat_fit_predict(X_fit, y_fit, X_es, y_es, X_va):
    model = CatBoostClassifier(
        iterations=8000, learning_rate=0.02, depth=5, eval_metric="AUC",
        cat_features=CAT_COLS, early_stopping_rounds=ES_ROUNDS,
        random_seed=SEED, verbose=False)
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    p = model.predict_proba(X_va)[:, 1]
    return p, model.get_best_iteration() or model.tree_count_


if time.time() - START < WALL_BUDGET_S - 240:
    print("\n=== CatBoost (single config) ===")
    cat_oof, cat_folds, cat_iters = cv_oof(cat_fit_predict, Xtr_raw)
    report("CatBoost", cat_oof, cat_folds)
    results["catboost"] = (cat_oof, cat_folds, cat_iters, None)
else:
    print("\nskipped CatBoost (wall budget)")

# --- rank, then write a submission for anything beating baseline ----------
BASELINE = 0.9416
print("\n=== summary ===")
ranked = sorted(results.items(),
                key=lambda kv: np.mean(kv[1][1]), reverse=True)
for name, (_, folds, _, _) in ranked:
    print(f"  {name:16s} mean {np.mean(folds):.5f}  min {min(folds):.5f}  "
          f"OOF {roc_auc_score(y, results[name][0]):.5f}")


def refit_predict(name, params, n_est):
    if name.startswith("lgbm"):
        p = dict(params)
        p.pop("n_estimators", None)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=n_est,
                               random_state=SEED, n_jobs=-1, verbose=-1, **p)
        m.fit(Xtr_cat, y)
        return m.predict_proba(Xte_cat)[:, 1]
    if name == "xgboost":
        m = XGBClassifier(
            tree_method="hist", enable_categorical=True, max_bin=512,
            n_estimators=n_est, learning_rate=0.02, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=4.0, eval_metric="auc",
            random_state=SEED, n_jobs=-1)
        m.fit(Xtr_cat, y)
        return m.predict_proba(Xte_cat)[:, 1]
    m = CatBoostClassifier(
        iterations=n_est, learning_rate=0.02, depth=5, eval_metric="AUC",
        cat_features=CAT_COLS, random_seed=SEED, verbose=False)
    m.fit(Xtr_raw, y)
    return m.predict_proba(Xte_raw)[:, 1]


for name, (oof, folds, iters, params) in ranked:
    mean_auc = float(np.mean(folds))
    if mean_auc <= BASELINE:
        print(f"\n{name}: {mean_auc:.5f} does not beat baseline {BASELINE} "
              f"- no submission written")
        continue
    n_est = int(np.median(iters))  # fix boosting length from the CV folds
    print(f"\n{name}: {mean_auc:.5f} beats baseline - refit on all rows "
          f"with n_estimators={n_est}")
    prob = np.clip(refit_predict(name, params, n_est), *CLIP)
    out = f"submission_{name}.csv"
    pd.DataFrame({"id": test_ids.to_numpy(), "Will_Buy_EV": prob}).to_csv(
        out, index=False)
    print(f"  wrote {out}  test mean {prob.mean():.4f} "
          f"(train base rate {y.mean():.4f})")

print(f"\ntotal wall time {time.time() - START:.0f}s")
