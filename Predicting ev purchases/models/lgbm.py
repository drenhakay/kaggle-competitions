"""LightGBM base model - base + digit features, native categorical handling.

Params from the Optuna search (see ../lgbm_digits_tuned_best.json and tuning/).
"""
import lightgbm as lgb

from dataset import freeze_categoricals

NAME = "lgbm"

# n_estimators is a high cap here; the real length comes from early stopping
MAX_ROUNDS = 5000
ES_ROUNDS = 150

# ../lgbm_digits_tuned_best.json -> params (minus n_estimators)
PARAMS = {
    "objective": "binary",
    "bagging_freq": 1,
    "learning_rate": 0.03112203095091409,
    "max_bin": 412,
    "num_leaves": 16,
    "max_depth": 7,
    "min_child_samples": 111,
    "feature_fraction": 0.8358987982014802,
    "bagging_fraction": 0.9041861138184736,
    "reg_alpha": 0.005206906899338946,
    "reg_lambda": 5.322281077059205,
}


def prepare(X, X_test, cat_cols):
    """Cast cat_cols to 'category' (native LightGBM categorical handling)."""
    return freeze_categoricals(X, X_test, cat_cols)


def fit_predict(X_fit, y_fit, X_es, y_es, X_val, X_test):
    """Train with early stopping on (X_es, y_es); return
    (val_prob, test_prob, best_iteration)."""
    model = lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, random_state=42,
                               n_jobs=-1, verbose=-1, **PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    best_it = model.best_iteration_          # None if early stopping never fired
    return val_prob, test_prob, best_it if best_it is not None else MAX_ROUNDS
