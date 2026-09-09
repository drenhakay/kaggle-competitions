"""XGBoost base model - base + digit features, enable_categorical.

Single sensible config from tuning/tune_gbm_digits.py; not separately tuned,
it is here for blend diversity rather than as a standalone contender.
"""
from xgboost import XGBClassifier

from dataset import freeze_categoricals

NAME = "xgb"

MAX_ROUNDS = 4000
ES_ROUNDS = 150

PARAMS = {
    "tree_method": "hist",
    "enable_categorical": True,
    "max_bin": 384,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "eval_metric": "auc",
}


def prepare(X, X_test, cat_cols):
    """Cast cat_cols to 'category' (XGBoost reads it via enable_categorical)."""
    return freeze_categoricals(X, X_test, cat_cols)


def fit_predict(X_fit, y_fit, X_es, y_es, X_val, X_test):
    """Train with early stopping on (X_es, y_es); return
    (val_prob, test_prob, best_iteration + 1)."""
    model = XGBClassifier(n_estimators=MAX_ROUNDS, early_stopping_rounds=ES_ROUNDS,
                          random_state=42, n_jobs=-1, **PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    bi = model.best_iteration
    return val_prob, test_prob, (bi + 1 if bi is not None else MAX_ROUNDS)
