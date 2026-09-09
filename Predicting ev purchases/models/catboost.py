"""CatBoost base model - base + digit features, native cat_features.

Single sensible config from tuning/tune_gbm_digits.py; not separately tuned.
Slowest model on this CPU-only setup, so iterations are capped and early
stopping does the real work.
"""
from catboost import CatBoostClassifier

from dataset import CAT_COLS

NAME = "cat"

MAX_ROUNDS = 4000
ES_ROUNDS = 150

PARAMS = {
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "AUC",
}


def prepare(X, X_test, cat_cols):
    """CatBoost wants categorical columns as plain object; cast them."""
    X, X_test = X.copy(), X_test.copy()
    for c in cat_cols:
        X[c] = X[c].astype("object")
        X_test[c] = X_test[c].astype("object")
    return X, X_test


def fit_predict(X_fit, y_fit, X_es, y_es, X_val, X_test):
    """Train with early stopping on (X_es, y_es); return
    (val_prob, test_prob, best_iteration)."""
    cat_features = [c for c in CAT_COLS if c in X_fit.columns]
    model = CatBoostClassifier(
        iterations=MAX_ROUNDS, early_stopping_rounds=ES_ROUNDS,
        cat_features=cat_features, random_seed=42, verbose=False, **PARAMS)
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    best_it = model.get_best_iteration()     # None if early stopping never fired
    return val_prob, test_prob, best_it if best_it is not None else MAX_ROUNDS
