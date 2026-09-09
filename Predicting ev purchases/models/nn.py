"""Neural-net base model: the BatchNorm MLP (n_features -> 64 -> 32 -> 1).

Unlike the tree models the NN's encoding (PowerTransform + one-hot + scaling)
has to be fit per fold to stay leak-free, so `prepare()` here only strips the
digit block (the NN uses the base frame, as model_nn.py did) and `fit_predict()`
fits preprocessing.make_preprocessor() on X_fit alone.

Loop code is reused from train.py so the architecture lives in one place.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

import dataset
from preprocessing import make_preprocessor
from train import fit, make_loaders, predict

NAME = "nn"

_GBDT_LIBS = ("lightgbm", "xgboost", "catboost")


def _guard_threads():
    """A multi-threaded torch segfaults on macOS once a GBDT lib's OpenMP
    runtime is also in the process. Checked at fit time (not import) so it still
    fires when a booster was imported after this module. Run the NN on its own
    (`python run.py nn`) and it keeps every core."""
    if torch.get_num_threads() > 1 and any(m in sys.modules for m in _GBDT_LIBS):
        torch.set_num_threads(1)

PARAMS = {
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "epochs": 50,
    "patience": 5,
    "batch_size": 512,
}


def make_model(n_features):
    """The MLP; outputs a raw logit."""
    return nn.Sequential(
        nn.BatchNorm1d(n_features),
        nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    )


def prepare(X, X_test, cat_cols):
    """Drop the digit block - the NN runs on the base cleaned frame only.
    No encoding here; that is fold-local (see fit_predict)."""
    drop = [c for c in dataset.DIGIT_FEATURE_COLS if c in X.columns]
    return X.drop(columns=drop), X_test.drop(columns=drop)


def fit_predict(X_fit, y_fit, X_es, y_es, X_val, X_test):
    """Fit the preprocessor on X_fit only, train the MLP with early stopping on
    (X_es, y_es), then return sigmoid probabilities for X_val / X_test plus the
    number of epochs run."""
    _guard_threads()
    torch.manual_seed(42)                             # per-fold reproducibility

    prep = make_preprocessor()
    Xf = np.asarray(prep.fit_transform(X_fit, y_fit), dtype="float32")
    Xe = np.asarray(prep.transform(X_es), dtype="float32")
    Xv = np.asarray(prep.transform(X_val), dtype="float32")
    Xt = np.asarray(prep.transform(X_test), dtype="float32")

    model = make_model(Xf.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"],
                                 weight_decay=PARAMS["weight_decay"])
    train_loader, val_loader = make_loaders(
        Xf, np.asarray(y_fit), Xe, np.asarray(y_es),
        batch_size=PARAMS["batch_size"])

    hist = fit(model, train_loader, val_loader,
               loss_fn=nn.BCEWithLogitsLoss(), optimizer=optimizer,
               epochs=PARAMS["epochs"], patience=PARAMS["patience"],
               device="cpu", verbose=0)

    # train.predict returns raw logits; sigmoid to probabilities for the blend
    val_prob = expit(predict(model, Xv))
    test_prob = expit(predict(model, Xt))
    return val_prob, test_prob, len(hist["train"])
