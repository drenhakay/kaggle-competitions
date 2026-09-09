"""Reusable tabular training utilities — your competition template.

Fill the TODOs by lifting the loop bodies straight out of your finished
exercises 03/04/05. Once this works, `from train import ...` in any notebook
and stop retyping the loop.

Keep it flat. Don't turn `fit` into a Trainer class until a real need forces it.
"""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _take(X, idx):
    """Row-slice by positional index, whether X is a DataFrame or an ndarray."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _as_xy(X, y):
    """Any array-like -> float32 tensors, y shaped (N, 1)."""
    Xt = torch.as_tensor(np.asarray(X, dtype="float32"))
    yt = torch.as_tensor(np.asarray(y, dtype="float32")).reshape(-1, 1)
    return Xt, yt


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size=512):
    """Train loader (shuffled) + val loader (not). Returns (train_loader, val_loader)."""
    Xtr, ytr = _as_xy(X_tr, y_tr)
    Xva, yva = _as_xy(X_va, y_va)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _run_epoch(model, loader, loss_fn, optimizer, device, train):
    """One pass. If train: update weights. Returns mean loss over the split."""
    model.train(train)
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)

        if train:
            loss.backward()
            optimizer.step()

        total += loss.item() * len(xb)
        n += len(xb)

    torch.set_grad_enabled(True)
    return total / n


def fit(model, train_loader, val_loader, *, loss_fn, optimizer,
        epochs=200, patience=20, device="cpu", verbose=20):
    """Train with early stopping + best-weight restore.

    Returns history dict {"train": [...], "val": [...], "best_val": float}.
    The model is left holding the best weights seen.
    """
    model.to(device)
    hist = {"train": [], "val": []}
    best_val, best_state, waited = float("inf"), None, 0

    for epoch in range(epochs):
        tr = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va = _run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        hist["train"].append(tr)
        hist["val"].append(va)
        
        if va < best_val:
            best_val = va 
            best_state = copy.deepcopy(model.state_dict())
            waited = 0

        else:
            waited += 1

        if patience and waited >= patience:
            break

        if verbose and epoch % verbose == 0:
            print(f"epoch {epoch:4d}  train {tr:.4f}  val {va:.4f}")

    # summary — how far it got, and where it ended vs the best it saw
    if verbose:
        print(f"stopped at epoch {len(hist['train']) - 1}/{epochs}  "
              f"final train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}  "
              f"best val {best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    hist["best_val"] = best_val
    return hist


@torch.no_grad()
def predict(model, X, device="cpu", batch_size=4096):
    """Model outputs for X as a 1-D numpy array (raw — apply sigmoid yourself
    for classification probabilities)."""
    model.eval().to(device)
    Xt = torch.as_tensor(np.asarray(X, dtype="float32"))
    out = []
    for i in range(0, len(Xt), batch_size):
        out.append(model(Xt[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).squeeze(-1).numpy()


def cross_validate(make_model, make_optimizer, X, y, *, loss_fn, splitter,
                   groups=None, X_test=None, make_preprocessor=None,
                   device="cpu", **fit_kw):
    """Run `fit` on each fold of `splitter`.

    make_model():      -> a fresh model (new weights every fold)
    make_optimizer(m): -> optimizer bound to that model's params
    splitter:          any sklearn splitter (KFold, GroupKFold, your purged CV)
    X_test:            optional held-out test features; if given, every fold
                       model predicts it and the results are averaged (a simple
                       ensemble, usually better than one model refit on all data)
    make_preprocessor(): -> a fresh sklearn-style transformer, fit on each fold's
                       TRAIN slice only then applied to its val slice and X_test.
                       Keeps scaling stats from leaking across the split. If None,
                       X is used as-is.

    X may be a DataFrame or an ndarray. It is NOT flattened to numpy, so a
    name-based ColumnTransformer works — pass X and X_test as DataFrames when
    make_preprocessor selects columns by name.

    Returns a dict:
      scores      list of per-fold best_val   (pick a config by MIN, not mean)
      oof         out-of-fold predictions aligned to X's rows — feed these to a
                  stacker, or score them for one honest CV number
      test_preds  mean of the fold models' predictions on X_test, or None
      models      the trained fold models, in fold order

    Raw model outputs are stored (logits for classification) — apply sigmoid
    yourself on oof / test_preds if you need probabilities.
    """
    # keep X as-is (DataFrame stays a DataFrame) so name-based transformers work
    y = np.asarray(y)
    scores, models = [], []
    oof = np.full(len(X), np.nan, dtype="float32")
    test_sum = np.zeros(len(X_test), dtype="float32") if X_test is not None else None

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr, X_va, X_te = _take(X, tr_idx), _take(X, va_idx), X_test

        if make_preprocessor is not None:
            prep = make_preprocessor()
            X_tr = np.asarray(prep.fit_transform(X_tr, y[tr_idx]), dtype="float32")
            X_va = np.asarray(prep.transform(X_va), dtype="float32")
            if X_test is not None:
                X_te = np.asarray(prep.transform(X_test), dtype="float32")

        train, valid = make_loaders(X_tr, y[tr_idx], X_va, y[va_idx])
        model = make_model()
        optimizer = make_optimizer(model)
        hist = fit(model, train, valid, loss_fn=loss_fn, optimizer=optimizer,
                   device=device, **fit_kw)

        scores.append(hist["best_val"])
        models.append(model)
        oof[va_idx] = predict(model, X_va, device=device)  # best weights restored by fit
        if X_test is not None:
            test_sum += predict(model, X_te, device=device)
        print(f"fold {fold}: {hist['best_val']:.4f}")

    print(f"folds: {[round(s, 4) for s in scores]}  "
          f"min {min(scores):.4f}  mean {np.mean(scores):.4f}")
    return {
        "scores": scores,
        "oof": oof,
        "test_preds": test_sum / len(scores) if test_sum is not None else None,
        "models": models,
    }


