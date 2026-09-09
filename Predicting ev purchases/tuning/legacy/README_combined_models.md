# Combined models — OOF blend

Blends the base models (LightGBM x2, XGBoost, CatBoost, NN) on a shared CV split.

## Contract

Every base model writes two files into `./oof/`, both on
`StratifiedKFold(5, shuffle=True, random_state=42)` — the same splitter as
`baseline.py` and the tune scripts:

- `oof_<model>.csv` — `id, pred` for every **train** row (out-of-fold probability)
- `test_<model>.csv` — `id, pred` for every **test** row (mean over the 5 fold models)

`oof_utils.save_oof(name, oof_pred, test_pred)` writes both and checks the
lengths / NaNs.

## Workflow

```
cd "Combined models"
python3 make_oof_trees.py     # LightGBM base + (LGBM/XGB/CatBoost on base+digits)
python3 make_oof_nn.py        # the torch MLP, same arch as model_nn.py
python3 blend.py              # scores each model + 3 blends, writes submission_blend.csv
```

`blend.py` rank-transforms each model's predictions (AUC only cares about order,
and it puts the NN's sigmoid outputs on the same scale as the trees), then
compares:

1. **equal** — equal-weight mean
2. **hillclimb** — non-negative weights added greedily (Caruana ensemble selection)
3. **stacker** — logistic regression on the rank columns, scored out-of-fold

Selection is by **minimum fold AUC** (the project rule). A `submission_blend.csv`
is written to the parent folder only if the best blend beats the best single model.

## Notes

- `make_oof_trees.py` uses fixed `n_estimators` per model (near where the tuned
  runs' early stopping landed) so the loop stays simple and leak-free. The
  base+digits LightGBM reuses the tuned params from `lgbm_digits_tuned_best.json`.
- The three GBDTs will be ~0.97+ rank-correlated with each other; the NN is the
  real source of blend diversity. Check the correlation table `blend.py` prints.
- To add a model: write `oof_<name>.csv` / `test_<name>.csv` into `./oof/` in the
  same format and re-run `blend.py` — it picks up everything it finds.
