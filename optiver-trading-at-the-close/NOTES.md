# Optiver — Trading at the Close — decision record

Running log of what was tried, what cross-validation said, and the verdict. `Final_code.ipynb`
is the optimisation workflow only (validation, features, model); the submission pipeline is
not included here.

## Task

Predict each stock's 60-second-ahead return *relative to a synthetic index* during the
Nasdaq closing auction. Metric: **MAE**. Data: ~5.24M rows, 200 stocks x ~481 days x 55
ten-second buckets.

## Validation

Custom **purged, expanding-window CV keyed on `date_id`** (`time_series_folds`): 5 folds,
first training block 100 days, 10-day embargo gap before each validation block.

Shuffled K-fold is unusable here — it leaks through same-day contamination (all 200 stocks
on a day share that day's regime), through rolling features computed across the fold
boundary, and through any preprocessing fit on the full set. The embargo also removes the
near-duplicate rows either side of the boundary.

## Tracking metric: gain, not raw MAE

Raw MAE is dominated by the volatility regime of each fold's window — target std ranges
8.8–10.3 across folds, and `MAE(predict 0)` moves 7.21 → 5.93 fold to fold for reasons that
have nothing to do with the model. So every experiment is judged on

    gain = MAE(predict 0) - model MAE

per fold. The question for a new feature is "did mean gain rise and did no fold regress",
not "did raw MAE drop".

## Progression

| Stage | Features added | Mean gain | Note |
|---|---|---|---|
| Baseline | raw provided columns only | ~0.082 | model beats predict-0 by ~1–1.6% (low-SNR competition, normal) |
| Round 1 | `signed_imbalance`, `normalise_imb_del_1/2` (imbalance-delta / matched_size) | ~0.091 | all 5 folds improved |
| Round 2 | `wap_ret_1b/2b/3b` (log returns), `wap_rv_6b` (realized vol) | ~0.0955 | all 5 folds improved |
| Prune | dropped `imbalance_delta_1b/2b` (raw, redundant vs normalised), `wap_rv_6b` (bottom of importance), `log_wap` (monotonic transform of `wap`) | 0.0950 | no fold regressed; fewer columns, faster loop |

Current fold gains: `[0.133, 0.102, 0.067, 0.092, 0.081]` — mean **0.0950**, worst of folds
2–5 (fold 1 trains on only 100 days) **0.0672**. Fold 3's window is genuinely the hardest
(lowest gain in every run), not a data-size effect.

## Settled choices

- **LightGBM, `objective='regression_l1'`** to match the MAE metric — predicts the
  conditional median, robust to the fat-tailed target (rare huge auction dislocations that
  are not predictable from the features).
- `learning_rate=0.05`, `max_depth=8`, early stopping (100). Tree count per fold:
  `[177, 190, 162, 405, 465]`.
- `stock_id` as a pandas `category` — by far the highest feature importance; genuine
  per-stock effects (different target scales and auction behaviour), not an ordered-int
  artifact.
- **Final model**: retrained on all rows with `n_estimators = fold-5 best_iter x 1.15 ~=
  534`. A median across folds would underfit — the small early folds converge fast.

## Not submitted

Competition closed early 2024. A late submission (scored, no leaderboard) was not run.

## Next, if picking this back up

- Cross-sectional features (per `(date_id, seconds_in_bucket)`, ranked/z-scored across the
  200 stocks) — highest expected payoff because the target is index-relative.
- Near/far price spread, `wap - reference_price`, imbalance as a fraction of total auction
  size, `matched_size` growth rate.
- Hyperparameter tuning (`num_leaves` is the real complexity knob, not `max_depth`).
