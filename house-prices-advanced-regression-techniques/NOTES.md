# House Prices — decision record

Running log of what was tried, what cross-validation said, and the verdict. Kept so the
dead ends don't get re-explored.

## Current model

`Final_code.ipynb` — Lasso + XGBoost blend, `0.6 * Lasso + 0.4 * XGBoost` averaged in log
space. Repeated 5x3-fold CV RMSE (log target) **0.10753**. Kaggle public LB **0.12490**.
Consistent CV→LB gap of ~0.015–0.016; CV is optimistic but its *ordering* of models has
matched the LB every time.

Preprocessing: NA→"None"/0 fills, combined features (TotalSF, TotalBath, TotalPorchSF,
OverallScore), presence flags, Neighborhood-median LotFrontage impute, mode impute of stray
NaNs, ordinal maps for quality columns, one-hot for nominals, year→age transforms, drop
Id/Utilities, 2 GrLivArea outliers dropped from train, target log1p.

## Progression

| Change | CV RMSE | LB | Note |
|---|---|---|---|
| XGBoost only, narrow hand-tuned grid | 0.1115 | 0.13133 | CV overfit to the fold split |
| XGBoost only, widened search | 0.1124 | 0.12952 | XGB alone tapped out here |
| + combined features (TotalSF etc.) | — | 0.12931 | small gain |
| Lasso + XGBoost blend | 0.10753 | **0.12490** | the real lever — a scaled/skew-corrected linear model errs differently from the tree |

## Ruled out (all validated on repeated 5x3-fold, dual LightGBM + Lasso baseline)

| Experiment | Verdict | Why |
|---|---|---|
| **Feature round 1** — `Qual x GrLivArea`, OOF Neighborhood target encoding, GrLivArea vs neighbourhood median | No keepers | Deltas inside fold noise; every candidate hurt at least one fold. Trees already find the interactions; Lasso shrinks the interaction coefficient to ~0. |
| **Feature round 2** — replace the 25 Neighborhood one-hot dummies with target encoding / target-rank ordinal / 5 price tiers / frequency encoding | One-hot wins decisively | Ames neighbourhood premiums are not low-rank — each needs its own offset. For Lasso every compact encoding was clearly worse (0–1 of 15 folds better). 25 columns vs ~1450 rows is not a dimensionality problem. |
| **Feature round 3** — `NonArmsLength` flag (Abnorml/Family/Alloca/AdjLand), `IsCommercialZone` flag | No keepers | Residual analysis flagged these sales as mispriced, but the error is idiosyncratic (each off by a different amount and sign), not a consistent discount a flag can capture. `IsCommercialZone` is 10 rows — models ignore it. |
| **Stacking** — Ridge / NNLS / Lasso meta-model on OOF base predictions, with and without LightGBM | No gain over the fixed 0.6/0.4 blend | Best combiner (`mean_2`) was 0.0001 better = noise. Adding LightGBM made it worse — weakest base learner, too correlated with XGBoost, no diversity. |
| **Post-hoc de-bias** — linear / isotonic calibration of blend predictions | Makes it worse | Marginal pred-vs-truth slope is 1.008 (near identity). The tail bias is *conditional*, so a correction on the prediction alone just trades tail bias for middle-band bias and adds variance. |

## Known limitation

The blend is tail-biased: it over-predicts the cheapest quintile by ~0.039 log points and
under-predicts the most expensive quintile by ~0.034 (regularised models shrink toward the
mean). Feature rounds 1–3 confirmed the features do not contain the signal to fix this;
stacking and calibration can't fix it either. Treated as the accepted ceiling for this
approach.

## If picking this back up

The numeric feature representation is clean (no raw column correlates with the residual
above 0.043). Any further gain would need a genuinely different feature set (not tried:
polynomial features for the linear model) or a different modelling approach, and the
expected payoff is small.
