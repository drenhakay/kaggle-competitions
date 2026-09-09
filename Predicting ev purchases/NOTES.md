# Predicting EV Purchases — decision record

Running log of what was tried, what cross-validation said, and the verdict. Kept so the
dead ends don't get re-explored.

Binary classification (`Will_Buy_EV`, ~17.5% positive), metric ROC-AUC, ~668k train rows.
Kaggle Playground Series S6E9 — a **synthetic** dataset.

## The generator

Reverse-engineered from the data (and confirmed by a public write-up). `Will_Buy_EV`
depends on **only four inputs**:

```
buy_score = 1.2*(Annual_Income_USD / 1e5)
          + 0.6*Environmental_Concern_Level
          + 2.0*Subsidy_Available
          - {Low: 0, Medium: 1, High: 3}[Range_Anxiety_Level]
buy  if  buy_score + wobble > 5.5
```

`Range_Anxiety_Level` is itself a noisy 3-way bucketing of a latent
`worry = Daily_Commute_km - 5*(chargers_home + chargers_work) - 150*home_charging`.
So the commute / charger / home-charging columns reach the target *only* through the
anxiety bucket, which is observed directly — given the bucket they add nothing.

Everything else — `Age` (uniform 25–69), `Gender`, `City_Type`, `Current_Car_Type`,
`Number_of_Cars_Owned` — is noise for this target. `Annual_Income_USD` is drawn from a
bell curve then floor-censored at $30k. The `wobble` is irreducible: two identical rows
can decide differently, so AUC is capped well below 1.

## Current pipeline

`run_all.sh`: `folds.py` → `run.py` (trains lgbm + xgb + nn) → `blend.py`.

- **Features** (`dataset.py`): the cleaned frame + `Buy_Score` (the recipe as one column)
  + a per-digit decomposition of Age / Income / Commute. 28 columns.
- **Base models** (`models/`): LightGBM and XGBoost on the full frame with native
  categoricals; a small BatchNorm MLP for diversity. Fixed 5-fold split in `folds.npy`,
  stratified inner holdout for early stopping.
- **Blend** (`blend.py`): rank-transform each model, try equal / hill-climb / logistic
  stacker, pick the best. Writes two submissions:
  - `submission_private.csv` — highest **minimum fold AUC** (safe pick for the private 80%)
  - `submission_public.csv` — highest **pooled OOF AUC** (aggressive pick for public 20%),
    forced distinct from the private pick.

| | min-fold AUC | OOF AUC |
|---|---|---|
| LightGBM | 0.94262 | 0.94381 |
| XGBoost | 0.94245 | 0.94353 |
| MLP | ~0.9374 | ~0.9387 |
| stacker blend | **0.94268** | 0.94380 |

Kaggle public LB **0.94392**. CV ordering has matched the LB; the pooled-OOF number is
within ~0.0001 of public.

## Progression

| Change | min-fold AUC | Note |
|---|---|---|
| LightGBM, hand-tuned | 0.9416 | baseline |
| LightGBM, Optuna round 1 | 0.94323 (OOF) | `lgbm_fe_best.json` |
| + digit decomposition, Optuna round 2 | 0.94386 (OOF) | `lgbm_digits_tuned_best.json` — the last real gain |
| + XGBoost + MLP, stacker blend | 0.94268 | +0.00006 over LightGBM alone — the three trees are 0.998 correlated, so blending barely helps |

## Ruled out

All validated on `RepeatedKFold(5x3)`, fixed untuned LightGBM, reporting mean **and**
minimum-fold delta.

| Experiment | Verdict | Why |
|---|---|---|
| **Feature round 2** — `worry_score`, `buy_score_smooth`, `total_chargers`, `commute_per_charger`, `age_sq`, `age_x_income`, `log_income`, `income_x_env`, `income_floor_dist`, `commute_infra_gap`, and OOF target encoding of `City_Type` / `City_Type × Current_Car_Type` | No keepers | Every candidate helps mean AUC by ~0.0001 (noise) and hurts at least one fold. All are functions of the noise columns or redundant with `Buy_Score` / the anxiety bucket. |
| **Prune to the recipe** — model on just the 4 generator inputs (± `Buy_Score`) | Costs 0.0022 min-fold | The full 28-column frame wins consistently across all 15 folds. The recipe explains ~99% of the signal but the last ~0.004 AUC is real and lives in the "noise" columns (mostly the income digits). |
| **City home-charging features** — `P(home charging \| City_Type)` lookup, and `home_charging - that probability` | Marginal, negative on min fold | `City_Type` reaches the target only through the observed anxiety bucket; four independent encodings of it all land in the same place. |
| **Recipe recovery (logistic regression)** — fit LR on the recipe terms, then add `Age` / `Gender` / `City_Type` / commute / chargers one at a time | No missing linear terms | Every added term contributes <0.0002 AUC. The recipe is fully recovered; there is no hidden additive signal. |
| **Neural net — 5 architectures** — plain 64/32 MLP, wide 256/128, `no_power` preprocessing + digit block, embeddings + linear skip (2 hidden layers), embeddings + skip (1 layer, dropout 0.5) | Blend unchanged at **0.94266** in every case | The MLP sits ~0.005 AUC below the trees and 0.98-correlated with them, so the stacker gives it ~zero weight. Better features / architecture raise its standalone score (best ~0.9375) but never move the blend. |

Kept from the pruning tests: dropping `income_is_30k`, `Daily_Commute_is_5km` and the
fractional commute digits is neutral; **`Annual_Income_USD_d2` genuinely helps** (a
quantisation artefact of how income was drawn) and stays.

CatBoost was dropped from the default run — it tracks LightGBM/XGBoost almost exactly,
barely moves the blend, and roughly doubles the run time. Still runnable via
`python run.py cat`.

## Known limitation

A clean model tops out around 0.944 here — the recipe is recovered, feature engineering
is exhausted, and the NN can't add diversity. The top of the leaderboard (~0.947) is
almost certainly reconstructing the generator's RNG stream (seed 101) to compute exact
labels, which is not a modelling problem. Treated as the accepted ceiling for this
approach.

## If picking this back up

The remaining ~0.003 to the top would need either the seed reconstruction (not attempted)
or a genuinely different model with decorrelated errors — and the NN experiments suggest
that is hard to get on 4 signal variables plus a large wobble.
