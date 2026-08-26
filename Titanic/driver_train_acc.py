import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from pipeline import (
    RANDOM_STATE, DATA_DIR, load_raw, add_deterministic_features,
    fill_fare_and_compute_fare_per_person, fit_title_encoder, apply_title_encoder,
    fit_cabin_sex_encoders, apply_cabin_sex_encoders, fit_age_medians, apply_age_fill,
    add_family_survival_oof, fit_family_survival_stats, apply_family_survival,
    FEATURE_COLS, build_logreg_pipeline, build_tree_pipeline,
)

t0 = time.time()
train_raw, test_raw = load_raw()
y_full = train_raw['Survived']

# Ticket-group counts from combined train+test identities (unchanged design choice)
all_tickets = pd.concat([train_raw['Ticket'], test_raw['Ticket']])
ticket_counts = all_tickets.value_counts()

train_det = add_deterministic_features(train_raw, ticket_counts)
test_det = add_deterministic_features(test_raw, ticket_counts)

# ---- Lockbox split: carved out BEFORE any fitting, stratified on the target ----
dev_idx, lockbox_idx = train_test_split(
    train_det.index, test_size=0.15, stratify=y_full, random_state=RANDOM_STATE
)
dev_df, lockbox_df = train_det.loc[dev_idx].copy(), train_det.loc[lockbox_idx].copy()
y_dev, y_lockbox = y_full.loc[dev_idx], y_full.loc[lockbox_idx]

# Grouped on Ticket (fix for an ml-guard finding: a plain StratifiedKFold let
# a family/ticket span two folds, leaking one member's label into another
# member's FamilySurvivalRate feature within the same outer fold).
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def fit_all_encoders(fit_df, fit_y):
    # NOTE: PassengerCluster is intentionally NOT fit here -- it's added by
    # ClusterFeatureAdder inside each model's own Pipeline (see pipeline.py)
    # so it gets refit on each CV fold's training rows automatically.
    fare_median_by_pclass = fit_df.groupby('Pclass')['Fare'].median()
    fit_df = fill_fare_and_compute_fare_per_person(fit_df, fare_median_by_pclass)
    title_map, title_unseen = fit_title_encoder(fit_df)
    fit_df = apply_title_encoder(fit_df, title_map, title_unseen)
    cabin_enc, sex_enc = fit_cabin_sex_encoders(fit_df)
    fit_df = apply_cabin_sex_encoders(fit_df, cabin_enc, sex_enc)
    age_by_title, age_overall = fit_age_medians(fit_df)
    fit_df = apply_age_fill(fit_df, age_by_title, age_overall)
    fam_group_sum, fam_group_count, fam_global_rate = fit_family_survival_stats(fit_y, fit_df['Ticket'])
    encoders = dict(fare_median_by_pclass=fare_median_by_pclass, title_map=title_map,
                     title_unseen=title_unseen, cabin_enc=cabin_enc, sex_enc=sex_enc,
                     age_by_title=age_by_title, age_overall=age_overall,
                     fam_group_sum=fam_group_sum, fam_group_count=fam_group_count,
                     fam_global_rate=fam_global_rate)
    return fit_df, encoders


def apply_all_encoders(df, encoders):
    df = fill_fare_and_compute_fare_per_person(df, encoders['fare_median_by_pclass'])
    df = apply_title_encoder(df, encoders['title_map'], encoders['title_unseen'])
    df = apply_cabin_sex_encoders(df, encoders['cabin_enc'], encoders['sex_enc'])
    df = apply_age_fill(df, encoders['age_by_title'], encoders['age_overall'])
    df = apply_family_survival(df, df['Ticket'], encoders['fam_group_sum'],
                                encoders['fam_group_count'], encoders['fam_global_rate'])
    return df


# =====================================================================
# PASS A: fit everything on the DEV partition only.
#   - dev gets its FamilySurvivalRate computed OOF (so dev's own CV folds
#     used for search/selection never see a fold's held-out labels through
#     that feature).
#   - lockbox gets every fit-dependent feature from a single dev-only fit
#     (lockbox rows never influence any encoder/target-stat).
#   - PassengerCluster is added later, per-fold, inside each model Pipeline.
# =====================================================================
dev_df, dev_encoders = fit_all_encoders(dev_df, y_dev)
dev_df = add_family_survival_oof(dev_df, y_dev, dev_df['Ticket'], cv)
dev_groups = dev_df['Ticket']
lockbox_df = apply_all_encoders(lockbox_df, dev_encoders)

X_dev = dev_df[FEATURE_COLS]
X_lockbox = lockbox_df[FEATURE_COLS]

print(f"[{time.time()-t0:.0f}s] dev={X_dev.shape}, lockbox={X_lockbox.shape}")
print("NaNs in X_dev:", X_dev.isnull().sum().sum(), " X_lockbox:", X_lockbox.isnull().sum().sum())

# =====================================================================
# Candidate models: narrower / more-regularized grids than the previous
# 60-iter wide search. n_iter capped at 25 per model (guardrail). Every
# model is wrapped so ClusterFeatureAdder is refit inside each CV fold.
# =====================================================================
xgb_param_dist = {
    'clf__n_estimators': [100, 150, 200, 300],
    'clf__max_depth': [2, 3, 4],
    'clf__learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08],
    'clf__subsample': [0.7, 0.85, 1.0],
    'clf__colsample_bytree': [0.7, 0.85, 1.0],
    'clf__reg_alpha': [0.5, 1, 2, 5],
    'clf__reg_lambda': [1, 2, 5, 10],
}
xgb_search = RandomizedSearchCV(
    build_tree_pipeline(XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', tree_method='hist')),
    param_distributions=xgb_param_dist, n_iter=25, cv=cv, scoring='accuracy',
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_search.fit(X_dev, y_dev, groups=dev_groups)
best_xgb = xgb_search.best_estimator_
print(f"[{time.time()-t0:.0f}s] XGB best dev-CV acc: {xgb_search.best_score_:.4f}  params={xgb_search.best_params_}")

rf_param_dist = {
    'clf__n_estimators': [200, 300, 400],
    'clf__max_depth': [3, 4, 5, 6],
    'clf__min_samples_split': [4, 6, 8, 10],
    'clf__min_samples_leaf': [2, 3, 5, 8],
    'clf__max_features': ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    build_tree_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
    param_distributions=rf_param_dist, n_iter=25, cv=cv, scoring='accuracy',
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf_search.fit(X_dev, y_dev, groups=dev_groups)
best_rf = rf_search.best_estimator_
print(f"[{time.time()-t0:.0f}s] RF best dev-CV acc: {rf_search.best_score_:.4f}  params={rf_search.best_params_}")

# build_logreg_pipeline nests: outer Pipeline('cluster', 'clf'=inner Pipeline('pre', 'clf'=LogisticRegression))
lr_param_dist = {
    'clf__clf__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    'clf__clf__class_weight': [None, 'balanced'],
}
lr_search = RandomizedSearchCV(
    build_logreg_pipeline(),
    param_distributions=lr_param_dist, n_iter=14, cv=cv, scoring='accuracy',
    random_state=RANDOM_STATE, n_jobs=-1,
)
lr_search.fit(X_dev, y_dev, groups=dev_groups)
best_lr = lr_search.best_estimator_
print(f"[{time.time()-t0:.0f}s] LogReg best dev-CV acc: {lr_search.best_score_:.4f}  params={lr_search.best_params_}")

# Soft-voting ensemble of the tuned RF+XGB pipelines (each still refits its
# own ClusterFeatureAdder per fold when scored below).
ensemble_model = VotingClassifier(estimators=[('xgb', best_xgb), ('rf', best_rf)], voting='soft')

candidates = {
    'logreg': best_lr,
    'rf': best_rf,
    'xgb': best_xgb,
    'ensemble_rf_xgb': ensemble_model,
}

fold_scores = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X_dev, y_dev, cv=cv, scoring='accuracy', groups=dev_groups)
    fold_scores[name] = scores
    print(f"[{time.time()-t0:.0f}s] {name:16s} dev-CV folds={np.round(scores,4)} "
          f"mean={scores.mean():.4f} min={scores.min():.4f} max={scores.max():.4f} std={scores.std():.4f}")

# Selection rule: highest MEAN fold accuracy (changed from min per user request)
winner_name = max(fold_scores, key=lambda k: fold_scores[k].mean())
winner_model = candidates[winner_name]
print(f"\n[{time.time()-t0:.0f}s] Selected '{winner_name}' by highest mean dev-CV fold accuracy "
      f"({fold_scores[winner_name].mean():.4f})")

# =====================================================================
# Honest, single-shot lockbox read on the winner (never touched during
# tuning/selection above).
# =====================================================================
winner_model.fit(X_dev, y_dev)
lockbox_preds = winner_model.predict(X_lockbox)
lockbox_acc = (lockbox_preds == y_lockbox.values).mean()
print(f"[{time.time()-t0:.0f}s] Lockbox holdout accuracy for '{winner_name}': {lockbox_acc:.4f} "
      f"(n={len(y_lockbox)})")

# =====================================================================
# PASS B: refit encoders + winning model on the FULL training set
# (dev+lockbox combined) for the actual test-set submission.
# =====================================================================
train_full_df, full_encoders = fit_all_encoders(train_det.copy(), y_full)
train_full_df = add_family_survival_oof(train_full_df, y_full, train_full_df['Ticket'], cv)
test_final_df = apply_all_encoders(test_det.copy(), full_encoders)

X_full = train_full_df[FEATURE_COLS]
X_test_final = test_final_df[FEATURE_COLS]
print("NaNs in X_test_final:", X_test_final.isnull().sum().sum())

winner_model.fit(X_full, y_full)
train_preds = winner_model.predict(X_full)
train_acc = (train_preds == y_full.values).mean()
print(f"TRAIN_ACCURACY={train_acc:.4f}")

test_preds = winner_model.predict(X_test_final)

submission = pd.DataFrame({'PassengerId': test_final_df.index, 'Survived': test_preds})
out_path = str(Path(DATA_DIR) / 'submission_phase2_candidate.csv')
submission.to_csv(out_path, index=False)
print(f"[{time.time()-t0:.0f}s] Wrote {out_path}")

print("\n=== SUMMARY ===")
for name, scores in fold_scores.items():
    marker = ' <== selected' if name == winner_name else ''
    print(f"{name:16s} folds={np.round(scores,4)} mean={scores.mean():.4f} min={scores.min():.4f} max={scores.max():.4f}{marker}")
print(f"Lockbox accuracy (winner, single honest read): {lockbox_acc:.4f}")
print(f"Total time: {time.time()-t0:.0f}s")
