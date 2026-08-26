"""
Titanic Phase-2 bounded model comparison.

Design decisions for this pipeline:
- Metric: plain accuracy (matches Kaggle scoring for this competition).
- Selection rule: highest MEAN CV fold accuracy (average fold performance,
  not the worst-case fold).
- Candidates (max 4): LogisticRegression, RandomForest, XGBoost, soft-voting
  ensemble of tuned RF+XGB.
- Search budget: RandomizedSearchCV n_iter=25 per model (narrower, more
  regularized grids than the previous 60-iter wide search).
- CV: StratifiedGroupKFold(5, shuffle=True, random_state=42), grouped on
  Ticket -- same splitter used for the OOF family-survival-rate feature and
  for model search/selection, run over a DEVELOPMENT partition only (85% of
  train) so it no longer shares folds with the held-out lockbox. Grouping on
  Ticket stops a family/ticket from spanning two folds, which would
  otherwise let one member's label leak into another member's
  FamilySurvivalRate feature within the same outer fold.
- New: ~15% stratified "lockbox" holdout, carved out before ANY fitting, used
  exactly once at the end for an honest read on the winning model. Never used
  for selection.
- Bug fixes applied: (1) Title encoding is now fit once (on the dev partition
  for the dev/lockbox pass, on full train for the final test pass) and mapped
  onto every other split through that single fitted mapping, with an explicit
  "unseen title" bucket -- previously train and test were factorized
  independently, silently producing different codes for the same title.
  (2) The single missing Fare value in the real test set is now filled
  (median Fare by Pclass, fit on whichever partition is doing the fitting)
  before FarePerPerson is computed, instead of silently propagating a NaN.
- To make the lockbox honest, every fit-dependent step (Cabin/Sex/Title
  encoders, per-Title Age median, StandardScaler+KMeans cluster) is now fit
  ONLY on the partition that is allowed to be fit on for that pass (dev-only
  for the dev/lockbox pass; full-train for the final test-prediction pass) --
  this fixes a "fit on entire X_train before CV split" leakage issue for
  the cluster feature, as a side effect of building the lockbox correctly
  (not a separate scope item).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
# Resolved relative to this file so the pipeline still finds train/test.csv
# when run from another machine or a Kaggle kernel, not just this exact path.
DATA_DIR = str(Path(__file__).resolve().parent)


# ---------------------------------------------------------------------------
# Raw loading + feature-engineering steps that do NOT require fitting on a
# specific partition (deterministic, safe to apply identically everywhere).
# ---------------------------------------------------------------------------
def load_raw():
    train = pd.read_csv(f'{DATA_DIR}/train.csv', index_col='PassengerId')
    test = pd.read_csv(f'{DATA_DIR}/test.csv', index_col='PassengerId')
    return train, test


def add_deterministic_features(df, ticket_counts):
    df = df.copy()
    # Embarked: fill + one-hot, fixed column set so dev/lockbox/test always align
    df['Embarked'] = df['Embarked'].fillna('S')
    for cat in ['C', 'Q', 'S']:
        df[f'Embarked_{cat}'] = (df['Embarked'] == cat).astype(int)
    df = df.drop(columns=['Embarked'])

    # Ticket-group size uses combined train+test identities (not labels) --
    # kept as in the original pipeline; this is a defensible transductive
    # choice for a fixed Kaggle test set and wasn't part of the agreed bug fixes.
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title: raw string extraction only here; the fitted train/test-consistent
    # mapping happens later in fit_title_encoder/apply_title_encoder.
    df['TitleRaw'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    return df


def fill_fare_and_compute_fare_per_person(df, fare_median_by_pclass):
    # Bug fix: previously the single missing test Fare silently propagated
    # as NaN into FarePerPerson. Now filled by median Fare for that Pclass,
    # computed from whichever partition is doing the fitting for this pass.
    df = df.copy()
    df['Fare'] = df['Fare'].fillna(df['Pclass'].map(fare_median_by_pclass))
    df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
    return df


# ---------------------------------------------------------------------------
# Fit-dependent steps: each takes an explicit fit source and returns a
# fitted object (or fitted stats) plus a transform function.
# ---------------------------------------------------------------------------
def fit_title_encoder(fit_df):
    known_titles = fit_df['TitleRaw'].unique().tolist()
    mapping = {t: i for i, t in enumerate(sorted(known_titles))}
    unseen_code = len(mapping)  # bucket for any title never seen at fit time
    return mapping, unseen_code


def apply_title_encoder(df, mapping, unseen_code):
    df = df.copy()
    df['Title'] = df['TitleRaw'].map(mapping).fillna(unseen_code).astype(int)
    return df.drop(columns=['TitleRaw'])


def fit_cabin_sex_encoders(fit_df):
    cabin_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    cabin_deck = fit_df['Cabin'].fillna('U').str[0]
    cabin_enc.fit(cabin_deck.to_frame(name='Cabin'))

    sex_enc = OrdinalEncoder()
    sex_enc.fit(fit_df[['Sex']])
    return cabin_enc, sex_enc


def apply_cabin_sex_encoders(df, cabin_enc, sex_enc):
    df = df.copy()
    deck = df['Cabin'].fillna('U').str[0]
    df['Cabin'] = cabin_enc.transform(deck.to_frame(name='Cabin'))
    df['Sex'] = sex_enc.transform(df[['Sex']])
    return df


def fit_age_medians(fit_df):
    by_title = fit_df.groupby('Title')['Age'].median()
    overall = fit_df['Age'].median()
    return by_title, overall


def apply_age_fill(df, by_title, overall):
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Title'].map(by_title))
    df['Age'] = df['Age'].fillna(overall)
    return df


class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds a PassengerCluster feature via StandardScaler+KMeans.

    Wrapped as a Pipeline step (rather than a precomputed static column) so
    RandomizedSearchCV/cross_val_score refit the scaler+KMeans on each CV
    fold's OWN training rows -- this fixes a "cluster fit on all of dev
    before the inner CV split" leakage issue (fitting a static column once
    let every inner validation fold's cluster label be influenced by that
    fold's own held-out rows).
    """
    def __init__(self, cluster_features, n_clusters=4, random_state=RANDOM_STATE):
        self.cluster_features = cluster_features
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        vals = X[self.cluster_features].fillna(0)
        self.scaler_ = StandardScaler().fit(vals)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        ).fit(self.scaler_.transform(vals))
        return self

    def transform(self, X):
        X = X.copy()
        scaled = self.scaler_.transform(X[self.cluster_features].fillna(0))
        X['PassengerCluster'] = self.kmeans_.predict(scaled)
        return X


def family_survival_rate(tickets, group_sum, group_count, global_rate):
    total_count = tickets.map(group_count).fillna(0.0)
    total_sum = tickets.map(group_sum).fillna(0.0)
    known = total_count > 0
    safe_count = total_count.where(known, 1)
    rate = (total_sum / safe_count).where(known, global_rate)
    return rate, known.astype(int)


def add_family_survival_oof(df, y, tickets_raw, cv):
    """OOF family-survival-rate for a partition that will itself be used in CV
    (dev partition). Group stats for each fold built only from that fold's
    training rows -- same design as the original notebook, just scoped to dev.

    `cv` must be a group-aware splitter (e.g. StratifiedGroupKFold) and is
    split here with `groups=tickets_raw`. A plain StratifiedKFold would let a
    ticket/family split across folds, so a held-out passenger's label could
    leak into a training passenger's FamilySurvivalRate via a shared ticket --
    exactly the leak this grouping closes.
    """
    df = df.copy()
    df['FamilySurvivalRate'] = np.nan
    df['FamilySurvivalKnown'] = 0
    for tr_idx, ho_idx in cv.split(df, y, groups=tickets_raw):
        fold_tickets = tickets_raw.iloc[tr_idx]
        fold_y = y.iloc[tr_idx]
        fold_group_sum = fold_y.groupby(fold_tickets).sum()
        fold_group_count = fold_tickets.value_counts()
        fold_global_rate = fold_y.mean()
        ho_tickets = tickets_raw.iloc[ho_idx]
        rate, known = family_survival_rate(ho_tickets, fold_group_sum, fold_group_count, fold_global_rate)
        df.iloc[ho_idx, df.columns.get_loc('FamilySurvivalRate')] = rate.values
        df.iloc[ho_idx, df.columns.get_loc('FamilySurvivalKnown')] = known.values
    return df


def fit_family_survival_stats(fit_y, fit_tickets_raw):
    group_sum = fit_y.groupby(fit_tickets_raw).sum()
    group_count = fit_tickets_raw.value_counts()
    global_rate = fit_y.mean()
    return group_sum, group_count, global_rate


def apply_family_survival(df, tickets_raw, group_sum, group_count, global_rate):
    df = df.copy()
    rate, known = family_survival_rate(tickets_raw, group_sum, group_count, global_rate)
    df['FamilySurvivalRate'] = rate.values
    df['FamilySurvivalKnown'] = known.values
    return df


# Raw feature columns fed INTO every model pipeline. PassengerCluster is
# deliberately excluded here -- it doesn't exist yet; ClusterFeatureAdder
# (prepended to every model's Pipeline below) synthesizes it fresh from
# whatever rows that Pipeline is fit on, so it gets refit per CV fold.
NUMERIC_COLS_BASE = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex', 'FamilySize', 'IsAlone',
                      'TicketGroupSize', 'FarePerPerson', 'FamilySurvivalRate',
                      'FamilySurvivalKnown', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
ORDINAL_NOMINAL_COLS = ['Title', 'Cabin']  # already-fitted ordinal codes, come in via FEATURE_COLS
CLUSTER_FEATURES = ['Age', 'FarePerPerson', 'Pclass', 'FamilySize']
FEATURE_COLS = NUMERIC_COLS_BASE + ORDINAL_NOMINAL_COLS
# Columns seen by the one-hot step, AFTER ClusterFeatureAdder has run and
# added PassengerCluster -- used only inside the LogReg pipeline below.
NOMINAL_COLS_FOR_ONEHOT = ORDINAL_NOMINAL_COLS + ['PassengerCluster']


def _with_cluster_step(clf):
    # Every candidate model gets its own ClusterFeatureAdder instance so
    # RandomizedSearchCV/cross_val_score refit it on each fold's training
    # rows only -- no fold's held-out rows influence its own cluster label.
    return Pipeline([
        ('cluster', ClusterFeatureAdder(CLUSTER_FEATURES)),
        ('clf', clf),
    ])


def build_tree_pipeline(clf):
    # Trees see the raw ordinal codes for Title/Cabin/PassengerCluster
    # directly (as in the original notebook) -- no one-hot needed.
    return _with_cluster_step(clf)


def build_logreg_pipeline(**lr_kwargs):
    # Trees can split on arbitrary integer codes for Title/Cabin/PassengerCluster,
    # but LogisticRegression would falsely assume an ordering between those
    # codes, so it gets its own one-hot + scaling step (fit fresh inside every
    # CV fold via the Pipeline -- no leakage).
    pre = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_COLS_BASE),
        ('nom', OneHotEncoder(handle_unknown='ignore'), NOMINAL_COLS_FOR_ONEHOT),
    ])
    inner = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=2000, **lr_kwargs))])
    return _with_cluster_step(inner)
