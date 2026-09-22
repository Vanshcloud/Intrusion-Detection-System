"""Preprocessing that is fitted on the training partition only (inside an sklearn Pipeline).

Columns are selected by name from a feature-manifest list — never by position.
"""
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ids.prep import CATEGORICAL


def preprocessor(features, kind):
    """kind='tree': drop train-constant columns, no scaling (codes kept numeric).
    kind='linear': one-hot categorical codes (unknown -> all zeros), drop train-constant numeric, standardise."""
    features = list(features)
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    if kind == "tree":
        ct = ColumnTransformer([("all", VarianceThreshold(0.0), features)], remainder="drop",
                               verbose_feature_names_out=False)
    elif kind == "linear":
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", Pipeline([("const", VarianceThreshold(0.0)), ("scale", StandardScaler())]), num),
        ], remainder="drop", verbose_feature_names_out=False)
    else:
        raise ValueError(kind)
    return ct.set_output(transform="pandas")
