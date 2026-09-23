"""Preprocessing that is fitted on the training partition only (inside an sklearn Pipeline).

Columns are selected by name from a feature-manifest list — never by position.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ids.prep import CATEGORICAL


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def preprocessor(features, kind, log=False):
    """kind='tree': drop train-constant columns, no scaling (codes kept numeric).
    kind='linear': one-hot categorical codes (unknown -> all zeros), drop train-constant numeric, standardise.
    log=True (linear only): stateless sign(x)*log1p(|x|) before scaling — counts, bytes, durations and rates
    span many orders of magnitude; the signed form keeps the sentinel -1 and the (overflowed) negative
    header lengths finite and ordered."""
    features = list(features)
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    if kind == "tree":
        ct = ColumnTransformer([("all", VarianceThreshold(0.0), features)], remainder="drop",
                               verbose_feature_names_out=False)
    elif kind == "linear":
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", Pipeline([("const", VarianceThreshold(0.0))]
                             + ([("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))] if log else [])
                             + [("scale", StandardScaler())]), num),
        ], remainder="drop", verbose_feature_names_out=False)
    else:
        raise ValueError(kind)
    return ct.set_output(transform="pandas")
