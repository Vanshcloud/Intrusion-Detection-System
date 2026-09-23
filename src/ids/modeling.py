"""Milestone 4: data loading from the frozen v1 artifacts, model construction, fitting and scoring.

- Features are selected by name from configs/feature_manifest_v1.json and validated with check_features().
- Preprocessing (src/ids/pipeline.py) is fitted on the training rows only, then frozen inside the returned Pipeline.
- The only data a model ever sees at fit time is the training partition; validation is used for LightGBM
  early stopping (allowed by the protocol addendum) and, outside this module, for selection and thresholds.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ids.data import PROCESSED, ROOT
from ids.pipeline import preprocessor
from ids.prep import check_features

KEY = ["source_file", "source_row"]
META = ["label_original", "family", "is_attempted", "y_binary", "stratum"]
SEED = 20170703
N_JOBS = 8  # of 10 cores; leaves headroom for the OS


def manifest():
    return json.loads((ROOT / "configs/feature_manifest_v1.json").read_text())


def feature_list(name):
    m = manifest()
    return check_features(m["feature_sets"][name], m)


def load(protocol, feature_set, variant="primary", partitions=("train", "val", "test")):
    """Return {partition: (X, y, meta)} for one frozen split. X has exactly the manifest columns, in order."""
    feats = feature_list(feature_set)
    ds = pd.read_parquet(PROCESSED / "dataset_v1.parquet", columns=KEY + META + feats)
    split = pd.read_parquet(PROCESSED / f"split_{protocol}_v1.parquet", columns=KEY + ["partition"])
    novel = pd.read_parquet(PROCESSED / "novel_test_v1.parquet",
                            columns=KEY + [f"test_novel__{protocol}__{variant}__{feature_set}"])
    if not (novel[KEY].to_numpy() == ds[KEY].to_numpy()).all():
        raise ValueError("novel manifest not aligned with dataset")
    ds["novel"] = novel.iloc[:, -1].to_numpy()
    ds = ds.merge(split, on=KEY, how="left", validate="1:1")
    if ds["partition"].isna().any():
        raise ValueError("rows without partition")
    if variant == "excl_attempted":
        ds = ds[~ds["is_attempted"]]
    elif variant != "primary":
        raise ValueError(variant)
    out = {}
    for p in partitions:
        d = ds[ds["partition"] == p]
        X = d[feats]  # native dtypes (int64/float64) as stored; no lossy casting
        assert list(X.columns) == feats
        out[p] = (X.reset_index(drop=True), d["y_binary"].to_numpy().astype(int),
                  d[KEY + META + ["novel"]].reset_index(drop=True))
    return out


# ------------------------------------------------------------------ models
def make_estimator(family, params):
    p = dict(params)
    if family == "lr":
        return LogisticRegression(solver="lbfgs", max_iter=p.pop("max_iter", 2000), random_state=SEED, **p)
    if family == "rf":
        return RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **p)
    if family == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", n_jobs=N_JOBS, random_state=SEED, deterministic=True,
                                  force_col_wise=True, verbose=-1, **p)
    raise ValueError(family)


def fit(family, params, feats, X_tr, y_tr, X_va=None, y_va=None, early_stopping_rounds=None):
    """Fit preprocessing on train only, then the estimator. Returns (Pipeline, info)."""
    pre = preprocessor(feats, "linear" if family == "lr" else "tree", log=(family == "lr")).fit(X_tr)
    Xt = pre.transform(X_tr)
    est = make_estimator(family, params)
    info = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if family == "lgbm" and early_stopping_rounds:
            import lightgbm as lgb
            est.fit(Xt, y_tr, eval_set=[(pre.transform(X_va), y_va)], eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            info["best_iteration"] = int(est.best_iteration_)
        else:
            est.fit(Xt, y_tr)
    info["fit_seconds"] = time.perf_counter() - t0
    conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    info["convergence_warnings"] = conv
    if family == "lr":
        info["n_iter"] = int(np.max(est.n_iter_))
        info["converged"] = not conv and info["n_iter"] < est.max_iter
    info["n_inputs_after_preprocessing"] = int(Xt.shape[1])
    info["dropped_constant_in_train"] = sorted(set(feats) - set(pre.get_feature_names_out()) - _onehot_sources(pre))
    return Pipeline([("pre", pre), ("est", est)]), info


def _onehot_sources(pre):
    return {c for name, _, cols in pre.transformers_ if name == "cat" for c in cols}


def attack_score(model, X):
    """Probability of the attack class (y=1); higher = more likely attack."""
    est = model.named_steps["est"]
    classes = list(est.classes_)
    if classes != [0, 1]:
        raise ValueError(f"unexpected class order {classes}")
    # sklearn forests sum per-tree probabilities across threads in nondeterministic order (1-ulp
    # differences, measured). Scores that tie with a frozen threshold could flip, so predict single-threaded.
    n_jobs = getattr(est, "n_jobs", None)
    if isinstance(est, RandomForestClassifier):
        est.n_jobs = 1
    try:
        t0 = time.perf_counter()
        s = model.predict_proba(X)[:, 1]
        secs = time.perf_counter() - t0
    finally:
        if isinstance(est, RandomForestClassifier):
            est.n_jobs = n_jobs
    return s, secs
