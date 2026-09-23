"""Milestone 4: training pipeline, scoring, evaluation-from-predictions (synthetic + real artifacts)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ids.data import PROCESSED, ROOT
from ids.metrics import select_threshold, wilson
from ids.modeling import attack_score, feature_list, fit, load

sys.path.insert(0, str(ROOT / "scripts"))
from m4_evaluate import metrics_from_predictions  # noqa: E402

FEATS = ["Protocol", "Flow Duration", "Total Fwd Packet", "Fwd URG Flags"]


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"Protocol": rng.choice([6, 17], n), "Flow Duration": rng.exponential(1e5, n) * (1 + 50 * y),
                      "Total Fwd Packet": rng.integers(1, 50, n) + 30 * y, "Fwd URG Flags": np.zeros(n, int)})
    return X, y


@pytest.mark.parametrize("family,params", [("lr", {"C": 1.0}), ("rf", {"n_estimators": 20, "min_samples_leaf": 2}),
                                           ("lgbm", {"n_estimators": 30, "num_leaves": 7})])
def test_fit_score_orientation_serialisation_and_determinism(family, params, tmp_path):
    X, y = synthetic()
    m1, info = fit(family, params, FEATS, X[:300], y[:300])
    s1, _ = attack_score(m1, X[300:])
    assert np.corrcoef(s1, y[300:])[0, 1] > 0.5  # score = P(attack): higher for attacks
    assert np.allclose(s1, m1.predict_proba(X[300:])[:, 1])
    assert info["dropped_constant_in_train"] == ["Fwd URG Flags"]  # constant in train -> dropped
    joblib.dump(m1, tmp_path / "m.joblib")
    assert np.array_equal(s1, attack_score(joblib.load(tmp_path / "m.joblib"), X[300:])[0])
    m2, _ = fit(family, params, FEATS, X[:300], y[:300])
    assert np.array_equal(s1, attack_score(m2, X[300:])[0])  # same seed -> identical predictions


def test_lr_preprocessing_uses_training_rows_only():
    X, y = synthetic()
    m, info = fit("lr", {"C": 1.0}, FEATS, X[:300], y[:300])
    sc = m.named_steps["pre"].named_transformers_["num"].named_steps["scale"]
    logged = np.sign(X[:300][["Flow Duration", "Total Fwd Packet"]]) * np.log1p(np.abs(X[:300][["Flow Duration", "Total Fwd Packet"]]))
    assert np.allclose(sc.mean_, logged.mean().to_numpy())
    assert info["converged"] is True


def test_threshold_ties_are_all_on_one_side():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.9])  # three rows tied at 0.5, one benign-heavy tie
    t = select_threshold(y, s, max_fpr=0.0)
    assert t == 0.9  # 0.5 would flag two benign rows at once; no partial tie is possible


def test_wilson_interval():
    lo, hi = wilson(0, 3)
    assert lo == 0.0 and hi == pytest.approx(0.5615, abs=1e-3)
    lo, hi = wilson(3, 3)
    assert lo == pytest.approx(0.4385, abs=1e-3) and hi == 1.0
    assert wilson(0, 0) == (None, None)


def test_metrics_from_predictions_known_values():
    p = pd.DataFrame({"y_binary": [0, 0, 0, 0, 1, 1, 1, 1],
                      "score": [.1, .2, .7, .1, .9, .8, .3, .95],
                      "label_original": ["BENIGN"] * 3 + ["X - Attempted", "A", "A", "B", "Portscan"],
                      "family": ["BENIGN"] * 4 + ["A", "A", "B", "Portscan"],
                      "novel": [True, False, True, True, False, False, True, False]})
    summ, lab = metrics_from_predictions(p, {"0.5": 0.5}, "t")
    full = next(r for r in summ if r["subset"] == "full_test")
    assert (full["tp"], full["fp"], full["tn"], full["fn"]) == (3, 1, 3, 1) and full["fpr"] == 0.25
    nov = next(r for r in summ if r["subset"] == "novel_test")
    assert (nov["n"], nov["attack"], nov["tp"], nov["fn"]) == (4, 1, 0, 1) and nov["recall"] == 0.0
    a = next(r for r in lab if r["subset"] == "full_test" and r["label"] == "A" and r["level"] == "label_original")
    assert (a["n"], a["flagged"], a["rate"]) == (2, 2, 1.0) and a["unstable"]
    # a label absent from the novel subset has no row at all (undefined), never a 0 or 100 % rate
    assert not [r for r in lab if r["subset"] == "novel_test" and r["label"] == "A"]


# ---------------------------------------------------------------- real artifacts
needs_data = pytest.mark.skipif(not (PROCESSED / "dataset_v1.parquet").exists(), reason="processed data missing")


@needs_data
def test_load_real_split_features_and_targets():
    d = load("chrono", "no_header_length", partitions=("val", "test"))
    feats = feature_list("no_header_length")
    for p, (X, y, meta) in d.items():
        assert list(X.columns) == feats and set(np.unique(y)) == {0, 1}
        assert not {"Label", "y_binary", "label_original", "Attempted Category", "Src IP", "Dst IP"} & set(X.columns)
        assert np.array_equal(y, meta["y_binary"].to_numpy())
    assert not d["val"][2]["novel"].any() and d["test"][2]["novel"].any()  # novel flags exist on test rows only
    e = load("chrono", "default", "excl_attempted", partitions=("test",))["test"]
    assert not e[2]["is_attempted"].any()


FROZEN = ROOT / "configs" / "m4_frozen.json"
TEST_DIR = ROOT / "artifacts" / "m4" / "test"


@pytest.mark.skipif(not (TEST_DIR / "summary.csv").exists(), reason="test evaluation not run")
def test_reported_metrics_recompute_from_stored_predictions():
    fz = json.loads(FROZEN.read_text())
    stored = pd.read_csv(TEST_DIR / "summary.csv")
    for e in fz["experiments"]:
        tag = f"{e['experiment']}__{e['family']}"
        p = pd.read_parquet(TEST_DIR / f"{tag}__test_predictions.parquet")
        summ, _ = metrics_from_predictions(p, e["thresholds"], tag)
        again = pd.DataFrame(summ)
        mine = stored[stored.experiment == tag].reset_index(drop=True)
        pd.testing.assert_frame_equal(again[mine.columns], mine, check_dtype=False, atol=1e-12)
        run = json.loads((ROOT / "artifacts/m4/runs" / f"{e['run_id']}.json").read_text())
        assert {k: run["val"][f"b{k}"]["threshold"] for k in e["thresholds"]} == e["thresholds"]
