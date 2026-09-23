"""Milestone 5: SHAP helpers (synthetic) and the stored explanation artifacts (real, skipped when absent)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ids.data import ROOT, sha256
from ids.explain import (FEATURE_GROUP, attack_pairs, contributions, group_shap, importance, is_reverse_of_attack,
                         model_inputs, outcome, quantiles, stratified_sample, topk_overlap)
from ids.modeling import KEY, feature_list, fit

FEATS = ["Protocol", "Flow Duration", "Total Fwd Packet", "Fwd URG Flags"]


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"Protocol": rng.choice([6, 17], n), "Flow Duration": rng.exponential(1e5, n) * (1 + 50 * y),
                      "Total Fwd Packet": rng.integers(1, 50, n), "Fwd URG Flags": np.zeros(n, int)})
    return X, y


@pytest.mark.parametrize("family,params,space", [("lgbm", {"n_estimators": 30, "num_leaves": 7}, "log_odds"),
                                                 ("rf", {"n_estimators": 20, "min_samples_leaf": 2}, "probability")])
def test_contributions_additivity_order_and_mapping(family, params, space):
    X, y = synthetic()
    m, _ = fit(family, params, FEATS, X[:300], y[:300])
    Xe = X[300:]
    sv, base, sp = contributions(m, Xe)
    assert sp == space
    assert list(sv.columns) == model_inputs(m) == ["Protocol", "Flow Duration", "Total Fwd Packet"]  # URG constant -> dropped
    assert sv.index.equals(Xe.index)  # rows keep their identity
    s = m.predict_proba(Xe)[:, 1]
    recon = base + sv.sum(axis=1).to_numpy()
    assert np.allclose(1 / (1 + np.exp(-recon)) if space == "log_odds" else recon, s, atol=1e-9)
    assert importance(sv).index[0] == "Flow Duration"  # the only informative feature carries the attribution


def test_contributions_rejects_feature_order_mismatch():
    X, y = synthetic()
    m, _ = fit("lgbm", {"n_estimators": 10, "num_leaves": 7}, FEATS, X[:300], y[:300])
    m.named_steps["est"].booster_.feature_name = lambda: ["Flow_Duration", "Protocol", "Total_Fwd_Packet"]
    with pytest.raises(ValueError, match="order"):
        contributions(m, X[300:])


def test_outcome_groups():
    o = outcome([1, 1, 0, 0, 0], [0.9, 0.1, 0.5, 0.49, 0.5], 0.5)
    assert list(o) == ["TP", "FN", "FP", "TN", "FP"]  # score == threshold is flagged


def test_stratified_sample_deterministic_and_order_independent():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"source_file": "a.csv", "source_row": np.arange(1000),
                       "g": rng.choice(["x", "y", "z"], 1000, p=[0.8, 0.19, 0.01])})
    a = stratified_sample(df, ["g"], 50, seed=7)
    b = stratified_sample(df.sample(frac=1, random_state=3), ["g"], 50, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = a.g.value_counts()
    assert c["x"] == 50 and c["y"] == 50 and c["z"] == (df.g == "z").sum()  # small groups are kept whole
    assert list(a.columns) == list(df.columns) and a.source_row.is_unique
    assert not stratified_sample(df, ["g"], 50, seed=8).equals(a)


def test_reverse_pair_flags():
    meta = pd.DataFrame({"day": ["thu", "thu", "thu", "fri"], "Src IP": ["A", "B", "B", "B"],
                         "Dst IP": ["B", "A", "C", "A"], "y_binary": [1, 0, 0, 0]})
    assert list(is_reverse_of_attack(meta, attack_pairs(meta))) == [False, True, False, False]  # same day only


def test_group_shap_is_additive_and_covers_all_features():
    assert set(feature_list("default")) == set(FEATURE_GROUP)
    sv = pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=["Flow Duration", "Flow IAT Max", "Protocol", "SYN Flag Count"])
    g = group_shap(sv)
    assert np.allclose(g.sum(axis=1), sv.sum(axis=1)) and np.allclose(g["timing"], [1.0, 9.0])


def test_quantiles_and_topk():
    q = quantiles([1, 2, 3, 4, 10], ["a", "a", "a", "b", "b"], qs=(0.5,))
    assert q.loc["a", "n"] == 3 and q.loc["a", "q0.5"] == 2 and q.loc["b", "q0.5"] == 7
    a = pd.Series([3, 2, 1], index=["x", "y", "z"])
    b = pd.Series([3, 2, 1], index=["y", "z", "x"])
    assert topk_overlap(a, b, 2) == 0.5


# ---------------------------------------------------------------- real artifacts
ART = ROOT / "artifacts" / "m5"
needs_m5 = pytest.mark.skipif(not (ART / "run.json").exists(), reason="Milestone 5 artifacts not computed")


@needs_m5
def test_frozen_model_and_preprocessor_match_manifest():
    fz = json.loads((ROOT / "configs/m4_frozen.json").read_text())
    e = next(x for x in fz["experiments"] if x["run_id"] == fz["overall_selected"])
    assert sha256(ROOT / e["model_path"]) == e["model_sha256"]
    m = joblib.load(ROOT / e["model_path"])
    run = json.loads((ART / "run.json").read_text())
    assert model_inputs(m) == run["model_inputs"] == [f for f in feature_list("default") if f != "Bwd URG Flags"]
    assert run["E1_lgbm"]["space"] == "log_odds" and run["checks"]["reload_bitwise_identical"]
    assert run["checks"]["shap_library_vs_native_max_abs_diff"] < 1e-9


@needs_m5
def test_stored_shap_census_additivity_ids_and_novelty():
    run = json.loads((ART / "run.json").read_text())
    s = pd.read_parquet(ART / "shap_E1_lgbm_test.parquet")
    p = pd.read_parquet(ROOT / "artifacts/m4/test/E1_primary__lgbm__test_predictions.parquet")
    assert (s[KEY].to_numpy() == p[KEY].to_numpy()).all()  # row IDs preserved, same order as the frozen predictions
    recon = run["E1_lgbm"]["base"] + s[run["model_inputs"]].sum(axis=1).to_numpy()
    assert np.abs(1 / (1 + np.exp(-recon)) - p.score.to_numpy()).max() < 1e-9
    strat = pd.read_parquet(ART / "sample_stratified.parquet").merge(p, on=KEY, validate="1:1")
    fp = (p.y_binary == 0) & (p.score >= run["provenance"]["models"]["E1_primary__lgbm"]["threshold_0.001"])
    assert (strat.group == "FP").sum() == fp.sum()  # every false positive is in the explanation sample
    nov = strat.group.str.endswith("|novel")
    assert strat.novel[nov].all() and not strat.novel[strat.group.str.endswith("|dup")].any()


@needs_m5
def test_generated_tables_consistent_with_m4():
    g = ROOT / "reports" / "generated"
    f = json.loads((g / "m5_facts.json").read_text())
    m4 = pd.read_csv(g / "m4_test_summary.csv")
    r = m4[(m4.experiment == "E1_primary__lgbm") & (m4.subset == "full_test") & (m4.budget == 0.001)].iloc[0]
    assert f["outcomes"] == {"TN": r.tn, "FP": r.fp, "TP": r.tp, "FN": r.fn}
    assert len(pd.read_csv(g / "m5_fn_cases.csv")) == r.fn
    d = pd.read_csv(g / "m5_fp_by_day.csv")
    assert d[d.partition == "test"].fp.sum() == r.fp
    old = pd.read_csv(g / "m4_test_fp_by_day.csv").query("experiment == 'E1_primary__lgbm'").set_index("day")
    assert (d[d.partition == "test"].set_index("day").fp == old.fp).all()
