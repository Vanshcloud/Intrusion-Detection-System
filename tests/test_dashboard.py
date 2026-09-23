"""Milestone 6 dashboard: data-layer consistency with the frozen artifacts, number-provenance guard, app smoke test.

Tests marked `needs_local` also use git-ignored artifacts (stored predictions, frozen model) and are skipped in a clone.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import data as D
from ids.data import ROOT
from ids.metrics import UNSTABLE_SUPPORT

PRED = ROOT / "artifacts/m4/test/E1_primary__lgbm__test_predictions.parquet"
needs_local = pytest.mark.skipif(not PRED.exists(), reason="local stored predictions (artifacts/m4) missing")
needs_cases = pytest.mark.skipif(not D.demo_cases_path().exists(),
                                 reason="local case rows (artifacts/m6/demo_cases.csv) missing; exact rows are not committed")
PAGES = ["overview", "evaluation", "attack_analysis", "explainability", "false_positives", "novel_vectors",
         "prediction_demo", "methodology"]


# ---------------------------------------------------------------- metric consistency (committed artifacts only)
def test_headline_matches_frozen_config_and_m4_tables():
    h, fz = D.headline(), json.loads((ROOT / "configs/m4_frozen.json").read_text())
    e = next(x for x in fz["experiments"] if x["run_id"] == fz["overall_selected"])
    assert h["run_id"] == fz["overall_selected"] and h["threshold"] == e["thresholds"]["0.001"]
    assert D.primary_budget() == "0.001"  # strictest pre-declared budget = the selection operating point
    s = pd.read_csv(ROOT / "reports/generated/m4_test_summary.csv", float_precision="round_trip")
    r = s[(s.experiment == "E1_primary__lgbm") & (s.subset == "full_test") & (s.budget == 0.001)].iloc[0]
    for k in ("tp", "fp", "tn", "fn", "recall", "fpr", "precision", "pr_auc"):
        assert h[k] == r[k], k
    assert h["recall"] == pytest.approx(h["tp"] / (h["tp"] + h["fn"]), abs=1e-15)
    assert h["fpr"] == pytest.approx(h["fp"] / (h["fp"] + h["tn"]), abs=1e-15)
    t = pd.read_csv(ROOT / "reports/generated/m4_frozen_thresholds.csv", float_precision="round_trip")
    assert h["val_fpr"] == pytest.approx(t[(t.experiment == "E1_primary__lgbm") & (t.target_fpr == 0.001)].val_fpr.iloc[0])


def test_dataset_partition_sizes_add_up():
    o, h = D.dataset_overview(), D.headline()
    assert o["n_train"] + o["n_val"] + o["n_test"] + o["n_purged"] == o["rows_clean"] == o["rows_raw"] - o["rows_removed"]
    assert o["n_test"] == h["n"]


def test_model_comparison_and_confusion():
    c = D.model_comparison()
    assert len(c) == 6 and c.selected.sum() == 1 and c[c.selected].model.iloc[0] == D.headline()["family"]
    cm, h = D.confusion(), D.headline()
    assert cm.to_numpy().tolist() == [[h["tn"], h["fp"]], [h["fn"], h["tp"]]] and cm.to_numpy().sum() == h["n"]


def test_family_recall_counts_and_small_support():
    fr, h = D.family_recall(), D.headline()
    assert (fr.detected + fr.missed == fr.n).all()
    assert fr.detected.sum() == h["tp"] and fr.missed.sum() == h["fn"]
    assert set(fr[fr.small_support].label) == set(fr[fr.n < UNSTABLE_SUPPORT].label)
    hb = fr.set_index("label").loc["Heartbleed"]
    assert hb.detected_of_n == f"{int(hb.detected)}/{int(hb.n)} detected"
    ba = D.benign_alerts()
    assert ba.flagged.sum() == h["fp"] and ba.n.sum() == h["benign"]


def test_fp_story_decomposes_and_matches_milestone4():
    fs, h = D.fp_story(), D.headline()
    assert fs["rev"]["fp"] + fs["other"]["fp"] == h["fp"]
    assert fs["rev"]["benign"] + fs["other"]["benign"] == h["benign"]
    m4 = pd.read_csv(ROOT / "reports/generated/m4_test_fp_breakdown.csv").set_index("experiment").loc["E1_primary__lgbm"]
    assert fs["rev"]["fp"] == m4.fp_reverse_of_same_day_attack_pair  # independent Milestone 4 computation
    assert fs["by_daygroup"].fp.sum() == h["fp"] and fs["by_daygroup"].benign_test.sum() == h["benign"]


def test_novel_vector_statistics():
    r, t = D.novel_overview()
    lg = t[(t.model == "LightGBM") & (t.subset == "novel_test")].iloc[0]
    assert (lg.n, lg.attack) == (r["novel_test"], r["attack_novel"])
    comp = D.novel_composition()
    assert comp.novel.sum() == r["attack_novel"] and comp.full.sum() == r["attack_full"]
    assert comp.share_of_attacks_novel.sum() == pytest.approx(1.0)


def test_shap_ranking_and_groups():
    s = D.shap_global()
    imp = s["importance"]
    assert list(imp["rank"]) == list(range(1, len(imp) + 1)) and imp.share_all.is_monotonic_decreasing
    assert imp.share_all.sum() == pytest.approx(1.0, abs=1e-5) and s["groups"].sum() == pytest.approx(1.0, abs=1e-5)
    assert len(imp) == s["n_inputs"]


def test_score_histograms_cover_populations_exactly():
    h = D.headline()
    assert D.score_hist("test outcome").n.sum() == h["n"]
    assert D.score_hist("test benign by day group").n.sum() == h["benign"]
    assert D.score_hist("test benign by day group and direction").n.sum() == h["benign"]
    v = D.selected()["val_at_threshold"][D.primary_budget()]
    assert D.score_hist("val benign by day group").n.sum() == v["tn"] + v["fp"]
    o = D.score_hist("test outcome").groupby("group").n.sum()
    assert o.to_dict() == {"FN": h["fn"], "FP": h["fp"], "TN": h["tn"], "TP": h["tp"]}


# ---------------------------------------------------------------- demo cases (local) and live model
@needs_cases
def test_demo_cases_are_self_consistent():
    c = D.demo_cases()
    assert c.source_row.notna().all() and not c.duplicated(["source_file", "source_row"]).any()
    from ids.modeling import feature_list
    assert D.demo_features() == feature_list("default")  # the frozen model's raw inputs, in manifest order
    assert set(c.outcome) == {"TP", "FP", "TN", "FN"}
    thr = D.headline()["threshold"]
    exp = np.select([c.y_binary.eq(1) & (c.score >= thr), c.y_binary.eq(0) & (c.score >= thr), c.y_binary.eq(0)],
                    ["TP", "FP", "TN"], "FN")
    assert (c.outcome == exp).all()
    for i in range(len(c)):  # stored SHAP values reproduce the stored score (log-odds additivity)
        assert D.reconstructed_score(c.iloc[i]) == pytest.approx(c.score.iloc[i], abs=1e-9)
    assert c.case.notna().sum() == len(pd.read_csv(ROOT / "reports/generated/m5_casebook.csv"))


@needs_cases
def test_case_contributions_and_index_validation():
    row = D.demo_cases().iloc[0]
    pos, neg = D.case_contributions(row, 5)
    assert (pos.shap > 0).all() and (neg.shap < 0).all() and pos.shap.is_monotonic_decreasing
    assert len(pos) <= 5 and len(neg) <= 5
    for bad in (-1, len(D.demo_cases()), 1.5, "3", True, None):
        with pytest.raises(ValueError):
            D.validate_case_index(bad)
    assert D.validate_case_index(np.int64(0)) == 0


def test_load_model_missing_and_hash_mismatch(tmp_path):
    m, msg = D.load_model(tmp_path / "absent.joblib")
    assert m is None and "not found" in msg
    f = tmp_path / "fake.joblib"
    f.write_bytes(b"not the frozen model")
    m, msg = D.load_model(f)
    assert m is None and "does not match" in msg


@needs_local
@needs_cases
def test_demo_scores_equal_stored_predictions():
    c = D.demo_cases()
    p = pd.read_parquet(PRED).merge(c[["source_file", "source_row", "score"]], on=["source_file", "source_row"],
                                    suffixes=("", "_demo"))
    assert len(p) == len(c) and (p.score == p.score_demo).all()  # exact, thanks to round-trip CSV parsing


def test_committed_model_export_record_is_consistent():
    """The text export is a representation of the frozen model: hashes link it to configs/m4_frozen.json."""
    import hashlib
    rec, sel = D.model_export(), D.selected()
    assert rec["source_model_sha256"] == sel["model_sha256"] and rec["source_run_id"] == sel["run_id"]
    assert rec["frozen_threshold_0.001"] == sel["thresholds"]["0.001"] and rec["best_iteration"] == sel["best_iteration"]
    f = ROOT / rec["export_path"]
    assert hashlib.sha256(f.read_bytes()).hexdigest() == rec["export_sha256"] and f.stat().st_size == rec["export_bytes"]
    from ids.modeling import feature_list
    assert set(rec["input_features"]) == set(D.shap_global()["importance"].feature)  # same inputs as the SHAP census
    assert rec["input_features"] == [f for f in feature_list("default") if f in set(rec["input_features"])]  # manifest order
    assert len(rec["input_features"]) == D.facts()["n_model_inputs"]
    v = rec["verification"]
    assert v["test_rows"] == D.headline()["n"] and all(x for x in v.values() if isinstance(x, bool))


@needs_cases
def test_live_export_reproduces_every_demo_case_exactly():
    """Committed export vs stored frozen scores and SHAP values of all local demo cases."""
    booster, msg = D.load_model()
    assert booster is not None, msg
    c = D.demo_cases()
    for i in range(len(c)):
        live = D.live_explain(booster, c.iloc[i])
        assert live["score_equal_stored"] and live["base_equal_stored"] and live["shap_max_abs_diff"] < D.SHAP_TOL


# ---------------------------------------------------------------- number-provenance guard
ALLOWED = re.compile(
    r"CIC-IDS2017|SHA-256|\b\w*sha256\w*|(?:configs|reports|scripts|dashboard|data|artifacts)/[\w./-]*|\bm[3-6]_\w+|protocol-v1|\bv1\b|§\d+(?:–§\d+)?|"
    r"Milestones? \d(?:[-–]\d)?|\b95 %|\b\w*top-?\d+\w*|\bci95_\w+|\bE[12]_\w+|\be1e2\b|\b[fF]1\b|"
    r"#[0-9a-f]{6}|^[{:,+.]*\d*[a-z%]}?$|\{:\+\.\d+f\}|\(1\)|\(0\)|fontsize=\d+")
LAYOUT_KW = {"height", "width", "labelLimit", "dx", "dy"}


def _numbers_in_dashboard_code():
    files = sorted((ROOT / "dashboard/pages").glob("*.py")) + [ROOT / "dashboard/app.py", ROOT / "dashboard/ui.py"]
    bad = []
    for f in files:
        tree = ast.parse(f.read_text())
        parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        specs = {id(c) for n in ast.walk(tree) if isinstance(n, ast.FormattedValue) and n.format_spec
                 for c in ast.walk(n.format_spec)}

        def layout(n):
            while n in parent:
                n = parent[n]
                if isinstance(n, ast.keyword) and n.arg in LAYOUT_KW:
                    return True
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "head":
                    return True
            return False
        for n in ast.walk(tree):
            if not isinstance(n, ast.Constant) or id(n) in specs or isinstance(n.value, bool):
                continue
            if isinstance(n.value, str) and re.search(r"\d", ALLOWED.sub("", n.value)):
                bad.append(f"{f.name}:{n.lineno} {n.value[:60]!r}")
            elif isinstance(n.value, float) or (isinstance(n.value, int) and n.value > 10 and not layout(n)):
                bad.append(f"{f.name}:{n.lineno} {n.value!r}")
    return bad


def test_dashboard_pages_contain_no_typed_numbers():
    """Every data value shown must come from dashboard.data; pages may only hold identifiers and layout sizes."""
    assert _numbers_in_dashboard_code() == []


def test_guard_catches_a_typed_number():
    assert re.search(r"\d", ALLOWED.sub("", "Test FPR was 0.58 % on test"))
    assert not re.search(r"\d", ALLOWED.sub("", "see reports/generated/m4_test_summary.csv §17"))


# ---------------------------------------------------------------- figures and app smoke test
def test_dashboard_figures_exist_and_are_the_report_figures():
    report = (ROOT / "reports/EXPLAINABILITY_ANALYSIS.md").read_text()
    for key, name in D.FIGURES.items():
        assert D.figure(key) is not None, key
        assert f"figures/m5/{name}" in report, name


def test_app_pages_render_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for p in PAGES:
        at.switch_page(f"pages/{p}.py")
        at.run()
        assert not at.exception, (p, [e.message for e in at.exception])
        assert at.title and "not a live network monitor" in at.info[0].value


@needs_cases
def test_case_explorer_without_model(monkeypatch, tmp_path):
    """Missing frozen model: the explorer still renders from stored values and says why the live check is skipped."""
    from streamlit.testing.v1 import AppTest
    import dashboard.ui as ui
    monkeypatch.setattr(D, "model_path", lambda: tmp_path / "absent.joblib")
    ui.model.clear()
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=120)
    at.run()
    at.switch_page("pages/prediction_demo.py")
    at.run()
    assert not at.exception
    assert any("live check is skipped" in i.value for i in at.info)
    ui.model.clear()


def test_every_cited_repository_path_exists():
    """Figure/table/doc paths cited by the dashboard pages and its README resolve to real files or directories."""
    text = "".join(f.read_text() for f in sorted((ROOT / "dashboard").rglob("*.py"))) + (ROOT / "dashboard/README.md").read_text()
    paths = set(re.findall(r"(?:configs|reports|scripts|dashboard|data)/[\w./-]*\w/?", text))
    missing = [p for p in sorted(paths)
               if not ((ROOT / p).exists() or (p.endswith("_") and any(ROOT.glob(p + "*"))))]  # "m3_*" style prefixes
    assert paths and missing == []


def test_case_explorer_public_mode_without_local_rows(monkeypatch, tmp_path):
    """Clone without the dataset: no exact rows; the explorer shows the committed Milestone 5 casebook and says why."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(D, "demo_cases_path", lambda: tmp_path / "absent.csv")
    assert D.demo_cases() is None
    with pytest.raises(ValueError):
        D.validate_case_index(0)
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=120)
    at.run()
    at.switch_page("pages/prediction_demo.py")
    at.run()
    assert not at.exception
    assert any("not included in this public repository" in i.value for i in at.info)
    cb = D.casebook()
    assert list(at.selectbox[0].options) == list(cb.case)
    assert at.metric[0].value == f"{cb.score.iloc[0]:.4f}"


def test_no_committed_table_contains_exact_benchmark_rows():
    """Exact source rows (row IDs + most model input values) must not be committed (no verified redistribution licence)."""
    import subprocess
    inputs = set(D.model_export()["input_features"])
    tracked = subprocess.run(["git", "ls-files", "*.csv"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    bad = []
    for f in tracked:
        cols = {c.removeprefix("f__") for c in pd.read_csv(ROOT / f, nrows=0).columns}
        if "source_row" in cols and len(cols & inputs) > len(inputs) // 2:
            bad.append(f)
    assert tracked and bad == []
