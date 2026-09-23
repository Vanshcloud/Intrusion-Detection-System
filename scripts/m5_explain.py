"""Milestone 5, compute stage: SHAP values for the frozen models (configs/m4_frozen.json); nothing is trained.

Writes artifacts/m5/ (git-ignored):
  shap_E1_lgbm_test.parquet   selected LightGBM, every chronological-test row (census, log-odds space)
  shap_E1_lgbm_val.parquet    selected LightGBM, every validation row (for the validation->test FPR shift)
  shap_E2_lgbm_test.parquet   random-split LightGBM, every random-split test row
  sample_uniform.parquet      uniform random sample of chronological-test rows (plots, bootstrap)
  sample_stratified.parquet   stratified explanation sample (all FP, all FN, capped TN/TP groups)
  shap_E1_rf_sample.parquet   frozen Random Forest on the stratified sample (probability space)
  interactions_uniform.parquet mean |SHAP interaction| of the selected LightGBM on the first rows of the uniform sample
  run.json                    versions, hashes, seeds, checks
Every SHAP table passes an additivity check against the model's own scores, and the recomputed scores must equal
the stored Milestone 4 predictions exactly.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import PROCESSED, ROOT, sha256  # noqa: E402
from ids.explain import contributions, model_inputs, outcome, stratified_sample  # noqa: E402
from ids.modeling import KEY, SEED, load  # noqa: E402

OUT = ROOT / "artifacts" / "m5"
M4 = ROOT / "artifacts" / "m4"
FROZEN = ROOT / "configs" / "m4_frozen.json"
BUDGET = "0.001"  # the headline 0.1 % FPR threshold of Milestone 4
N_UNIFORM, N_INTERACT, CAP = 5000, 1000, 150
EXTRA_META = ["day", "Src IP", "Dst IP", "Src Port", "Dst Port", "t_start"]


def frozen_entry(experiment, family):
    fz = json.loads(FROZEN.read_text())
    e = next(x for x in fz["experiments"] if x["experiment"] == experiment and x["family"] == family)
    if sha256(ROOT / e["model_path"]) != e["model_sha256"]:
        sys.exit(f"frozen model changed: {e['model_path']}")
    return e


def with_meta(meta):
    extra = pd.read_parquet(PROCESSED / "dataset_v1.parquet", columns=KEY + EXTRA_META)
    return meta.merge(extra, on=KEY, how="left", validate="1:1")


def check_scores(meta, model, X, stored_path):
    """Recompute scores from the frozen model and require exact equality with the stored M4 predictions."""
    from ids.modeling import attack_score
    s, _ = attack_score(model, X)
    st = pd.read_parquet(stored_path)
    if not (st[KEY].to_numpy() == meta[KEY].to_numpy()).all():
        raise ValueError(f"row order differs from {stored_path}")
    if not np.array_equal(st["score"].to_numpy(), s):
        raise ValueError(f"scores differ from {stored_path}")
    return s


def save(sv, meta, name):
    sv.reset_index(drop=True).assign(**{k: meta[k].to_numpy() for k in KEY})[KEY + list(sv.columns)] \
        .to_parquet(OUT / name, index=False)


def verify(other):
    """Compare artifacts/m5 with an independent rerun (--out DIR) and hash the generated report outputs.

    Writes reports/generated/m5_reproducibility.json. Output hashes are compared with the previous verify run,
    so: verify -> rerun scripts/m5_analyze.py -> verify shows whether tables/figures are byte-identical."""
    from ids.data import GENERATED
    other = Path(other)
    res = {"compared_with": "independent rerun of scripts/m5_explain.py --out <dir>",
           "artifact_contents_equal": {p.name: bool(pd.read_parquet(p).equals(pd.read_parquet(other / p.name)))
                                       for p in sorted(OUT.glob("*.parquet"))}}
    r1, r2 = (json.loads((d / "run.json").read_text()) for d in (OUT, other))
    res["run_json_equal_except_time"] = all(r1[k] == r2[k] for k in r1 if k not in ("created_utc", "timing_s"))
    rep = GENERATED / "m5_reproducibility.json"
    prev = json.loads(rep.read_text()).get("output_sha256") if rep.exists() else None
    files = sorted([*GENERATED.glob("m5_*.csv"), GENERATED / "m5_facts.json", *(ROOT / "reports/figures/m5").glob("*.png")])
    res["output_sha256"] = {str(f.relative_to(ROOT)): sha256(f) for f in files}
    res["outputs_identical_to_previous_verify"] = None if prev is None else prev == res["output_sha256"]
    rep.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "output_sha256"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="alternative output dir (reproducibility check)")
    ap.add_argument("--verify", default=None, help="compare artifacts/m5 with a rerun in this dir; hash outputs")
    a = ap.parse_args()
    if a.verify:
        return verify(a.verify)
    global OUT
    OUT = Path(a.out) if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    run = {"created_utc": pd.Timestamp.now("UTC").isoformat(), "seed": SEED, "budget": float(BUDGET),
           "checks": {}, "timing_s": {}}
    e1, rf, e2 = frozen_entry("E1_primary", "lgbm"), frozen_entry("E1_primary", "rf"), frozen_entry("E2_random", "lgbm")
    thr = e1["thresholds"][BUDGET]

    # ---- selected LightGBM: validation + test census
    d = load("chrono", "default", partitions=("val", "test"))
    m = joblib.load(ROOT / e1["model_path"])
    run["model_inputs"] = model_inputs(m)
    for part, stored in (("test", M4 / "test" / "E1_primary__lgbm__test_predictions.parquet"),
                         ("val", M4 / "preds" / f"{e1['run_id']}__val.parquet")):
        X, y, meta = d[part]
        check_scores(meta, m, X, stored)
        t0 = time.perf_counter()
        sv, base, space = contributions(m, X)
        run["timing_s"][f"E1_lgbm_{part}"] = time.perf_counter() - t0
        run["checks"][f"E1_lgbm_{part}_scores_equal_stored"] = True
        save(sv, meta, f"shap_E1_lgbm_{part}.parquet")
        run["E1_lgbm"] = {"base": base, "space": space}
        print("E1 lgbm", part, sv.shape, flush=True)
    X, y, meta = d["test"]
    meta = with_meta(meta)
    score = pd.read_parquet(M4 / "test" / "E1_primary__lgbm__test_predictions.parquet")["score"].to_numpy()
    rows = meta.assign(score=score, outcome=outcome(y, score, thr))

    # ---- samples (IDs recorded; deterministic from the frozen predictions)
    uni = rows.sort_values(KEY, kind="mergesort")  # keeps the positional index of X / the census tables
    uni = uni.iloc[np.sort(np.random.default_rng(SEED).choice(len(uni), N_UNIFORM, replace=False))]
    uni[KEY].to_parquet(OUT / "sample_uniform.parquet", index=False)
    err = rows[rows.outcome.isin(["FP", "FN"])].assign(group=lambda r: r.outcome)
    tn = stratified_sample(rows[rows.outcome == "TN"], ["day", "novel"], CAP, SEED).assign(
        group=lambda r: "TN|" + r.day + "|" + np.where(r.novel, "novel", "dup"))
    tp = stratified_sample(rows[rows.outcome == "TP"], ["label_original", "novel"], CAP, SEED + 1).assign(
        group=lambda r: "TP|" + r.label_original + "|" + np.where(r.novel, "novel", "dup"))
    strat = pd.concat([err, tn, tp]).sort_values(KEY, kind="mergesort")[KEY + ["group"]].reset_index(drop=True)
    strat.to_parquet(OUT / "sample_stratified.parquet", index=False)
    run["samples"] = {"uniform": {"n": N_UNIFORM, "rule": "uniform without replacement from all test rows sorted by key, seed"},
                      "stratified": {"n": len(strat), "cap": CAP, "rule": "all FP and FN; TN capped per (day, novel); "
                                     "TP capped per (label_original, novel); seeded, key-sorted"}}

    # ---- stability: reload the model from disk, native vs shap library, bitwise repeatability
    Xu = X.loc[uni.index]
    full = pd.read_parquet(OUT / "shap_E1_lgbm_test.parquet").iloc[uni.index][run["model_inputs"]].to_numpy()
    m2 = joblib.load(ROOT / e1["model_path"])
    again = contributions(m2, Xu)[0].to_numpy()
    run["checks"]["reload_bitwise_identical"] = bool(np.array_equal(full, again))
    import shap
    lib = shap.TreeExplainer(m2.named_steps["est"]).shap_values(m2.named_steps["pre"].transform(Xu))
    run["checks"]["shap_library_vs_native_max_abs_diff"] = float(np.abs(np.asarray(lib) - full).max())
    t0 = time.perf_counter()
    inter = np.asarray(shap.TreeExplainer(m2.named_steps["est"]).shap_interaction_values(
        m2.named_steps["pre"].transform(Xu.iloc[:N_INTERACT])))
    run["timing_s"]["E1_lgbm_interactions"] = time.perf_counter() - t0
    run["checks"]["interaction_rows_sum_to_shap_max_abs_diff"] = float(np.abs(inter.sum(2) - full[:N_INTERACT]).max())
    pd.DataFrame(np.abs(inter).mean(0), index=run["model_inputs"], columns=run["model_inputs"]) \
        .to_parquet(OUT / "interactions_uniform.parquet")
    print("stability + interactions done", flush=True)

    # ---- frozen Random Forest on the stratified sample
    idx = rows.reset_index().merge(strat[KEY], on=KEY)["index"].to_numpy()
    mrf = joblib.load(ROOT / rf["model_path"])
    t0 = time.perf_counter()
    svr, base_rf, space_rf = contributions(mrf, X.loc[idx])
    run["timing_s"]["E1_rf_sample"] = time.perf_counter() - t0
    save(svr, meta.loc[idx].reset_index(drop=True), "shap_E1_rf_sample.parquet")
    run["E1_rf"] = {"base": base_rf, "space": space_rf, "rows": len(idx)}
    print("rf sample", svr.shape, flush=True)

    # ---- random-split LightGBM on its own test partition
    Xr, yr, mr = load("random", "default", partitions=("test",))["test"]
    m_e2 = joblib.load(ROOT / e2["model_path"])
    check_scores(mr, m_e2, Xr, M4 / "test" / "E2_random__lgbm__test_predictions.parquet")
    t0 = time.perf_counter()
    sv2, base2, space2 = contributions(m_e2, Xr)
    run["timing_s"]["E2_lgbm_test"] = time.perf_counter() - t0
    run["checks"]["E2_lgbm_test_scores_equal_stored"] = True
    save(sv2, mr, "shap_E2_lgbm_test.parquet")
    run["E2_lgbm"] = {"base": base2, "space": space2}

    # ---- provenance
    import lightgbm
    import matplotlib
    import sklearn
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    run["provenance"] = {
        "python": platform.python_version(), "shap": shap.__version__, "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__, "git_commit": commit, "git_tracked_changes": dirty,
        "m4_frozen_sha256": sha256(FROZEN),
        "protocol_sha256": sha256(ROOT / "configs/experiment_protocol_v1.json"),
        "feature_manifest_sha256": sha256(ROOT / "configs/feature_manifest_v1.json"),
        "label_mapping_sha256": sha256(ROOT / "configs/label_mapping_v1.json"),
        "models": {f"{e['experiment']}__{e['family']}": {"path": e["model_path"], "sha256": e["model_sha256"],
                                                         "threshold_0.001": e["thresholds"][BUDGET]}
                   for e in (e1, rf, e2)},
        "preprocessor": "embedded in each frozen Pipeline artifact (step 'pre'); covered by the model SHA-256",
    }
    (OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({k: run[k] for k in ("checks", "timing_s")}, indent=1))


if __name__ == "__main__":
    main()
