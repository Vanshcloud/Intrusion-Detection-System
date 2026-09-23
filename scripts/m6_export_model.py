"""Milestone 6: compact text export of the already-frozen selected LightGBM model (no training).

The frozen artifact (configs/m4_frozen.json -> overall_selected) is a scikit-learn Pipeline whose only preprocessing is a
train-fitted VarianceThreshold column selection. Its prediction function is therefore fully described by
  (a) LightGBM's text dump of the booster (the early-stopped trees), and
  (b) the ordered list of input columns the booster receives.
This script regenerates (a) from the frozen joblib, verifies exact equivalence with the stored frozen results, and
writes dashboard/model/<run_id>.txt plus a separate record configs/m6_model_export.json (m4_frozen.json is untouched).

  .venv/bin/python scripts/m6_export_model.py           # verify; write only if no export exists yet
  .venv/bin/python scripts/m6_export_model.py --check   # verify an existing export is byte-identical, write nothing
Needs the local artifacts: frozen joblib, data/processed, stored test predictions, artifacts/m5 SHAP census and
the local case rows artifacts/m6/demo_cases.csv (scripts/m6_dashboard_data.py).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import ROOT, sha256  # noqa: E402
from ids.modeling import KEY, load  # noqa: E402

OUT_DIR = ROOT / "dashboard" / "model"
RECORD = ROOT / "configs" / "m6_model_export.json"


def export_text(model):
    est = model.named_steps["est"]
    return est.booster_.model_to_string(num_iteration=est.best_iteration_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed export only; write nothing")
    a = ap.parse_args()
    fz = json.loads((ROOT / "configs/m4_frozen.json").read_text())
    e = next(x for x in fz["experiments"] if x["run_id"] == fz["overall_selected"])
    src = ROOT / e["model_path"]
    if sha256(src) != e["model_sha256"]:
        sys.exit("frozen model hash does not match configs/m4_frozen.json")
    model = joblib.load(src)
    pre, est = model.named_steps["pre"], model.named_steps["est"]
    if [type(t).__name__ for _, t, _ in pre.transformers_] != ["VarianceThreshold"]:
        sys.exit("preprocessor is not a pure column selection; a booster-only export would not be equivalent")
    inputs = list(pre.get_feature_names_out())
    txt = export_text(model)
    if export_text(joblib.load(src)) != txt:
        sys.exit("text export is not deterministic")
    path = OUT_DIR / f"{e['run_id']}.txt"
    if path.exists() and path.read_text() != txt:
        sys.exit(f"{path} differs from a fresh export of the frozen model")
    booster = lgb.Booster(model_str=txt)
    if booster.model_to_string() != txt:
        sys.exit("text export does not round-trip")
    if booster.feature_name() != [c.replace(" ", "_") for c in inputs]:
        sys.exit("booster feature order differs from the frozen preprocessor output")

    # --- exact equivalence on every test row: scores, predicted labels at the frozen threshold
    X, y, meta = load("chrono", "default", partitions=("test",))["test"]
    stored = pd.read_parquet(ROOT / "artifacts/m4/test/E1_primary__lgbm__test_predictions.parquet")
    if not (stored[KEY].to_numpy() == meta[KEY].to_numpy()).all():
        sys.exit("stored predictions not aligned with test rows")
    s = booster.predict(X[inputs].to_numpy(dtype=float))
    thr = e["thresholds"]["0.001"]
    checks = {"test_rows": int(len(s)),
              "scores_bitwise_equal_stored": bool(np.array_equal(s, stored.score.to_numpy())),
              "predictions_equal_at_frozen_threshold": bool(np.array_equal(s >= thr, stored.score.to_numpy() >= thr)),
              "scores_bitwise_equal_original_pipeline": bool(np.array_equal(s, model.predict_proba(X)[:, 1]))}
    # --- SHAP on the dashboard case set: export vs original model vs stored Milestone 5 census
    demo = ROOT / "artifacts/m6/demo_cases.csv"  # local only; built by scripts/m6_dashboard_data.py
    if not demo.exists():
        sys.exit(f"{demo.relative_to(ROOT)} missing: run scripts/m6_dashboard_data.py first")
    cases = pd.read_csv(demo, float_precision="round_trip")
    Xc = cases[[f"f__{f}" for f in X.columns]].set_axis(list(X.columns), axis=1).astype(X.dtypes.to_dict())
    c_export = booster.predict(Xc[inputs].to_numpy(dtype=float), pred_contrib=True)
    c_orig = est.booster_.predict(pre.transform(Xc), pred_contrib=True, num_iteration=est.best_iteration_)
    census = cases[KEY].merge(pd.read_parquet(ROOT / "artifacts/m5/shap_E1_lgbm_test.parquet"), on=KEY)[inputs].to_numpy()
    checks.update({"demo_cases": int(len(cases)),
                   "demo_shap_bitwise_equal_original_model": bool(np.array_equal(c_export, c_orig)),
                   "demo_shap_bitwise_equal_m5_census": bool(np.array_equal(c_export[:, :-1], census))})
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        sys.exit(f"equivalence audit failed: {checks}")

    record = {"what": "LightGBM text export of the frozen protocol-v1 selected model (not a newly trained model)",
              "source_run_id": e["run_id"], "source_model_path": e["model_path"],
              "source_model_sha256": e["model_sha256"], "source_record": "configs/m4_frozen.json (unchanged)",
              "export_path": str(path.relative_to(ROOT)), "export_sha256": hashlib.sha256(txt.encode()).hexdigest(),
              "export_bytes": len(txt.encode()), "num_trees": booster.num_trees(), "best_iteration": e["best_iteration"],
              "frozen_threshold_0.001": thr,
              "input_features": inputs,
              "input_note": "raw feature values in this order (the frozen preprocessor only drops the train-constant "
                            "column(s) not listed); score = booster.predict(X)[i] = probability of attack",
              "method": "Booster.model_to_string(num_iteration=best_iteration_); regenerated twice (identical) and "
                        "round-tripped through lightgbm.Booster(model_str=...)",
              "lightgbm": lgb.__version__, "verification": checks,
              "regenerate": "`.venv/bin/python scripts/m6_export_model.py --check` (needs local frozen artifacts)"}
    if a.check:
        if not path.exists() or json.loads(RECORD.read_text())["export_sha256"] != record["export_sha256"]:
            sys.exit("committed export or record missing / different")
        print("check passed:", json.dumps(checks))
        return
    if not path.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(txt)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("export_path", "export_sha256", "export_bytes", "verification")}, indent=1))


if __name__ == "__main__":
    main()
