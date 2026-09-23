"""Milestone 4, development stage (train + validation only; never reads test labels or features for scoring).

Usage:
  m4_develop.py benchmark   one initial configuration per family (runtime + validation metrics)
  m4_develop.py search      the recorded search spaces (configs/model_selection_v1.json)
  m4_develop.py freeze      selection (addendum rule), comparison experiments, thresholds -> configs/m4_frozen.json

Runs are cached by (experiment, family, params): an identical configuration is never retrained.
"""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import GENERATED, ROOT  # noqa: E402
from ids.metrics import binary_report, select_threshold  # noqa: E402
from ids.modeling import KEY, attack_score, feature_list, fit, load  # noqa: E402

ART = ROOT / "artifacts" / "m4"
SEL = ROOT / "configs" / "model_selection_v1.json"
FROZEN = ROOT / "configs" / "m4_frozen.json"
BUDGETS = [0.001, 0.01]
ES_ROUNDS = 50
EXPERIMENTS = {  # name: (protocol, variant, feature_set)
    "E1_primary": ("chrono", "primary", "default"),
    "E2_random": ("random", "primary", "default"),
    "E3_dst_port": ("chrono", "primary", "default_plus_dst_port"),
    "E3_no_tcp_window": ("chrono", "primary", "no_tcp_window"),
    "E3_no_header_length": ("chrono", "primary", "no_header_length"),
    "E4_excl_attempted": ("chrono", "excl_attempted", "default"),
}
BENCHMARK = {
    "lr": {"C": 1.0, "class_weight": None, "max_iter": 2000},
    "rf": {"n_estimators": 25, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
    "lgbm": {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100},
}


def run_id(exp, family, params):
    h = hashlib.sha256(json.dumps([exp, family, params], sort_keys=True).encode()).hexdigest()[:10]
    return f"{exp}__{family}__{h}"


def val_summary(y, s):
    r = {"pr_auc": binary_report(y, s, 0.5)["pr_auc"], "roc_auc": binary_report(y, s, 0.5)["roc_auc"]}
    for b in BUDGETS:
        t = select_threshold(y, s, max_fpr=b)
        rep = binary_report(y, s, t)
        r[f"b{b}"] = {k: rep[k] for k in ("threshold", "fpr", "recall", "precision", "f1", "fp", "fn", "tp", "tn")}
        r[f"b{b}"]["budget_met"] = rep["fpr"] is not None and rep["fpr"] <= b
    return r


def run(exp, family, params, data, save_model=False):
    rid = run_id(exp, family, params)
    out = ART / "runs" / f"{rid}.json"
    if out.exists() and (not save_model or (ART / "models" / f"{rid}.joblib").exists()):
        return json.loads(out.read_text())
    protocol, variant, fs = EXPERIMENTS[exp]
    feats = feature_list(fs)
    (Xtr, ytr, _), (Xva, yva, mva) = data["train"], data["val"]
    model, info = fit(family, params, feats, Xtr, ytr, Xva, yva, ES_ROUNDS if family == "lgbm" else None)
    s, pred_s = attack_score(model, Xva)
    rec = {"run_id": rid, "experiment": exp, "family": family, "params": params, "protocol": protocol,
           "variant": variant, "feature_set": fs, "n_features": len(feats), "n_train": len(ytr), "n_val": len(yva),
           "val_prevalence": float(yva.mean()), **info, "val_predict_seconds": pred_s,
           "val_predict_us_per_row": 1e6 * pred_s / len(yva),
           "peak_rss_gb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,  # macOS: bytes
           "val": val_summary(yva, s)}
    (ART / "preds").mkdir(parents=True, exist_ok=True)
    mva.assign(score=s)[KEY + ["y_binary", "score"]].to_parquet(ART / "preds" / f"{rid}__val.parquet", index=False)
    if save_model:
        (ART / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ART / "models" / f"{rid}.joblib", compress=3)
        rec["model_path"] = f"artifacts/m4/models/{rid}.joblib"
    (ART / "runs").mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(f"{rid}: fit {info['fit_seconds']:.0f}s  PR-AUC {rec['val']['pr_auc']:.4f}  "
          f"recall@0.1% {rec['val']['b0.001']['recall']:.4f}  recall@1% {rec['val']['b0.01']['recall']:.4f}", flush=True)
    return rec


def load_dev(exp):
    protocol, variant, fs = EXPERIMENTS[exp]
    return load(protocol, fs, variant, partitions=("train", "val"))  # test is not loaded here


def key(rec):
    """Addendum rule: recall@0.1% FPR, then PR-AUC, then fit time (as a sortable tuple)."""
    return (round(rec["val"]["b0.001"]["recall"], 3), rec["val"]["pr_auc"], -rec["fit_seconds"])


def pick(recs):
    best = max(recs, key=lambda r: r["val"]["b0.001"]["recall"])
    close = [r for r in recs if best["val"]["b0.001"]["recall"] - r["val"]["b0.001"]["recall"] < 0.001]
    return max(close, key=lambda r: (r["val"]["pr_auc"], -r["fit_seconds"]))


def grid(space):
    import itertools
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]


def main(stage):
    sel = json.loads(SEL.read_text())
    if stage == "benchmark":
        data = load_dev("E1_primary")
        for fam, p in BENCHMARK.items():
            run("E1_primary", fam, p, data)
    elif stage == "search":
        data = load_dev("E1_primary")
        for fam, space in sel["search_spaces"].items():
            if fam.startswith("_"):
                continue
            for p in grid(space):
                run("E1_primary", fam, p, data)
    elif stage == "freeze":
        freeze(sel)
    else:
        sys.exit(__doc__)


def freeze(sel):
    if FROZEN.exists():
        sys.exit(f"{FROZEN} exists; the development stage is frozen (delete only with a documented protocol revision)")
    recs = [json.loads(p.read_text()) for p in (ART / "runs").glob("E1_primary__*.json")]
    searched = {fam: [r for r in recs if r["family"] == fam and r["params"] in grid(sp)]
                for fam, sp in sel["search_spaces"].items() if not fam.startswith("_")}
    winners = {fam: pick(rs) for fam, rs in searched.items()}
    overall = pick(list(winners.values()))
    pd.DataFrame([{"family": f, "run_id": w["run_id"], **{k: w["val"]["b0.001"][k] for k in ("recall", "fpr")},
                   "pr_auc": w["val"]["pr_auc"]} for f, w in winners.items()]).to_csv(GENERATED / "m4_family_winners.csv", index=False)
    plan = [("E1_primary", fam, w["params"]) for fam, w in winners.items()]
    plan += [("E2_random", fam, w["params"]) for fam, w in winners.items()]
    plan += [(e, overall["family"], overall["params"]) for e in EXPERIMENTS if e.startswith(("E3", "E4"))]
    frozen = {"created_utc": pd.Timestamp.now("UTC").isoformat(), "selection_rule_file": "configs/model_selection_v1.json",
              "selection_rule_sha256": hashlib.sha256(SEL.read_bytes()).hexdigest(),
              "family_winners": {f: w["run_id"] for f, w in winners.items()}, "overall_selected": overall["run_id"],
              "experiments": []}
    cache = {}
    for exp, fam, params in plan:
        if exp not in cache:
            cache.clear()
            cache[exp] = load_dev(exp)
        rec = run(exp, fam, params, cache[exp], save_model=True)
        frozen["experiments"].append({
            "experiment": exp, "family": fam, "run_id": rec["run_id"], "params": params,
            "protocol": rec["protocol"], "variant": rec["variant"], "feature_set": rec["feature_set"],
            "model_path": rec["model_path"],
            "model_sha256": hashlib.sha256((ROOT / rec["model_path"]).read_bytes()).hexdigest(),
            "thresholds": {str(b): rec["val"][f"b{b}"]["threshold"] for b in BUDGETS},
            "val_at_threshold": {str(b): rec["val"][f"b{b}"] for b in BUDGETS},
            "best_iteration": rec.get("best_iteration")})
    FROZEN.write_text(json.dumps(frozen, indent=2, default=float) + "\n")
    print("frozen:", FROZEN)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
