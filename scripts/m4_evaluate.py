"""Milestone 4, final test evaluation of the frozen experiments (configs/m4_frozen.json).

Scores the test partition once per frozen model, stores predictions, and computes all metrics from the
stored predictions. Refuses to overwrite an existing test run: a second run must use --out <dir>
(used only to verify reproducibility) and never changes a frozen decision.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import GENERATED, ROOT  # noqa: E402
from ids.metrics import UNSTABLE_SUPPORT, binary_report, wilson  # noqa: E402
from ids.modeling import KEY, attack_score, load  # noqa: E402

FROZEN = ROOT / "configs" / "m4_frozen.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label_rows(pred, meta, subset_name, exp, budget):
    """Per original label and per family: flagged / n with Wilson CI. Rate = recall for attacks, alert rate otherwise."""
    rows = []
    for col in ("label_original", "family"):
        for lab, g in meta.assign(pred=pred).groupby(col):
            n, k = len(g), int(g["pred"].sum())
            lo, hi = wilson(k, n)
            rows.append({"experiment": exp, "budget": budget, "subset": subset_name, "level": col, "label": lab,
                         "role": "attack" if g["y_binary"].max() == 1 else "benign/attempted",
                         "n": n, "flagged": k, "missed_or_clear": n - k, "rate": k / n if n else None,
                         "ci95_low": lo, "ci95_high": hi, "unstable": n < UNSTABLE_SUPPORT})
    return rows


def metrics_from_predictions(pred_df, thresholds, exp):
    """Everything is computed from a stored prediction table (so it can be independently recomputed)."""
    y, s = pred_df["y_binary"].to_numpy(), pred_df["score"].to_numpy()
    nov = pred_df["novel"].to_numpy()
    summary, labels = [], []
    for subset, m in (("full_test", np.ones(len(y), bool)), ("novel_test", nov)):
        for b, t in thresholds.items():
            r = binary_report(y[m], s[m], t)
            summary.append({"experiment": exp, "subset": subset, "budget": float(b), "threshold": t,
                            "n": r["n"], "attack": r["support_attack"], "benign": r["support_benign"],
                            "prevalence": r["support_attack"] / r["n"] if r["n"] else None,
                            **{k: r[k] for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")}})
            labels += label_rows(s[m] >= t, pred_df.loc[m, ["label_original", "family", "y_binary"]], subset, exp, float(b))
    return summary, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="alternative output dir (reproducibility check only)")
    a = ap.parse_args()
    fz = json.loads(FROZEN.read_text())
    if sha(ROOT / fz["selection_rule_file"]) != fz["selection_rule_sha256"]:
        sys.exit("selection rule changed after freezing")
    out = Path(a.out) if a.out else ROOT / "artifacts" / "m4" / "test"
    if not a.out and (out / "summary.csv").exists():
        sys.exit(f"test evaluation already exists in {out}; use --out for a reproducibility check")
    out.mkdir(parents=True, exist_ok=True)
    summary, labels, runtime = [], [], []
    loaded = {}
    for e in fz["experiments"]:
        if sha(ROOT / e["model_path"]) != e["model_sha256"]:
            sys.exit(f"model file changed after freezing: {e['model_path']}")
        k = (e["protocol"], e["feature_set"], e["variant"])
        if k not in loaded:
            loaded.clear()
            loaded[k] = load(e["protocol"], e["feature_set"], e["variant"], partitions=("test",))["test"]
        X, y, meta = loaded[k]
        model = joblib.load(ROOT / e["model_path"])
        s, secs = attack_score(model, X)
        tag = f"{e['experiment']}__{e['family']}"
        pred = meta[KEY + ["label_original", "family", "y_binary", "novel"]].assign(score=s)
        pred.to_parquet(out / f"{tag}__test_predictions.parquet", index=False)
        su, la = metrics_from_predictions(pred, e["thresholds"], tag)
        summary += su
        labels += la
        runtime.append({"experiment": tag, "test_rows": len(y), "predict_seconds": secs,
                        "predict_us_per_row": 1e6 * secs / len(y)})
        print(tag, "scored", flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(labels).to_csv(out / "per_label.csv", index=False)
    pd.DataFrame(runtime).to_csv(out / "runtime.csv", index=False)
    (out / "run.json").write_text(json.dumps({"frozen_sha256": sha(FROZEN),
                                              "created_utc": pd.Timestamp.now("UTC").isoformat()}, indent=2) + "\n")
    if not a.out:  # the official run is copied to reports/generated for rendering
        for f in ("summary.csv", "per_label.csv", "runtime.csv"):
            (GENERATED / f"m4_test_{f}").write_text((out / f).read_text())


if __name__ == "__main__":
    main()
