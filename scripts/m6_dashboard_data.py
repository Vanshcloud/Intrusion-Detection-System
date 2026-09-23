"""Milestone 6: compact, clone-friendly tables for the dashboard, derived from frozen artifacts only.

Needs the local artifacts (data/processed, artifacts/m4 predictions, artifacts/m5 SHAP census); writes
  reports/generated/m6_demo_cases.csv   benchmark test rows for the prediction demo: raw feature values (f__*),
                                         stored frozen score, stored SHAP values (s__*), outcome and row metadata
  reports/generated/m6_score_hist.csv   counts of logit(score) in fixed bins, per population (no raw scores)
Nothing is trained or re-scored; scores and SHAP values are copied from the stored Milestone 4/5 artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ids.data import GENERATED, ROOT  # noqa: E402
from ids.explain import logit, outcome, stratified_sample  # noqa: E402
from ids.modeling import KEY, SEED  # noqa: E402
from m5_analyze import ART, FEATS, THR, load_all, shap_table  # noqa: E402

CAP = 3  # demo rows per (outcome, label, novel, day group)
BINS = np.arange(-12.0, 12.5, 0.5)  # logit(score) bin edges; scores are clipped into [-12, 12]
META = ["day", "daygroup", "label_original", "family", "y_binary", "novel", "reverse",
        "Src IP", "Src Port", "Dst IP", "Dst Port"]


def demo_cases(X, meta, sv, oc):
    strat = pd.read_parquet(ART / "sample_stratified.parquet")[KEY]
    cb = pd.read_csv(GENERATED / "m5_casebook.csv")[KEY + ["case"]]
    rows = meta.reset_index().assign(outcome=oc).merge(strat, on=KEY)
    pick = stratified_sample(rows, ["outcome", "label_original", "novel", "daygroup"], CAP, SEED)
    pick = pd.concat([pick, rows.merge(cb[KEY], on=KEY)]).drop_duplicates(KEY)
    pick = pick.merge(cb, on=KEY, how="left").sort_values(KEY, kind="mergesort")
    i = pick["index"].to_numpy()
    out = pick[KEY + META + ["score", "outcome", "case"]].reset_index(drop=True)
    feats = X.loc[i].reset_index(drop=True).add_prefix("f__")
    shap = sv.loc[i].reset_index(drop=True).add_prefix("s__")
    return pd.concat([out, feats, shap], axis=1)


def score_hist(parts):
    rows = []

    def add(set_, group, s):
        c, _ = np.histogram(np.clip(logit(s), BINS[0], BINS[-1]), BINS)
        rows.extend({"set": set_, "group": group, "bin_lo": lo, "bin_hi": hi, "n": int(k)}
                    for lo, hi, k in zip(BINS[:-1], BINS[1:], c))
    for part in ("val", "test"):
        m = parts[part][2]
        b = m[m.y_binary == 0]
        for dg, g in b.groupby("daygroup"):
            add(f"{part} benign by day group", dg, g.score)
            for rv, gg in g.groupby("reverse"):
                add(f"{part} benign by day group and direction", f"{dg}|{'reverse' if rv else 'other'}", gg.score)
    m = parts["test"][2]
    for o, g in m.assign(o=outcome(m.y_binary, m.score, THR)).groupby("o"):
        add("test outcome", o, g.score)
    return pd.DataFrame(rows)


def main():
    fz = json.loads((ROOT / "configs/m4_frozen.json").read_text())
    run_id = fz["overall_selected"]
    _, _, _, parts = load_all()
    for p, stored in (("test", ART.parent / "m4" / "test" / "E1_primary__lgbm__test_predictions.parquet"),
                      ("val", ART.parent / "m4" / "preds" / f"{run_id}__val.parquet")):
        st, m = pd.read_parquet(stored), parts[p][2]
        if not (st[KEY].to_numpy() == m[KEY].to_numpy()).all():
            raise ValueError(f"{stored} not aligned")
        m["score"] = st["score"].to_numpy()
    X, y, meta = parts["test"]
    sv = shap_table("shap_E1_lgbm_test.parquet", meta)
    assert list(sv.columns) == FEATS
    demo_cases(X, meta, sv, outcome(y, meta.score, THR)).to_csv(GENERATED / "m6_demo_cases.csv", index=False)
    score_hist(parts).to_csv(GENERATED / "m6_score_hist.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
