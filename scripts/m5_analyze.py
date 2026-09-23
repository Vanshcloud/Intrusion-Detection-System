"""Milestone 5, analysis stage: tables and figures from artifacts/m5 (scripts/m5_explain.py) and the stored
Milestone 4 predictions. No model is trained, no threshold is changed, no label is changed.

Writes reports/generated/m5_*.csv, reports/generated/m5_facts.json and reports/figures/m5/*.png.
Metadata (IPs, ports, times) is used only to describe rows; it is never a model input.
"benign" = y_binary 0 (BENIGN and '- Attempted' rows, as in protocol v1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import GENERATED, PROCESSED, ROOT  # noqa: E402
from ids.explain import (FEATURE_GROUP, attack_pairs, group_shap, importance, is_reverse_of_attack, logit,  # noqa: E402
                         outcome, quantiles, rank_corr, topk_overlap)
from ids.metrics import UNSTABLE_SUPPORT, wilson  # noqa: E402
from ids.modeling import KEY, SEED, load  # noqa: E402

ART = ROOT / "artifacts" / "m5"
M4 = ROOT / "artifacts" / "m4"
FIG = ROOT / "reports" / "figures" / "m5"
RUN = json.loads((ART / "run.json").read_text())
FEATS = RUN["model_inputs"]
THR = RUN["provenance"]["models"]["E1_primary__lgbm"]["threshold_0.001"]
THR_E2 = RUN["provenance"]["models"]["E2_random__lgbm"]["threshold_0.001"]
DAYGROUP = {"monday": "mon-wed", "tuesday": "mon-wed", "wednesday": "mon-wed", "thursday": "thursday", "friday": "friday"}
DG_ORDER = ["mon-wed", "thursday", "friday"]
B, BOOT_N = 200, (1000, 5000)
FACTS = {}

# palette: reference categorical order (dataviz skill, light surface)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def out(df, name, index=False):
    df.to_csv(GENERATED / f"m5_{name}.csv", index=index, float_format="%.6g")
    return df


def ci(k, n):
    lo, hi = wilson(int(k), int(n))
    return pd.Series({"ci95_low": lo, "ci95_high": hi})


def fmt_contrib(row_sv, row_x, sign, k=3):
    s = row_sv.sort_values(ascending=(sign < 0), kind="mergesort")
    s = s[s * sign > 0][:k]
    return "; ".join(f"{f}={row_x[f]:.6g} ({v:+.2f})" for f, v in s.items())


# ------------------------------------------------------------------ loading
def load_all():
    ds = pd.read_parquet(PROCESSED / "dataset_v1.parquet",
                         columns=KEY + ["day", "Src IP", "Dst IP", "Src Port", "Dst Port", "t_start", "t_end",
                                        "y_binary", "label_original"])
    pairs = attack_pairs(ds)
    att = ds[ds.y_binary == 1]
    windows = att.groupby(["day", "label_original"]).agg(w0=("t_start", "min"), w1=("t_end", "max")).reset_index()
    split = pd.read_parquet(PROCESSED / "split_chrono_v1.parquet", columns=KEY + ["partition"])
    ds = ds.merge(split, on=KEY, how="left", validate="1:1")
    ds["reverse"] = is_reverse_of_attack(ds, pairs)
    d = load("chrono", "default", partitions=("train", "val", "test"))
    parts = {}
    for p in ("train", "val", "test"):
        X, y, meta = d[p]
        meta = meta.merge(ds.drop(columns=["y_binary", "label_original", "partition"]), on=KEY, how="left", validate="1:1")
        meta["daygroup"] = meta["day"].map(DAYGROUP)
        parts[p] = (X, y, meta)
    return ds, pairs, windows, parts


def shap_table(name, meta):
    s = pd.read_parquet(ART / name)
    if not (s[KEY].to_numpy() == meta[KEY].to_numpy()).all():
        raise ValueError(f"{name} not aligned with metadata")
    return s[FEATS]


# ------------------------------------------------------------------ analyses
def global_importance(sv, X, meta, uni_idx):
    ben, att = meta.y_binary.to_numpy() == 0, meta.y_binary.to_numpy() == 1
    imp = importance(sv)
    g = pd.DataFrame({"feature": imp.index, "group": imp.index.map(FEATURE_GROUP), "rank": np.arange(1, len(imp) + 1),
                      "mean_abs_all": imp.values, "share_all": imp.values / imp.sum(),
                      "mean_abs_benign": importance(sv, ben)[imp.index].values,
                      "mean_abs_attack": importance(sv, att)[imp.index].values,
                      "mean_signed_all": sv.mean()[imp.index].values})
    # directionality on the uniform sample: Spearman between the feature value and its SHAP value
    from scipy.stats import spearmanr
    rho = []
    for f in imp.index:
        x, s = X.loc[uni_idx, f].to_numpy(), sv.loc[uni_idx, f].to_numpy()
        rho.append(np.nan if np.ptp(x) == 0 or np.ptp(s) == 0 else spearmanr(x, s).statistic)
    g["spearman_value_vs_shap_uniform"] = rho
    # numeric summaries of the dependence plots (uniform sample), so the report text is not read off a figure
    u = sv.loc[uni_idx]
    xu = X.loc[uni_idx]
    rst = u["RST Flag Count"][xu["RST Flag Count"] >= 1]
    short = xu["Flow Duration"] < 100
    win = u["Bwd Init Win Bytes"] > 1
    FACTS["dep"] = {"rst_ge1_shap_q05_q95": [float(rst.quantile(0.05)), float(rst.quantile(0.95))],
                    "rst_0_shap_q05_q95": [float(u["RST Flag Count"][xu["RST Flag Count"] == 0].quantile(q)) for q in (0.05, 0.95)],
                    "duration_lt100us_shap_median": float(u["Flow Duration"][short].median()),
                    "duration_ge100us_shap_median": float(u["Flow Duration"][~short].median()),
                    "bwd_win_shap_gt1_value_q05_q95": [float(xu["Bwd Init Win Bytes"][win].quantile(q)) for q in (0.05, 0.95)],
                    "bwd_win_shap_gt1_rows": int(win.sum())}
    g["direction"] = np.select([g.spearman_value_vs_shap_uniform >= 0.5, g.spearman_value_vs_shap_uniform <= -0.5],
                               ["higher value -> toward attack", "higher value -> toward benign"], "non-monotone / mixed")
    return out(g, "global_importance")


def population_groups(sv, meta, oc):
    gs = group_shap(sv).abs()
    pops = {"all_test": np.ones(len(meta), bool), "TN": oc == "TN", "FP": oc == "FP", "TP": oc == "TP", "FN": oc == "FN"}
    for dg in DG_ORDER:
        pops[f"FP_{dg}"] = (oc == "FP") & (meta.daygroup == dg).to_numpy()
        pops[f"TN_{dg}"] = (oc == "TN") & (meta.daygroup == dg).to_numpy()
    rows = []
    for name, m in pops.items():
        r = gs[m].mean()
        rows.append({"population": name, "n": int(m.sum()), **(r / r.sum()).round(6).to_dict()})
    return out(pd.DataFrame(rows), "group_share_by_population")


def per_family(sv, X, meta, oc):
    att = meta.y_binary == 1
    top, shares, rare = [], [], []
    gs = group_shap(sv).abs()
    for lab, g in meta[att].groupby("label_original"):
        i = g.index
        n = len(i)
        if n >= UNSTABLE_SUPPORT:
            imp = importance(sv.loc[i])
            for r, f in enumerate(imp.index[:10], 1):
                top.append({"label": lab, "n": n, "rank": r, "feature": f, "group": FEATURE_GROUP[f],
                            "mean_abs": imp[f], "share": imp[f] / imp.sum(), "mean_signed": sv.loc[i, f].mean(),
                            "sd": sv.loc[i, f].std(), "median_value": X.loc[i, f].median()})
            s = gs.loc[i].mean()
            shares.append({"label": lab, "n": n, **(s / s.sum()).round(6).to_dict()})
        else:
            for j in i:
                rare.append({"label": lab, "n_label": n, "source_file": meta.at[j, "source_file"],
                             "source_row": meta.at[j, "source_row"], "novel": bool(meta.at[j, "novel"]),
                             "score": meta.at[j, "score"], "outcome": oc[j],
                             "top_toward_attack": fmt_contrib(sv.loc[j], X.loc[j], +1),
                             "top_toward_benign": fmt_contrib(sv.loc[j], X.loc[j], -1)})
    top, shares = out(pd.DataFrame(top), "family_top_features"), out(pd.DataFrame(shares), "family_group_share")
    rare = out(pd.DataFrame(rare).sort_values(["n_label", "label", "source_file", "source_row"]), "rare_family_cases")
    # how similar are family rankings? pairwise top-10 overlap
    labs = top.label.unique()
    sets = {l: set(top[top.label == l].feature) for l in labs}
    ov = pd.DataFrame([[len(sets[a] & sets[b]) / 10 for b in labs] for a in labs], index=labs, columns=labs)
    out(ov, "family_top10_overlap", index=True)
    return top, shares, rare, ov


def fp_tables(sv, X, meta, oc, parts, windows):
    fp, tn = oc == "FP", oc == "TN"
    ben = meta.y_binary.to_numpy() == 0
    # FPR by day, test and validation (validation outcomes from the stored validation scores, same threshold)
    rows = []
    for part in ("val", "test"):
        m = parts[part][2]
        o = outcome(m.y_binary, m.score, THR)
        for day, g in m[m.y_binary == 0].assign(o=o[m.y_binary.to_numpy() == 0]).groupby("day"):
            k = int((g.o == "FP").sum())
            rows.append({"partition": part, "day": day, "benign": len(g), "fp": k, "fpr": k / len(g), **ci(k, len(g)),
                         "fp_reverse": int(((g.o == "FP") & g.reverse).sum()),
                         "fp_attempted": int(((g.o == "FP") & g.label_original.str.endswith("Attempted")).sum()),
                         "benign_reverse_share": g.reverse.mean()})
    byday = out(pd.DataFrame(rows), "fp_by_day")

    # time position: FP inside the time window [first start, last end] of a same-day successful-attack label
    def in_window(m):
        res = np.zeros(len(m), bool)
        for _, w in windows.iterrows():
            res |= ((m.day == w.day) & (m.t_start >= w.w0) & (m.t_start <= w.w1)).to_numpy()
        return res
    meta["in_attack_window"] = in_window(meta)

    # day-group comparison
    cmp_rows, feat_rows = [], []
    for dg in DG_ORDER:
        d = (meta.daygroup == dg).to_numpy()
        b, f, t = d & ben, d & fp, d & tn
        k, n = int(f.sum()), int(b.sum())
        fpm = meta[f]
        imp_fp = importance(sv, f)
        abs_sum = sv.abs().sum(axis=1)
        pr = fpm.assign(pair=fpm["Src IP"] + " -> " + fpm["Dst IP"]).pair.value_counts()
        cmp_rows.append({
            "daygroup": dg, "benign_test": n, "fp": k, "fpr": k / n, **ci(k, n),
            "fp_share_reverse": fpm.reverse.mean(), "fp_share_in_attack_window": fpm.in_attack_window.mean(),
            "tn_share_in_attack_window": meta[t].in_attack_window.mean(),
            "fp_share_attempted": fpm.label_original.str.endswith("Attempted").mean(),
            "benign_score_q0.5": np.quantile(meta.score[b], 0.5), "benign_score_q0.99": np.quantile(meta.score[b], 0.99),
            "benign_score_q0.999": np.quantile(meta.score[b], 0.999), "fp_score_median": fpm.score.median(),
            "fp_median_sum_abs_shap": abs_sum[f].median(), "tn_median_sum_abs_shap": abs_sum[t].median(),
            "fp_top5_features": ", ".join(imp_fp.index[:5]),
            "fp_protocols": ", ".join(f"{p}:{c}" for p, c in X.loc[f, "Protocol"].value_counts().head(3).items()),
            "fp_top_dst_ports": ", ".join(f"{p}:{c}" for p, c in fpm["Dst Port"].value_counts().head(5).items()),
            "fp_top_src_ports": ", ".join(f"{p}:{c}" for p, c in fpm["Src Port"].value_counts().head(5).items()),
            "fp_n_host_pairs": len(pr), "fp_top3_pairs_share": pr.head(3).sum() / max(k, 1),
            "fp_top_pairs": "; ".join(f"{p} ({c})" for p, c in pr.head(3).items())})
        for r, fe in enumerate(imp_fp.index[:10], 1):
            feat_rows.append({"daygroup": dg, "rank": r, "feature": fe, "fp_mean_abs": imp_fp[fe],
                              "fp_mean_signed": sv.loc[f, fe].mean(), "tn_mean_signed": sv.loc[t, fe].mean(),
                              "fp_median_value": X.loc[f, fe].median(), "tn_median_value": X.loc[t, fe].median()})
    cmp = out(pd.DataFrame(cmp_rows), "fp_daygroup_compare")
    feats = out(pd.DataFrame(feat_rows), "fp_top_features_by_daygroup")

    # repeated host pairs (ordered src -> dst, per day)
    trm, vam = parts["train"][2], parts["val"][2]
    vo = outcome(vam.y_binary, vam.score, THR)
    key3 = ["day", "Src IP", "Dst IP"]
    fpm = meta[fp]
    top = fpm.groupby(key3).size().sort_values(ascending=False, kind="mergesort").head(15).rename("fp").reset_index()
    att_pairs_lab = pd.concat([parts[p][2] for p in parts])
    att_pairs_lab = att_pairs_lab[att_pairs_lab.y_binary == 1].groupby(key3).label_original.agg(
        lambda s: ", ".join(sorted(s.unique())))
    rows = []
    for _, r in top.iterrows():
        sel = lambda m: (m.day == r.day) & (m["Src IP"] == r["Src IP"]) & (m["Dst IP"] == r["Dst IP"])
        tb = meta[sel(meta) & (meta.y_binary == 0)]
        vb = sel(vam) & (vam.y_binary == 0)
        rows.append({**r.to_dict(), "benign_test_rows": len(tb), "fpr_within_pair": r.fp / len(tb),
                     "reverse_of_attack": bool(tb.reverse.iloc[0]),
                     "reverse_attack_labels": att_pairs_lab.get((r.day, r["Dst IP"], r["Src IP"]), ""),
                     "train_benign_rows": int((sel(trm) & (trm.y_binary == 0)).sum()),
                     "val_benign_rows": int(vb.sum()), "val_fp": int((vb.to_numpy() & (vo == "FP")).sum()),
                     "top_dst_ports": ", ".join(f"{p}:{c}" for p, c in fpm[sel(fpm)]["Dst Port"].value_counts().head(3).items())})
    pairs = out(pd.DataFrame(rows), "fp_host_pairs")
    FACTS["fp_top15_pairs_share"] = float(top.fp.sum() / fp.sum())
    FACTS["fp_n_pairs"] = int(fpm.groupby(key3).ngroups)

    # how many distinct inputs / scores do the FPs collapse onto; exact train matches; closest family SHAP profile
    Xtr, ytr, mtr = parts["train"]
    htr = pd.util.hash_pandas_object(Xtr + 0.0, index=False)
    tr_lab = pd.DataFrame({"h": htr.to_numpy(), "y": ytr}).groupby("h").y.agg(["min", "max"])
    att = (meta.y_binary == 1).to_numpy()
    fam_mean = {l: sv[(meta.label_original == l).to_numpy()].mean() for l in meta.label_original[att].unique()
                if (meta.label_original == l).sum() >= UNSTABLE_SUPPORT}
    prof = []
    for dg in DG_ORDER:
        f = fp & (meta.daygroup == dg).to_numpy()
        hf = pd.util.hash_pandas_object(X[f] + 0.0, index=False)
        m = tr_lab.reindex(hf.to_numpy())
        mv = sv[f].mean()
        cos = {l: float(mv @ v / (np.linalg.norm(mv) * np.linalg.norm(v))) for l, v in fam_mean.items()}
        best = sorted(cos.items(), key=lambda t: -t[1])[:3]
        prof.append({"daygroup": dg, "fp": int(f.sum()), "distinct_vectors": int(hf.nunique()),
                     "distinct_scores": int(meta.score[f].nunique()),
                     "top1_vector_share": float(hf.value_counts().iloc[0] / f.sum()),
                     "in_train_as_benign_only": int((m["max"] == 0).sum()),
                     "in_train_as_attack_only": int((m["min"] == 1).sum()),
                     "in_train_both": int(((m["min"] == 0) & (m["max"] == 1)).sum()),
                     "closest_family_profiles": "; ".join(f"{l} ({c:.2f})" for l, c in best)})
    out(pd.DataFrame(prof), "fp_profile")
    return byday, cmp, feats, pairs


def shift_decomposition(parts, sv, svv, windows):
    """Benign FPR split by reverse-of-attack direction, validation vs test; score quantiles; SHAP of val vs test FPs."""
    rows, q = [], []
    for part in ("val", "test"):
        m = parts[part][2]
        b = m[m.y_binary == 0]
        fl = b.score >= THR
        for dg in DG_ORDER + ["all"]:
            for rv in (True, False):
                s = (b.reverse == rv) & ((b.daygroup == dg) if dg != "all" else True)
                n, k = int(s.sum()), int((s & fl).sum())
                rows.append({"partition": part, "daygroup": dg, "reverse_of_attack": rv, "benign": n, "fp": k,
                             "fpr": k / n if n else np.nan, **ci(k, n)})
        g = np.where(b.reverse, "reverse", "other") + "|" + b.daygroup
        qq = quantiles(b.score, g).reset_index().rename(columns={"g": "group"}).assign(partition=part)
        qq["share_ge_threshold"] = [float((b.score[g == x] >= THR).mean()) for x in qq.group]
        q.append(qq)
    dec = out(pd.DataFrame(rows), "shift_decomposition")
    sq = out(pd.concat(q), "shift_score_quantiles")
    # where the BENIGN stratum's train/val/test time windows fall relative to the Thursday/Friday attack windows
    bnd = pd.read_csv(GENERATED / "m3_split_boundaries_chrono.csv")
    rows = []
    for day in ("thursday", "friday"):
        b = bnd[bnd.stratum == f"{day}|BENIGN"].iloc[0]
        for part in ("train", "val", "test"):
            n_rev = int(parts[part][2].query("y_binary == 0 and day == @day").reverse.sum())
            rows.append({"day": day, "row": f"BENIGN {part} window", "start": b[f"{part}_min"][11:19],
                         "end": b[f"{part}_max"][11:19], "benign_reverse_rows": n_rev})
        for _, wr in windows[windows.day == day].iterrows():
            rows.append({"day": day, "row": f"attack: {wr.label_original}",
                         "start": pd.to_datetime(wr.w0, unit="s").strftime("%H:%M:%S"),
                         "end": pd.to_datetime(wr.w1, unit="s").strftime("%H:%M:%S"), "benign_reverse_rows": None})
    out(pd.DataFrame(rows), "shift_time_windows")
    trm = parts["train"][2]
    tr = trm[trm.y_binary == 0].groupby("daygroup").reverse.mean()
    FACTS["train_benign_reverse_share"] = {k: float(v) for k, v in tr.items()}
    # validation FPs vs test FPs: do the same features carry them?
    vm, tm = parts["val"][2], parts["test"][2]
    vfp = ((vm.y_binary == 0) & (vm.score >= THR)).to_numpy()
    tfp = ((tm.y_binary == 0) & (tm.score >= THR)).to_numpy()
    iv, it = importance(svv, vfp), importance(sv, tfp)
    FACTS["val_fp_n"] = int(vfp.sum())
    FACTS["val_vs_test_fp_top10_overlap"] = topk_overlap(iv, it, 10)
    FACTS["val_vs_test_fp_spearman"] = rank_corr(iv, it)
    FACTS["val_fp_top5"] = list(iv.index[:5])
    FACTS["val_fp_reverse_share"] = float(vm.reverse[vfp].mean())
    return dec, sq


def feature_shift(parts, fp_feats):
    """KS distance of benign feature distributions, validation vs test, for the features that carry test FPs."""
    from scipy.stats import ks_2samp
    (Xv, _, mv), (Xt, _, mt) = parts["val"], parts["test"]
    rows = []
    for f in fp_feats:
        for dg in DG_ORDER:
            for rv in ("all", False):
                a = (mv.y_binary == 0) & (mv.daygroup == dg) & ((mv.reverse == rv) if rv != "all" else True)
                b = (mt.y_binary == 0) & (mt.daygroup == dg) & ((mt.reverse == rv) if rv != "all" else True)
                r = ks_2samp(Xv.loc[a, f], Xt.loc[b, f])
                rows.append({"feature": f, "daygroup": dg, "rows": "all benign" if rv == "all" else "non-reverse benign",
                             "n_val": int(a.sum()), "n_test": int(b.sum()), "ks": r.statistic,
                             "val_median": Xv.loc[a, f].median(), "test_median": Xt.loc[b, f].median()})
    return out(pd.DataFrame(rows), "feature_shift_val_test")


def fn_cases(sv, X, meta, oc, parts):
    Xtr, ytr, mtr = parts["train"]
    fn = np.flatnonzero(oc == "FN")

    def h(df):  # exact-vector hash; -0.0 canonicalised to 0.0 (as in the protocol's novelty definition)
        return pd.util.hash_pandas_object(df + 0.0, index=False).to_numpy()
    htr = h(Xtr)
    hf = h(X.iloc[fn])
    tp = oc == "TP"
    rows = []
    for j, hv in zip(fn, hf):
        lab = meta.at[j, "label_original"]
        ref = Xtr[(mtr.label_original == lab).to_numpy()]
        lo, hi = ref.quantile(0.005), ref.quantile(0.995)
        outside = int(((X.loc[j] < lo) | (X.loc[j] > hi)).sum()) if len(ref) else None
        tps = X[tp & (meta.label_original == lab).to_numpy()]
        base = int(np.median(((tps < lo) | (tps > hi)).sum(axis=1))) if len(ref) and len(tps) else None
        cand = np.flatnonzero(htr == hv)  # hash candidates, then exact per-column equality (no collision trust)
        exact = cand[((Xtr.iloc[cand] + 0.0).to_numpy() == (X.loc[[j]] + 0.0).to_numpy()).all(1)] if len(cand) else cand
        ys = pd.Series(ytr[exact])
        rows.append({"source_file": meta.at[j, "source_file"], "source_row": meta.at[j, "source_row"],
                     "day": meta.at[j, "day"], "label": lab, "novel": bool(meta.at[j, "novel"]),
                     "label_test_n": int((meta.label_original == lab).sum()),
                     "score": meta.at[j, "score"], "threshold": THR,
                     "logit_margin": float(logit(meta.at[j, "score"]) - logit(THR)),
                     "top_toward_benign": fmt_contrib(sv.loc[j], X.loc[j], -1),
                     "top_toward_attack": fmt_contrib(sv.loc[j], X.loc[j], +1),
                     "train_rows_same_label": len(ref),
                     "features_outside_train_label_p0.5_p99.5": outside, "median_same_for_label_TPs": base,
                     "exact_vector_in_train_benign": int((ys == 0).sum()), "exact_vector_in_train_attack": int((ys == 1).sum())})
    return out(pd.DataFrame(rows).sort_values(["label", "score"]), "fn_cases")


def novel_dup(sv, meta, oc):
    abs_sum = sv.abs().sum(axis=1)
    lg = logit(meta.score)
    nov = meta.novel.to_numpy()
    rows = []
    full = importance(sv)
    for name, m in (("full_test", np.ones(len(meta), bool)), ("novel_test", nov), ("duplicated_test", ~nov)):
        mm = meta[m]
        rows.append({"population": name, "n": int(m.sum()), "attack_share": mm.y_binary.mean(),
                     "fp": int((oc[m] == "FP").sum()), "mean_sum_abs_shap": abs_sum[m].mean(),
                     "top10": ", ".join(importance(sv, m).index[:10]),
                     "top10_overlap_with_full": topk_overlap(importance(sv, m), full, 10),
                     "spearman_with_full": rank_corr(importance(sv, m), full)})
    pop = out(pd.DataFrame(rows), "novel_vs_full")
    fpn = (oc == "FP") & nov
    FACTS["fp_novel"] = int(fpn.sum())
    FACTS["fp_novel_vs_full_fp_top10_overlap"] = topk_overlap(importance(sv, fpn), importance(sv, oc == "FP"), 10)
    # duplicated vs novel within label (controls for family mix)
    rows = []
    for lab, g in meta.groupby("label_original"):
        nv, dp = g.index[g.novel], g.index[~g.novel]
        if min(len(nv), len(dp)) < UNSTABLE_SUPPORT:
            continue
        a, b = importance(sv.loc[nv]), importance(sv.loc[dp])
        rows.append({"label": lab, "n_novel": len(nv), "n_dup": len(dp),
                     "mean_logit_novel": lg[nv].mean(), "mean_logit_dup": lg[dp].mean(),
                     "median_logit_novel": np.median(lg[nv]), "median_logit_dup": np.median(lg[dp]),
                     "mean_sum_abs_shap_novel": abs_sum[nv].mean(), "mean_sum_abs_shap_dup": abs_sum[dp].mean(),
                     "flag_rate_novel": (meta.score[nv] >= THR).mean(), "flag_rate_dup": (meta.score[dp] >= THR).mean(),
                     "top5_overlap": topk_overlap(a, b, 5), "spearman": rank_corr(a, b)})
    return pop, out(pd.DataFrame(rows), "dup_vs_novel_by_label")


def random_split(sv, meta, oc):
    X2, y2, m2 = load("random", "default", partitions=("test",))["test"]
    ds = pd.read_parquet(PROCESSED / "dataset_v1.parquet", columns=KEY + ["day", "Src IP", "Dst IP", "y_binary"])
    pairs = attack_pairs(ds)
    m2 = m2.merge(ds.drop(columns="y_binary"), on=KEY, how="left", validate="1:1")
    m2["score"] = pd.read_parquet(M4 / "test" / "E2_random__lgbm__test_predictions.parquet")["score"].to_numpy()
    sv2 = shap_table("shap_E2_lgbm_test.parquet", m2)
    o2 = outcome(y2, m2.score, THR_E2)
    rows = []
    for name, a, b in (("all_test", importance(sv), importance(sv2)),
                       ("benign", importance(sv, meta.y_binary == 0), importance(sv2, y2 == 0)),
                       ("attack", importance(sv, meta.y_binary == 1), importance(sv2, y2 == 1)),
                       ("FP", importance(sv, oc == "FP"), importance(sv2, o2 == "FP"))):
        rows.append({"population": name, "top10_overlap": topk_overlap(a, b, 10), "spearman": rank_corr(a, b),
                     "E1_top5": ", ".join(a.index[:5]), "E2_top5": ", ".join(b.index[:5]),
                     "E1_mean_sum_abs": sv[_mask(name, meta.y_binary, oc)].abs().sum(axis=1).mean(),
                     "E2_mean_sum_abs": sv2[_mask(name, pd.Series(y2), o2)].abs().sum(axis=1).mean()})
    summ = out(pd.DataFrame(rows), "e1_vs_e2_summary")
    a, b = importance(sv), importance(sv2)
    feat = out(pd.DataFrame({"feature": a.index, "E1_mean_abs": a.values, "E1_rank": np.arange(1, len(a) + 1),
                             "E2_mean_abs": b[a.index].values,
                             "E2_rank": b.rank(ascending=False, method="first")[a.index].astype(int).values}),
               "e1_vs_e2_features")
    fp2 = o2 == "FP"
    FACTS["E2_fp"] = int(fp2.sum())
    FACTS["E2_fp_reverse_share"] = float(is_reverse_of_attack(m2[fp2], pairs).mean())
    FACTS["E2_benign_reverse_share"] = float(is_reverse_of_attack(m2[y2 == 0], pairs).mean())
    rev2 = is_reverse_of_attack(m2, pairs)
    b2 = y2 == 0
    FACTS["E2_fpr_reverse"] = float((fp2 & rev2).sum() / (b2 & rev2).sum())
    FACTS["E2_fpr_other"] = float((fp2 & ~rev2).sum() / (b2 & ~rev2).sum())
    FACTS["E2_benign_score_q50_q99_q999"] = [float(v) for v in np.quantile(m2.score[b2], [0.5, 0.99, 0.999])]
    return summ, feat, sv2, X2, m2


def _mask(name, y, oc):
    y = np.asarray(y)
    return {"all_test": np.ones(len(y), bool), "benign": y == 0, "attack": y == 1, "FP": np.asarray(oc) == "FP"}[name]


def rf_compare(sv, meta, oc):
    strat = pd.read_parquet(ART / "sample_stratified.parquet")
    rf = pd.read_parquet(ART / "shap_E1_rf_sample.parquet")
    idx = meta.reset_index().merge(rf[KEY], on=KEY)["index"].to_numpy()
    if not (meta.loc[idx, KEY].to_numpy() == rf[KEY].to_numpy()).all():
        raise ValueError("RF sample not aligned")
    rsv = rf[FEATS].set_index(pd.Index(idx))
    lsv = sv.loc[idx]
    rf_score = pd.read_parquet(M4 / "test" / "E1_primary__rf__test_predictions.parquet")["score"].to_numpy()
    rf_thr = RUN["provenance"]["models"]["E1_primary__rf"]["threshold_0.001"]
    ocs = oc[idx]
    rows = []
    cos = (rsv.to_numpy() * lsv.to_numpy()).sum(axis=1) / (
        np.linalg.norm(rsv.to_numpy(), axis=1) * np.linalg.norm(lsv.to_numpy(), axis=1))
    for name in ("all_sample", "TN", "FP", "TP", "FN"):
        m = np.ones(len(idx), bool) if name == "all_sample" else ocs == name
        a, b = importance(lsv[m]), importance(rsv[m])
        rows.append({"population": name, "n": int(m.sum()), "top10_overlap": topk_overlap(a, b, 10),
                     "spearman": rank_corr(a, b), "lgbm_top5": ", ".join(a.index[:5]), "rf_top5": ", ".join(b.index[:5]),
                     "median_row_cosine": float(np.median(cos[m]))})
    summ = out(pd.DataFrame(rows), "lgbm_vs_rf_summary")
    a, b = importance(lsv), importance(rsv)
    feat = out(pd.DataFrame({"feature": a.index, "lgbm_share": (a / a.sum()).values,
                             "rf_share": (b / b.sum())[a.index].values,
                             "lgbm_rank": np.arange(1, len(a) + 1),
                             "rf_rank": b.rank(ascending=False, method="first")[a.index].astype(int).values}),
               "lgbm_vs_rf_features")
    gl, gr = group_shap(lsv).abs().mean(), group_shap(rsv).abs().mean()
    out(pd.DataFrame({"lgbm_share": gl / gl.sum(), "rf_share": gr / gr.sum()}).sort_values("lgbm_share", ascending=False),
        "lgbm_vs_rf_groups", index=True)
    both = (meta.y_binary == 0).to_numpy() & (meta.score.to_numpy() >= THR) & (rf_score >= rf_thr)
    FACTS["fp_both_lgbm_rf"] = int(both.sum())
    FACTS["fp_rf"] = int(((meta.y_binary == 0).to_numpy() & (rf_score >= rf_thr)).sum())
    FACTS["rf_sample_n"] = len(idx)
    FACTS["stratified_group_counts"] = strat.group.str.split("|").str[0].value_counts().to_dict()
    return summ, feat, rsv, rf_score, rf_thr


def stability(sv, meta, oc):
    full = importance(sv)
    A = sv.abs().to_numpy()
    rng = np.random.default_rng(SEED)
    rows, ranks = [], {}
    for n in BOOT_N:
        R, sp, ov = [], [], []
        for _ in range(B):
            i = rng.integers(0, len(A), n)
            m = pd.Series(A[i].mean(0), index=sv.columns).sort_values(ascending=False, kind="mergesort")
            R.append(m.rank(ascending=False, method="first")[full.index[:15]].to_numpy())
            sp.append(rank_corr(m, full))
            ov.append(topk_overlap(m, full, 10))
        R = np.array(R)
        ranks[n] = R
        for k, f in enumerate(full.index[:15]):
            rows.append({"bootstrap_n": n, "feature": f, "census_rank": k + 1, "rank_p5": np.percentile(R[:, k], 5),
                         "rank_median": np.median(R[:, k]), "rank_p95": np.percentile(R[:, k], 95),
                         "top10_freq": float((R[:, k] <= 10).mean())})
        FACTS[f"boot{n}_spearman_min"] = float(np.min(sp))
        FACTS[f"boot{n}_spearman_median"] = float(np.median(sp))
        FACTS[f"boot{n}_top10_overlap_min"] = float(np.min(ov))
        FACTS[f"boot{n}_top10_overlap_median"] = float(np.median(ov))
        FACTS[f"boot{n}_top1_always"] = bool((R[:, 0] == 1).all())
    st = out(pd.DataFrame(rows), "stability_bootstrap")
    # population sensitivity: rankings over different populations
    pops = {"all": np.ones(len(meta), bool), "benign": (meta.y_binary == 0).to_numpy(),
            "attack": (meta.y_binary == 1).to_numpy(), "FP": oc == "FP"}
    for d in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        pops[d] = (meta.day == d).to_numpy()
    imps = {k: importance(sv, m) for k, m in pops.items()}
    ps = pd.DataFrame([[topk_overlap(imps[a], imps[b], 10) for b in pops] for a in pops], index=list(pops), columns=list(pops))
    out(ps, "stability_population_top10_overlap", index=True)
    # disjoint halves split by key parity (deterministic)
    half = (meta.source_row.to_numpy() % 2) == 0
    FACTS["halves_top10_overlap"] = topk_overlap(importance(sv, half), importance(sv, ~half), 10)
    FACTS["halves_spearman"] = rank_corr(importance(sv, half), importance(sv, ~half))
    return st, ps, ranks


def interactions():
    I = pd.read_parquet(ART / "interactions_uniform.parquet")
    rows = []
    for i, a in enumerate(I.index):
        for b in I.columns[i + 1:]:
            rows.append({"feature_a": a, "feature_b": b, "mean_abs_interaction_x2": 2 * I.at[a, b]})
    r = pd.DataFrame(rows).sort_values("mean_abs_interaction_x2", ascending=False, kind="mergesort").head(15)
    diag = pd.Series(np.diag(I), index=I.index)
    FACTS["interaction_offdiag_share"] = float(2 * np.triu(I.to_numpy(), 1).sum() /
                                               (2 * np.triu(I.to_numpy(), 1).sum() + diag.sum()))
    return out(r, "interactions_top")


def score_quantiles(meta, oc, parts):
    groups = pd.Series(oc, index=meta.index).copy()
    tab = [quantiles(meta.score, oc).assign(set="test outcome")]
    b = meta.y_binary == 0
    tab.append(quantiles(meta.score[b], "test benign " + meta.daygroup[b]).assign(set="test benign by day group"))
    vm = parts["val"][2]
    vb = vm.y_binary == 0
    tab.append(quantiles(vm.score[vb], "val benign " + vm.daygroup[vb]).assign(set="validation benign by day group"))
    q = pd.concat(tab).reset_index().rename(columns={"g": "group"})
    return out(q, "score_quantiles")


def casebook(sv, X, meta, oc, rsv):
    strat = pd.read_parquet(ART / "sample_stratified.parquet")
    idx = meta.reset_index().merge(strat[KEY], on=KEY)["index"].to_numpy()
    m = meta.loc[idx].assign(o=oc[idx])

    def median_pick(c):
        if c.empty:
            return None
        med = c.score.median()
        return c.assign(d=(c.score - med).abs()).sort_values(["d"] + KEY, kind="mergesort").index[0]

    def lowest(c):
        return None if c.empty else c.sort_values(["score"] + KEY, kind="mergesort").index[0]
    cases = {
        "1 high-confidence TP (median of TP with score >= 0.99)": median_pick(m[(m.o == "TP") & (m.score >= 0.99)]),
        "2 borderline TP (lowest-scoring TP in sample)": lowest(m[m.o == "TP"]),
        "3 Thursday FP (median-score Thursday FP)": median_pick(m[(m.o == "FP") & (m.day == "thursday")]),
        "4 Friday FP (median-score Friday FP)": median_pick(m[(m.o == "FP") & (m.day == "friday")]),
        "5 Mon-Wed FP (median-score Mon-Wed FP)": median_pick(m[(m.o == "FP") & (m.daygroup == "mon-wed")]),
        "6 FN (median-score FN)": median_pick(m[m.o == "FN"]),
        "7 novel-vector TP (median-score novel TP)": median_pick(m[(m.o == "TP") & m.novel]),
        "8 exact-duplicate TP (median-score duplicated TP)": median_pick(m[(m.o == "TP") & ~m.novel]),
        "9 rare family: Heartbleed (median-score row)": median_pick(m[m.label_original == "Heartbleed"]),
        "10 rare family: Web Attack - SQL Injection (median-score row)": median_pick(m[m.label_original == "Web Attack - SQL Injection"]),
    }
    rows = []
    for name, j in cases.items():
        if j is None:
            continue
        rows.append({"case": name, "source_file": meta.at[j, "source_file"], "source_row": meta.at[j, "source_row"],
                     "day": meta.at[j, "day"], "label_original": meta.at[j, "label_original"],
                     "y_binary": int(meta.at[j, "y_binary"]), "score": meta.at[j, "score"], "threshold": THR,
                     "prediction": int(meta.at[j, "score"] >= THR), "novel": bool(meta.at[j, "novel"]),
                     "protocol": int(X.at[j, "Protocol"]), "src_dst": f"{meta.at[j, 'Src IP']}:{meta.at[j, 'Src Port']} -> "
                                                                     f"{meta.at[j, 'Dst IP']}:{meta.at[j, 'Dst Port']}",
                     "reverse_of_attack_pair": bool(meta.at[j, "reverse"]),
                     "lgbm_top_toward_attack": fmt_contrib(sv.loc[j], X.loc[j], +1),
                     "lgbm_top_toward_benign": fmt_contrib(sv.loc[j], X.loc[j], -1),
                     "rf_top_toward_attack": fmt_contrib(rsv.loc[j], X.loc[j], +1) if j in rsv.index else "",
                     "rf_top_toward_benign": fmt_contrib(rsv.loc[j], X.loc[j], -1) if j in rsv.index else ""})
    return out(pd.DataFrame(rows), "casebook")


# ------------------------------------------------------------------ figures
def figures(sv, X, meta, oc, uni_idx, g, fam_share, dec, parts, st, feat12, feat_rf, fp_feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False})

    def save(fig, name):
        fig.savefig(FIG / name, dpi=130, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
    n_all = len(meta)

    # 1 global importance (census)
    top = g.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(top.feature, top.mean_abs_all, color=C[0], height=0.7)
    ax.set_xlabel("mean |SHAP| (log-odds of attack)")
    ax.set_title(f"Selected LightGBM — global importance, full chronological test (n = {n_all:,})", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "global_importance.png")

    # 2 beeswarm (uniform sample)
    import shap
    fig = plt.figure(figsize=(7, 6))
    np.random.seed(SEED)  # shap's beeswarm jitter uses the global NumPy RNG
    shap.summary_plot(sv.loc[uni_idx].to_numpy(), X.loc[uni_idx, FEATS], max_display=15, show=False,
                      plot_size=None, color_bar_label="feature value (rank-coloured)")
    plt.title(f"SHAP values, uniform random test sample (n = {len(uni_idx):,})", loc="left", fontsize=10)
    plt.xlabel("SHAP value (log-odds of attack)")
    save(plt.gcf(), "beeswarm_uniform.png")

    # 3 dependence plots for top features + flagged groups
    dep = list(g.feature[:4])
    for f in ("FWD Init Win Bytes", "Bwd Init Win Bytes", "Bwd Header Length", "Fwd Header Length"):
        if f not in dep and len(dep) < 6:
            dep.append(f)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    y = meta.y_binary.to_numpy()
    for ax, f in zip(axes.ravel(), dep):
        for cls, col, lab in ((0, C[0], "benign"), (1, C[1], "attack")):
            i = uni_idx[y[uni_idx] == cls]
            ax.scatter(X.loc[i, f], sv.loc[i, f], s=4, alpha=0.4, color=col, label=lab, linewidths=0)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(f, loc="left")
        ax.set_xlabel("feature value (symlog)")
        ax.set_ylabel("SHAP (log-odds)")
        ax.axhline(0, color=INK2, lw=0.6)
    axes[0, 0].legend(markerscale=3, loc="best")
    fig.suptitle(f"Dependence: feature value vs SHAP, uniform test sample (n = {len(uni_idx):,})", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "dependence.png")

    # 4 benign score ECDF, validation vs test, per day group
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, dg in zip(axes, DG_ORDER):
        for part, col in (("val", C[0]), ("test", C[1])):
            mm = parts[part][2]
            s = np.sort(logit(mm.score[(mm.y_binary == 0) & (mm.daygroup == dg)]))
            ax.plot(s, 1 - np.arange(1, len(s) + 1) / len(s), color=col, lw=2, label=f"{part} (n = {len(s):,})")
        ax.axvline(logit(THR), color=INK2, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_title(dg, loc="left")
        ax.set_xlabel("logit(score)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("share of benign rows with score above x")
    fig.suptitle("Benign score tails, validation vs test (dashed = frozen 0.1 % threshold)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "benign_score_tails_val_test.png")

    # 5 FPR decomposition by direction
    d = dec[dec.daygroup != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, rv, title in ((axes[0], True, "reverse of a same-day attack pair"), (axes[1], False, "all other benign rows")):
        for k, (part, col) in enumerate((("val", C[0]), ("test", C[1]))):
            r = d[(d.partition == part) & (d.reverse_of_attack == rv)].set_index("daygroup").loc[DG_ORDER]
            x = np.arange(3) + (k - 0.5) * 0.36
            ax.bar(x, 100 * r.fpr, width=0.34, color=col, label=part)
            ax.errorbar(x, 100 * r.fpr, yerr=[100 * (r.fpr - r.ci95_low), 100 * (r.ci95_high - r.fpr)],
                        fmt="none", ecolor=INK2, lw=1, capsize=2)
        ax.set_xticks(range(3), DG_ORDER)
        ax.set_title(title, loc="left")
        ax.set_ylabel("FPR at frozen threshold (%)")
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    fig.suptitle("Benign false-positive rate by traffic direction (95 % Wilson intervals)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fpr_by_direction_val_test.png")

    # 6 family x feature-group heatmap (families with n >= 30)
    h = fam_share.set_index("label").drop(columns="n")
    h = h[[c for c in h.mean().sort_values(ascending=False).index]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(h.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(h.shape[1]), h.columns, rotation=35, ha="right")
    ax.set_yticks(range(h.shape[0]), [f"{l} (n = {n:,})" for l, n in zip(fam_share.label, fam_share.n)])
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.iat[i, j]
            if v >= 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else INK)
    fig.colorbar(im, ax=ax, label="share of mean |SHAP|")
    ax.set_title("Per-family share of |SHAP| by feature group (test attack rows, labels with n ≥ 30)", loc="left")
    save(fig, "family_group_share.png")

    # 7 score distribution by outcome
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, o, col in zip(axes, ("TN", "FP", "TP", "FN"), (C[0], C[1], C[2], C[3])):
        s = logit(meta.score[oc == o])
        ax.hist(s, bins=60, color=col)
        ax.axvline(logit(THR), color=INK2, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{o} (n = {len(s):,})", loc="left")
        ax.set_xlabel("logit(score)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("rows (log scale)")
    fig.suptitle("Test score distribution by outcome (dashed = frozen threshold)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "score_by_outcome.png")

    # 8 E1 vs E2 and LightGBM vs RF (normalised shares, top 15 of the selected model)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, df, a, b, la, lb, title in (
            (axes[0], feat12.assign(a=feat12.E1_mean_abs / feat12.E1_mean_abs.sum(), b=feat12.E2_mean_abs / feat12.E2_mean_abs.sum()),
             "a", "b", "E1 chronological", "E2 random split", "LightGBM: chronological vs random-split model (own test sets)"),
            (axes[1], feat_rf, "lgbm_share", "rf_share", "LightGBM", "Random Forest",
             f"Selected LightGBM vs frozen RF (stratified sample, n = {FACTS['rf_sample_n']:,})")):
        t = df.head(15).iloc[::-1]
        yy = np.arange(len(t))
        ax.barh(yy + 0.2, t[a], height=0.38, color=C[0], label=la)
        ax.barh(yy - 0.2, t[b], height=0.38, color=C[1], label=lb)
        ax.set_yticks(yy, t.feature)
        ax.set_xlabel("share of total mean |SHAP|")
        ax.set_title(title, loc="left", fontsize=9)
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "model_comparison_importance.png")

    # 9 bootstrap rank stability
    s = st[st.bootstrap_n == BOOT_N[0]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.errorbar(s.rank_median, s.feature, xerr=[s.rank_median - s.rank_p5, s.rank_p95 - s.rank_median],
                fmt="o", color=C[0], ecolor=C[0], ms=4, lw=1.5)
    ax.plot(s.census_rank, s.feature, "|", color=INK, ms=10, label="census rank")
    ax.set_xlabel("rank by mean |SHAP| (1 = most important)")
    ax.set_title(f"Rank stability, {B} bootstrap samples of n = {BOOT_N[0]:,} test rows (5–95 %)", loc="left")
    ax.legend(loc="lower right")
    save(fig, "stability_bootstrap_ranks.png")

    # 10 FP top features by day group
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, dg, col in zip(axes, DG_ORDER, C):
        t = fp_feats[fp_feats.daygroup == dg].head(8).iloc[::-1]
        n = int(((oc == "FP") & (meta.daygroup == dg).to_numpy()).sum())
        ax.barh(t.feature, t.fp_mean_abs, color=col)
        ax.set_title(f"{dg} FPs (n = {n:,})", loc="left")
        ax.set_xlabel("mean |SHAP| (log-odds)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Features carrying false positives, by day group (selected LightGBM, full test)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fp_features_by_daygroup.png")


def main():
    ds, pairs, windows, parts = load_all()
    for p, stored in (("test", M4 / "test" / "E1_primary__lgbm__test_predictions.parquet"),
                      ("val", M4 / "preds" / f"{RUN['provenance']['models']['E1_primary__lgbm']['path'].split('/')[-1].replace('.joblib', '')}__val.parquet")):
        st = pd.read_parquet(stored)
        m = parts[p][2]
        if not (st[KEY].to_numpy() == m[KEY].to_numpy()).all():
            raise ValueError(p)
        m["score"] = st["score"].to_numpy()
    X, y, meta = parts["test"]
    sv = shap_table("shap_E1_lgbm_test.parquet", meta)
    svv = shap_table("shap_E1_lgbm_val.parquet", parts["val"][2])
    base = RUN["E1_lgbm"]["base"]
    # re-verify additivity on every stored row (log-odds -> probability)
    err = np.abs(1 / (1 + np.exp(-(base + sv.sum(axis=1).to_numpy()))) - meta.score.to_numpy()).max()
    if err > 1e-9:
        raise ValueError(f"stored SHAP additivity error {err}")
    FACTS["stored_additivity_max_abs_prob_error"] = float(err)
    import joblib
    est = joblib.load(ROOT / RUN["provenance"]["models"]["E1_primary__lgbm"]["path"]).named_steps["est"]
    FACTS["booster_trees"], FACTS["best_iteration"] = int(est.booster_.num_trees()), int(est.best_iteration_)
    oc = outcome(y, meta.score, THR)
    uni = pd.read_parquet(ART / "sample_uniform.parquet")
    uni_idx = meta.reset_index().merge(uni, on=KEY)["index"].to_numpy()
    FACTS.update({"n_test": len(meta), "n_val": len(parts["val"][2]), "threshold": THR,
                  "outcomes": pd.Series(oc).value_counts().to_dict(), "base_log_odds": base,
                  "base_probability": float(1 / (1 + np.exp(-base))), "n_uniform": len(uni_idx),
                  "n_model_inputs": len(FEATS)})

    g = global_importance(sv, X, meta, uni_idx)
    population_groups(sv, meta, oc)
    top, fam_share, rare, ov = per_family(sv, X, meta, oc)
    byday, cmp, fp_feats, hp = fp_tables(sv, X, meta, oc, parts, windows)
    dec, sq = shift_decomposition(parts, sv, svv, windows)
    feature_shift(parts, list(importance(sv, oc == "FP").index[:8]))
    fn_cases(sv, X, meta, oc, parts)
    novel_dup(sv, meta, oc)
    summ12, feat12, *_ = random_split(sv, meta, oc)
    summ_rf, feat_rf, rsv, *_ = rf_compare(sv, meta, oc)
    st, ps, _ = stability(sv, meta, oc)
    interactions()
    score_quantiles(meta, oc, parts)
    casebook(sv, X, meta, oc, rsv)
    FACTS["fp_top10_features"] = list(importance(sv, oc == "FP").index[:10])
    FACTS["tn_top10_features"] = list(importance(sv, oc == "TN").index[:10])
    FACTS["fp_vs_tn_top10_overlap"] = topk_overlap(importance(sv, oc == "FP"), importance(sv, oc == "TN"), 10)
    FACTS["run_checks"] = RUN["checks"]
    FACTS["timing_s"] = {k: round(v, 1) for k, v in RUN["timing_s"].items()}
    figures(sv, X, meta, oc, uni_idx, g, fam_share, dec, parts, st, feat12, feat_rf, fp_feats)
    (GENERATED / "m5_facts.json").write_text(json.dumps(FACTS, indent=2, sort_keys=True, default=float) + "\n")
    print("done")


if __name__ == "__main__":
    main()
