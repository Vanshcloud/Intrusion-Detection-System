"""Cross-file analyses on data/interim/*.parquet: labels, documented-window check, episodes,
duplicates, identifier / single-feature screening and split-strategy simulation.

Exploratory only: nothing here is feature selection or model evaluation. Deterministic (SEED).
Outputs go to reports/generated/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import DAYS, GENERATED, INTERIM, model_features  # noqa: E402

SEED = 0
BURST_GAPS_S = [60, 300, 900]  # gap thresholds used to split a label's flows into bursts
EMBARGO_S = [0, 120, 600]      # 120 s = CICFlowMeter active timeout (max Flow Duration observed)
TIMING = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Total TCP Flow Time"]
TIMING_PREFIX = ("Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle", "Fwd Bulk Rate", "Bwd Bulk Rate")
# Documented labelling windows (UTC, first/last packet) transcribed from
# https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html (unix seconds as printed there).
DOC_WINDOWS = {
    "FTP-Patator": [(1499170672.838272, 1499174416.931403)],
    "SSH-Patator": [(1499188141.049616, 1499195059.018486)],
    "DoS Hulk": [(1499262203.194704, 1499263641.326171)],
    "DoS GoldenEye": [(1499263803.231753, 1499264408.915718)],
    "DoS Slowloris": [(1499258934.539220, 1499260278.500956)],
    "DoS Slowhttptest": [(1499260537.936810, 1499261869.331517)],
    "Heartbleed": [(1499278335.650811, 1499279563.294455)],
    "Web Attack - SQL Injection": [(1499348145.732950, 1499348575.320284)],
    "Web Attack - XSS": [(1499346935.283859, 1499348121.341704)],
    "Web Attack - Brute Force": [(1499343567.660566, 1499346011.622209)],
    "Infiltration": [(1499361542.547210, 1499366769.364731), (1499363616.453990, 1499371339.347892)],
    "Infiltration - Portscan": [(1499360431.706755, 1499360445.728887), (1499362410.884008, 1499362444.285175),
                                (1499364314.425162, 1499366764.331875)],
    "Portscan": [(1499446532.117090, 1499447948.582083), (1499449905.450532, 1499451841.699238)],
    "Botnet": [(1499432653.990571, 1499436122.903736)],
    "DDoS": [(1499453791.796937, 1499454972.216560)],
}
HULK_BWD_VALUES = [11595, 23190, 11606, 23201]  # Liu et al. documentation, DoS Hulk section


def load():
    df = pd.concat([pd.read_parquet(INTERIM / f"{d}.parquet") for d in DAYS], ignore_index=True)
    df["day"] = df["source_file"].str.removesuffix(".csv")
    df["attempted"] = df["Label"].str.endswith(" - Attempted")
    df["base_label"] = df["Label"].str.removesuffix(" - Attempted")
    # Authors' default policy ("when in doubt"): Attempted -> Benign. Used here only for screening.
    df["y"] = ((df["Label"] != "BENIGN") & ~df["attempted"]).astype(int)
    df["t"] = df["ts"].astype("int64") / 1e6  # seconds (naive timestamps interpreted as-is)
    return df


def write(name, obj):
    p = GENERATED / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False)
    else:
        p.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def feature_hash(df, cols):
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


# ------------------------------------------------------------------ labels
def labels(df):
    tab = df.groupby(["day", "Label", "Attempted Category"]).size().rename("n").reset_index()
    write("labels_by_day.csv", tab)
    consistent = bool(((df["Attempted Category"] != -1) == df["attempted"]).all())
    per_label = df.groupby("Label").size().sort_values(ascending=False)
    out = {
        "total_rows": len(df),
        "label_counts": per_label.to_dict(),
        "attempted_category_counts": df["Attempted Category"].value_counts().sort_index().to_dict(),
        "attempted_flag_consistent_with_label_suffix": consistent,
        "benign": int((df["Label"] == "BENIGN").sum()),
        "successful_attack": int(df["y"].sum()),
        "attempted": int(df["attempted"].sum()),
        "labels_on_more_than_one_day": df.groupby("Label")["day"].nunique().loc[lambda s: s > 1].to_dict(),
        "unexpected_labels": sorted(set(df["Label"]) - set(DOC_WINDOWS) - {"BENIGN"}
                                    - {f"{k} - Attempted" for k in DOC_WINDOWS} - {"Botnet - Attempted"}),
    }
    write("labels.json", out)
    return out


# ------------------------------------------------------------------ timestamps
def timestamps(df):
    rows = []
    for lab, w in DOC_WINDOWS.items():
        x = df.loc[df["Label"] == lab]
        if x.empty:
            rows.append({"label": lab, "n": 0})
            continue
        inside = np.zeros(len(x), bool)
        for a, b in w:
            inside |= (x["t"].to_numpy() >= a - 1e-6) & (x["t"].to_numpy() <= b + 1e-6)
        rows.append({"label": lab, "n": len(x), "share_start_inside_documented_utc_window": inside.mean(),
                     "first": x["ts"].min(), "last": x["ts"].max(),
                     "doc_first": pd.to_datetime(min(a for a, _ in w), unit="s"),
                     "doc_last": pd.to_datetime(max(b for _, b in w), unit="s")})
    write("doc_window_check.csv", pd.DataFrame(rows))
    out = {}
    for d, g in df.groupby("day", sort=False):
        s = g.sort_values("t")
        gaps = np.diff(s["t"].to_numpy())
        out[d] = {"min": g["ts"].min(), "max": g["ts"].max(),
                  "span_hours": (g["t"].max() - g["t"].min()) / 3600,
                  "file_order_is_time_order": bool(g["ts"].is_monotonic_increasing),
                  "spearman_file_order_vs_time": float(pd.Series(g["source_row"].to_numpy()).corr(
                      pd.Series(g["t"].to_numpy()), method="spearman")),
                  "rows_with_non_unique_timestamp": int(g["ts"].duplicated(keep=False).sum()),
                  "largest_gap_between_consecutive_flow_starts_s": float(gaps.max()),
                  "flow_duration_max_us": int(g["Flow Duration"].max()),
                  "rows_at_120s_timeout": int((g["Flow Duration"] == 120_000_000).sum())}
    # identical timestamps: which labels are involved
    dup_ts = df[df.duplicated(["day", "ts"], keep=False)]
    out["non_unique_timestamp_rows_by_label"] = dup_ts["Label"].value_counts().to_dict()
    write("timestamps.json", out)
    return out


# ------------------------------------------------------------------ episodes
def episodes(df):
    rows = []
    for (day, lab), g in df[df["Label"] != "BENIGN"].groupby(["day", "Label"]):
        t = np.sort(g["t"].to_numpy())
        gaps = np.diff(t)
        end = (g["t"] + g["Flow Duration"] / 1e6).max()
        r = {"day": day, "label": lab, "n": len(t),
             "first": pd.to_datetime(t[0], unit="s"), "last_start": pd.to_datetime(t[-1], unit="s"),
             "last_end": pd.to_datetime(end, unit="s"), "span_min": (t[-1] - t[0]) / 60,
             "src_ips": g["Src IP"].nunique(), "dst_ips": g["Dst IP"].nunique(),
             "dst_ports": g["Dst Port"].nunique(), "median_gap_s": float(np.median(gaps)) if len(gaps) else np.nan,
             "top_gaps_min": ", ".join(f"{x / 60:.1f}" for x in np.sort(gaps)[::-1][:3])}
        for thr in BURST_GAPS_S:
            r[f"bursts_gap>{thr}s"] = int((gaps > thr).sum() + 1)
        rows.append(r)
    ep = pd.DataFrame(rows).sort_values(["day", "first"])
    write("episodes.csv", ep)
    return ep


def burst_ids(g, gap):
    order = np.argsort(g["t"].to_numpy(), kind="stable")
    t = g["t"].to_numpy()[order]
    b = np.concatenate([[0], np.cumsum(np.diff(t) > gap)])
    out = np.empty(len(g), int)
    out[order] = b
    return out


# ------------------------------------------------------------------ duplicates
def duplicates(df, feats):
    coarse = [c for c in feats if c not in TIMING and not c.startswith(TIMING_PREFIX)]
    df["h"] = feature_hash(df, feats)
    df["h_dport"] = feature_hash(df, feats + ["Dst Port"])
    df["h_coarse"] = feature_hash(df, coarse)
    out = {"n_features_hashed": len(feats), "coarse_feature_count": len(coarse),
           "coarse_features_excluded_timing": sorted(set(feats) - set(coarse))}
    for key, name in [("h", "features"), ("h_dport", "features_plus_dst_port"), ("h_coarse", "coarse_no_timing")]:
        g = df.groupby(key)
        size, nlab, ny = g.size(), g["Label"].nunique(), g["y"].nunique()
        dup = df[key].map(size) > 1
        out[name] = {
            "distinct_vectors": int(len(size)),
            "rows_in_duplicated_vectors": int(dup.sum()),
            "redundant_rows_if_deduplicated": int(len(df) - len(size)),
            "vectors_with_multiple_raw_labels": int((nlab > 1).sum()),
            "rows_in_vectors_with_multiple_raw_labels": int(size[nlab > 1].sum()),
            "vectors_with_conflicting_binary_y": int((ny > 1).sum()),
            "rows_in_vectors_with_conflicting_binary_y": int(size[ny > 1].sum()),
            "vectors_on_multiple_days": int((g["day"].nunique() > 1).sum()),
        }
    # per-label duplicate structure (model features)
    size = df.groupby("h").size()
    lab_per_h = df.groupby("h")["Label"].nunique()
    rows = []
    for lab, g in df.groupby("Label"):
        s = g.sort_values("t")
        same_prev = s.groupby("h")["t"].diff()  # time since previous identical vector in same label
        rows.append({"label": lab, "n": len(g), "distinct_vectors": g["h"].nunique(),
                     "share_rows_duplicated_within_label": float(g["h"].duplicated(keep=False).mean()),
                     "share_rows_vector_also_under_other_label": float((g["h"].map(lab_per_h) > 1).mean()),
                     "share_rows_coarse_duplicated_within_label": float(g["h_coarse"].duplicated(keep=False).mean()),
                     "repeat_within_1s": float((same_prev <= 1).sum() / max(same_prev.notna().sum(), 1)),
                     "repeat_within_60s": float((same_prev <= 60).sum() / max(same_prev.notna().sum(), 1)),
                     "median_repeat_gap_s": float(same_prev.median()) if same_prev.notna().any() else np.nan,
                     "largest_group": int(g["h"].map(size).max())})
    write("duplicates_per_label.csv", pd.DataFrame(rows).sort_values("n", ascending=False))
    # most frequent conflicting vectors (raw labels)
    conf = df[df["h"].map(lab_per_h) > 1]
    top = (conf.groupby(["h", "Label"]).size().rename("n").reset_index()
           .assign(total=lambda x: x.groupby("h")["n"].transform("sum"))
           .sort_values(["total", "h", "n"], ascending=[False, True, False]))
    top["h"] = top["h"].astype(str)
    top = top[top["h"].isin(top["h"].drop_duplicates().head(25))]
    examples = conf.drop_duplicates("h").set_index("h")[["Protocol", "Dst Port", "Total Fwd Packet", "Total Bwd packets",
                                                       "Total Length of Fwd Packet", "Total Length of Bwd Packet",
                                                       "Flow Duration"]]
    examples.index = examples.index.astype(str)
    write("conflicting_vectors_top.csv", top.join(examples, on="h"))
    pair = conf.groupby("h")["Label"].agg(lambda s: " + ".join(sorted(set(s))))
    write("conflicting_label_pairs.csv", conf.assign(pair=conf["h"].map(pair)).groupby("pair").size()
          .rename("rows").reset_index().sort_values("rows", ascending=False))
    write("duplicates.json", out)
    return out


# ------------------------------------------------------------------ identifiers
def identifiers(df):
    rows = []
    for lab, g in df[df["y"] == 1].groupby("Label"):
        src = set(g["Src IP"])
        same_day_benign = df[(df["day"] == g["day"].iloc[0]) & (df["Label"] == "BENIGN")]
        rows.append({"label": lab, "n": len(g), "src_ips": ", ".join(sorted(src)[:6]),
                     "dst_ips": ", ".join(sorted(set(g["Dst IP"]))[:4]) + (" ..." if g["Dst IP"].nunique() > 4 else ""),
                     "dst_ports_top": ", ".join(map(str, g["Dst Port"].value_counts().head(3).index)),
                     "share_same_day_benign_from_these_src_ips": float(same_day_benign["Src IP"].isin(src).mean()),
                     "benign_rows_from_these_src_ips_same_day": int(same_day_benign["Src IP"].isin(src).sum())})
    write("identifier_per_attack.csv", pd.DataFrame(rows))
    ext = df[df["Src IP"] == "172.16.0.1"]
    out = {"rows_src_172.16.0.1": len(ext),
           "labels_src_172.16.0.1": ext["Label"].value_counts().to_dict(),
           "attack_rows_from_172.16.0.1_share": float((df.loc[df.y == 1, "Src IP"] == "172.16.0.1").mean())}
    write("identifier_summary.json", out)
    return out


# ------------------------------------------------------------------ single-feature screening
def one_feature_scores(x, y, rng, discrete):
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    if discrete:  # value lookup learned on half A, unseen values -> majority class
        m = pd.Series(y[a]).groupby(x[a]).mean()
        prior = y[a].mean()
        p = pd.Series(x[b]).map(m).fillna(prior).to_numpy()
        pred = (p > 0.5).astype(int)
    else:
        big = np.finfo(np.float32).max  # sklearn trees work in float32; +/-inf -> float32 extremes
        xa = np.clip(np.nan_to_num(x[a].astype(float), posinf=big, neginf=-big), -big, big).reshape(-1, 1)
        xb = np.clip(np.nan_to_num(x[b].astype(float), posinf=big, neginf=-big), -big, big).reshape(-1, 1)
        tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED).fit(xa, y[a])
        pred = tree.predict(xb)
    return balanced_accuracy_score(y[b], pred)


def screening(df, feats):
    rng = np.random.default_rng(SEED)
    ident_discrete = ["Src IP", "Dst IP", "Flow ID", "Src Port", "Dst Port", "Protocol"]
    ident_numeric = ["id", "source_row", "t"]
    cols = [(c, True) for c in ident_discrete] + [(c, False) for c in ident_numeric + feats if c not in ident_discrete]
    y = df["y"].to_numpy()
    rows = []
    for c, disc in cols:
        rows.append({"feature": c, "role": "candidate" if c in feats else "identifier/metadata/under experiment",
                     "bal_acc_binary": one_feature_scores(df[c].to_numpy(), y, rng, disc)})
    # mutual information on a fixed stratified sample
    samp = df.groupby("y", group_keys=False).sample(n=50_000, random_state=SEED)
    X = np.nan_to_num(samp[feats].to_numpy(float), posinf=1e300, neginf=-1e300)
    mi = mutual_info_classif(X, samp["y"].to_numpy(), random_state=SEED)
    mi = dict(zip(feats, mi))
    bin_df = pd.DataFrame(rows)
    bin_df["mi_binary_sample"] = bin_df["feature"].map(mi)
    # per-attack: attack label vs benign of the same day (benign subsampled to 50k per day)
    per = []
    for lab, g in df[df["y"] == 1].groupby("Label"):
        day = g["day"].iloc[0]
        ben = df[(df["day"] == day) & (df["Label"] == "BENIGN")]
        ben = ben.sample(n=min(50_000, len(ben)), random_state=SEED)
        sub = pd.concat([g, ben])
        yy = sub["y"].to_numpy()
        for c, disc in cols:
            if c in ("id", "source_row", "Flow ID"):
                continue
            per.append({"label": lab, "n_attack": len(g), "feature": c,
                        "bal_acc": one_feature_scores(sub[c].to_numpy(), yy, rng, disc)})
    per = pd.DataFrame(per)
    bin_df["n_attacks_bal_acc>=0.99"] = bin_df["feature"].map(per[per["bal_acc"] >= 0.99].groupby("feature").size()).fillna(0).astype(int)
    write("single_feature_binary.csv", bin_df.sort_values("bal_acc_binary", ascending=False))
    top = (per[~per["feature"].isin(ident_discrete + ident_numeric)].sort_values(["label", "bal_acc"], ascending=[True, False])
           .groupby("label").head(3))
    write("single_feature_per_attack_top3.csv", top)
    write("single_feature_per_attack_all.csv", per)
    # Liu et al. DoS Hulk claim
    hulk = df["Total Length of Bwd Packet"].isin(HULK_BWD_VALUES)
    is_hulk = df["Label"] == "DoS Hulk"
    write("hulk_check.json", {"hulk_rows": int(is_hulk.sum()), "hulk_rows_with_4_values": int((hulk & is_hulk).sum()),
                              "non_hulk_rows_with_4_values": int((hulk & ~is_hulk).sum()),
                              "non_hulk_labels_with_4_values": df.loc[hulk & ~is_hulk, "Label"].value_counts().to_dict()})
    return bin_df, per


# ------------------------------------------------------------------ split simulation
def split_masks(df):
    """Return {strategy: boolean is_test} for candidate designs. Benign handled per day."""
    rng = np.random.default_rng(SEED)
    strata = df["day"] + "|" + df["Label"]
    out = {}
    # D. random stratified 70/30 (by day+label)
    u = pd.Series(rng.random(len(df)))
    out["D_random_stratified"] = (u.groupby(strata.to_numpy()).rank(pct=True) > 0.7).to_numpy()
    # A. chronological within each (day, label): latest 30 % of flow starts -> test
    r = df.groupby(strata)["t"].rank(method="first", pct=True)
    out["A_chrono_within_label"] = (r > 0.7).to_numpy()
    # A with embargo: drop rows within E s before the per-stratum cut (mark as neither)
    cut = df.assign(r=r).loc[r <= 0.7].groupby(strata[r <= 0.7])["t"].max()
    df["_cut"] = strata.map(cut).to_numpy()
    # B. grouped temporal: bursts (gap > 300 s) inside each stratum; whole bursts assigned chronologically
    b = np.zeros(len(df), int)
    for _, idx in df.groupby(strata).indices.items():
        b[idx] = burst_ids(df.iloc[idx], 300)
    df["_burst"] = b
    nb = df.groupby(strata)["_burst"].transform("max") + 1
    out["B_burst_holdout_last_burst"] = ((df["_burst"] == nb - 1) & (nb > 1)).to_numpy() | (
        (nb == 1).to_numpy() & out["A_chrono_within_label"])  # single-burst strata fall back to A
    return out


def leakage_metrics(df, test, train, name):
    rows = []
    tr, te = df[train], df[test]
    tr_h, tr_hc, tr_fid = set(tr["h"]), set(tr["h_coarse"]), set(tr["Flow ID"])
    tr_lab_h = set(zip(tr["h"], tr["Label"]))
    for lab, g in te.groupby("Label"):
        same = tr[tr["Label"] == lab]["t"].to_numpy()
        gt = g["t"].to_numpy()
        if len(same):
            s = np.sort(same)
            pos = np.clip(np.searchsorted(s, gt), 1, len(s) - 1) if len(s) > 1 else np.zeros(len(gt), int)
            near = np.minimum(np.abs(gt - s[pos]), np.abs(gt - s[np.maximum(pos - 1, 0)]))
        else:
            near = np.full(len(gt), np.nan)
        rows.append({"strategy": name, "label": lab, "n_train": int((tr["Label"] == lab).sum()), "n_test": len(g),
                     "test_vector_in_train_any_label": float(g["h"].isin(tr_h).mean()),
                     "test_vector_in_train_same_label": float(pd.Series(list(zip(g["h"], g["Label"]))).isin(tr_lab_h).mean()),
                     "test_coarse_vector_in_train": float(g["h_coarse"].isin(tr_hc).mean()),
                     "test_flow_id_in_train": float(g["Flow ID"].isin(tr_fid).mean()),
                     "median_s_to_nearest_train_same_label": float(np.nanmedian(near)) if len(near) else np.nan})
    return rows


def splits(df):
    masks = split_masks(df)
    rows = []
    for name, test in masks.items():
        rows += leakage_metrics(df, test, ~test, name)
        if name == "A_chrono_within_label":
            # purge: drop train flows whose [start, start + duration] reaches past the stratum's cut
            end = (df["t"] + df["Flow Duration"] / 1e6).to_numpy()
            purged = ~test & (end <= df["_cut"].to_numpy())
            rows += leakage_metrics(df, test, purged, "A_chrono_purged_train")
            write("purge_cost.json", {"train_rows": int((~test).sum()), "train_rows_purged": int((~test & ~purged).sum()),
                                      "attack_train_rows_purged": int((~test & ~purged & (df["y"] == 1)).sum())})
            for e in EMBARGO_S[1:]:
                keep = ~test | (df["t"] - df["_cut"] > e).to_numpy()  # drop test rows starting < e s after cut
                t2 = test & keep
                rows += leakage_metrics(df, t2, ~test, f"A_chrono_embargo_{e}s")
    res = pd.DataFrame(rows)
    write("split_simulation_per_label.csv", res)
    # overall (attack rows only, weighted by n_test) and benign
    agg = []
    for name, g in res.groupby("strategy", sort=False):
        for part, gg in [("attacks", g[g["label"] != "BENIGN"]), ("benign", g[g["label"] == "BENIGN"])]:
            w = gg["n_test"]
            agg.append({"strategy": name, "part": part, "n_test": int(w.sum()),
                        **{c: float(np.average(gg[c], weights=w)) for c in
                           ["test_vector_in_train_any_label", "test_vector_in_train_same_label",
                            "test_coarse_vector_in_train", "test_flow_id_in_train"]},
                        "labels_with_empty_train": int((gg["n_train"] == 0).sum())})
    write("split_simulation_summary.csv", pd.DataFrame(agg))
    return res


def anomalies(df):
    neg = df[(df["Fwd Header Length"] < 0) | (df["Bwd Header Length"] < 0)]
    inf = df[np.isinf(df["Flow Bytes/s"]) | np.isinf(df["Flow Packets/s"])]
    p0 = df[df["Protocol"] == 0]
    out = {"negative_header_length_rows": len(neg), "negative_header_length_labels": neg["Label"].value_counts().to_dict(),
           "negative_header_length_protocols": neg["Protocol"].value_counts().to_dict(),
           "negative_header_length_flow_duration_min_s": float(neg["Flow Duration"].min() / 1e6),
           "inf_rate_rows": len(inf), "inf_rate_labels": inf["Label"].value_counts().to_dict(),
           "inf_rate_flow_duration_values": inf["Flow Duration"].value_counts().to_dict(),
           "protocol0_rows": len(p0), "protocol0_labels": p0["Label"].value_counts().to_dict(),
           "protocol0_top_flow_ids": p0["Flow ID"].value_counts().head(5).to_dict(),
           "zero_duration_rows": int((df["Flow Duration"] == 0).sum()),
           "zero_duration_labels": df.loc[df["Flow Duration"] == 0, "Label"].value_counts().to_dict(),
           "icmp_fields_set_when_protocol_not_1": int(((df["ICMP Type"] != -1) & (df["Protocol"] != 1)).sum())}
    write("anomalies.json", out)
    return out


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    df = load()
    feats = model_features([c for c in df.columns if c not in
                            ("ts", "day", "attempted", "base_label", "y", "t")])
    write("feature_roles.json", {"candidate_model_features": feats, "n": len(feats)})
    only = set(sys.argv[1:])  # optional: run selected steps, e.g. `audit_deep.py screening`
    steps = [(labels, ()), (timestamps, ()), (episodes, ()), (identifiers, ()), (anomalies, ()),
             (duplicates, (feats,)), (splits, ()), (screening, (feats,))]
    for step, args in steps:
        if only and step.__name__ not in only:
            continue
        if step is splits and "h" not in df:
            duplicates(df, feats)  # splits need the feature hashes
        step(df, *args)
        print(step.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
