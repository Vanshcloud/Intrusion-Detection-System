"""Milestone 3 entry point: cleaned dataset, targets, feature manifest, split manifests,
novel-vector subsets, statistics and the frozen experiment protocol (v1). No model training.

Inputs : data/interim/*.parquet (lossless CSV copies), data/DATASET_PROVENANCE.json,
         reports/generated/anomalies.json (Milestone 2 finding re-checked here)
Outputs: data/processed/*_v1.parquet (git-ignored), configs/*_v1.json, reports/generated/m3_*
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import DAYS, GENERATED, INTERIM, PROCESSED, PROVENANCE, ROOT, sha256  # noqa: E402
from ids.metrics import UNSTABLE_SUPPORT  # noqa: E402
from ids.prep import (PROTOCOL_VERSION as V, SEED, TEST_FRAC, VAL_FRAC_OF_REST, add_targets, check_features,  # noqa: E402
                      chrono_split, clean, coarse_features, content_hash, feature_manifest, label_mapping,
                      random_split, seen_in, split_boundaries)

CONFIGS = ROOT / "configs"
KEY = ["source_file", "source_row"]
VARIANTS = {"primary": "Attempted rows kept, labelled benign (y_binary=0)",
            "excl_attempted": "Attempted rows removed from train, val and test (sensitivity)"}
FPR_BUDGETS = [0.001, 0.01]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def main():
    for p in (PROCESSED, CONFIGS, GENERATED):
        p.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([pd.read_parquet(INTERIM / f"{d}.parquet") for d in DAYS], ignore_index=True)
    raw_cols = [c for c in raw.columns if c not in ("source_file", "source_row", "ts")]

    # ---------------------------------------------------------------- cleaning (fresh re-audit)
    kept, removed = clean(raw)
    m2 = json.loads((GENERATED / "anomalies.json").read_text())
    if len(removed) != m2["inf_rate_rows"] or set(removed["Label"]) != set(m2["inf_rate_labels"]):
        sys.exit(f"non-finite rows differ from Milestone 2: {len(removed)} vs {m2['inf_rate_rows']}")
    if (~np.isfinite(kept.select_dtypes("number").to_numpy(dtype=float))).any():
        sys.exit("non-finite values remain in other columns")
    df = add_targets(kept).reset_index(drop=True)
    rem_cols = KEY + ["id", "Label", "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Timestamp"]
    removed[rem_cols].to_csv(GENERATED / "m3_removed_rows.csv", index=False)
    dump(GENERATED / "m3_cleaning.json", {
        "rows_in": len(raw), "rows_removed_non_finite": len(removed), "rows_out": len(df),
        "removed_rule": "Flow Bytes/s or Flow Packets/s not finite (all have Flow Duration == 0)",
        "removed_flow_duration_values": removed["Flow Duration"].value_counts().to_dict(),
        "kept_flag_protocol0": int(df["flag_protocol0"].sum()),
        "kept_flag_protocol0_by_label": df.loc[df.flag_protocol0, "Label"].value_counts().to_dict(),
        "kept_flag_negative_header_length": int(df["flag_negative_header_length"].sum()),
        "kept_flag_negative_header_length_by_label": df.loc[df.flag_negative_header_length, "Label"].value_counts().to_dict(),
    })

    # ---------------------------------------------------------------- labels and features
    mapping = label_mapping(df)
    dump(CONFIGS / f"label_mapping_{V}.json", {
        "version": V,
        "binary_positive_class": "attack (y_binary = 1) = successful attack labels; BENIGN and every '- Attempted' label = 0",
        "binary_rule_source": "Liu et al. CNS2022 Dataset_Download.html: 'When in doubt, we recommend to relabel all "
                              "Attempted flows as Benign'; Attempted must never be a separate class",
        "family_rule": "family = original label for successful attacks, except 'Infiltration - Portscan' -> 'Portscan'; "
                       "BENIGN for benign and Attempted rows. No other classes merged.",
        "sensitivity_variant": VARIANTS["excl_attempted"],
        "labels": mapping})
    manifest = feature_manifest(raw_cols)
    for fs in manifest["feature_sets"].values():
        check_features(fs, manifest)
    dump(CONFIGS / f"feature_manifest_{V}.json", manifest)

    # ---------------------------------------------------------------- splits
    chrono = chrono_split(df)
    rnd = random_split(df)
    splits = {"chrono": chrono, "random": rnd}
    for name, s in splits.items():
        assert len(s) == len(df) and not s.duplicated(KEY).any()
    part = {name: df[KEY].merge(s[KEY + ["partition"]], on=KEY, how="left")["partition"].to_numpy()
            for name, s in splits.items()}

    # purge verification
    x = df[KEY + ["stratum", "t_start", "t_end", "Label"]].merge(chrono, on=KEY + ["stratum"])
    t0 = {p: x[x.partition_before_purge == p].groupby("stratum")["t_start"].min() for p in ("val", "test")}
    tr, va = x[x.partition == "train"], x[x.partition == "val"]
    purge = {
        "rule": "drop train rows with t_end >= first val t_start of the stratum; drop val rows with t_end >= first "
                "test t_start. Test rows are never removed.",
        "test_rows_before_purge": int((x.partition_before_purge == "test").sum()),
        "test_rows_after_purge": int((x.partition == "test").sum()),
        "purged_by_reason": x.loc[x.partition == "purged", "purge_reason"].value_counts().to_dict(),
        "purged_by_label": x[x.partition == "purged"].groupby(["Label", "purge_reason"]).size()
                           .rename("n").reset_index().to_dict("records"),
        "check_train_ends_before_val": bool((tr.t_end < tr.stratum.map(t0["val"])).all()),
        "check_val_ends_before_test": bool((va.t_end < va.stratum.map(t0["test"])).all()),
    }
    assert purge["test_rows_before_purge"] == purge["test_rows_after_purge"]
    assert purge["check_train_ends_before_val"] and purge["check_val_ends_before_test"]
    dump(GENERATED / "m3_purge.json", purge)
    bnd = split_boundaries(df, chrono[KEY + ["stratum", "partition"]])
    bnd.to_csv(GENERATED / "m3_split_boundaries_chrono.csv", index=False)

    # cross-stratum temporal overlap (why this is not a deployment simulation)
    x["partition"] = x["partition"].astype(str)
    x["day"] = x["stratum"].str.split("|").str[0]
    rows = []
    for (day, st), g in x[x.partition == "test"].groupby(["day", "stratum"]):
        if st.endswith("|BENIGN"):
            continue
        t_first = g["t_start"].min()
        later_train = x[(x.day == day) & (x.partition == "train") & (x.t_start > t_first)]
        rows.append({"stratum": st, "test_rows": len(g), "test_first_start": pd.to_datetime(t_first, unit="s"),
                     "train_rows_same_day_starting_later": len(later_train),
                     "of_which_benign": int((later_train.Label == "BENIGN").sum())})
    day_rows = []
    for day, g in x.groupby("day"):
        t_first_test = g.loc[g.partition == "test", "t_start"].min()
        trn = g[g.partition == "train"]
        day_rows.append({"day": day, "first_test_start": pd.to_datetime(t_first_test, unit="s"),
                         "train_rows": len(trn), "train_rows_starting_after_first_test": int((trn.t_start > t_first_test).sum())})
    pd.DataFrame(rows).to_csv(GENERATED / "m3_cross_stratum_overlap.csv", index=False)
    pd.DataFrame(day_rows).to_csv(GENERATED / "m3_day_overlap.csv", index=False)

    # ---------------------------------------------------------------- partition / class accounting
    acc = []
    for name, p in part.items():
        t = df[["label_original", "family", "y_binary", "is_attempted"]].assign(protocol=name, partition=p)
        acc.append(t.groupby(["protocol", "partition", "label_original", "family", "y_binary", "is_attempted"])
                   .size().rename("n").reset_index())
    acc = pd.concat(acc)
    acc.to_csv(GENERATED / "m3_partition_counts.csv", index=False)
    fam = (acc[acc.partition.isin(["train", "val", "test"])]
           .groupby(["protocol", "family", "partition"])["n"].sum().unstack("partition").fillna(0).astype(int).reset_index())
    fam["test_unstable"] = fam["test"] < UNSTABLE_SUPPORT
    fam.to_csv(GENERATED / "m3_family_distribution.csv", index=False)
    totals = acc.groupby(["protocol", "partition"])["n"].sum().unstack("partition").fillna(0).astype(int)
    assert (totals.sum(axis=1) == len(df)).all()  # every row accounted for exactly once

    # ---------------------------------------------------------------- novel-vector subsets
    novel = df[KEY].copy()
    summ, by_label = [], []
    att = df["is_attempted"].to_numpy()
    for name, p in part.items():
        variants = ["primary", "excl_attempted"] if name == "chrono" else ["primary"]
        for var in variants:
            keep = ~att if var == "excl_attempted" else np.ones(len(df), bool)
            ref, q = (p == "train") & keep, (p == "test") & keep
            for fs_name, fs in manifest["feature_sets"].items():
                seen = seen_in(df, fs, ref, q)
                near = seen_in(df, coarse_features(fs), ref, q)
                col = f"test_novel__{name}__{var}__{fs_name}"
                novel[col] = q & ~seen
                y = df["y_binary"].to_numpy()
                summ.append({"protocol": name, "variant": var, "feature_set": fs_name, "n_features": len(fs),
                             "full_test": int(q.sum()), "novel_test": int((q & ~seen).sum()), "excluded_seen": int(seen.sum()),
                             "attack_full": int((q & (y == 1)).sum()), "attack_novel": int((q & ~seen & (y == 1)).sum()),
                             "benign_full": int((q & (y == 0)).sum()), "benign_novel": int((q & ~seen & (y == 0)).sum()),
                             "attack_exact_overlap_share": float(seen[q & (y == 1)].mean()),
                             "attack_near_overlap_share": float(near[q & (y == 1)].mean()),
                             "benign_exact_overlap_share": float(seen[q & (y == 0)].mean())})
                t = pd.DataFrame({"label_original": df["label_original"], "family": df["family"], "q": q, "seen": seen, "near": near})
                g = t[t.q].groupby(["label_original", "family"]).agg(full=("q", "size"), exact_seen=("seen", "sum"),
                                                                      near_seen=("near", "sum")).reset_index()
                g["novel"] = g["full"] - g["exact_seen"]
                g["exact_overlap_share"] = g["exact_seen"] / g["full"]
                g["near_overlap_share"] = g["near_seen"] / g["full"]
                g["novel_unstable"] = g["novel"] < UNSTABLE_SUPPORT
                by_label.append(g.assign(protocol=name, variant=var, feature_set=fs_name))
    pd.DataFrame(summ).to_csv(GENERATED / "m3_novel_summary.csv", index=False)
    pd.concat(by_label).to_csv(GENERATED / "m3_novel_by_label.csv", index=False)

    # ---------------------------------------------------------------- write processed artifacts
    arts = {"dataset": df, "split_chrono": chrono, "split_random": rnd, "novel_test": novel}
    for name, frame in arts.items():
        frame.to_parquet(PROCESSED / f"{name}_{V}.parquet", index=False)
    hashes = {f"data/processed/{k}_{V}.parquet": {"rows": len(v), "content_sha256": content_hash(v)} for k, v in arts.items()}

    # ---------------------------------------------------------------- frozen protocol
    prov = json.loads(PROVENANCE.read_text())
    protocol = {
        "version": V,
        "status": "frozen before any model training; changes require a new version with a reason",
        "source": {"dataset": prov["variant"], "url": prov["source_url"], "archive_sha256": prov["archive_sha256"],
                   "files": {f["archive_member"]: f["sha256"] for f in prov["extracted_files"]}},
        "cleaning": {"removed": "rows with non-finite Flow Bytes/s or Flow Packets/s (list: reports/generated/m3_removed_rows.csv)",
                     "kept_and_flagged": ["Protocol == 0 (flag_protocol0)", "negative Fwd/Bwd Header Length (flag_negative_header_length)"],
                     "duplicates": "no rows deleted; novelty evaluated separately"},
        "targets": {"binary": "y_binary (configs/label_mapping_v1.json)", "multiclass": "family (same file)",
                    "variants": VARIANTS},
        "features": {"manifest": f"configs/feature_manifest_{V}.json", "sets": manifest["feature_set_sizes"],
                     "selection": "by column name from the manifest only; check_features() rejects target/metadata/excluded columns"},
        "splits": {
            "primary": {"name": "chrono", "strata": "(day, original Label)", "order": "t_start, source_file, source_row",
                        "sizes": f"per stratum: test = round({TEST_FRAC} n), val = round({VAL_FRAC_OF_REST} * (n - test)), "
                                 "each >= 1; rest train (~56/14/30)",
                        "purge": purge["rule"], "manifest": f"data/processed/split_chrono_{V}.parquet",
                        "boundaries": "reports/generated/m3_split_boundaries_chrono.csv"},
            "comparison": {"name": "random", "strata": "(day, original Label)", "seed": SEED,
                           "sizes": "identical per-stratum sizes to chrono (before purge); no purge",
                           "manifest": f"data/processed/split_random_{V}.parquet"}},
        "novel_vector": {"definition": "test row whose exact ordered feature vector (for that feature set) does not occur "
                                       "in the training partition of the same protocol/variant; floats canonicalised "
                                       "(-0.0 -> 0.0, single NaN); 128-bit hash match confirmed by exact per-column comparison",
                         "reference_set": "train partition only (models are fitted on train only in v1)",
                         "near_duplicate_definition": "same equality on non-timing features only (reported, not used as a subset)",
                         "manifest": f"data/processed/novel_test_{V}.parquet"},
        "preprocessing": "src/ids/pipeline.py: fitted on train rows only (constant-column filter; one-hot + scaling for linear models)",
        "model_selection": "hyper-parameters and thresholds chosen on val only; test untouched until final evaluation",
        "metrics": {"positive_class": "attack (y_binary = 1); score = model probability of attack",
                    "threshold_free": ["PR-AUC (average precision) - primary threshold-free summary, prevalence-dependent",
                                       "ROC-AUC - secondary; can look high while false alarms are operationally large"],
                    "operating_points": f"threshold = highest validation recall with validation FPR <= budget, budgets {FPR_BUDGETS}",
                    "at_threshold": ["precision", "recall", "F1", "FPR", "false-positive count", "confusion matrix",
                                     "per-original-label detection / alert rate with support"],
                    "multiclass": "per-class precision, recall, F1, support; macro-F1 over classes present in test",
                    "undefined": "reported as null, never replaced by 0/1",
                    "unstable_support_below": UNSTABLE_SUPPORT,
                    "evaluation_sets": ["full test", "novel-vector test"]},
        "experiments_planned": ["chrono x primary x {default, default_plus_dst_port, no_tcp_window, no_header_length}",
                                "chrono x excl_attempted x default", "random x primary x default (comparison)",
                                "held-out attack family (separate, later milestone)"],
        "artifacts": hashes | {f"configs/{p.name}": {"sha256": sha256(p)} for p in
                               [CONFIGS / f"feature_manifest_{V}.json", CONFIGS / f"label_mapping_{V}.json"]},
    }
    dump(CONFIGS / f"experiment_protocol_{V}.json", protocol)
    print(json.dumps({"rows": len(df), "removed": len(removed), "purged": purge["purged_by_reason"],
                      "test": purge["test_rows_after_purge"]}, default=str))


if __name__ == "__main__":
    main()
