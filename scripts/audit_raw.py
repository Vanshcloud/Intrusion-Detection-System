"""Per-file raw inspection of the extracted CSVs (no transformation, no merging).

Writes reports/generated/raw_<day>.json and reports/generated/raw_summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import GENERATED, csv_paths, sha256  # noqa: E402

TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
NEAR_CONSTANT_SHARE = 0.999  # top value covers >= 99.9 % of rows
CATEGORICAL = ["Protocol", "Label", "Attempted Category", "ICMP Code", "ICMP Type"]


def counts(s):
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}


def inspect(path):
    df = pd.read_csv(path, low_memory=False)
    out = {"file": path.name, "sha256_at_audit": sha256(path), "rows": len(df), "columns": len(df.columns),
           "column_names": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}

    # --- timestamps
    ts_raw = df["Timestamp"].astype(str)
    ts = pd.to_datetime(ts_raw, format=TS_FORMAT, errors="coerce")
    bad = ts.isna()
    lens = ts_raw.str.len().value_counts().sort_index()
    diffs = ts.diff().dt.total_seconds()
    out["timestamp"] = {
        "format_tried": TS_FORMAT,
        "parse_failures": int(bad.sum()),
        "parse_failure_examples": ts_raw[bad].head(5).tolist(),
        "string_length_counts": {str(k): int(v) for k, v in lens.items()},
        "has_timezone_suffix": bool(ts_raw.str.contains(r"[+-]\d\d:?\d\d$|Z$", regex=True).any()),
        "min": str(ts.min()), "max": str(ts.max()),
        "monotonic_non_decreasing": bool(ts.is_monotonic_increasing),
        "rows_earlier_than_previous_row": int((diffs < 0).sum()),
        "rows_earlier_than_running_max": int((ts < ts.cummax().shift()).sum()),
        "max_backwards_jump_seconds": float(-diffs.min()) if (diffs < 0).any() else 0.0,
        "rows_sharing_timestamp_with_another_row": int(ts.duplicated(keep=False).sum()),
        "distinct_timestamps": int(ts.nunique()),
    }

    # --- missing / non-finite
    num = df.select_dtypes(include="number")
    out["missing_per_column"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    obj = df.select_dtypes(exclude="number")
    out["empty_string_per_column"] = {c: int(v) for c, v in (obj.apply(lambda s: s.astype(str).str.strip() == "")).sum().items() if v}
    arr = num.to_numpy(dtype="float64")
    out["pos_inf_per_column"] = {c: int(v) for c, v in zip(num.columns, np.isposinf(arr).sum(0)) if v}
    out["neg_inf_per_column"] = {c: int(v) for c, v in zip(num.columns, np.isneginf(arr).sum(0)) if v}
    out["negative_values_per_column"] = {c: int(v) for c, v in zip(num.columns, (arr < 0).sum(0)) if v}

    # --- duplicates / identifiers
    no_id = [c for c in df.columns if c != "id"]
    out["duplicates"] = {
        "exact_rows_all_columns": int(df.duplicated().sum()),
        "exact_rows_excluding_id": int(df.duplicated(subset=no_id).sum()),
        "duplicate_id_values": int(df["id"].duplicated().sum()),
        "id_is_1_to_n_in_file_order": bool((df["id"].to_numpy() == np.arange(1, len(df) + 1)).all()),
        "id_min": int(df["id"].min()), "id_max": int(df["id"].max()),
        "rows_with_repeated_flow_id": int(df["Flow ID"].duplicated(keep=False).sum()),
        "distinct_flow_ids": int(df["Flow ID"].nunique()),
        "repeated_flow_id_and_timestamp": int(df.duplicated(subset=["Flow ID", "Timestamp"]).sum()),
    }

    # --- constant / near-constant
    nunique = df.nunique(dropna=False)
    top_share = {c: float(df[c].value_counts(dropna=False, normalize=True).iloc[0]) for c in df.columns}
    out["constant_columns"] = [c for c in df.columns if nunique[c] == 1]
    out["near_constant_columns"] = {c: round(top_share[c], 6) for c in df.columns
                                    if nunique[c] > 1 and top_share[c] >= NEAR_CONSTANT_SHARE}
    out["nunique"] = {c: int(v) for c, v in nunique.items()}

    # --- categorical / labels
    out["categorical_values"] = {c: counts(df[c]) for c in CATEGORICAL}
    out["label_by_attempted_category"] = {
        f"{lab} | {att}": int(n) for (lab, att), n in df.groupby(["Label", "Attempted Category"]).size().items()}
    out["numeric_summary"] = {c: {"min": float(num[c].replace([np.inf, -np.inf], np.nan).min()),
                                  "max": float(num[c].replace([np.inf, -np.inf], np.nan).max())} for c in num.columns}
    return out


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    summary = []
    headers = {}
    for p in csv_paths():
        r = inspect(p)
        (GENERATED / f"raw_{p.stem}.json").write_text(json.dumps(r, indent=2) + "\n")
        headers[p.name] = r["column_names"]
        summary.append({k: r[k] for k in ("file", "rows", "columns")} | {
            "ts_min": r["timestamp"]["min"], "ts_max": r["timestamp"]["max"],
            "ts_parse_failures": r["timestamp"]["parse_failures"],
            "ts_monotonic": r["timestamp"]["monotonic_non_decreasing"],
            "exact_dup_rows_excl_id": r["duplicates"]["exact_rows_excluding_id"]})
        print(summary[-1], flush=True)
    first = next(iter(headers.values()))
    res = {"files": summary, "total_rows": sum(s["rows"] for s in summary),
           "identical_headers_across_files": all(h == first for h in headers.values()),
           "header": first}
    (GENERATED / "raw_summary.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
