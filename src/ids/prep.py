"""Milestone 3 data preparation: cleaning, targets, feature manifest, splits, novel-vector subsets.

Pure functions on DataFrames so they can be unit-tested on small synthetic frames.
Rows are identified by (source_file, source_row) throughout; nothing is re-indexed away.
"""
import numpy as np
import pandas as pd

from ids.data import METADATA, PROVENANCE_COLS, REQUIRES_EXPERIMENT, TARGET

PROTOCOL_VERSION = "v1"
SEED = 20170703  # fixed; used only by the random-split comparison
TEST_FRAC = 0.30
VAL_FRAC_OF_REST = 0.20  # of the pre-test part -> overall ~56 / 14 / 30
MIN_STRATUM = 3          # every stratum must yield >= 1 train, val and test row
ORDER = ["t_start", "source_file", "source_row"]  # deterministic chronological order with tie-break

TCP_WINDOW = ["FWD Init Win Bytes", "Bwd Init Win Bytes"]
HEADER_LENGTH = ["Fwd Header Length", "Bwd Header Length"]
CATEGORICAL = ["Protocol", "ICMP Code", "ICMP Type"]
EXCLUDED_LEAKAGE = ["Src IP", "Dst IP", "Src Port", "Flow ID"]
TIMING = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Total TCP Flow Time"]
TIMING_PREFIX = ("Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle", "Fwd Bulk Rate", "Bwd Bulk Rate")

# Multiclass family mapping. Only Portscan is unified (identical vectors under two labels,
# DATA_AUDIT §7/§9); rare attacks are NOT merged. Attempted flows -> BENIGN (primary policy).
FAMILY_OVERRIDES = {"Infiltration - Portscan": "Portscan"}


# ------------------------------------------------------------------ cleaning and targets
def clean(df):
    """Drop rows with non-finite rate values. Returns (kept, removed)."""
    bad = ~np.isfinite(df["Flow Bytes/s"]) | ~np.isfinite(df["Flow Packets/s"])
    return df.loc[~bad].copy(), df.loc[bad].copy()


def add_targets(df):
    df = df.copy()
    df["label_original"] = df["Label"]
    df["is_attempted"] = df["Label"].str.endswith(" - Attempted")
    # Primary binary target (authors' default): successful attack = 1; benign and Attempted = 0.
    df["y_binary"] = ((df["Label"] != "BENIGN") & ~df["is_attempted"]).astype("int8")
    df["family"] = np.where(df["y_binary"] == 1, df["Label"].replace(FAMILY_OVERRIDES), "BENIGN")
    df["t_start"] = (df["ts"] - pd.Timestamp(0)) / pd.Timedelta(seconds=1)  # unit-independent seconds
    df["t_end"] = df["t_start"] + df["Flow Duration"] / 1e6
    df["day"] = df["source_file"].str.removesuffix(".csv")
    df["stratum"] = df["day"] + "|" + df["Label"]
    df["flag_protocol0"] = df["Protocol"] == 0
    df["flag_negative_header_length"] = (df["Fwd Header Length"] < 0) | (df["Bwd Header Length"] < 0)
    return df


def label_mapping(df):
    m = (df.groupby(["label_original", "is_attempted", "y_binary", "family"]).size()
         .rename("rows").reset_index().sort_values("label_original"))
    return [dict(label_original=r.label_original, is_attempted=bool(r.is_attempted), y_binary=int(r.y_binary),
                 family=r.family, rows=int(r.rows)) for r in m.itertuples()]


# ------------------------------------------------------------------ feature manifest
def feature_manifest(raw_columns):
    """Explicit allow-lists. Every raw column gets exactly one class."""
    targets = ["Label"]
    metadata = ["id", "Timestamp", "Attempted Category"]  # Attempted Category: target-related metadata
    excluded = EXCLUDED_LEAKAGE
    experimental = REQUIRES_EXPERIMENT
    default = [c for c in raw_columns if c not in targets + metadata + excluded + experimental]
    classes = {c: "target" for c in targets} | {c: "metadata" for c in metadata} | \
              {c: "excluded" for c in excluded} | {c: "experimental" for c in experimental} | \
              {c: "default" for c in default}
    assert set(classes) == set(raw_columns) and len(classes) == len(raw_columns)
    assert set(TARGET + METADATA) <= set(targets + metadata + excluded)
    sets = {
        "default": default,
        "default_plus_dst_port": default + ["Dst Port"],
        "no_tcp_window": [c for c in default if c not in TCP_WINDOW],
        "no_header_length": [c for c in default if c not in HEADER_LENGTH],
    }
    return {"version": PROTOCOL_VERSION, "column_class": classes, "categorical": CATEGORICAL,
            "added_columns_never_features": PROVENANCE_COLS + [
                "ts", "t_start", "t_end", "day", "stratum", "label_original", "is_attempted", "y_binary",
                "family", "flag_protocol0", "flag_negative_header_length"],
            "feature_sets": sets, "feature_set_sizes": {k: len(v) for k, v in sets.items()}}


def forbidden_inputs(manifest):
    cc = manifest["column_class"]
    return {c for c, k in cc.items() if k in ("target", "metadata", "excluded")} | set(manifest["added_columns_never_features"])


def check_features(features, manifest):
    bad = set(features) & forbidden_inputs(manifest)
    if bad:
        raise ValueError(f"forbidden model inputs: {sorted(bad)}")
    unknown = set(features) - set(manifest["column_class"])
    if unknown:
        raise ValueError(f"columns not in manifest: {sorted(unknown)}")
    return list(features)


# ------------------------------------------------------------------ splits
def _round(x):
    return int(np.floor(x + 0.5))


def partition_sizes(n):
    if n < MIN_STRATUM:
        raise ValueError(f"stratum too small for train/val/test: {n}")
    n_test = max(1, _round(TEST_FRAC * n))
    n_val = max(1, _round(VAL_FRAC_OF_REST * (n - n_test)))
    return n - n_test - n_val, n_val, n_test


def _assign(pos, n):
    n_tr, n_va, _ = partition_sizes(n)
    return np.where(pos < n_tr, "train", np.where(pos < n_tr + n_va, "val", "test"))


def chrono_split(df):
    """Chronological train/val/test inside each (day, raw label) stratum, then purge.

    Purge: a train row whose [t_start, t_end] reaches the first val start (t_end >= t_val0) of its
    stratum is removed; likewise a val row reaching the first test start. Test rows are never removed.
    """
    s = df.sort_values(["stratum"] + ORDER, kind="mergesort")
    pos = s.groupby("stratum").cumcount().to_numpy()
    n = s.groupby("stratum")["stratum"].transform("size").to_numpy()
    part = np.empty(len(s), dtype=object)
    for k in np.unique(n):
        m = n == k
        part[m] = _assign(pos[m], k)
    out = pd.DataFrame({"source_file": s["source_file"].to_numpy(), "source_row": s["source_row"].to_numpy(),
                        "stratum": s["stratum"].to_numpy(), "partition_before_purge": part,
                        "t_start": s["t_start"].to_numpy(), "t_end": s["t_end"].to_numpy()})
    first = lambda p: out[out["partition_before_purge"] == p].groupby("stratum")["t_start"].min()  # noqa: E731
    t_val0 = out["stratum"].map(first("val")).to_numpy()
    t_test0 = out["stratum"].map(first("test")).to_numpy()
    pb = out["partition_before_purge"].to_numpy()
    reason = np.where((pb == "train") & (out["t_end"].to_numpy() >= t_val0), "train_reaches_val",
                      np.where((pb == "val") & (out["t_end"].to_numpy() >= t_test0), "val_reaches_test", ""))
    out["purge_reason"] = reason
    out["partition"] = np.where(reason == "", pb, "purged")
    return out.drop(columns=["t_start", "t_end"])


def random_split(df, seed=SEED):
    """Optimistic comparison: same per-stratum sizes as chrono_split, random order, no purge."""
    rng = np.random.default_rng(seed)
    s = df.sort_values(["stratum", "source_file", "source_row"], kind="mergesort").reset_index(drop=True)
    s["_u"] = rng.random(len(s))
    s = s.sort_values(["stratum", "_u"], kind="mergesort")
    pos = s.groupby("stratum").cumcount().to_numpy()
    n = s.groupby("stratum")["stratum"].transform("size").to_numpy()
    part = np.empty(len(s), dtype=object)
    for k in np.unique(n):
        m = n == k
        part[m] = _assign(pos[m], k)
    return pd.DataFrame({"source_file": s["source_file"].to_numpy(), "source_row": s["source_row"].to_numpy(),
                         "stratum": s["stratum"].to_numpy(), "partition": part, "purge_reason": ""})


def split_boundaries(df, split):
    x = df[["source_file", "source_row", "stratum", "t_start"]].merge(split, on=["source_file", "source_row", "stratum"])
    g = x.groupby(["stratum", "partition"])["t_start"].agg(["size", "min", "max"]).unstack("partition")
    g.columns = [f"{b}_{a}" for a, b in g.columns]
    for p in ("train", "val", "test", "purged"):
        g[f"{p}_size"] = g.get(f"{p}_size", pd.Series(0, index=g.index)).fillna(0).astype(int)
    keep = [c for c in ["train_size", "val_size", "test_size", "purged_size", "train_min", "train_max",
                        "val_min", "val_max", "test_min", "test_max"] if c in g]
    g = g[keep].reset_index()
    for c in g.columns:
        if c.endswith(("_min", "_max")):
            g[c] = pd.to_datetime(g[c], unit="s")
    return g


# ------------------------------------------------------------------ exact feature-vector identity
def canonical(df, features):
    """Column-ordered frame with -0.0 -> 0.0 and one NaN representation; dtype preserved per column."""
    out = {}
    for c in features:
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.floating):
            v = v.astype("float64") + 0.0
            v[np.isnan(v)] = np.nan
        elif not np.issubdtype(v.dtype, np.integer):
            raise TypeError(f"{c}: non-numeric feature dtype {v.dtype}")
        out[c] = v
    return pd.DataFrame(out, columns=list(features))


def vector_keys(df, features):
    """128-bit key (two independent 64-bit hashes) of the canonical, ordered feature vector."""
    x = canonical(df, features)
    h1 = pd.util.hash_pandas_object(x, index=False, hash_key="ids-m3-key-one!!").to_numpy()
    h2 = pd.util.hash_pandas_object(x, index=False, hash_key="ids-m3-key-two!!").to_numpy()
    return h1, h2


def seen_in(df, features, ref_mask, query_mask):
    """For rows in query_mask: does the exact feature vector occur among ref_mask rows?

    Hash match is then confirmed by exact value comparison against one reference row,
    so a hash collision can never mark a novel row as seen.
    """
    h1, h2 = vector_keys(df, features)
    keys = pd.DataFrame({"h1": h1, "h2": h2, "i": np.arange(len(df))})
    ref = keys[ref_mask].drop_duplicates(["h1", "h2"])
    q = keys[query_mask].merge(ref, on=["h1", "h2"], how="left", suffixes=("", "_ref"))
    hit = q["i_ref"].notna().to_numpy()
    if hit.any():
        x = canonical(df, features)
        ia, ib = q.loc[hit, "i"].to_numpy(), q.loc[hit, "i_ref"].astype(int).to_numpy()
        same = np.ones(len(ia), bool)
        for c in features:  # native dtype per column: no int->float rounding
            a, b = x[c].to_numpy()[ia], x[c].to_numpy()[ib]
            eq = a == b
            if np.issubdtype(a.dtype, np.floating):
                eq |= np.isnan(a) & np.isnan(b)
            same &= eq
        if not same.all():  # would indicate a collision; treat as not seen
            hit[np.flatnonzero(hit)[~same]] = False
    seen = np.zeros(len(df), bool)
    seen[q["i"].to_numpy()] = hit
    return seen


def coarse_features(features):
    return [c for c in features if c not in TIMING and not c.startswith(TIMING_PREFIX)]


def content_hash(df):
    """Order-sensitive hash of values (independent of Parquet encoding / library metadata)."""
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
