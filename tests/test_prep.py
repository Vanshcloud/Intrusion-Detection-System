"""Unit tests on small synthetic frames (no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from ids.pipeline import preprocessor
from ids.prep import (add_targets, check_features, chrono_split, feature_manifest, forbidden_inputs,
                      partition_sizes, random_split, seen_in)

RAW_COLS = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
            "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Header Length", "Bwd Header Length",
            "FWD Init Win Bytes", "Bwd Init Win Bytes", "ICMP Code", "ICMP Type", "Label", "Attempted Category"]


def frame(n=20, label="DoS Hulk", day="wednesday", start=0.0, step=10.0, dur_s=1.0, shuffle_seed=None):
    t = start + step * np.arange(n)
    df = pd.DataFrame({
        "source_file": f"{day}.csv", "source_row": np.arange(n), "id": np.arange(1, n + 1),
        "Flow ID": "a-b", "Src IP": "1.1.1.1", "Src Port": 1000 + np.arange(n), "Dst IP": "2.2.2.2",
        "Dst Port": 80, "Protocol": 6, "Timestamp": "x", "ts": pd.to_datetime(t, unit="s"),
        "Flow Duration": np.full(n, int(dur_s * 1e6)), "Flow Bytes/s": 1.0, "Flow Packets/s": 1.0,
        "Fwd Header Length": 20, "Bwd Header Length": 20, "FWD Init Win Bytes": 1, "Bwd Init Win Bytes": 1,
        "ICMP Code": -1, "ICMP Type": -1, "Label": label, "Attempted Category": -1})
    if shuffle_seed is not None:  # source files are not time-ordered
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        df["source_row"] = np.arange(n)
    return add_targets(df)


def test_partition_sizes_keep_every_partition_nonempty():
    for n in range(3, 200):
        tr, va, te = partition_sizes(n)
        assert min(tr, va, te) >= 1 and tr + va + te == n
    assert partition_sizes(11) == (6, 2, 3)  # Heartbleed-sized stratum
    with pytest.raises(ValueError):
        partition_sizes(2)


def test_chrono_split_orders_by_time_not_file_order():
    df = frame(50, shuffle_seed=1)
    s = chrono_split(df).merge(df[["source_file", "source_row", "t_start", "t_end"]])
    tr, va, te = (s[s.partition_before_purge == p] for p in ("train", "val", "test"))
    assert tr.t_start.max() < va.t_start.min() and va.t_start.max() < te.t_start.min()


def test_split_is_deterministic_and_row_ids_preserved():
    df = frame(40, shuffle_seed=3)
    a, b = chrono_split(df), chrono_split(df.sample(frac=1, random_state=9))
    key = ["source_file", "source_row"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True))
    assert sorted(map(tuple, a[key].to_numpy())) == sorted(map(tuple, df[key].to_numpy()))
    r1, r2 = random_split(df), random_split(df.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(r1.sort_values(key).reset_index(drop=True), r2.sort_values(key).reset_index(drop=True))


def test_purge_removes_only_overlapping_earlier_rows_and_never_test_rows():
    df = frame(20, step=10.0, dur_s=1.0)
    df.loc[df.source_row == 9, "t_end"] = df.loc[df.source_row == 9, "t_start"] + 500  # long train flow
    s = chrono_split(df).merge(df[["source_file", "source_row", "t_start", "t_end"]])
    assert (s.partition_before_purge == "test").sum() == (s.partition == "test").sum()
    purged = s[s.partition == "purged"]
    assert list(purged.source_row) == [9] and purged.purge_reason.iloc[0] == "train_reaches_val"
    t_val0 = s.loc[s.partition_before_purge == "val", "t_start"].min()
    t_test0 = s.loc[s.partition_before_purge == "test", "t_start"].min()
    assert (s.loc[s.partition == "train", "t_end"] < t_val0).all()
    assert (s.loc[s.partition == "val", "t_end"] < t_test0).all()


def test_purge_handles_identical_timestamps_at_boundary():
    df = frame(10, step=0.0, dur_s=0.0)  # all flows start at the same instant
    s = chrono_split(df)
    assert set(s.partition) == {"purged", "test"}  # no earlier row can precede the tied test rows


def test_strata_split_independently():
    df = pd.concat([frame(30, label="DoS Hulk"), frame(10, label="BENIGN", start=1000.0)], ignore_index=True)
    df["source_row"] = np.arange(len(df))
    s = chrono_split(df)
    for st, g in s.groupby("stratum"):
        _, _, te = partition_sizes(len(g))
        assert (g.partition_before_purge == "test").sum() == te


def test_seen_in_is_exact_and_feature_set_specific():
    df = frame(6)
    df["Flow Duration"] = [1, 2, 3, 1, 2, 99]
    df["Fwd Header Length"] = [5, 5, 5, 6, 5, 5]
    ref = np.array([True, True, True, False, False, False])
    q = ~ref
    a = seen_in(df, ["Flow Duration", "Fwd Header Length"], ref, q)
    b = seen_in(df, ["Flow Duration"], ref, q)
    assert a.tolist() == [False, False, False, False, True, False]
    assert b.tolist() == [False, False, False, True, True, False]  # dropping a column changes novelty


def test_seen_in_canonicalises_negative_zero_and_nan():
    df = pd.DataFrame({"x": [0.0, -0.0, np.nan, np.nan], "y": [1, 1, 2, 2]})
    ref = np.array([True, False, True, False])
    assert seen_in(df, ["x", "y"], ref, ~ref).tolist() == [False, True, False, True]


def test_seen_in_column_order_matters_only_through_values():
    df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
    ref = np.array([True, False])
    assert seen_in(df, ["x", "y"], ref, ~ref).tolist() == [False, False]  # (1,2) != (2,1)


def test_manifest_excludes_targets_metadata_and_identifiers():
    m = feature_manifest(RAW_COLS)
    bad = {"Label", "Attempted Category", "id", "Timestamp", "Src IP", "Dst IP", "Src Port", "Flow ID",
           "y_binary", "family", "label_original", "t_start", "source_row"}
    for name, feats in m["feature_sets"].items():
        assert not bad & set(feats), name
        check_features(feats, m)
    assert "Dst Port" not in m["feature_sets"]["default"]
    assert m["feature_sets"]["default_plus_dst_port"] == m["feature_sets"]["default"] + ["Dst Port"]
    assert not {"FWD Init Win Bytes", "Bwd Init Win Bytes"} & set(m["feature_sets"]["no_tcp_window"])
    assert not {"Fwd Header Length", "Bwd Header Length"} & set(m["feature_sets"]["no_header_length"])
    with pytest.raises(ValueError):
        check_features(["Flow Duration", "y_binary"], m)
    assert bad <= forbidden_inputs(m) | {"Flow ID"}


def test_label_mapping():
    df = frame(4)
    df["Label"] = ["BENIGN", "DoS Hulk - Attempted", "Infiltration - Portscan", "Portscan"]
    t = add_targets(df)
    assert t.y_binary.tolist() == [0, 0, 1, 1]
    assert t.family.tolist() == ["BENIGN", "BENIGN", "Portscan", "Portscan"]
    assert t.label_original.tolist() == df.Label.tolist() and t.is_attempted.tolist() == [False, True, False, False]


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_preprocessing_fitted_on_training_rows_only(kind):
    train = pd.DataFrame({"Protocol": [6, 6, 17, 17], "a": [1.0, 2.0, 3.0, 4.0], "const": [5.0] * 4})
    test = pd.DataFrame({"Protocol": [1, 6], "a": [100.0, 200.0], "const": [0.0, 9.0]})
    p = preprocessor(["Protocol", "a", "const"], kind).fit(train)
    out = p.transform(test)
    assert "const" not in out.columns  # constant in train -> dropped even though it varies in test
    if kind == "linear":
        sc = p.named_transformers_["num"].named_steps["scale"]
        assert sc.mean_.tolist() == [2.5]  # train mean only
        assert (out.filter(like="Protocol").sum(axis=1) == [0, 1]).all()  # unseen category -> zeros
