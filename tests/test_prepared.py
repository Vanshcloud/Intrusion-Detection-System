"""Checks on the Milestone 3 artifacts built by scripts/prepare.py. Skipped when they are absent."""
import json

import numpy as np
import pandas as pd
import pytest

from ids.data import GENERATED, INTERIM, PROCESSED, ROOT
from ids.prep import check_features, content_hash, label_mapping, partition_sizes

pytestmark = pytest.mark.skipif(not (PROCESSED / "dataset_v1.parquet").exists(), reason="run scripts/prepare.py")
KEY = ["source_file", "source_row"]


@pytest.fixture(scope="module")
def protocol():
    return json.loads((ROOT / "configs/experiment_protocol_v1.json").read_text())


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ROOT / "configs/feature_manifest_v1.json").read_text())


@pytest.fixture(scope="module")
def ds():
    return pd.read_parquet(PROCESSED / "dataset_v1.parquet")


@pytest.fixture(scope="module")
def chrono(ds):
    s = pd.read_parquet(PROCESSED / "split_chrono_v1.parquet")
    return ds[KEY + ["stratum", "t_start", "t_end"]].merge(s, on=KEY + ["stratum"], validate="1:1")


def test_artifacts_match_frozen_protocol_hashes(protocol):
    for path, rec in protocol["artifacts"].items():
        if path.endswith(".parquet"):
            assert content_hash(pd.read_parquet(ROOT / path)) == rec["content_sha256"], path


def test_only_the_recorded_non_finite_rows_were_removed(ds):
    n_interim = sum(pd.read_parquet(p, columns=["id"]).shape[0] for p in INTERIM.glob("*.parquet"))
    removed = pd.read_csv(GENERATED / "m3_removed_rows.csv")
    assert len(ds) == n_interim - len(removed) == n_interim - 5
    assert ds.merge(removed[KEY], on=KEY).empty
    assert np.isfinite(ds.select_dtypes("number").to_numpy(dtype=float)).all()


def test_row_identifiers_preserved(ds):
    assert not ds.duplicated(KEY).any()
    assert (ds["id"] == ds["source_row"] + 1).all()  # id is 1..n in each source file


def test_every_row_in_exactly_one_partition(ds, chrono):
    rnd = pd.read_parquet(PROCESSED / "split_random_v1.parquet")
    for s in (chrono, rnd):
        assert len(s) == len(ds) and not s.duplicated(KEY).any()
        assert set(s["partition"]) <= {"train", "val", "test", "purged"}


def test_chronological_order_within_every_stratum(chrono):
    b = chrono.groupby(["stratum", "partition_before_purge"])["t_start"].agg(["min", "max"]).unstack()
    assert (b[("max", "train")] <= b[("min", "val")]).all()
    assert (b[("max", "val")] <= b[("min", "test")]).all()


def test_purge_removes_interval_overlap_and_never_test_rows(chrono):
    assert ((chrono.partition_before_purge == "test") == (chrono.partition == "test")).all()
    t_val0 = chrono.stratum.map(chrono[chrono.partition_before_purge == "val"].groupby("stratum")["t_start"].min())
    t_test0 = chrono.stratum.map(chrono[chrono.partition_before_purge == "test"].groupby("stratum")["t_start"].min())
    tr, va = chrono.partition == "train", chrono.partition == "val"
    assert (chrono.t_end[tr] < t_val0[tr]).all() and (chrono.t_end[va] < t_test0[va]).all()
    p = chrono[chrono.partition == "purged"]
    expected = ((p.partition_before_purge == "train") & (p.t_end >= t_val0[p.index])) | \
               ((p.partition_before_purge == "val") & (p.t_end >= t_test0[p.index]))
    assert expected.all()  # nothing purged without meeting the rule


def test_split_sizes_and_random_split_comparable(chrono):
    rnd = pd.read_parquet(PROCESSED / "split_random_v1.parquet")
    n = chrono.groupby("stratum").size()
    for part_col, s in (("partition_before_purge", chrono), ("partition", rnd)):
        got = s.groupby(["stratum", part_col]).size().unstack().fillna(0).astype(int)
        for st, k in n.items():
            assert tuple(got.loc[st, ["train", "val", "test"]]) == partition_sizes(k), st


def test_feature_sets_contain_only_allowed_numeric_columns(manifest, ds):
    for name, fs in manifest["feature_sets"].items():
        check_features(fs, manifest)
        assert all(np.issubdtype(ds[c].dtype, np.number) for c in fs), name
    assert {"Src IP", "Dst IP", "Flow ID", "Timestamp", "Label", "Attempted Category", "id"}.isdisjoint(
        set().union(*manifest["feature_sets"].values()))


def test_label_mapping_file_matches_data(ds):
    cfg = json.loads((ROOT / "configs/label_mapping_v1.json").read_text())["labels"]
    assert cfg == label_mapping(ds)
    m = {r["label_original"]: r for r in cfg}
    assert m["Infiltration - Portscan"]["family"] == m["Portscan"]["family"] == "Portscan"
    assert all(r["y_binary"] == 0 and r["family"] == "BENIGN" for r in cfg if r["is_attempted"])
    assert (ds["label_original"] == ds["Label"]).all()


def test_class_accounting_covers_every_row(ds):
    acc = pd.read_csv(GENERATED / "m3_partition_counts.csv")
    assert (acc.groupby("protocol")["n"].sum() == len(ds)).all()


def test_novel_flags_verified_by_exact_join(ds, chrono, manifest):
    nov = pd.read_parquet(PROCESSED / "novel_test_v1.parquet")
    assert (nov[KEY].to_numpy() == ds[KEY].to_numpy()).all()
    fs = manifest["feature_sets"]["default"]
    col = nov["test_novel__chrono__primary__default"].to_numpy()
    part = chrono.set_index(KEY).loc[list(map(tuple, ds[KEY].to_numpy())), "partition"].to_numpy()
    test_idx = np.flatnonzero(part == "test")
    rng = np.random.default_rng(0)
    sample = np.concatenate([rng.choice(test_idx[col[test_idx]], 300, replace=False),
                             rng.choice(test_idx[~col[test_idx]], 300, replace=False)])
    train = ds.loc[part == "train", fs].drop_duplicates()
    hit = ds.iloc[sample][fs].reset_index().merge(train, on=fs, how="left", indicator=True)
    found = hit.drop_duplicates("index").set_index("index")["_merge"].eq("both")
    assert (found.loc[ds.index[sample]].to_numpy() == ~col[sample]).all()
    assert not col[part != "test"].any()


def test_novel_subsets_are_feature_set_specific():
    s = pd.read_csv(GENERATED / "m3_novel_summary.csv").set_index(["protocol", "variant", "feature_set"])
    a, b = s.loc[("chrono", "primary", "default")], s.loc[("chrono", "primary", "default_plus_dst_port")]
    assert a.full_test == b.full_test and a.novel_test != b.novel_test
