"""Tests against the downloaded dataset. Skipped when data/raw is absent (e.g. fresh clone)."""
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ids.data import (EXPECTED_FILES, EXTRACTED, INTERIM, PROVENANCE, TARGET, csv_paths, model_features, sha256)

pytestmark = pytest.mark.skipif(not EXTRACTED.exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def prov():
    return json.loads(PROVENANCE.read_text())


@pytest.fixture(scope="module")
def headers():
    return {p.name: p.open().readline().rstrip("\n").split(",") for p in csv_paths()}


def test_expected_source_files_discovered():
    assert sorted(p.name for p in EXTRACTED.glob("*.csv")) == sorted(EXPECTED_FILES)


def test_schema_identical_across_days(headers):
    first = headers[EXPECTED_FILES[0]]
    assert len(first) == 91
    assert all(h == first for h in headers.values())


def test_target_columns_identified_and_excluded(headers):
    cols = headers[EXPECTED_FILES[0]]
    assert cols[-2:] == TARGET
    feats = model_features(cols)
    assert not set(TARGET) & set(feats)
    assert not {"Src IP", "Dst IP", "Flow ID", "Timestamp", "id", "Src Port", "Dst Port"} & set(feats)


def test_timestamps_parse_with_expected_format():
    for p in csv_paths():
        ts = pd.read_csv(p, usecols=["Timestamp"])["Timestamp"]
        parsed = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        assert parsed.notna().all(), p.name
        assert (parsed.dt.strftime("%Y-%m-%d") == parsed.iloc[0].strftime("%Y-%m-%d")).all()  # one day per file


def test_provenance_exists_and_is_complete(prov):
    assert prov["archive_sha256"] and len(prov["archive_sha256"]) == 64
    assert sorted(f["archive_member"] for f in prov["extracted_files"]) == sorted(EXPECTED_FILES)


def test_raw_files_unchanged_since_provenance(prov):
    for f in prov["extracted_files"]:
        p = EXTRACTED / f["archive_member"]
        assert p.stat().st_size == f["extracted_bytes"]
        assert sha256(p) == f["sha256"], f"{p.name} was modified"


def test_parquet_preserves_rows_and_provenance():
    for p in csv_paths():
        with p.open() as fh:
            n_csv = sum(1 for _ in fh) - 1
        pf = INTERIM / f"{p.stem}.parquet"
        assert pq.ParquetFile(pf).metadata.num_rows == n_csv
        df = pd.read_parquet(pf, columns=["source_file", "source_row", "id", "Timestamp", "Label"])
        assert (df["source_file"] == p.name).all()
        assert (df["source_row"].to_numpy() == range(n_csv)).all()
        assert df["source_row"].is_unique


def test_convert_is_lossless_on_sample(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from convert_parquet import convert

    src = csv_paths()[1]
    small = tmp_path / src.name
    small.write_text("".join(src.open().readlines()[:2001]))
    out, n = convert(small, tmp_path)
    back = pd.read_parquet(out).drop(columns=["source_file", "source_row", "ts"])
    pd.testing.assert_frame_equal(back, pd.read_csv(small, low_memory=False))
    assert n == 2000
