"""Tests that need no dataset download."""
import hashlib
import subprocess

from ids.data import METADATA, REQUIRES_EXPERIMENT, ROOT, TARGET, model_features, sha256


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    payload = b"abc" * 1_000_000  # spans several read chunks
    p.write_bytes(payload)
    assert sha256(p) == hashlib.sha256(payload).hexdigest()
    assert sha256(p, chunk=7) == hashlib.sha256(payload).hexdigest()


def test_known_sha256_vector(tmp_path):
    p = tmp_path / "abc"
    p.write_bytes(b"abc")
    assert sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_non_features_never_in_model_features():
    cols = TARGET + METADATA + REQUIRES_EXPERIMENT + ["source_file", "source_row", "Flow Duration"]
    assert model_features(cols) == ["Flow Duration"]


def test_raw_data_ignored_by_git():
    for path in ["data/raw/CICIDS2017_improved.zip", "data/raw/CICIDS2017_improved/monday.csv",
                 "data/interim/x.parquet", "data/processed/x.parquet"]:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert r.returncode == 0, f"{path} is not git-ignored"
    # provenance and docs must stay tracked
    r = subprocess.run(["git", "check-ignore", "-q", "data/DATASET_PROVENANCE.json"], cwd=ROOT)
    assert r.returncode == 1
