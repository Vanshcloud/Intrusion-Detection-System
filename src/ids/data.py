"""Dataset paths, column roles and small shared helpers for the improved CIC-IDS2017 data."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
ARCHIVE = RAW / "CICIDS2017_improved.zip"
EXTRACTED = RAW / "CICIDS2017_improved"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
PROVENANCE = DATA / "DATASET_PROVENANCE.json"
GENERATED = ROOT / "reports" / "generated"

SOURCE_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
EXPECTED_FILES = [f"{d}.csv" for d in DAYS]

# Column roles. Decisions and reasons are documented in reports/LEAKAGE_AUDIT.md.
TARGET = ["Label", "Attempted Category"]
METADATA = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
REQUIRES_EXPERIMENT = ["Dst Port"]
# Columns added by our Parquet conversion (provenance, never features).
PROVENANCE_COLS = ["source_file", "source_row"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def csv_paths():
    return [EXTRACTED / f for f in EXPECTED_FILES]


def model_features(columns):
    """Candidate model features: everything that is not target, metadata, provenance or under experiment."""
    excluded = set(TARGET) | set(METADATA) | set(REQUIRES_EXPERIMENT) | set(PROVENANCE_COLS)
    return [c for c in columns if c not in excluded]
