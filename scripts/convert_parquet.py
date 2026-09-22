"""Convert each raw CSV to data/interim/<day>.parquet without dropping or altering any value.

Adds provenance columns: source_file, source_row (0-based data-row index in the CSV) and a parsed
`ts` (timezone-naive; timezone is not encoded in the source). The raw `Timestamp` string is kept.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import INTERIM, csv_paths  # noqa: E402

TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def convert(csv_path, out_dir=INTERIM):
    df = pd.read_csv(csv_path, low_memory=False)
    df.insert(0, "source_row", range(len(df)))
    df.insert(0, "source_file", csv_path.name)
    df["ts"] = pd.to_datetime(df["Timestamp"], format=TS_FORMAT)  # raises on any unparseable value
    out = out_dir / f"{csv_path.stem}.parquet"
    df.to_parquet(out, index=False)
    return out, len(df)


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    for p in csv_paths():
        out, n = convert(p)
        print(f"{p.name} -> {out.name}: {n} rows", flush=True)


if __name__ == "__main__":
    main()
