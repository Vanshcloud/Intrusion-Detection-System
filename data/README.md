# data/

Nothing under `raw/`, `interim/` or `processed/` is committed (see `.gitignore`). Only this README and
`DATASET_PROVENANCE.json` are tracked.

| Folder | Contents | Written by |
|---|---|---|
| `raw/CICIDS2017_improved.zip` | Archive exactly as downloaded | `curl` (below) |
| `raw/CICIDS2017_improved/*.csv` | Extracted per-day CSVs, set read-only (`0444`) | `scripts/provenance.py` |
| `interim/<day>.parquet` | Lossless typed copy of each CSV + `source_file`, `source_row`, parsed `ts` | `scripts/convert_parquet.py` |
| `processed/*_v1.parquet` | Cleaned dataset + targets, split manifests (chrono, random), novel-vector flags | `scripts/prepare.py` |

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
mkdir -p data/raw && date -u +%Y-%m-%dT%H:%M:%SZ > data/raw/.download_started_utc
curl -fL -o data/raw/CICIDS2017_improved.zip \
  https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
.venv/bin/python scripts/provenance.py        # CRC check, extract, SHA-256, DATASET_PROVENANCE.json
.venv/bin/python scripts/audit_raw.py         # per-file raw inspection -> reports/generated/raw_*.json
.venv/bin/python scripts/convert_parquet.py   # CSV -> data/interim/*.parquet
.venv/bin/python scripts/audit_deep.py        # timestamps, episodes, duplicates, leakage screening
.venv/bin/python scripts/prepare.py           # data/processed/*_v1, configs/*_v1.json, reports/generated/m3_*
.venv/bin/python scripts/render_reports.py    # DATA_AUDIT, LEAKAGE_AUDIT, EXPERIMENT_PROTOCOL, DATA_CARD block
.venv/bin/python -m pytest
```

Compare your `archive_sha256` with the one in `DATASET_PROVENANCE.json`. The upstream authors publish
no checksum, so that value is only our own record of the file we downloaded.
