# Research dashboard (Milestone 6)

A read-only Streamlit view of the frozen protocol-v1 CIC-IDS2017 experiment. It is a benchmark and research
demonstration: it does not capture, scan or generate network traffic.

## Launch

From the repository root, with the project environment (Python 3.12):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python -m streamlit run dashboard/app.py
```

## What it needs

- **All research pages** use only committed files: `configs/*.json`, `reports/generated/` (Milestone 3–6 tables)
  and `reports/figures/m5/`. The dataset and the `artifacts/` directory are not needed.
- **Benchmark case explorer, live check:** `dashboard/model/E1_primary__lgbm__99966b1299.txt` is a committed LightGBM
  text export of the frozen selected model. It is not a newly trained model. `configs/m6_model_export.json` records
  its SHA-256, the original model hash from `configs/m4_frozen.json` (unchanged), its input order and the equivalence
  audit (bit-identical scores on all test rows). The dashboard loads it only if the hashes match. Without it, the
  explorer shows the stored frozen scores and SHAP values and skips the live check.

- **Benchmark case explorer, exact rows:** exact CIC-IDS2017 rows are not redistributed (the improved dataset has no
  verified redistribution licence). In a clone the explorer shows the committed Milestone 5 casebook
  (`reports/generated/m5_casebook.csv`: stored frozen scores and top SHAP contributions). With a locally obtained
  dataset and the local artifacts, `.venv/bin/python scripts/m6_dashboard_data.py` writes the git-ignored
  `artifacts/m6/demo_cases.csv`, which enables row exploration and the live check.

`reports/generated/m6_score_hist.csv` (aggregate counts) and the local case rows are rebuilt with
`.venv/bin/python scripts/m6_dashboard_data.py`; the export is re-verified (byte-identical regeneration, exact
scores) with `.venv/bin/python scripts/m6_export_model.py --check`.

## Checks

`tests/test_dashboard.py` checks the displayed values against the frozen artifacts. It also fails if a data
value is typed directly into a page, and it renders every page (Streamlit `AppTest`).
