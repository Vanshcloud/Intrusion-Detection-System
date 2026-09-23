import streamlit as st

from dashboard import data as D
from dashboard import ui

m, h, o = ui.cached("methodology"), ui.cached("headline"), ui.cached("dataset_overview")
ex = ui.cached("model_export")
s = ui.cached("shap_global")
ui.header("Methodology & limitations", "How every number in this dashboard was produced — and what it does not show.")

steps = [
    ("Dataset", f"{m['dataset']}: scripted lab traffic, one CSV per recorded day. Download: {m['url']}."),
    ("Data audit", "Milestone 2 checked file hashes, label counts, duplicates, conflicting labels, timestamps and "
                   "single-feature predictiveness before any modelling (`reports/DATA_AUDIT.md`, `reports/LEAKAGE_AUDIT.md`)."),
    ("Cleaning", f"{D.num(o['rows_removed'])} rows removed ({m['cleaning']['removed']}); "
                 f"{m['cleaning']['duplicates']}. Kept and flagged: {', '.join(m['cleaning']['kept_and_flagged'])}."),
    ("Feature exclusion policy", f"Model inputs are selected by name from the frozen manifest ({m['feature_sets']['default']} "
                                 f"default features). Never used as features: {', '.join(f'`{c}`' for c in m['excluded'])}. "
                                 f"Only in a separate ablation: {', '.join(f'`{c}`' for c in m['experimental'])}."),
    ("Chronological split", f"Within each stratum {m['split']['strata']}, rows ordered by {m['split']['order']}; "
                            f"{m['split']['sizes']}. Because the strata use the labels, the split is retrospective and "
                            "label-aware."),
    ("Purging across boundaries", f"{m['split']['purge']} Purged: {D.num(m['purge']['train_reaches_val'])} training and "
                                  f"{D.num(m['purge']['val_reaches_test'])} validation rows; test rows removed: "
                                  f"{D.num(m['test_rows_removed_by_purge'])}."),
    ("Validation-only model selection", f"{m['selection']['primary']}. Tie-break: {m['selection']['tie_break']}. "
                                        f"Fit: {m['data_use']['fit']}; early stopping: {m['data_use']['early_stopping']}."),
    ("Frozen threshold selection", f"{m['operating_points']}. Budgets: {', '.join(D.pct(b, 1) for b in m['budgets'])}. "
                                   f"Selected model threshold: {h['threshold']:.6f}."),
    ("Test-once policy", f"{m['data_use']['test']}."),
    ("Novel-vector evaluation", f"Novel row = {m['novel']}. Reported as a subset, never as unseen-attack detection."),
    ("Random-split comparison", f"{m['comparison']['strata']} strata, seed {m['comparison']['seed']}; "
                                f"{m['comparison']['sizes']}. Used only to show how optimistic a random split is."),
    ("SHAP methodology", f"Path-dependent TreeSHAP on the frozen model for all {D.num(s['n_test'])} test rows, in log-odds "
                         "space, additivity verified; Random Forest and random-split model as consistency checks "
                         "(`reports/EXPLAINABILITY_ANALYSIS.md`)."),
]
for i, (title, text) in enumerate(steps, 1):
    st.markdown(f"**{i}. {title}.** {text}")

st.subheader("Data flow")
st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style=rounded, fontname="Helvetica", fontsize=11];
  raw [label="CIC-IDS2017 improved\\ndaily CSVs"]; clean [label="cleaning +\\nfrozen manifests"];
  split [label="chronological split\\n+ purge (protocol v1)"]; sel [label="selection + threshold\\n(validation only)"];
  test [label="single test\\nevaluation"]; shap [label="SHAP census +\\nerror analysis"]; dash [label="this dashboard\\n(read-only)"];
  raw -> clean -> split -> sel -> test -> shap -> dash; }""")

ui.limitations()

st.subheader("Reproducing the artifacts")
st.markdown(
    "- **Research pages** need only files committed to the repository (`configs/`, `reports/generated/`, "
    "`reports/figures/m5/`).\n"
    "- **Tables and figures** are regenerated from the local artifacts by the Milestone 3–6 scripts "
    "(`data/README.md`, `reports/MODEL_EVALUATION.md` §17, `reports/EXPLAINABILITY_ANALYSIS.md` §22, "
    "`scripts/m6_dashboard_data.py`).\n"
    f"- **The frozen model** is included as a LightGBM text export (`{ex['export_path']}`, "
    f"{D.num(ex['export_bytes'])} bytes, SHA-256 `{ex['export_sha256'][:10]}…`) of the original frozen artifact "
    f"(SHA-256 `{ex['source_model_sha256'][:10]}…` in `configs/m4_frozen.json`, which is unchanged). It is not a new "
    f"model: before export it reproduced all {D.num(ex['verification']['test_rows'])} stored test scores bit for bit "
    "(record: `configs/m6_model_export.json`). The original joblib file stays outside the repository; "
    "`scripts/m4_develop.py freeze` deliberately refuses to run while `configs/m4_frozen.json` exists, so a clone cannot "
    "re-train it without a documented protocol revision. If the export is missing or its hash differs, the case "
    "explorer shows the stored frozen scores and SHAP values and skips the live check.")
st.code(".venv/bin/python -m streamlit run dashboard/app.py", language="bash")
ui.source("configs/experiment_protocol_v1.json", "configs/model_selection_v1.json", "configs/feature_manifest_v1.json",
          "reports/generated/m3_purge.json", "reports/EXPERIMENT_PROTOCOL.md")
