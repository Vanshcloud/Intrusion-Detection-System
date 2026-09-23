import streamlit as st

from dashboard import data as D
from dashboard import ui

h, o = ui.cached("headline"), ui.cached("dataset_overview")
ui.header("Explainable intrusion detection on CIC-IDS2017",
          "Research demonstration of a leakage-aware, explainable binary flow classifier — a frozen benchmark study.")

st.markdown("**Research objective.** Measure how reliably a supervised model separates benign from malicious flows in "
            "CIC-IDS2017 when the evaluation is designed to reduce temporal, duplicate and identifier leakage — and "
            "explain what the model relies on.")

c = st.columns(4)
c[0].metric("Cleaned flows", D.num(o["rows_clean"]), help=f"{D.num(o['rows_raw'])} raw rows; "
            f"{D.num(o['rows_removed'])} removed (non-finite rates)")
c[1].metric("Train rows", D.num(o["n_train"]))
c[2].metric("Validation rows", D.num(o["n_val"]))
c[3].metric("Test rows", D.num(o["n_test"]), help=f"{D.num(o['n_purged'])} rows purged at split boundaries "
            "(train/validation only; test rows are never removed)")

left, right = st.columns([3, 2])
with left:
    st.subheader("Frozen experiment")
    st.markdown(
        f"- **Dataset:** {o['dataset']}\n"
        f"- **Task:** binary — successful attack (1) vs benign, including `- Attempted` flows (0)\n"
        f"- **Selected model:** {h['family']} `{h['run_id']}` ({h['best_iteration']} trees)\n"
        f"- **Operating point:** frozen threshold {h['threshold']:.6f}, chosen on validation for a "
        f"{D.pct(h['budget'], 1)} false-positive-rate budget\n"
        "- **Evaluation protocol:** chronological split within each (day, original label) stratum, boundary purging, "
        "test partition evaluated once")
    st.warning("**The primary split is retrospective and label-aware.** Split points are placed inside each "
               "(day, label) stratum, which needs the labels. It controls temporal and duplicate leakage within the "
               "benchmark; it is not a deployment-style time split.", icon=":material/warning:")
with right:
    st.subheader("Selected model on the test partition")
    a, b = st.columns(2)
    a.metric("Recall (attacks detected)", D.pct(h["recall"], 3), help=f"{D.num(h['tp'])} of {D.num(h['attack'])} attack flows")
    b.metric("False-positive rate", D.pct(h["fpr"], 3),
             help=f"{D.num(h['fp'])} of {D.num(h['benign'])} benign flows flagged; validation FPR {D.pct(h['val_fpr'], 3)}")
    a.metric("False positives", D.num(h["fp"]))
    b.metric("False negatives", D.num(h["fn"]))
    st.caption(f"Validation FPR at the same frozen threshold: {D.pct(h['val_fpr'], 3)}; the test rate is "
               f"{h['fpr'] / h['val_fpr']:.1f}× that value — see *False-positive case study*.")

st.subheader("Pages")
st.markdown(
    "- **Evaluation** — model comparison, chronological vs random split, confusion matrix\n"
    "- **Attack families** — binary-detector recall per original attack label, with support\n"
    "- **Explainability (SHAP)** — what drives the frozen model's scores\n"
    "- **False-positive case study** — why false alarms concentrate on Thursday and Friday\n"
    "- **Novel-vector analysis** — results on test rows whose feature vector never occurs in training\n"
    "- **Benchmark case explorer** — individual stored test rows with their frozen score and SHAP explanation\n"
    "- **Methodology & limitations** — how the numbers were produced and what they do not show")

ui.limitations()
ui.source("configs/m4_frozen.json", "reports/generated/m4_test_summary.csv", "reports/generated/m3_cleaning.json",
          "reports/generated/m3_partition_counts.csv")
