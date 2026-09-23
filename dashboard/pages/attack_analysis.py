import streamlit as st

from dashboard import data as D
from dashboard import ui

h = ui.cached("headline")
ui.header("Attack-family analysis", f"Recall of the selected {h['family']} binary detector, per original attack label.")
st.info("**This is attack-family recall for a binary attack detector, not multiclass classification accuracy.** "
        "Each row answers: of the test flows carrying this label, how many did the binary model flag as attack?",
        icon=":material/info:")

c1, c2 = st.columns(2)
budget = c1.radio("FPR budget", D.budgets(), format_func=lambda b: f"{D.pct(float(b), 1)} budget", horizontal=True)
subset = c2.radio("Test rows", ["full_test", "novel_test"], horizontal=True,
                  format_func={"full_test": "full test set", "novel_test": "novel-vector subset"}.get)
fr = ui.cached("family_recall", None, subset, budget)
big, small = fr[~fr.small_support], fr[fr.small_support]

st.subheader("Labels with enough test rows for a rate")
ui.bar(big.assign(label=big.label + " (n = " + big.n.map(D.num) + ")"), "recall", "label",
       "Recall per attack label (95 % Wilson interval in the table)", "recall (share of flows flagged)", fmt=".2%")
st.dataframe(big.assign(recall=big.recall.map(lambda v: D.pct(v, 2)),
                        ci=big.ci95_low.map(lambda v: D.pct(v, 2)) + " – " + big.ci95_high.map(lambda v: D.pct(v, 2)))
             [["label", "n", "detected", "missed", "recall", "ci", "detected_of_n"]]
             .rename(columns={"n": "test support", "ci": "95 % Wilson interval", "detected_of_n": "count"}),
             hide_index=True, width="stretch")

st.subheader("Low-sample-size labels — counts only")
st.warning(f"These labels have fewer than {D.SMALL_SUPPORT} test flows — too few for a stable rate. A perfect count "
           "on a handful of flows is not evidence of reliable detection; read them as anecdotes.", icon=":material/warning:")
if small.empty:
    st.write("No attack label in this selection has small support.")
for r in small.itertuples():
    st.markdown(f"- **{r.label}: {r.detected_of_n}** (95 % Wilson interval {D.pct(r.ci95_low, 1)} – {D.pct(r.ci95_high, 1)})")

st.subheader("Alerts on benign and `- Attempted` labels")
st.caption("All of these are benign under the protocol-v1 target (attempted attacks count as benign). The rate is the "
           "share of flows flagged — a false-positive rate per label.")
ba = ui.cached("benign_alerts", None, budget)
st.dataframe(ba.assign(rate=ba.rate.map(lambda v: D.pct(v, 3)),
                       small_support=ba.unstable.map({True: "yes — anecdotal", False: "no"}))
             .drop(columns=["ci95_low", "ci95_high", "unstable"]).rename(columns={"n": "test flows", "rate": "flagged share"}),
             hide_index=True, width="stretch")

ui.limitations()
ui.source("reports/generated/m4_test_per_label.csv")
