import altair as alt
import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard import ui

h = ui.cached("headline")
cases = D.demo_cases()  # local only (not cached: it appears once the user generates it)
ui.header("Benchmark case explorer",
          "Pick a stored test flow from the benchmark and see the frozen model's score and its SHAP explanation.")
OUTCOME = {"TP": "true positive (attack detected)", "FP": "false positive (benign flagged)",
           "TN": "true negative (benign passed)", "FN": "false negative (attack missed)"}


def split_contrib(text):
    return pd.DataFrame({"feature = value (SHAP, log-odds)": str(text).split("; ")})


if cases is None:
    st.info("**Exact CIC-IDS2017 rows are not included in this public repository**: the dataset has no verified "
            "redistribution licence. This view shows the Milestone 5 casebook instead: the stored frozen score and the "
            "largest recorded SHAP contributions of representative test flows. Exploring exact benchmark rows, and the "
            "live check with the model export, becomes available after you obtain the dataset yourself and build the "
            "local artifacts (`data/README.md`, then `scripts/m6_dashboard_data.py`).", icon=":material/lock:")
    cb = ui.cached("casebook")
    i = st.selectbox("Casebook entry", cb.index, format_func=lambda k: cb.case[k])
    r = cb.loc[i]
    c = st.columns(4)
    c[0].metric("Frozen score (probability of attack)", f"{r.score:.4f}")
    c[1].metric("Frozen threshold", f"{h['threshold']:.4f}")
    c[2].metric("Predicted class", "attack" if r.score >= h["threshold"] else "benign")
    c[3].metric("Benchmark label", "attack" if r.y_binary else "benign", help=f"original label: {r.label_original}")
    st.dataframe(pd.DataFrame({"day": [r.day], "original label": [r.label_original],
                               "feature vector": ["novel" if r.novel else "exact match in training"],
                               "reverse of a same-day attack pair": ["yes" if r.reverse_of_attack_pair else "no"]}),
                 hide_index=True, width="stretch")
    st.subheader("Largest stored SHAP contributions (Milestone 5)")
    a, b = st.columns(2)
    a.markdown("**Toward attack**")
    a.dataframe(split_contrib(r.lgbm_top_toward_attack), hide_index=True, width="stretch")
    b.markdown("**Toward benign**")
    b.dataframe(split_contrib(r.lgbm_top_toward_benign), hide_index=True, width="stretch")
    model, status = ui.model()
    st.caption(f"Model export: {status}. Scores and contributions above are the stored frozen values; nothing is "
               "recomputed without the local rows.")
    ui.limitations()
    ui.source("reports/generated/m5_casebook.csv", "configs/m4_frozen.json", "configs/m6_model_export.json")
    st.stop()

st.markdown(f"These are **{D.num(len(cases))} real rows of the CIC-IDS2017 test partition**, built locally from your own "
            "copy of the dataset (`artifacts/m6/demo_cases.csv`, not part of the repository): a small curated set covering "
            "every outcome, attack label, day group and novel/duplicated status, plus the Milestone 5 casebook. There is no "
            "manual input, packet capture or upload: synthetic feature values would not be network traffic.")
with st.container(border=True):
    casebook = cases[cases.case.notna()].sort_values("case", key=lambda s: s.str.split(" ").str[0].astype(int))
    mode = st.radio("Choose a case", ["Milestone 5 casebook", "Filter all cases"], horizontal=True)
    if mode == "Milestone 5 casebook":
        pick = st.selectbox("Casebook entry", casebook.index, format_func=lambda i: cases.case[i])
    else:
        f = st.columns(4)
        oc = f[0].selectbox("Outcome", ["any"] + list(OUTCOME), format_func=lambda k: OUTCOME.get(k, k))
        lab = f[1].selectbox("Original label", ["any"] + sorted(cases.label_original.unique()))
        vec = f[2].selectbox("Feature vector", ["any", "novel", "seen in training"])
        day = f[3].selectbox("Day", ["any"] + sorted(cases.day.unique()))
        sel = cases[((cases.outcome == oc) | (oc == "any")) & ((cases.label_original == lab) | (lab == "any"))
                    & ((cases.novel == (vec == "novel")) | (vec == "any")) & ((cases.day == day) | (day == "any"))]
        if sel.empty:
            st.warning("No stored case matches these filters.")
            st.stop()
        pick = st.selectbox(f"Case ({D.num(len(sel))} matching)", sel.index,
                            format_func=lambda i: f"{cases.day[i]} · {cases.label_original[i]} · {cases.outcome[i]} · "
                                                  f"score {cases.score[i]:.4f}")

row = cases.loc[D.validate_case_index(int(pick))]
pred = int(row.score >= h["threshold"])
st.subheader(f"{OUTCOME[row.outcome].capitalize()}")
c = st.columns(4)
c[0].metric("Frozen score (probability of attack)", f"{row.score:.4f}")
c[1].metric("Frozen threshold", f"{h['threshold']:.4f}")
c[2].metric("Predicted class", "attack" if pred else "benign")
c[3].metric("Benchmark label", f"{'attack' if row.y_binary else 'benign'}", help=f"original label: {row.label_original}")
st.dataframe(pd.DataFrame({
    "row ID": [f"{row.source_file}:{row.source_row}"], "day": [row.day], "original label": [row.label_original],
    "binary target": [int(row.y_binary)], "feature vector": ["novel" if row.novel else "exact match in training"],
    "reverse of a same-day attack pair": ["yes" if row.reverse else "no"],
    "flow (metadata, not a model input)": [f"{row['Src IP']}:{int(row['Src Port'])} → {row['Dst IP']}:{int(row['Dst Port'])}"]}),
    hide_index=True, width="stretch")

st.subheader("Why the model scored it this way (stored SHAP values, log-odds)")
pos, neg = D.case_contributions(row, 8)
top = pd.concat([pos, neg]).assign(direction=lambda x: x.shap.gt(0).map({True: "toward attack", False: "toward benign"}),
                                   label=lambda x: x.feature + " = " + x.value.map(lambda v: f"{v:g}"))
chart = alt.Chart(top).mark_bar().encode(
    x=alt.X("shap:Q", title="SHAP contribution (log-odds of attack)"),
    y=alt.Y("label:N", sort=list(top.sort_values("shap", ascending=False).label), title=None,
            axis=alt.Axis(labelLimit=360, labelOverlap=False)),
    color=alt.Color("direction:N", scale=alt.Scale(domain=["toward attack", "toward benign"], range=[ui.ORANGE, ui.BLUE]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=["feature", alt.Tooltip("value:Q", format=",.6g"), alt.Tooltip("shap:Q", format="+.3f"), "direction"])
text = chart.mark_text(align="left", dx=3).encode(text=alt.Text("shap:Q", format="+.2f"), color=alt.value("#52514e"))
st.altair_chart((chart + text).properties(height=28 * len(top) + 40), width="stretch")
a, b = st.columns(2)
a.markdown("**Toward attack**")
a.dataframe(pos.assign(shap=pos.shap.map("{:+.3f}".format)), hide_index=True, width="stretch")
b.markdown("**Toward benign**")
b.dataframe(neg.assign(shap=neg.shap.map("{:+.3f}".format)), hide_index=True, width="stretch")
base = D.facts()["base_log_odds"]
st.caption(f"score = sigmoid({base:.3f} + sum of all SHAP values) = {D.reconstructed_score(row):.6f}; the chart shows the "
           "largest contributions in each direction. Feature values are the benchmark's CICFlowMeter statistics.")

st.subheader("Live check with the frozen model (committed text export)")
model, status = ui.model()
if model is None:
    st.info(f"{status}. The live check is skipped; the stored frozen scores and SHAP values above are unaffected "
            "(see *Methodology & limitations* → Reproducing the artifacts).", icon=":material/cloud_off:")
else:
    live = D.live_explain(model, row)
    ok = live["score_equal_stored"] and live["base_equal_stored"] and live["shap_max_abs_diff"] < D.SHAP_TOL
    (st.success if ok else st.error)(
        f"{status}. Live score {live['score']:.6f} — identical to the stored Milestone 4 score: "
        f"{live['score_equal_stored']}. Largest difference between live and stored SHAP values: "
        f"{live['shap_max_abs_diff']:.1e}.", icon=":material/verified:" if ok else ":material/error:")

ui.limitations()
ui.source("artifacts/m6/demo_cases.csv (local; built by scripts/m6_dashboard_data.py from the stored predictions and "
          "the SHAP census)", "configs/m4_frozen.json", "configs/m6_model_export.json")
