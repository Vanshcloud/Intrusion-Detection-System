import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.header("Evaluation", "Frozen test results of the three model families (single test evaluation, no re-tuning).")

budget = st.radio("False-positive-rate budget used to choose each model's threshold on validation",
                  D.budgets(), format_func=lambda b: f"{D.pct(float(b), 1)} FPR budget", horizontal=True)
cmp_ = ui.cached("model_comparison", budget)
h = ui.cached("headline")

st.subheader("Model comparison")
show = cmp_.assign(
    model=cmp_.model + cmp_.selected.map({True: " (selected)", False: ""}),
    **{c: cmp_[c].map(lambda v: D.pct(v, 3)) for c in ("precision", "recall", "f1", "fpr", "val_fpr")},
    **{c: cmp_[c].map(lambda v: D.num(v, 5)) for c in ("pr_auc", "roc_auc")},
    **{c: cmp_[c].map(D.num) for c in ("fp", "fn")}).drop(columns="selected")
st.dataframe(show.rename(columns={"fpr": "test FPR", "val_fpr": "validation FPR", "pr_auc": "PR-AUC",
                                  "roc_auc": "ROC-AUC", "fp": "false positives", "fn": "false negatives"}),
             hide_index=True, width="stretch")
st.caption("Precision, recall, F1 and FPR are at each model's frozen threshold; PR-AUC and ROC-AUC are threshold-free. "
           "Undefined values would be shown as 'not available'. Models were selected on validation only.")

st.subheader("Chronological vs random split")
chrono = cmp_[cmp_.protocol.str.startswith("chronological")].set_index("model")
rand = cmp_[cmp_.protocol.str.startswith("random")].set_index("model")
ui.grouped_bar(cmp_.assign(protocol=cmp_.protocol.str.split(" ").str[0]), "model", "fpr", "protocol",
               "Test false-positive rate by evaluation protocol (same features, hyper-parameters and threshold rule)",
               "test FPR")
if (rand.fpr < chrono.fpr).all():
    st.markdown("**Random splitting substantially reduced measured false-positive rates in this dataset**, "
                "illustrating how evaluation design can materially affect reported IDS performance. "
                + " ".join(f"{m}: {D.pct(chrono.fpr[m], 3)} chronological vs {D.pct(rand.fpr[m], 3)} random."
                           for m in chrono.index))
else:
    st.markdown("At this budget the random split does not give a lower FPR for every model family; see the table.")
st.caption("The random split places flows from the same attack windows in both training and test; this is the "
           "optimistic comparison protocol, not deployment performance. The finding is specific to this benchmark.")

st.subheader(f"Confusion matrix — selected model ({h['family']}), test partition")
cm = ui.cached("confusion", None, "full_test", budget)
a, b = st.columns([2, 3])
a.dataframe(cm.map(D.num), width="stretch")
r = cmp_[cmp_.selected].iloc[0]
b.markdown(f"- **{D.num(cm.iloc[1, 1])}** attack flows detected, **{D.num(cm.iloc[1, 0])}** missed\n"
           f"- **{D.num(cm.iloc[0, 1])}** benign flows flagged (false positives), **{D.num(cm.iloc[0, 0])}** correctly passed\n"
           f"- Test FPR **{D.pct(r.fpr, 3)}** vs validation FPR **{D.pct(r.val_fpr, 3)}** at the same threshold")

ui.limitations()
ui.source("reports/generated/m4_test_summary.csv", "configs/m4_frozen.json")
