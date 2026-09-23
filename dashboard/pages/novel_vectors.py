import altair as alt
import streamlit as st

from dashboard import data as D
from dashboard import ui

r, t = ui.cached("novel_overview")
comp = ui.cached("novel_composition")
ui.header("Novel-vector analysis", "Results on test rows whose exact feature vector never occurs in the training partition.")
st.info("**Novel vector means the exact model feature vector was not observed in training.** It is *not* unseen-attack "
        "detection, zero-day detection or external generalisation: every novel row belongs to an attack type, tool and "
        f"host that also appear in training, and {D.pct(r['attack_near_overlap_share'], 1)} of all attack test rows have a "
        "near-identical (non-timing) training neighbour.", icon=":material/info:")

c = st.columns(4)
c[0].metric("Test rows", D.num(r["full_test"]))
c[1].metric("Novel-vector rows", D.num(r["novel_test"]), help=f"{D.pct(r['novel_test'] / r['full_test'], 1)} retained")
c[2].metric("Novel attack rows", D.num(r["attack_novel"]),
            help=f"{D.pct(r['attack_novel'] / r['attack_full'], 1)} of {D.num(r['attack_full'])} attack test rows")
c[3].metric("Attack rows seen exactly", D.pct(r["attack_exact_overlap_share"], 1),
            help=f"share of attack test rows whose exact vector occurs in training; near-identical (non-timing features): "
                 f"{D.pct(r['attack_near_overlap_share'], 1)}")

st.subheader("Full test set vs novel-vector subset")
show = t.assign(subset=t.subset.map({"full_test": "full test", "novel_test": "novel subset"}),
                **{k: t[k].map(lambda v: D.pct(v, 3)) for k in ("recall", "fpr", "precision")},
                **{k: t[k].map(D.num) for k in ("n", "attack", "benign", "fp", "fn")})
st.dataframe(show.rename(columns={"fpr": "FPR", "fp": "false positives", "fn": "false negatives"}),
             hide_index=True, width="stretch")
st.caption("Differences mix two effects — removal of rows seen in training and a changed attack-family mix — and are not "
           "attributed to duplication alone.")

st.subheader("How the attack mix changes")
pscan = comp.set_index("label").loc["Portscan"]
top2 = comp.set_index("label").share_of_attacks_novel.nlargest(2)
st.markdown(f"**Portscan** makes up {D.pct(pscan.share_of_attacks_full, 1)} of attack test rows but only "
            f"{D.pct(pscan.share_of_attacks_novel, 1)} of the novel subset: {D.num(pscan.novel)} of {D.num(pscan.full)} "
            f"Portscan rows ({D.pct(pscan.retained, 1)}) have a vector not seen in training. In the novel subset the two "
            f"largest labels, {top2.index[0]} and {top2.index[1]}, make up {D.pct(top2.sum(), 1)} of attack rows, so novel-subset "
            "recall mostly reflects those families.")
long = comp.melt(id_vars=["label", "full", "novel"], value_vars=["share_of_attacks_full", "share_of_attacks_novel"],
                 var_name="subset", value_name="share")
long["subset"] = long.subset.map({"share_of_attacks_full": "full test", "share_of_attacks_novel": "novel subset"})
chart = alt.Chart(long).mark_bar().encode(
    y=alt.Y("label:N", sort=list(comp.label), title=None),
    yOffset=alt.YOffset("subset:N"),
    x=alt.X("share:Q", axis=alt.Axis(format=".0%"), title="share of attack rows in the subset"),
    color=alt.Color("subset:N", scale=alt.Scale(range=[ui.BLUE, ui.ORANGE]), legend=alt.Legend(title=None, orient="top")),
    tooltip=["label", "subset", alt.Tooltip("share:Q", format=".2%"), alt.Tooltip("full:Q", format=","),
             alt.Tooltip("novel:Q", format=",")])
st.altair_chart(chart.properties(title="Attack-family composition: full test vs novel subset", height=26 * len(comp) * 2),
                width="stretch")
st.dataframe(comp.assign(**{k: comp[k].map(lambda v: D.pct(v, 1)) for k in
                            ("exact_overlap_share", "share_of_attacks_full", "share_of_attacks_novel", "retained")})
             .rename(columns={"full": "test rows", "novel": "novel rows", "exact_overlap_share": "exact training match",
                              "share_of_attacks_full": "share (full)", "share_of_attacks_novel": "share (novel)"}),
             hide_index=True, width="stretch")

ui.limitations()
ui.source("reports/generated/m3_novel_summary.csv", "reports/generated/m3_novel_by_label.csv",
          "reports/generated/m4_test_summary.csv")
