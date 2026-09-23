import numpy as np
import streamlit as st

from dashboard import data as D
from dashboard import ui

h, fs = ui.cached("headline"), ui.cached("fp_story")
dg, prof = fs["by_daygroup"], fs["profile"]
rev, oth, vrev, voth = fs["rev"], fs["other"], fs["val_rev"], fs["val_other"]
ui.header("False-positive case study", "Why the validation false-positive rate did not hold on the test partition.")
st.caption("Terminology: *false positives under the benchmark labels* — benign-labelled flows the frozen model flags. "
           "Nothing here was relabelled.")

c = st.columns(4)
c[0].metric("Validation FPR", D.pct(fs["val_fpr"], 3), help=f"{D.num(fs['val_fp'])} validation false positives")
c[1].metric("Test FPR", D.pct(fs["test_fpr"], 3), help="same frozen threshold")
c[2].metric("Test false positives", D.num(fs["fp"]))
c[3].metric("…of which reverse-direction", D.pct(rev["fp"] / fs["fp"], 1),
            help=f"{D.num(rev['fp'])} flows whose (destination, source) is a successful-attack (source, destination) pair "
                 "on the same day")

st.subheader("Where the false positives are")
t = dg.reset_index()
ui.bar(t.assign(label=t.daygroup + " (n = " + t.benign_test.map(D.num) + " benign)"), "fpr", "label",
       "Test false-positive rate by day group (benign flows)", "test FPR", sort=None)
st.dataframe(t.assign(fpr=t.apply(lambda r: f"{D.pct(r.fpr, 3)} ({D.pct(r.ci95_low, 3)} – {D.pct(r.ci95_high, 3)})", axis=1),
                      fp_share_reverse=t.fp_share_reverse.map(lambda v: D.pct(v, 1)),
                      fp_share_attempted=t.fp_share_attempted.map(lambda v: D.pct(v, 1)),
                      fp_score_median=t.fp_score_median.map(lambda v: D.num(v, 3)))
             [["daygroup", "benign_test", "fp", "fpr", "fp_share_reverse", "fp_share_attempted", "fp_score_median", "fp_top_pairs"]]
             .rename(columns={"benign_test": "benign test flows", "fpr": "FPR (95 % Wilson interval)",
                              "fp_share_reverse": "FPs reverse-direction", "fp_share_attempted": "FPs '- Attempted'",
                              "fp_score_median": "median FP score", "fp_top_pairs": "top host pairs (FP count)"}),
             hide_index=True, width="stretch")

st.subheader("The decomposition: reverse-direction traffic vs everything else")
dec = ui.cached("read_csv", "m5_shift_decomposition")


def rd(part, daygroup):
    return dec[(dec.partition == part) & (dec.daygroup == daygroup) & dec.reverse_of_attack].iloc[0]

d = dec[dec.daygroup != "all"].assign(
    group=lambda x: x.partition.map({"val": "validation", "test": "test"}))
a, b = st.columns(2)
with a:
    ui.grouped_bar(d[d.reverse_of_attack], "daygroup", "fpr", "group", "Reverse-direction benign flows", "FPR", fmt=".1%",
                   order=list(dg.index))
with b:
    ui.grouped_bar(d[~d.reverse_of_attack], "daygroup", "fpr", "group", "All other benign flows", "FPR", fmt=".3%",
                   order=list(dg.index))
st.markdown(
    f"- Test, **reverse-direction** benign flows: FPR **{D.pct(rev['fpr'], 1)}** ({D.num(rev['fp'])} of {D.num(rev['benign'])}).\n"
    f"- Test, **all other** benign flows: FPR **{D.pct(oth['fpr'], 3)}** ({D.num(oth['fp'])} of {D.num(oth['benign'])}); "
    f"validation: {D.pct(voth['fpr'], 3)}. For this traffic the {D.pct(h['budget'], 1)} target approximately holds.\n"
    f"- Validation contained only {D.num(vrev['benign'])} reverse-direction benign flows; the training rows of Thursday and "
    f"Friday almost none (share {fs['train_reverse_share']['thursday']:.1e} and {fs['train_reverse_share']['friday']:.1e}).")
st.caption("Bars show the validation and test FPR per day group; 95 % intervals are in `m5_shift_decomposition.csv`. "
           f"Validation Friday has {D.num(rd('val', 'friday').benign)} reverse-direction flow(s), so its bar carries almost "
           "no information.")

st.subheader("Score distribution")
hist = ui.cached("score_hist", "test benign by day group and direction")
ui.histogram(hist, "Test benign flows: frozen score by direction",
             ["thursday|reverse", "friday|reverse", "thursday|other", "friday|other"],
             float(np.log(h["threshold"] / (1 - h["threshold"]))))
st.markdown(f"Median test false-positive score: Thursday **{D.num(dg.loc['thursday', 'fp_score_median'], 3)}**, Friday "
            f"**{D.num(dg.loc['friday', 'fp_score_median'], 3)}** against a threshold of {h['threshold']:.4f} — these are "
            f"high-confidence predictions. Thursday's {D.num(prof.loc['thursday', 'fp'])} false positives take only "
            f"{D.num(prof.loc['thursday', 'distinct_scores'])} distinct scores.")

st.subheader("What drives them (SHAP)")
st.image(str(D.figure("fp_features")), caption="Mean |SHAP| of the features carrying false positives, by day group "
         "(Milestone 5 figure).")
st.markdown("Closest attack-family SHAP profile (cosine similarity of mean SHAP vectors): "
            + "; ".join(f"**{k}** — {v}" for k, v in prof.closest_family_profiles.items()))

st.subheader("Host pairs and scenario timing")
hp = ui.cached("read_csv", "m5_fp_host_pairs")
st.dataframe(hp.drop(columns=["top_dst_ports"]).assign(fpr_within_pair=hp.fpr_within_pair.map(lambda v: D.pct(v, 1)),
                                                        reverse_attack_labels=hp.reverse_attack_labels.fillna("")),
             hide_index=True, width="stretch")
st.caption(f"The {len(hp)} largest (day, source, destination) pairs hold {D.pct(fs['fp_top15_pairs_share'], 1)} of the "
           f"{D.num(fs['fp'])} false positives, which fall in {D.num(fs['fp_n_pairs'])} pairs in total. IPs are testbed "
           "addresses from the benchmark.")
tw = ui.cached("read_csv", "m5_shift_time_windows")
st.dataframe(tw.assign(benign_reverse_rows=tw.benign_reverse_rows.map(lambda v: "" if np.isnan(v) else D.num(v))),
             hide_index=True, width="stretch")
st.caption("Time windows (dataset timestamps) of the BENIGN stratum's chronological partitions and of the same-day attack "
           "scenarios; the validation window covers only the start of the afternoon scans.")

k, u = st.columns(2)
with k:
    with st.container(border=True):
        st.subheader("What we know")
        st.markdown(
            f"- {D.pct(rev['fp'] / fs['fp'], 1)} of test false positives are reverse-direction traffic associated with "
            "attack scenarios (scanned hosts answering the scanner on Thursday; the Friday victim answering the external "
            "scanner/DDoS source).\n"
            "- They are high-confidence predictions, far above the threshold, and collapse onto few distinct scores.\n"
            "- Their SHAP profile resembles scan traffic: reset flags and microsecond-scale durations push them toward attack.\n"
            "- Excluding them, the test FPR is close to the validation FPR.\n"
            f"- A model trained on the random split flags {D.pct(fs['E2_fpr_reverse'], 1)} of such flows in its own test set "
            "(observational comparison of two frozen models).")
with u:
    with st.container(border=True):
        st.subheader("What remains unresolved")
        st.markdown(
            "- Whether the benchmark labels should classify these flows differently — a possible scenario-specific "
            "labelling asymmetry, not established.\n"
            "- Whether the pattern reflects flow-meter behaviour, host replies, scenario design or another artefact.\n"
            f"- Why {D.pct(rd('test', 'friday').fpr, 1)} of Friday reverse-direction flows are flagged but "
            f"{D.pct(rd('test', 'thursday').fpr, 1)} of Thursday ones.\n"
            "- Whether the same pattern holds for other fits of the model (not testable without retraining).")

ui.limitations()
ui.source("reports/generated/m5_shift_decomposition.csv", "reports/generated/m5_fp_daygroup_compare.csv",
          "reports/generated/m5_fp_profile.csv", "reports/generated/m5_fp_host_pairs.csv",
          "reports/generated/m6_score_hist.csv", "reports/EXPLAINABILITY_ANALYSIS.md §8–§9, §17")
