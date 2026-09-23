import streamlit as st

from dashboard import data as D
from dashboard import ui

h, s, mi = ui.cached("headline"), ui.cached("shap_global"), ui.cached("model_inputs")
assert mi["has_protocol"]
imp = s["importance"]
ui.header("Explainability (SHAP)", f"What drives the frozen {h['family']} model's scores — computed once in Milestone 5.")
st.info("**SHAP explains model behaviour. It does not prove that a feature causes malicious network activity.** "
        "A large SHAP value means the feature contributes strongly to this model's predictions.", icon=":material/info:")

with st.expander("Method and output space", expanded=True):
    st.markdown(
        f"- **Method:** path-dependent TreeSHAP on the frozen model, computed for every one of the "
        f"{D.num(s['n_test'])} test rows (a census, not a sample); nothing is recomputed in the dashboard.\n"
        f"- **Output space: log-odds of attack.** Each flow's score is sigmoid(base value + sum of its SHAP values). "
        f"Base value {s['base_log_odds']:.4f} log-odds (probability {s['base_probability']:.4f}); the stored scores are "
        f"reproduced with a maximum error of {s['additivity_error']:.1e}.\n"
        f"- **Inputs:** {mi['n']} — `Protocol` plus {mi['n'] - 1} flow statistics (the {mi['n_default']}-feature default "
        f"set minus {', '.join(mi['dropped'])}, constant in training); no IPs, ports, timestamps or identifiers.\n"
        "- Positive SHAP pushes a flow toward *attack*, negative toward *benign*.")

st.subheader("Global feature importance")
top = imp.head(15)
ui.bar(top.assign(feature=top["rank"].astype(str) + ". " + top.feature), "share_all", "feature",
       f"Share of total mean |SHAP| — top {len(top)} of {len(imp)} features (full test census)",
       "share of total mean |SHAP|", fmt=".1%", sort=None)
st.dataframe(top[["rank", "feature", "group", "mean_abs_all", "share_all", "mean_abs_benign", "mean_abs_attack", "direction"]]
             .assign(share_all=top.share_all.map(lambda v: D.pct(v, 1)),
                     **{c: top[c].map(lambda v: D.num(v, 3)) for c in ("mean_abs_all", "mean_abs_benign", "mean_abs_attack")})
             .rename(columns={"mean_abs_all": "mean |SHAP| (log-odds)", "share_all": "share",
                              "mean_abs_benign": "mean |SHAP| benign rows", "mean_abs_attack": "mean |SHAP| attack rows"}),
             hide_index=True, width="stretch")
g = s["groups"]
st.markdown("**By feature group:** " + " · ".join(f"{k.replace('_', ' ')} {D.pct(v, 1)}" for k, v in g.items()))
nonmono = (imp.direction.str.startswith("non-monotone")).mean()
st.caption("`direction` = Spearman correlation between a feature's value and its SHAP value on a uniform test sample "
           f"(threshold as defined in the explainability report); {D.pct(nonmono, 0)} of all features are non-monotone / mixed.")

st.subheader("Beeswarm")
st.image(str(D.figure("beeswarm")), width=720, caption=f"SHAP values on a uniform random sample of {D.num(s['n_uniform'])} "
         "test rows; colour = feature value rank (Milestone 5 figure).")
st.subheader("Per-family feature groups")
st.image(str(D.figure("family_groups")), caption="Share of |SHAP| by feature group for attack labels with enough test rows "
         "(Milestone 5 figure).")
st.subheader("Dependence plots")
st.image(str(D.figure("dependence")), caption="Feature value (symlog axis) vs its SHAP value, uniform test sample "
         "(Milestone 5 figure).")
st.caption(f"About {D.pct(s['interaction_share'], 0)} of the attribution arises from feature interactions (strongest pairs "
           f"involve `{s['interaction_hub']}`), so single-feature plots are incomplete descriptions.")

st.subheader("How far can these explanations be trusted?")
rf = s["rf"].loc["all_sample"]
st.markdown(
    f"- **Resampling:** over bootstrap resamples of the test rows the minimum top-10 overlap with the full ranking was "
    f"{s['boot_top10_min']:.1f}; `{imp.feature.iloc[0]}` was "
    f"{'ranked first in every resample' if s['boot_top1_always'] else 'not always ranked first'}.\n"
    f"- **Model-specific:** the frozen Random Forest ranks features differently (top-10 overlap {rf.top10_overlap:.1f}, "
    f"rank correlation {rf.spearman:.2f}). Which feature *gets the credit* depends on the model family, because many "
    "flow features are strongly correlated.\n"
    f"- **Random-split model:** top-10 overlap {s['e1e2'].loc['all_test', 'top10_overlap']:.1f}; mean total |SHAP| per "
    f"row {s['e1e2'].loc['all_test', 'E2_mean_sum_abs']:.1f} vs {s['e1e2'].loc['all_test', 'E1_mean_sum_abs']:.1f} "
    "for the chronological model.\n"
    "- Refit variability (other seeds) was not measured.")
st.image(str(D.figure("model_comparison")), caption="Normalised importance: chronological vs random-split LightGBM (left), "
         "selected LightGBM vs Random Forest (right) (Milestone 5 figure).")

ui.limitations()
ui.source("reports/generated/m5_global_importance.csv", "reports/generated/m5_facts.json",
          "reports/generated/m5_lgbm_vs_rf_summary.csv", "reports/figures/m5/", "reports/EXPLAINABILITY_ANALYSIS.md")
