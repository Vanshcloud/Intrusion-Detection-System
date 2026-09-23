"""Streamlit helpers shared by the pages: cached data access, a few chart builders, standard text blocks.

All numbers still come from dashboard.data; this module only caches and draws them.
"""
import altair as alt
import pandas as pd
import streamlit as st

from dashboard import data as D

# Reference categorical palette (validated for colour-vision deficiency; light surface). Identity is never
# colour-only: every chart also carries text labels, axis labels or tooltips.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DISCLAIMER = ("This dashboard visualises a frozen benchmark experiment on CIC-IDS2017. It is not a live network "
              "monitor or a production intrusion-detection system.")


@st.cache_data(show_spinner=False)
def cached(fn_name, *args):
    """Cache any dashboard.data function result by name and arguments (artifacts are read-only)."""
    return getattr(D, fn_name)(*args)


@st.cache_resource(show_spinner="Loading frozen model…")
def model():
    return D.load_model()


def header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.info(DISCLAIMER, icon=":material/science:")


def limitations():
    with st.container(border=True):
        st.subheader("Limitations that apply to every number on this page")
        st.markdown("\n".join(f"- {x}" for x in cached("limitations")))


def source(*files):
    st.caption("Source: " + ", ".join(f"`{f}`" for f in files))


def bar(df, x, y, title, x_title, y_title=None, fmt=".2%", color=BLUE, sort="-x", tooltip=None):
    """Horizontal bar chart with value labels (x = value, y = category)."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=fmt)),
        y=alt.Y(f"{y}:N", title=y_title, sort=sort, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
        tooltip=tooltip or [y, alt.Tooltip(f"{x}:Q", format=fmt)])
    chart = base.mark_bar(color=color, cornerRadiusEnd=3) + base.mark_text(align="left", dx=3).encode(
        text=alt.Text(f"{x}:Q", format=fmt))
    st.altair_chart(chart.properties(title=title, height=max(160, 30 * len(df))), width="stretch")


def grouped_bar(df, cat, val, group, title, y_title, fmt=".2%", colors=(BLUE, ORANGE), tooltip=None, order=None):
    """Vertical grouped bars (x = category, offset = group) with value labels."""
    groups = list(dict.fromkeys(df[group]))
    base = alt.Chart(df).encode(
        x=alt.X(f"{cat}:N", title=None, axis=alt.Axis(labelAngle=0), sort=order),
        xOffset=alt.XOffset(f"{group}:N", sort=groups),
        y=alt.Y(f"{val}:Q", title=y_title, axis=alt.Axis(format=fmt)),
        color=alt.Color(f"{group}:N", sort=groups, scale=alt.Scale(domain=groups, range=list(colors[:len(groups)])),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=tooltip or [cat, group, alt.Tooltip(f"{val}:Q", format=fmt)])
    chart = base.mark_bar(cornerRadiusEnd=3) + base.mark_text(dy=-6).encode(text=alt.Text(f"{val}:Q", format=fmt))
    st.altair_chart(chart.properties(title=title, height=320), width="stretch")


def histogram(h, title, groups, threshold_logit, colors=(BLUE, ORANGE, AQUA, YELLOW)):
    """Binned logit(score) counts, one panel per group (log-scale counts; empty bins omitted), threshold dashed."""
    totals = h.groupby("group").n.sum()
    x_scale = alt.Scale(domain=[float(h.bin_lo.min()), float(h.bin_hi.max())])
    panels = []
    for g, col in zip(groups, colors):
        d = h[(h.group == g) & (h.n > 0)]
        bars = alt.Chart(d).mark_bar(color=col).encode(
            x=alt.X("bin_lo:Q", bin="binned", scale=x_scale, title="logit(score) = log-odds of attack"),
            x2="bin_hi:Q",
            y=alt.Y("n:Q", scale=alt.Scale(type="symlog"), title="rows (symlog)"),
            tooltip=[alt.Tooltip("bin_lo:Q", title="from"), alt.Tooltip("bin_hi:Q", title="to"),
                     alt.Tooltip("n:Q", format=",", title="rows")])
        rule = alt.Chart(pd.DataFrame({"t": [threshold_logit]})).mark_rule(strokeDash=[5, 4], color="#52514e").encode(x="t:Q")
        panels.append(alt.layer(bars, rule).properties(title=f"{g} (n = {int(totals.get(g, 0)):,})", height=110, width=680))
    st.altair_chart(alt.vconcat(*panels).properties(title=title), width="content")
    st.caption(f"Dashed line: frozen threshold (logit = {threshold_logit:.2f}). Count axis is symmetric-log (linear near "
               "zero, logarithmic above) so single flows and hundreds of thousands are both visible; empty bins "
               f"are not drawn. Scores are binned in fixed {float((h.bin_hi - h.bin_lo).iloc[0]):g} log-odds steps "
               "(`reports/generated/m6_score_hist.csv`).")
