"""Research dashboard for the frozen protocol-v1 CIC-IDS2017 experiment (Milestones 1-5).

Launch from the repository root:  .venv/bin/python -m streamlit run dashboard/app.py
Offline and read-only: it shows committed artifacts; it does not capture, scan or generate traffic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard import ui  # noqa: E402

st.set_page_config(page_title="CIC-IDS2017 benchmark — research dashboard", page_icon=":material/lan:", layout="wide")

PAGES = [
    st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/evaluation.py", title="Evaluation", icon=":material/fact_check:"),
    st.Page("pages/attack_analysis.py", title="Attack families", icon=":material/category:"),
    st.Page("pages/explainability.py", title="Explainability (SHAP)", icon=":material/insights:"),
    st.Page("pages/false_positives.py", title="False-positive case study", icon=":material/report:"),
    st.Page("pages/novel_vectors.py", title="Novel-vector analysis", icon=":material/fiber_new:"),
    st.Page("pages/prediction_demo.py", title="Benchmark case explorer", icon=":material/search:"),
    st.Page("pages/methodology.py", title="Methodology & limitations", icon=":material/menu_book:"),
]

nav = st.navigation(PAGES)
with st.sidebar:
    st.caption(ui.DISCLAIMER)
    st.caption("All values are read from committed artifacts of the single frozen test evaluation (protocol v1).")
nav.run()
