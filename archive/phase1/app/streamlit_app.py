from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="XAI Metric Behavior Lab", layout="wide")
st.title("XAI Metric Behavior Lab")
st.caption("Phase 1 — controlled metric-response explorer")

root = Path(__file__).resolve().parents[1]
results = root/"results"
runs = sorted([p for p in results.glob("*") if p.is_dir() and (p/"summary.csv").exists()])

if not runs:
    st.info("No committed result snapshot yet. Run Phase 1 first, then commit a selected result folder.")
    st.stop()

run = st.sidebar.selectbox("Run",runs,format_func=lambda p:p.name)
summary = pd.read_csv(run/"summary.csv")
profiles = pd.read_csv(run/"behavior_profiles.csv") if (run/"behavior_profiles.csv").exists() else pd.DataFrame()

prop = st.sidebar.selectbox("Property",sorted(summary["property"].unique()))
view = summary[summary["property"]==prop].copy()
intervention = st.sidebar.selectbox("Intervention",sorted(view["intervention"].unique()))
view = view[view["intervention"]==intervention]
metric_names = sorted(view["metric_name"].unique())
selected = st.sidebar.multiselect("Metrics",metric_names,default=metric_names)
view = view[view["metric_name"].isin(selected)]

st.subheader(f"{prop.title()} — {intervention}")
st.write("Do not compare raw magnitudes across different metric scales. Compare response patterns.")
fig = px.line(view,x="level",y="score_mean",color="metric_name",markers=True,error_y="score_std")
st.plotly_chart(fig,use_container_width=True)

st.subheader("Condition summary")
st.dataframe(
    view[["metric_name","level","score_mean","score_std","n_scores","score_direction"]],
    use_container_width=True,hide_index=True
)

if not profiles.empty and "property" in profiles:
    pv = profiles[(profiles["property"]==prop)&(profiles["intervention"]==intervention)]
    if not pv.empty:
        st.subheader("Behavior profile")
        st.caption("Response alignment near +1 means the metric followed the expected ordinal direction.")
        st.dataframe(
            pv[["metric_name","score_direction","spearman_level_score","response_alignment",
                "start_score","end_score","n_levels"]],
            use_container_width=True,hide_index=True
        )
