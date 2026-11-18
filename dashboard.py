import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.forecaster import DemandForecaster
from src.inventory_logic import InventoryAnalyser, InventoryParams

st.set_page_config(page_title="SmartStock", page_icon="📦", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📦 SmartStock")
uploaded = st.sidebar.file_uploader("Upload sales CSV", type=["csv", "xlsx"])

st.sidebar.subheader("Inventory Settings")
lead_time    = st.sidebar.slider("Supplier lead time (days)", 1, 30, 7)
svc_level    = st.sidebar.selectbox("Service level", [0.90, 0.95, 0.99], index=1)
holding_cost = st.sidebar.number_input("Holding cost / unit / year (£)", 0.1, 100.0, 2.5)
order_cost   = st.sidebar.number_input("Cost per order (£)", 1.0, 500.0, 50.0)

st.sidebar.subheader("Model Settings")
model_type = st.sidebar.radio("Algorithm", ["random_forest", "gradient_boosting"])
horizon    = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("SmartStock — Inventory Demand Forecasting")

if uploaded is None:
    st.info("Upload a CSV with columns: `date`, `product_id`, `quantity_sold`")
    st.stop()

@st.cache_data
def run_pipeline(file_bytes, filename):
    import io
    prep = DataPreprocessor()
    raw = pd.read_csv(io.BytesIO(file_bytes)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))
    clean = prep.clean(raw)
    daily = prep.aggregate_daily(clean)
    fe = FeatureEngineer()
    return fe.build(daily)

df = run_pipeline(uploaded.read(), uploaded.name)

product = st.selectbox("Select product", sorted(df["product_id"].unique()))
pdf = df[df["product_id"] == product].copy()

# ── KPI strip ────────────────────────────────────────────────────────────────
params   = InventoryParams(lead_time, svc_level, holding_cost, order_cost)
analyser = InventoryAnalyser(params)

avg_d = pdf["quantity_sold"].mean()
std_d = pdf["quantity_sold"].std()
ss    = analyser.safety_stock(std_d)
rop   = analyser.reorder_point(avg_d, std_d)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Daily Demand",  f"{avg_d:.1f} units")
c2.metric("Safety Stock",      f"{ss:.0f} units")
c3.metric("Reorder Point",     f"{rop:.0f} units")
c4.metric("Demand Volatility", f"σ = {std_d:.1f}")

# ── History chart ─────────────────────────────────────────────────────────────
st.subheader("Sales History")
st.plotly_chart(
    px.line(pdf, x="date", y="quantity_sold",
            labels={"quantity_sold": "Units", "date": "Date"}),
    use_container_width=True,
)

# ── Forecast ──────────────────────────────────────────────────────────────────
st.subheader(f"Forecast — Next {horizon} Days")
with st.spinner("Training…"):
    fc = DemandForecaster(model_type)
    metrics = fc.walk_forward_validate(pdf)
    forecast_df = fc.forecast(pdf, horizon)

m1, m2, m3 = st.columns(3)
m1.metric("MAE",  metrics["mae"])
m2.metric("RMSE", metrics["rmse"])
m3.metric("MAPE", f"{metrics['mape']}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=pdf["date"], y=pdf["quantity_sold"],
                          name="Actual", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"],
                          name="Forecast", line=dict(color="#f59e0b", dash="dash")))
fig.update_layout(xaxis_title="Date", yaxis_title="Units Sold")
st.plotly_chart(fig, use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
if model_type == "random_forest":
    fi = fc.feature_importance()
    if not fi.empty:
        st.subheader("Feature Importance")
        st.plotly_chart(
            px.bar(fi.head(10), x="importance", y="feature", orientation="h"),
            use_container_width=True,
        )