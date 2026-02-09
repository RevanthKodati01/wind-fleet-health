from pathlib import Path
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="Asset Drilldown", page_icon="🔍", layout="wide")

RISK_CSV = Path("data/processed/fleet_risk.csv")
API = "http://127.0.0.1:8000"

@st.cache_data(ttl=60)
def api_score(farm_id: str, parquet_file: str, lookback_hours: int):
    payload = {"farm_id": farm_id, "parquet_file": parquet_file, "lookback_hours": lookback_hours}
    r = requests.post(f"{API}/score", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

st.title("🔍 Asset Drilldown (API)")
st.caption("Streamlit → FastAPI /score → model inference")

if not RISK_CSV.exists():
    st.error("Missing data/processed/fleet_risk.csv. Run: python src/scoring/fleet_risk.py")
    st.stop()

fleet = pd.read_csv(RISK_CSV)
fleet["t_end"] = pd.to_datetime(fleet["t_end"], errors="coerce")

# Sidebar selectors
st.sidebar.header("Select asset")
farms = sorted(fleet["farm_id"].dropna().unique().tolist())
farm_id = st.sidebar.selectbox("Farm", farms)

assets = fleet.loc[fleet["farm_id"] == farm_id, "asset_id"].dropna().unique().tolist()
assets = sorted(assets, key=lambda x: int(x) if str(x).isdigit() else str(x))
asset_id = st.sidebar.selectbox("Asset ID", assets)

lookback_days = st.sidebar.slider("Lookback days", 1, 30, 30)
run = st.sidebar.button("Run Score")
if not run:
    st.info("Select inputs and click **Run Score**.")
    st.stop()

row = fleet[(fleet["farm_id"] == farm_id) & (fleet["asset_id"] == asset_id)].head(1)
if row.empty:
    st.error("Could not find asset in fleet_risk.csv")
    st.stop()

parquet_file = row["parquet_file"].iloc[0]

# Call API
try:
    resp = api_score(farm_id, parquet_file, int(lookback_days * 24))
except Exception as e:
    st.error(f"API call failed. Is uvicorn running on {API}? Error: {e}")
    st.stop()

st.subheader("📌 Score Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Farm", resp["farm_id"])
c2.metric("Asset", resp["asset_id"])
c3.metric("Risk score", f"{resp['risk_score']:.1f}")
c4.metric("Alert rate", f"{resp['alert_rate']:.3f}")

c5, c6, c7 = st.columns(3)
c5.metric("Max anomaly score", f"{resp['max_anomaly_score']:.4f}")
c6.metric("Threshold", f"{resp['threshold']:.4f}")
c7.metric("Lookback (hours)", f"{resp['lookback_hours']}")

st.subheader("🚨 Recent alerts (tail)")
alerts_tail = pd.DataFrame(resp.get("alerts_tail", []))
if alerts_tail.empty:
    st.write("No alerts in the selected lookback window ✅")
else:
    st.dataframe(alerts_tail, use_container_width=True, height=280)

st.subheader("🧠 Top contributing sensors")
top = pd.DataFrame(resp.get("top_contributors", []))
if top.empty:
    st.write("No contributors available.")
else:
    st.dataframe(top, use_container_width=True, height=380)

st.divider()
#st.info("Tip: Keep the API running in another terminal: uvicorn src.api.main:app --reload --port 8000")
