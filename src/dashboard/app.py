# import pandas as pd
# import streamlit as st
# from pathlib import Path

# st.set_page_config(page_title="Wind Fleet Health", page_icon="🌀", layout="wide")

# RISK_CSV = Path("data/processed/fleet_risk.csv")

# st.title("🌀 Wind Fleet Health Monitoring")
# st.caption("Fleet-level anomaly risk scoring (Isolation Forest baseline)")

# if not RISK_CSV.exists():
#     st.error(f"Missing {RISK_CSV}. Run: python src/scoring/fleet_risk.py")
#     st.stop()

# df = pd.read_csv(RISK_CSV)
# df["t_end"] = pd.to_datetime(df["t_end"], errors="coerce")

# # Sidebar filters
# st.sidebar.header("Filters")
# farms = sorted(df["farm_id"].dropna().unique().tolist())
# selected_farms = st.sidebar.multiselect("Farm", farms, default=farms)

# min_risk, max_risk = float(df["risk_score"].min()), float(df["risk_score"].max())
# risk_range = st.sidebar.slider("Risk score range", 0.0, 100.0, (max(0.0, min_risk), min(100.0, max_risk)))

# df_f = df[
#     (df["farm_id"].isin(selected_farms)) &
#     (df["risk_score"] >= risk_range[0]) &
#     (df["risk_score"] <= risk_range[1])
# ].copy()

# # Risk buckets (simple, tune later)
# def bucket(r):
#     if r >= 80: return "High"
#     if r >= 50: return "Medium"
#     return "Low"

# df_f["risk_bucket"] = df_f["risk_score"].apply(bucket)

# # KPI row
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Assets in view", f"{len(df_f)}")
# c2.metric("High risk (≥80)", f"{(df_f['risk_bucket']=='High').sum()}")
# c3.metric("Avg risk", f"{df_f['risk_score'].mean():.1f}")
# c4.metric("Max risk", f"{df_f['risk_score'].max():.1f}" if len(df_f) else "—")

# # Main table + chart
# left, right = st.columns([2, 1])

# with left:
#     st.subheader("Fleet ranking")
#     show_cols = ["farm_id","asset_id","risk_score","risk_bucket","alert_rate","max_anomaly_score","threshold","n_points_scored","t_end"]
#     st.dataframe(
#         df_f.sort_values("risk_score", ascending=False)[show_cols],
#         use_container_width=True,
#         height=520
#     )

# with right:
#     st.subheader("Risk distribution")
#     st.bar_chart(df_f["risk_bucket"].value_counts())
# st.subheader("Top risky assets")
# top = df_f.sort_values("risk_score", ascending=False).head(10)[["farm_id","asset_id","risk_score","alert_rate"]]
# st.dataframe(top, use_container_width=True, height=320)

# st.divider()
import pandas as pd
import streamlit as st
from pathlib import Path

# 1. CHANGED LAYOUT TO "centered"
st.set_page_config(page_title="Wind Fleet Health", page_icon="🌀", layout="centered")

# 2. ADD CUSTOM CSS FOR "SOMEWHAT WIDE" LOOK
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px !important;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

RISK_CSV = Path("data/processed/fleet_risk.csv")

st.title("🌀 Wind Fleet Health Monitoring")
st.caption("Fleet-level anomaly risk scoring (Isolation Forest baseline)")

if not RISK_CSV.exists():
    st.error(f"Missing {RISK_CSV}. Run: python src/scoring/fleet_risk.py")
    st.stop()

df = pd.read_csv(RISK_CSV)
df["t_end"] = pd.to_datetime(df["t_end"], errors="coerce")

# 3. MOVED SIDEBAR FILTERS TO A COLLAPSIBLE EXPANDER
with st.expander("⚙️ Filter Options", expanded=False):
    c_filt1, c_filt2 = st.columns(2)
    farms = sorted(df["farm_id"].dropna().unique().tolist())
    with c_filt1:
        selected_farms = st.multiselect("Farm", farms, default=farms)

    min_risk, max_risk = float(df["risk_score"].min()), float(df["risk_score"].max())
    with c_filt2:
        risk_range = st.slider("Risk score range", 0.0, 100.0, (max(0.0, min_risk), min(100.0, max_risk)))

df_f = df[
    (df["farm_id"].isin(selected_farms)) &
    (df["risk_score"] >= risk_range[0]) &
    (df["risk_score"] <= risk_range[1])
].copy()

# Risk buckets (simple, tune later)
def bucket(r):
    if r >= 80: return "High"
    if r >= 50: return "Medium"
    return "Low"

df_f["risk_bucket"] = df_f["risk_score"].apply(bucket)

st.divider()

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets in view", f"{len(df_f)}")
c2.metric("High risk (≥80)", f"{(df_f['risk_bucket']=='High').sum()}")
c3.metric("Avg risk", f"{df_f['risk_score'].mean():.1f}")
c4.metric("Max risk", f"{df_f['risk_score'].max():.1f}" if len(df_f) else "—")

st.divider()

# 4. STACKED LAYOUT (Removed left/right split)
st.subheader("📊 Risk Distribution")
st.bar_chart(df_f["risk_bucket"].value_counts())

st.subheader("📋 Fleet Ranking")
show_cols = ["farm_id","asset_id","risk_score","risk_bucket","alert_rate","max_anomaly_score","threshold","n_points_scored","t_end"]
st.dataframe(
    df_f.sort_values("risk_score", ascending=False)[show_cols],
    use_container_width=True,
    height=300
)

st.subheader("⚠️ Top Risky Assets")
top = df_f.sort_values("risk_score", ascending=False).head(10)[["farm_id","asset_id","risk_score","alert_rate"]]
st.dataframe(top, use_container_width=True, height=320)