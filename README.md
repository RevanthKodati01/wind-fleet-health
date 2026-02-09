# 🌀 Wind Fleet Health Monitoring (SCADA) — End-to-End ML + API + Dashboard

Fleet-level anomaly risk scoring for wind turbines using SCADA sensor data (CARE-to-Compare).  
This repo includes a complete pipeline: **data ingestion → model training → threshold calibration → fleet risk scoring → FastAPI inference → Streamlit dashboard**.

---

## ✨ What this project does

- Converts raw SCADA CSVs into efficient **Parquet** files
- Trains **per-farm Isolation Forest** anomaly detection models
- Calibrates **alert thresholds** per wind farm
- Generates **fleet ranking** for operations triage (`fleet_risk.csv`)
- Serves scoring + explainability via **FastAPI** (`POST /score`)
- Visualizes fleet and asset drilldown via **Streamlit** (API-powered)

---

## 📦 Dataset

Download **CARE-To-Compare** (Zenodo) and place files in this expected structure:

```text
data/raw/zenodo/CARE_To_Compare/
  Wind Farm A/
    datasets/*.csv
  Wind Farm B/
    datasets/*.csv
  Wind Farm C/
    datasets/*.csv
The dataset has schema differences across farms/files (feature drift). This repo handles it via feature alignment.
🏗️ Architecture
Raw CSVs (Zenodo)
      │
      ▼
src/data/load.py
  → data/processed/scada_parquet/*.parquet
  → data/processed/scada_index.csv
      │
      ▼
src/models/baseline_isolation_forest.py
  → models/baseline/isoforest_<farm>.joblib
      │
      ▼
src/models/thresholding.py
  → models/baseline/thresholds.json
      │
      ▼
src/scoring/fleet_risk.py
  → data/processed/fleet_risk.csv
      │
      ├────────► FastAPI (src/api/main.py)  POST /score
      └────────► Streamlit (src/dashboard/app.py + pages)
📁 Project Structure
wind-fleet-health/
  src/
    data/
      load.py
      catalog.py
      split.py
    models/
      baseline_isolation_forest.py
      thresholding.py
    scoring/
      fleet_risk.py
    api/
      main.py
    dashboard/
      app.py
      pages/
        2_Asset_Drilldown.py
  data/
    raw/zenodo/...
    processed/
      scada_parquet/
      scada_index.csv
      fleet_risk.csv
  models/
    baseline/
      isoforest_<farm>.joblib
      thresholds.json
✅ Setup (Mac/Linux)
1) Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
2) Install dependencies
python -m pip install pandas numpy pyarrow scikit-learn joblib fastapi uvicorn streamlit requests
▶️ One-time full build (first run)
Run this once after placing the dataset under data/raw/zenodo/...:
python src/data/load.py
python src/data/catalog.py
python src/data/split.py
python src/models/baseline_isolation_forest.py
python src/models/thresholding.py
python src/scoring/fleet_risk.py
Generated outputs
data/processed/scada_parquet/*.parquet
data/processed/scada_index.csv
models/baseline/isoforest_<farm>.joblib
models/baseline/thresholds.json
data/processed/fleet_risk.csv
🚀 Run (normal daily run)
Once outputs exist, you do not rebuild every time.
You typically run only the API + dashboard.
Terminal A — Start API
source .venv/bin/activate
python -m uvicorn src.api.main:app --reload --port 8000
Health check:
curl -s http://127.0.0.1:8000/health
Swagger UI:
http://127.0.0.1:8000/docs
Terminal B — Start Dashboard
source .venv/bin/activate
streamlit run src/dashboard/app.py
🔌 API
POST /score
Scores a single turbine parquet file for a lookback window and returns:
risk_score (0–100)
alert_rate
max_anomaly_score
threshold
alerts_tail (last N alert timestamps)
top_contributors (simple explainability)
Example request:
python - << 'EOF'
import requests, json

payload = {
  "farm_id": "Wind_Farm_C",
  "parquet_file": "Wind_Farm_C__43.parquet",
  "lookback_hours": 720
}

r = requests.post("http://127.0.0.1:8000/score", json=payload, timeout=120)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2)[:1500])
EOF
📊 Dashboard
Fleet Overview
KPI cards: assets, high-risk count, avg risk, max risk
Fleet ranking table
Risk distribution chart
Asset Drilldown (API)
Calls FastAPI /score
Shows risk metrics, recent alerts, and top contributing sensors
🧠 Modeling (Baseline)
Model: Isolation Forest (per wind farm)
Handles schema differences via feature alignment
Thresholding: calibrated per farm and stored in models/baseline/thresholds.json
Fleet risk score: blends alert-rate and max anomaly score normalized by threshold
🛠️ Common issues
ModuleNotFoundError: joblib (API)
You likely started uvicorn from (base) instead of (.venv).
Fix:

source .venv/bin/activate
python -m pip install joblib
python -m uvicorn src.api.main:app --reload --port 8000
Drilldown slow on 30 days
30 days can be heavy depending on rows/features. Recommended:
keep API running in a separate terminal
add caching/downsampling in the API if needed
✅ Resume-ready bullet
Built an end-to-end wind turbine fleet health monitoring platform using SCADA data, including Parquet ingestion, per-farm Isolation Forest anomaly models, threshold calibration, fleet risk ranking, and a FastAPI + Streamlit interface for fleet triage and asset drilldown with explainability.
📜 License
MIT (optional — add a LICENSE file)
::contentReference[oaicite:0]{index=0}

