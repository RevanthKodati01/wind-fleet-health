# 🌬️ Wind Fleet Health Monitoring Platform

An **end-to-end, production-style ML system** for monitoring wind turbine fleet health using real SCADA data.
The platform detects early anomalies at **asset and fleet level**, exposes a **model inference API**, and provides an **interactive dashboard** for engineers and decision-makers.

> Built to mirror **real industrial workflows** used in energy and renewables companies (e.g., fleet health, condition monitoring, early fault detection).

---

## 🚀 Key Capabilities

* 📊 **Fleet-level risk ranking** across multiple wind farms
* 🔍 **Asset drilldown** with anomaly trends and explainability
* 🤖 **Per-farm anomaly detection models** (Isolation Forest)
* 🚨 **Threshold-based alerting** calibrated from historical data
* ⚙️ **FastAPI inference service** (`/score` endpoint)
* 🖥️ **Streamlit dashboard** (frontend → backend → model)
* 📦 Efficient **Parquet-based data pipeline** for large SCADA datasets

---

## 🧠 Problem Statement

Wind turbines generate high-frequency **SCADA sensor data** (temperatures, wind speed, power, vibration, etc.).

Failures are:

* Rare
* Expensive
* Often preceded by **subtle multivariate anomalies**

This project builds a **scalable ML system** that:

* Learns normal turbine behavior
* Detects abnormal operating patterns early
* Prioritizes risky assets across a fleet
* Provides **explainable signals** for engineers

---

## 📂 Dataset

* **Source:** Zenodo – CARE-to-Compare Wind Turbine SCADA Dataset
* **Structure:**

  * Multiple wind farms (A, B, C)
  * Multiple turbines per farm
  * ~10-minute resolution time series
  * 80–900+ sensor-derived features per asset

Raw CSVs are transformed into **per-asset Parquet files** for efficient analytics.

---

## 🏗️ Architecture

```
Raw SCADA CSVs (Zenodo)
        │
        ▼
Data Ingestion & Cleaning
(Pandas, Python)
        │
        ▼
Parquet Storage (per asset)
        │
        ▼
Isolation Forest Models
(per wind farm)
        │
        ▼
Threshold Calibration
        │
        ▼
Fleet Risk Scoring (24–30 day window)
        │
        ▼
FastAPI Inference Service (/score)
        │
        ▼
Streamlit Dashboard
(Fleet View + Asset Drilldown)
```

---

## 🤖 Modeling Approach

### Baseline Model

* **Isolation Forest** (unsupervised anomaly detection)
* Trained **per wind farm** to capture site-specific behavior

### Why Isolation Forest?

* Handles high-dimensional sensor data
* No need for labeled failures
* Efficient on large datasets
* Widely used in industrial anomaly detection

### Evaluation

* Train/Test splits using historical data
* ROC-AUC computed using abnormal operating labels
* Thresholds calibrated per farm for alerting

---

## 📊 Risk Scoring Logic

For each turbine (asset):

* Compute anomaly scores over last **N hours** (default: 24–30 days)
* Generate alerts when score ≥ calibrated threshold
* Aggregate into a **fleet risk score**:

```
Risk = 0.7 × Alert Rate + 0.3 × Normalized Max Score
```

Final score scaled to **0–100** for intuitive ranking.

---

## 🖥️ Dashboard Features

### Fleet Overview

* Asset ranking by risk score
* Risk buckets: Low / Medium / High
* Filters by wind farm and risk range
* Risk distribution visualization

### Asset Drilldown (API-backed)

* On-demand scoring via FastAPI
* Alert timestamps
* Top contributing sensors (explainability)
* Configurable lookback window (default: **30 days**)

---

## ⚙️ API

### Health Check

```
GET /health
```

### Score Asset

```
POST /score
```

**Payload**

```json
{
  "farm_id": "Wind_Farm_C",
  "parquet_file": "Wind_Farm_C__43.parquet",
  "lookback_hours": 720
}
```

**Response (excerpt)**

```json
{
  "risk_score": 100.0,
  "alert_rate": 1.0,
  "max_anomaly_score": 0.62,
  "top_contributors": [
    {"feature": "sensor_11_avg", "z_shift": 4.3}
  ]
}
```

---

## ▶️ How to Run

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start API

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Start Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
wind-fleet-health/
│
├── src/
│   ├── data/          # ingestion, catalog, splits
│   ├── models/        # training + thresholding
│   ├── scoring/       # fleet risk computation
│   ├── api/           # FastAPI service
│   └── dashboard/     # Streamlit UI
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── baseline/
│
├── requirements.txt
└── README.md
```

---

## 🎯 Why This Project Stands Out

* Uses **real industrial SCADA data**
* Full ML lifecycle: data → model → API → UI
* Designed for **scale and production realism**
* Clear separation of concerns (data, model, serving, UI)
* Directly applicable to **energy, renewables, and asset health monitoring** roles

---

## 🚧 Future Improvements

* Lead-time evaluation (hours before failure)
* Concept/data drift monitoring
* Model versioning (MLflow)
* Dockerized deployment
* Online scoring (streaming)

---

## 👤 Author

**Kodati Revanth**
M.S. Data Science – SUNY Albany
📧 [kodatirevanth@gmail.com](mailto:kodatirevanth@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/revanth-kodati](https://www.linkedin.com/in/revanth-kodati)

---

⭐ If you found this project interesting, feel free to star the repository!
