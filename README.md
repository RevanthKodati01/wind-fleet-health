# Fleet Health Monitoring + Early Fault Detection (Wind Turbine SCADA)

End-to-end ML system for detecting early fault signals in wind turbine SCADA time series and ranking turbines by near-term risk.

## What this project includes
- Data pipeline: ingestion → cleaning → labeling → windowing
- Models:
  - Baseline: Isolation Forest (trained on healthy data)
  - Improved: GRU/LSTM Autoencoder for multivariate time-series anomaly detection
- Early warning evaluation: lead time (hours before fault) + false alarms/day
- FastAPI service: score turbine data → risk + top contributing sensors
- Streamlit dashboard: fleet overview + turbine drilldown + event replay
- Monitoring: drift + alert quality trends

## Dataset
Wind Turbine SCADA Early Fault Detection dataset (CARE).  
Download (Zenodo/Kaggle), place files into `data/raw/`.

## Quickstart (after data is downloaded)
1) Create venv + install deps
2) Run preprocessing to generate parquet files
3) Train baseline + deep model
4) Run API + dashboard

## Repo structure
- `src/data/` ingestion, cleaning, labeling, features, splits
- `src/models/` baseline + autoencoder + thresholding + explainability
- `src/scoring/` metrics + early warning evaluation
- `src/api/` FastAPI inference service
- `src/dashboard/` Streamlit app
- `src/monitoring/` drift + false alarms
