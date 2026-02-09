import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache

PARQUET_DIR = Path("data/processed/scada_parquet")
RISK_CSV = Path("data/processed/fleet_risk.csv")
THR_PATH = Path("models/baseline/thresholds.json")
MODEL_DIR = Path("models/baseline")

app = FastAPI(title="Wind Fleet Health API", version="0.1.0")

class ScoreRequest(BaseModel):
    farm_id: str
    parquet_file: Optional[str] = None   # preferred
    asset_id: Optional[str] = None       # fallback
    lookback_hours: int = 24

def align_features(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    return df[feats]

def compute_risk(scores: np.ndarray, threshold: float) -> Dict[str, float]:
    alerts = (scores >= threshold).astype(int)
    alert_rate = float(alerts.mean()) if len(alerts) else 0.0
    max_score = float(scores.max()) if len(scores) else 0.0
    risk = 100 * (0.7 * alert_rate + 0.3 * (max_score / (threshold + 1e-6)))
    return {
        "risk_score": float(np.clip(risk, 0, 100)),
        "alert_rate": alert_rate,
        "max_anomaly_score": max_score,
    }

def top_contributors(df_recent: pd.DataFrame, feats: List[str], tmax: pd.Timestamp) -> List[Dict[str, Any]]:
    # baseline = earlier period (or first 30%) for simple explainability
    baseline = df_recent.iloc[: max(200, int(0.3 * len(df_recent)))].copy()
    recent_24h = df_recent[df_recent["timestamp"] >= (tmax - pd.Timedelta(hours=24))].copy()
    if len(recent_24h) < 50:
        recent_24h = df_recent.tail(200).copy()

    base_mean = baseline[feats].mean(numeric_only=True)
    base_std = baseline[feats].std(numeric_only=True).replace(0, np.nan)
    rec_mean = recent_24h[feats].mean(numeric_only=True)

    z_shift = ((rec_mean - base_mean).abs() / base_std).replace([np.inf, -np.inf], np.nan).dropna()
    top = z_shift.sort_values(ascending=False).head(10)

    out = []
    for f in top.index:
        out.append({
            "feature": f,
            "z_shift": float(top.loc[f]),
            "recent_mean": float(rec_mean.loc[f]),
            "baseline_mean": float(base_mean.loc[f]),
        })
    return out

@app.get("/health")
def health():
    return {"status": "ok"}
@lru_cache(maxsize=16)
def load_model(farm_id: str):
    pack = joblib.load(MODEL_DIR / f"isoforest_{farm_id}.joblib")
    model = pack["model"]
    feats = list(pack["features"])
    n_expected = int(getattr(model, "n_features_in_", len(feats)))
    feats = feats[:n_expected]
    return model, feats

@app.post("/score")
def score(req: ScoreRequest):
    if not THR_PATH.exists():
        raise HTTPException(status_code=500, detail="Missing thresholds.json. Run thresholding step.")

    thr = json.loads(THR_PATH.read_text())

    if req.farm_id not in thr:
        raise HTTPException(status_code=400, detail=f"Unknown farm_id {req.farm_id}")

    model_pack = joblib.load(MODEL_DIR / f"isoforest_{req.farm_id}.joblib")
    model = model_pack["model"]
    feats = list(model_pack["features"])

    n_expected = int(getattr(model, "n_features_in_", len(feats)))
    if len(feats) != n_expected:
        feats = feats[:n_expected]

    threshold = float(thr[req.farm_id]["threshold"])

    # resolve parquet file
    parquet_file = req.parquet_file
    if parquet_file is None:
        if req.asset_id is None:
            raise HTTPException(status_code=400, detail="Provide either parquet_file or asset_id")
        if not RISK_CSV.exists():
            raise HTTPException(status_code=500, detail="Missing fleet_risk.csv to map asset_id->parquet_file")
        fleet = pd.read_csv(RISK_CSV)
        match = fleet[(fleet["farm_id"] == req.farm_id) & (fleet["asset_id"].astype(str) == str(req.asset_id))].head(1)
        if match.empty:
            raise HTTPException(status_code=404, detail="asset_id not found in fleet_risk.csv for this farm")
        parquet_file = match["parquet_file"].iloc[0]

    path = PARQUET_DIR / parquet_file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Parquet not found: {path}")

    cols = list(set(feats + ["timestamp", "asset_id"]))
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise HTTPException(status_code=500, detail="No timestamped rows in parquet.")

    tmax = df["timestamp"].max()
    tmin = tmax - pd.Timedelta(hours=req.lookback_hours)
    recent = df[df["timestamp"] >= tmin].copy()
    
    if recent.empty:
        raise HTTPException(status_code=400, detail="No rows in lookback window.")
    MAX_POINTS = 50_000  # safe + fast
    if len(recent) > MAX_POINTS:
        recent = recent.sample(MAX_POINTS, random_state=42).sort_values("timestamp")
    X = align_features(recent, feats).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    scores = -model.score_samples(X)
    metrics = compute_risk(scores, threshold)

    # return a few alert timestamps (last 50)
    recent["anomaly_score"] = scores
    recent["alert"] = (recent["anomaly_score"] >= threshold).astype(int)
    alert_times = recent[recent["alert"] == 1][["timestamp", "anomaly_score"]].tail(50)
    

    return {
        "farm_id": req.farm_id,
        "parquet_file": parquet_file,
        "asset_id": str(recent["asset_id"].iloc[0]),
        "t_end": str(tmax),
        "lookback_hours": req.lookback_hours,
        "threshold": threshold,
        **metrics,
        "alerts_tail": [
            {"timestamp": str(r["timestamp"]), "anomaly_score": float(r["anomaly_score"])}
            for _, r in alert_times.iterrows()
        ],
        "top_contributors": top_contributors(recent, feats, tmax),
    }
