
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PARQUET_DIR = Path("data/processed/scada_parquet")
INDEX_CSV = Path("data/processed/scada_index.csv")
THR_PATH = Path("models/baseline/thresholds.json")
MODEL_DIR = Path("models/baseline")
OUT_CSV = Path("data/processed/fleet_risk.csv")

HOURS_LOOKBACK = 24

def align_features(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    # add missing expected columns
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    # select + order exactly
    return df[feats]

def compute_risk(scores: np.ndarray, threshold: float):
    alerts = (scores >= threshold).astype(int)
    alert_rate = float(alerts.mean()) if len(alerts) else 0.0
    max_score = float(scores.max()) if len(scores) else 0.0
    # risk score: simple blend (tune later)
    risk = 100 * (0.7 * alert_rate + 0.3 * (max_score / (threshold + 1e-6)))
    return float(np.clip(risk, 0, 100)), alert_rate, max_score

def main():
    if not THR_PATH.exists():
        raise FileNotFoundError(f"Missing {THR_PATH}. Run: python src/models/thresholding.py")

    thr = json.loads(THR_PATH.read_text())
    idx = pd.read_csv(INDEX_CSV)

    rows = []
    for fname in idx["parquet_file"]:
        farm_id = fname.split("__")[0]
        if farm_id not in thr:
            continue

        model_pack = joblib.load(MODEL_DIR / f"isoforest_{farm_id}.joblib")
        model = model_pack["model"]
        feats = list(model_pack["features"])

        # 🔒 ensure feature list length matches fitted model
        n_expected = int(getattr(model, "n_features_in_", len(feats)))
        if len(feats) != n_expected:
            # keep it deterministic
            feats = feats[:n_expected]

        threshold = float(thr[farm_id]["threshold"])

        # load only what's needed
        df = pd.read_parquet(PARQUET_DIR / fname, columns=list(set(feats + ["timestamp", "asset_id"])))
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            continue

        # last 24 hours slice
        tmax = df["timestamp"].max()
        tmin = tmax - pd.Timedelta(hours=HOURS_LOOKBACK)
        recent = df[df["timestamp"] >= tmin]
        if recent.empty:
            continue

        Xdf = align_features(recent, feats).fillna(0.0)
        X = Xdf.to_numpy(dtype=np.float32, copy=False)

        scores = -model.score_samples(X)

        risk, alert_rate, max_score = compute_risk(scores, threshold)

        rows.append({
            "farm_id": farm_id,
            "parquet_file": fname,
            "asset_id": recent["asset_id"].iloc[0],
            "t_end": tmax,
            "lookback_hours": HOURS_LOOKBACK,
            "risk_score": risk,
            "alert_rate": alert_rate,
            "max_anomaly_score": max_score,
            "threshold": threshold,
            "n_points_scored": int(len(scores)),
        })

    out = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    if len(out):
        print(out.head(10)[["farm_id","asset_id","risk_score","alert_rate","max_anomaly_score","threshold","n_points_scored"]])
    else:
        print("No rows written. Check thresholds/models/parquet paths.")

if __name__ == "__main__":
    main()
