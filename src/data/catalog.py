import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PARQUET_DIR = Path("data/processed/scada_parquet")
INDEX_CSV = Path("data/processed/scada_index.csv")
OUT_CSV = Path("data/processed/dataset_catalog.csv")

def main():
    idx = pd.read_csv(INDEX_CSV)
    rows = []

    for i, fname in enumerate(idx["parquet_file"], 1):
        p = PARQUET_DIR / fname
        df = pd.read_parquet(p, columns=["asset_id","train_test","status_type_id","timestamp"])

        farm_id, dataset_id = fname.replace(".parquet","").split("__", 1)

        n = len(df)
        abnormal_rate = float((df["status_type_id"] != 0).mean())
        train_rate = float((df["train_test"].astype(str).str.lower() == "train").mean())

        rows.append({
            "parquet_file": fname,
            "farm_id": farm_id,
            "dataset_id": dataset_id,
            "asset_id": df["asset_id"].iloc[0] if n else None,
            "n_rows": n,
            "ts_min": df["timestamp"].min(),
            "ts_max": df["timestamp"].max(),
            "abnormal_rate": abnormal_rate,
            "train_rate": train_rate,
        })

        if i % 10 == 0:
            logging.info(f"Cataloged {i}/{len(idx)} files...")

    cat = pd.DataFrame(rows)
    cat.to_csv(OUT_CSV, index=False)
    logging.info(f"Saved catalog: {OUT_CSV} (rows={len(cat)})")
    logging.info("Abnormal rate summary:")
    logging.info(cat["abnormal_rate"].describe().to_string())

if __name__ == "__main__":
    main()
