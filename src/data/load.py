# src/data/load.py
import pandas as pd
from pathlib import Path
from typing import List
import logging

RAW_ROOT = Path("data/raw/zenodo/CARE_To_Compare")
OUT_DIR = Path("data/processed/scada_parquet")   # per-file parquet output
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

def discover_dataset_csvs() -> List[Path]:
    csvs = list(RAW_ROOT.rglob("datasets/*.csv"))
    logging.info(f"Discovered {len(csvs)} dataset CSV files under datasets/")
    return csvs

def farm_from_path(p: Path) -> str:
    parts = p.parts
    farm = next((x for x in parts if x.lower().startswith("wind farm")), "unknown_farm")
    return farm.replace(" ", "_")

def load_single_dataset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "time_stamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.drop(columns=["time_stamp"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"No time_stamp/timestamp column in {path}")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    df["farm_id"] = farm_from_path(path)
    df["dataset_id"] = path.stem  # filename without .csv (e.g., 0, 40, ...)
    return df

def main():
    csvs = discover_dataset_csvs()
    written = 0

    for i, csv_path in enumerate(csvs, 1):
        try:
            df = load_single_dataset_csv(csv_path)

            farm_id = df["farm_id"].iloc[0]
            dataset_id = df["dataset_id"].iloc[0]
            out_path = OUT_DIR / f"{farm_id}__{dataset_id}.parquet"

            df.to_parquet(out_path, index=False)
            written += 1

            if i % 10 == 0:
                logging.info(f"Processed {i}/{len(csvs)} files... (written {written})")

        except Exception as e:
            logging.warning(f"Skipping {csv_path}: {e}")

    # Create an index so we can load all parquet files later efficiently
    index_path = Path("data/processed/scada_index.csv")
    pd.DataFrame({"parquet_file": sorted([p.name for p in OUT_DIR.glob("*.parquet")])}).to_csv(index_path, index=False)

    logging.info(f"Done. Parquet written: {written}")
    logging.info(f"Index saved: {index_path}")
    logging.info(f"Parquet folder: {OUT_DIR}")

if __name__ == "__main__":
    main()
