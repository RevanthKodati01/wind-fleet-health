# src/data/load.py
import pandas as pd
from pathlib import Path
from typing import List
import logging

RAW_ROOT = Path("data/raw/zenodo/CARE_To_Compare")
PROCESSED_ROOT = Path("data/processed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SKIP_FILES = {"event_info.csv", "feature_description.csv", "readme.csv"}

def discover_dataset_csvs() -> List[Path]:
    # only load time-series datasets/*.csv
    csvs = list(RAW_ROOT.rglob("datasets/*.csv"))
    logging.info(f"Discovered {len(csvs)} dataset CSV files under datasets/")
    return csvs

def farm_from_path(p: Path) -> str:
    # .../CARE_To_Compare/Wind Farm A/datasets/0.csv -> Wind_Farm_A
    parts = p.parts
    farm = next((x for x in parts if x.lower().startswith("wind farm")), "unknown_farm")
    return farm.replace(" ", "_")

def load_single_dataset_csv(path: Path) -> pd.DataFrame:
    # CARE files are semicolon-separated
    df = pd.read_csv(path, sep=";", engine="python")

    # normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # parse timestamp
    if "time_stamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.drop(columns=["time_stamp"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"No time_stamp/timestamp column in {path}")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # add farm_id and dataset_id for traceability
    df["farm_id"] = farm_from_path(path)
    df["dataset_id"] = path.stem  # "0", "40", etc.

    return df

def main():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for i, csv_path in enumerate(discover_dataset_csvs(), 1):
        try:
            df = load_single_dataset_csv(csv_path)
            all_dfs.append(df)
            if i % 10 == 0:
                logging.info(f"Loaded {i} files...")
        except Exception as e:
            logging.warning(f"Skipping {csv_path}: {e}")

    if not all_dfs:
        raise RuntimeError("No dataset CSVs loaded. Check paths and separator.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    out_path = PROCESSED_ROOT / "scada_all.parquet"
    full_df.to_parquet(out_path, index=False)

    logging.info(f"Saved processed dataset to {out_path}")
    logging.info(f"Final shape: {full_df.shape}")
    logging.info(f"Columns: {len(full_df.columns)}")
    if "asset_id" in full_df.columns:
        logging.info(f"Unique asset_id: {full_df['asset_id'].nunique()}")
    logging.info(f"timestamp range: {full_df['timestamp'].min()} -> {full_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
