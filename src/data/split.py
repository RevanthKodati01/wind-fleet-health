import pandas as pd
from pathlib import Path
import json

CATALOG = Path("data/processed/dataset_catalog.csv")
OUT_DIR = Path("data/processed/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    cat = pd.read_csv(CATALOG)

    splits = {}

    for farm, df in cat.groupby("farm_id"):
        # Prefer datasets marked as train by metadata
        train_files = df[df["train_rate"] > 0.9]["parquet_file"].tolist()
        test_files  = df[df["train_rate"] <= 0.9]["parquet_file"].tolist()

        # Fallback: if everything looks like train, do time-based split
        if len(test_files) == 0:
            train_files = df.sample(frac=0.8, random_state=42)["parquet_file"].tolist()
            test_files = list(set(df["parquet_file"]) - set(train_files))

        splits[farm] = {
            "train": train_files,
            "test": test_files
        }

    with open(OUT_DIR / "file_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("Saved splits to", OUT_DIR / "file_splits.json")
    for farm, s in splits.items():
        print(f"{farm}: train={len(s['train'])}, test={len(s['test'])}")

if __name__ == "__main__":
    main()
