# src/data/label.py
import pandas as pd
from pathlib import Path
import logging

RAW_ROOT = Path("data/raw/zenodo/CARE_To_Compare")
SCADA_PATH = Path("data/processed/scada_all.parquet")
OUT_PATH = Path("data/processed/scada_labeled.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

LEAD_HOURS = 48  # pre-fault window length

def load_event_info() -> pd.DataFrame:
    paths = list(RAW_ROOT.rglob("event_info.csv"))
    if not paths:
        raise FileNotFoundError("No event_info.csv found under dataset root.")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df["source_path"] = str(p)
        dfs.append(df)

    events = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(events)} event rows from {len(paths)} files")

    # Find likely time columns
    # We'll try common names; if yours differ, we’ll adjust after you run it once.
    for col in ["start_time", "startdate", "start_date", "event_start", "start"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")
    for col in ["end_time", "enddate", "end_date", "event_end", "end"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")

    return events

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    scada = pd.read_parquet(SCADA_PATH)
    scada["timestamp"] = pd.to_datetime(scada["timestamp"], errors="coerce")
    scada = scada.dropna(subset=["timestamp"])
    scada["label"] = 0  # healthy by default

    events = load_event_info()

    # identify columns in events file
    start_col = pick_col(events, ["start_time","start_date","event_start","start"])
    end_col   = pick_col(events, ["end_time","end_date","event_end","end"])
    turb_col  = pick_col(events, ["turbine_id","turbine","turbine_name"])
    farm_col  = pick_col(events, ["farm_id","wind_farm","farm","windfarm"])

    if start_col is None:
        raise ValueError(f"Could not find event start column. Available columns: {events.columns.tolist()}")
    if turb_col is None:
        logging.warning("No turbine_id column found in event_info. We will label only by time windows if possible.")

    # Normalize ids if they exist
    if turb_col:
        events[turb_col] = events[turb_col].astype(str).str.strip().str.replace(" ", "_")
    if farm_col:
        events[farm_col] = events[farm_col].astype(str).str.strip().str.replace(" ", "_")

    n_marked_prefault = 0
    n_marked_inevent = 0

    # iterate events and label scada
    for _, ev in events.iterrows():
        t0 = pd.to_datetime(ev[start_col], errors="coerce")
        t1 = pd.to_datetime(ev[end_col], errors="coerce") if end_col else pd.NaT
        if pd.isna(t0):
            continue

        pre_start = t0 - pd.Timedelta(hours=LEAD_HOURS)
        pre_end = t0

        mask = (scada["timestamp"] >= pre_start) & (scada["timestamp"] < pre_end)

        # optionally filter by turbine/farm
        if turb_col and (turb_col in events.columns) and ("turbine_id" in scada.columns):
            turb = str(ev[turb_col])
            if turb and turb != "nan":
                mask = mask & (scada["turbine_id"] == turb)

        if farm_col and (farm_col in events.columns) and ("farm_id" in scada.columns):
            farm = str(ev[farm_col])
            if farm and farm != "nan":
                mask = mask & (scada["farm_id"] == farm)

        scada.loc[mask, "label"] = 1
        n_marked_prefault += int(mask.sum())

        # in-event label if end exists
        if end_col and not pd.isna(t1):
            mask2 = (scada["timestamp"] >= t0) & (scada["timestamp"] <= t1)
            if turb_col and ("turbine_id" in scada.columns):
                turb = str(ev[turb_col])
                if turb and turb != "nan":
                    mask2 = mask2 & (scada["turbine_id"] == turb)
            if farm_col and ("farm_id" in scada.columns):
                farm = str(ev[farm_col])
                if farm and farm != "nan":
                    mask2 = mask2 & (scada["farm_id"] == farm)

            scada.loc[mask2, "label"] = 2
            n_marked_inevent += int(mask2.sum())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scada.to_parquet(OUT_PATH, index=False)

    logging.info(f"Saved labeled parquet: {OUT_PATH}")
    logging.info(f"Rows labeled pre-fault: {n_marked_prefault}")
    logging.info(f"Rows labeled in-event: {n_marked_inevent}")
    logging.info("Label distribution:")
    logging.info(scada["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
