#!/usr/bin/env python
"""Baseline: gradient-boosted trees on engineered features.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` and `od_medians.json` 
which `predict.py` loads at inference.
"""

from __future__ import annotations

import pickle
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
MEDIANS_PATH = Path(__file__).parent / "od_medians.json"
COORDS_PATH = Path(__file__).parent / "zone_coords.json"

FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "minute", "dow", "month", 
    "passenger_count", "is_weekend", "is_rush_hour", "historical_median",
    "haversine_dist"
]

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))

def engineer_features(df: pd.DataFrame, od_medians: pd.DataFrame, global_median: float, coords: dict) -> pd.DataFrame:
    """Turn raw request columns into model features."""
    df = df.merge(od_medians, on=["pickup_zone", "dropoff_zone"], how="left")
    df["historical_median"] = df["historical_median"].fillna(global_median).astype("float32")
    
    ts = pd.to_datetime(df["requested_at"])
    
    df["pickup_zone"] = df["pickup_zone"].astype("int32")
    df["dropoff_zone"] = df["dropoff_zone"].astype("int32")
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["passenger_count"] = df["passenger_count"].astype("int8")
    
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")
    
    is_rush_morning = (df["hour"] >= 7) & (df["hour"] <= 10) & (df["dow"] < 5)
    is_rush_evening = (df["hour"] >= 16) & (df["hour"] <= 19) & (df["dow"] < 5)
    df["is_rush_hour"] = (is_rush_morning | is_rush_evening).astype("int8")
    
    coords_df = pd.DataFrame.from_dict(coords, orient="index")
    coords_df.index = coords_df.index.astype(int)
    
    df = df.merge(coords_df, left_on="pickup_zone", right_index=True, how="left")
    df.rename(columns={"lat": "pickup_lat", "lon": "pickup_lon"}, inplace=True)
    df = df.merge(coords_df, left_on="dropoff_zone", right_index=True, how="left")
    df.rename(columns={"lat": "dropoff_lat", "lon": "dropoff_lon"}, inplace=True)
    
    # Fill missing coordinates with center of NYC roughly
    df["pickup_lat"] = df["pickup_lat"].fillna(40.7128)
    df["pickup_lon"] = df["pickup_lon"].fillna(-74.0060)
    df["dropoff_lat"] = df["dropoff_lat"].fillna(40.7128)
    df["dropoff_lon"] = df["dropoff_lon"].fillna(-74.0060)
    
    df["haversine_dist"] = haversine_np(
        df["pickup_lon"].to_numpy(), df["pickup_lat"].to_numpy(),
        df["dropoff_lon"].to_numpy(), df["dropoff_lat"].to_numpy()
    ).astype("float32")
    
    return df[FEATURES]

def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("Computing target encoding...")
    od_medians_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index()
    od_medians_df.rename(columns={"duration_seconds": "historical_median"}, inplace=True)
    global_median = float(train["duration_seconds"].median())
    
    with open(MEDIANS_PATH, "w") as f:
        encoding_dict = {"global_median": global_median, "medians": {}}
        for _, row in od_medians_df.iterrows():
            encoding_dict["medians"][f"{int(row['pickup_zone'])}_{int(row['dropoff_zone'])}"] = float(row["historical_median"])
        json.dump(encoding_dict, f)
        
    with open(COORDS_PATH, "r") as f:
        coords = json.load(f)

    print("Engineering features...")
    X_train = engineer_features(train, od_medians_df, global_median, coords)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev, od_medians_df, global_median, coords)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
