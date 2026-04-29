"""Submission interface — this is what Gobblecube's grader imports."""

from __future__ import annotations

import pickle
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_MEDIANS_PATH = Path(__file__).parent / "od_medians.json"
_COORDS_PATH = Path(__file__).parent / "zone_coords.json"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

with open(_MEDIANS_PATH, "r") as _f:
    _MEDIANS_DICT = json.load(_f)
    
with open(_COORDS_PATH, "r") as _f:
    _COORDS = json.load(_f)
    
_GLOBAL_MEDIAN = _MEDIANS_DICT["global_median"]
_OD_MEDIANS = _MEDIANS_DICT["medians"]

if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    return 6367 * 2 * math.asin(math.sqrt(a))

def predict(request: dict) -> float:
    """Predict trip duration in seconds."""
    ts = datetime.fromisoformat(request["requested_at"])
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    
    hour = ts.hour
    minute = ts.minute
    dow = ts.weekday()
    month = ts.month
    
    is_weekend = 1 if dow in (5, 6) else 0
    is_rush_hour = 1 if (dow < 5 and ((7 <= hour <= 10) or (16 <= hour <= 19))) else 0
    
    key = f"{pz}_{dz}"
    historical_median = _OD_MEDIANS.get(key, _GLOBAL_MEDIAN)
    
    p_coord = _COORDS.get(str(pz), {"lat": 40.7128, "lon": -74.0060})
    d_coord = _COORDS.get(str(dz), {"lat": 40.7128, "lon": -74.0060})
    dist = haversine(p_coord["lon"], p_coord["lat"], d_coord["lon"], d_coord["lat"])
    
    x = np.array(
        [[
            pz,
            dz,
            hour,
            minute,
            dow,
            month,
            int(request["passenger_count"]),
            is_weekend,
            is_rush_hour,
            historical_median,
            dist
        ]],
        dtype=np.float32,
    )
    return float(_MODEL.predict(x)[0])
