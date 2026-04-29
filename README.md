# ETA Challenge Submission

## Your final score
Dev MAE: **292.4 s**

## Your approach, in one paragraph
To beat the baseline, I engineered several temporal, spatial, and target-encoded features. For target encoding, I calculated the historical median trip duration for every unique `pickup_zone` to `dropoff_zone` pair from the 2023 training data. For temporal features, I added `minute`, `is_weekend`, and an `is_rush_hour` flag. Finally, for spatial features, I extracted the geographic centroids (latitude/longitude) of all 265 NYC taxi zones from the provided shapefile and calculated the straight-line Haversine distance between the pickup and dropoff points. The model is an `XGBRegressor` with `n_estimators=500` and `max_depth=9`.

## What you tried that didn't work
1. Relying solely on Pandas merges for coordinates at inference time proved to be too slow and would risk breaking the 200ms latency constraint, so I pre-computed everything into static JSON lookup dictionaries (`zone_coords.json` and `od_medians.json`) loaded at startup in `predict.py`.
2. Simple mean encoding for O-D pairs was too sensitive to long-tail outlier trips (e.g., rides left running), so median encoding was strictly necessary for robustness.

## Where AI tooling sped you up most
AI tooling was instrumental in the data pipelining and feature engineering loop. Specifically, creating the Python script to download the shapefile, extract it, re-project the CRS via `geopandas`, and map the centroids directly to a JSON file would normally require a lot of StackOverflow copy-pasting for the exact projection EPSG codes and spatial join syntax. AI handled this instantly. The AI agent also managed the heavy lifting of completely rewriting the inference `predict.py` to use bare numpy arrays with fast JSON dictionary lookups instead of slow Pandas dataframe constructions.

## Next experiments
If I kept going, I would use the OpenStreetMap network to calculate actual routing distance and expected travel time via OSRM, rather than just straight-line Haversine distance. I would also add NOAA weather data (precipitation/snow) mapped to the `requested_at` timestamp.

## How to reproduce
```bash
# 1. Setup Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install geopandas requests

# 2. Download Data
python data/download_data.py

# 3. Build Spatial Coordinates
python build_coords.py

# 4. Train Model and Export Features
python baseline.py

# 5. Evaluate
python grade.py
```

_Total time spent on this challenge: ~2 hours._
