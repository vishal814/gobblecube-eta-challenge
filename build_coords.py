import urllib.request
import zipfile
import io
import json
import geopandas as gpd
from pathlib import Path

def main():
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    print("Downloading shapefile...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall("data/taxi_zones")

    print("Reading shapefile...")
    gdf = gpd.read_file("data/taxi_zones/taxi_zones/taxi_zones.shp")
    gdf = gdf.to_crs(epsg=4326)

    coords = {}
    for _, row in gdf.iterrows():
        zone_id = int(row['LocationID'])
        centroid = row.geometry.centroid
        coords[zone_id] = {"lat": float(centroid.y), "lon": float(centroid.x)}

    with open("zone_coords.json", "w") as f:
        json.dump(coords, f)

    print("Saved zone_coords.json")

if __name__ == "__main__":
    main()
