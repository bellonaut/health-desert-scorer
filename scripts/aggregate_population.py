import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

# Paths
lga_path = "data/raw/nigeria_lga.geojson"
raster_path = "data/raw/nga_ppp_2020_constrained.tif"
out_path = "data/raw/population_lga.csv"

print("Loading LGAs...")
lgas = gpd.read_file(lga_path)

with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    raster_nodata = src.nodata
    raster_dtype = src.dtypes[0]
    print(f"Raster CRS: {raster_crs} | dtype: {raster_dtype} | nodata: {raster_nodata}")

if lgas.crs != raster_crs:
    print(f"Reprojecting LGAs from {lgas.crs} to {raster_crs}")
    lgas = lgas.to_crs(raster_crs)

print("Running zonal population sum (this may take 1–2 minutes)...")

zs = zonal_stats(
    lgas,
    raster_path,
    stats=["sum"],
    nodata=raster_nodata,      # <<< CRITICAL FIX
    all_touched=True
)

pop = [z["sum"] if z["sum"] is not None else 0 for z in zs]
# hard guard (shouldn’t be needed after nodata fix, but keeps pipeline sane)
pop = [0 if (p is None or (isinstance(p, float) and np.isnan(p)) or p < 0) else p for p in pop]

lgas["population"] = pop

out = lgas[["NAME_2", "NAME_1", "population"]].rename(columns={
    "NAME_2": "lga_name",
    "NAME_1": "state_name"
})

out["population"] = out["population"].round().astype("int64")
out.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(out))
print(out.population.describe())
print(out.head())
