import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("🔄 Loading data for buffer comparison...")
# Load datasets
df_bikes = pd.read_parquet("bicycle_data_prepared.parquet")
df_accidents = pd.read_parquet("bike_accidents_filtered.parquet")

# Prepare Sites
df_sites = df_bikes[['site_id', 'long', 'lat']].drop_duplicates().dropna(subset=['long', 'lat'])
geometry_sites = [Point(xy) for xy in zip(df_sites['long'], df_sites['lat'])]
gdf_sites = gpd.GeoDataFrame(df_sites, geometry=geometry_sites, crs="EPSG:4326").to_crs("EPSG:31370")

# Prepare Accidents
geometry_acc = [Point(xy) for xy in zip(df_accidents['MS_X_COORD'], df_accidents['MS_Y_COORD'])]
gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_acc, crs="EPSG:31370")

# --- Test 250m Buffer ---
gdf_sites['buffer_250m'] = gdf_sites.geometry.buffer(250)
gdf_sites.set_geometry('buffer_250m', inplace=True)
acc_250m = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_250 = acc_250m.groupby('site_id').size()

# --- Test 500m Buffer ---
gdf_sites.set_geometry('geometry', inplace=True) # Reset geometry back to points
gdf_sites['buffer_500m'] = gdf_sites.geometry.buffer(500)
gdf_sites.set_geometry('buffer_500m', inplace=True)
acc_500m = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_500 = acc_500m.groupby('site_id').size()

# --- Merge and Compare ---
df_compare = df_sites[['site_id']].copy()
df_compare['accidents_250m'] = df_compare['site_id'].map(counts_250).fillna(0).astype(int)
df_compare['accidents_500m'] = df_compare['site_id'].map(counts_500).fillna(0).astype(int)

# --- Print Diagnostics ---
total_sites = len(df_compare)

zero_acc_250 = (df_compare['accidents_250m'] == 0).sum()
zero_acc_500 = (df_compare['accidents_500m'] == 0).sum()

print("\n📊 --- BUFFER RADIUS COMPARISON --- 📊")
print(f"Total Bike Counter Sites: {total_sites}")
print("-" * 40)
print(f"🔹 250m Radius:")
print(f"   - Sites with 0 accidents: {zero_acc_250} ({round(zero_acc_250/total_sites*100, 1)}%)")
print(f"   - Max accidents at a single site: {df_compare['accidents_250m'].max()}")
print(f"   - Average accidents per site: {round(df_compare['accidents_250m'].mean(), 2)}")
print("-" * 40)
print(f"🔹 500m Radius:")
print(f"   - Sites with 0 accidents: {zero_acc_500} ({round(zero_acc_500/total_sites*100, 1)}%)")
print(f"   - Max accidents at a single site: {df_compare['accidents_500m'].max()}")
print(f"   - Average accidents per site: {round(df_compare['accidents_500m'].mean(), 2)}")