import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

print("🔄 Loading datasets...")
df_bikes = pd.read_parquet("bicycle_data_prepared.parquet")
df_accidents = pd.read_parquet("bike_accidents_filtered.parquet")

print("📍 Extracting unique bike sites...")
df_sites = df_bikes[['site_id', 'long', 'lat', 'naam_site']].drop_duplicates().dropna(subset=['long', 'lat'])

# Convert sites to a GeoDataFrame (Initially WGS84 - EPSG:4326)
geometry_sites = [Point(xy) for xy in zip(df_sites['long'], df_sites['lat'])]
gdf_sites = gpd.GeoDataFrame(df_sites, geometry=geometry_sites, crs="EPSG:4326")

# Reproject sites to Belgian Lambert 72 (EPSG:31370) so measurements are in METERS
gdf_sites = gdf_sites.to_crs("EPSG:31370")
gdf_sites['site_x_lambert'] = gdf_sites.geometry.x
gdf_sites['site_y_lambert'] = gdf_sites.geometry.y

print("💥 Preparing accident geometries in Lambert 72...")
geometry_acc = [Point(xy) for xy in zip(df_accidents['MS_X_COORD'], df_accidents['MS_Y_COORD'])]
gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_acc, crs="EPSG:31370")

print("🧠 Calculating Continuous Spatial Risk Score (KDE)...")
kde = KernelDensity(bandwidth=500.0, kernel='gaussian')
accident_coords = np.vstack([gdf_accidents.geometry.x, gdf_accidents.geometry.y]).T
kde.fit(accident_coords)

site_coords = np.vstack([gdf_sites['site_x_lambert'], gdf_sites['site_y_lambert']]).T
log_density = kde.score_samples(site_coords)
density = np.exp(log_density)
scaler = MinMaxScaler()
gdf_sites['spatial_risk_score'] = scaler.fit_transform(density.reshape(-1, 1))

# --- Calculate Multi-Scale Buffers ---
print("📏 Calculating 250m and 500m hard buffer counts...")
gdf_sites['buffer_250m'] = gdf_sites.geometry.buffer(250)
gdf_sites['buffer_500m'] = gdf_sites.geometry.buffer(500)

# Join 250m (Micro-risk)
gdf_sites.set_geometry('buffer_250m', inplace=True)
sjoin_250 = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_250 = sjoin_250.groupby('site_id').size().reset_index(name='historical_accidents_250m')

# Join 500m (Total overlapping neighborhood risk)
gdf_sites.set_geometry('buffer_500m', inplace=True)
sjoin_500 = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_500 = sjoin_500.groupby('site_id').size().reset_index(name='total_500m')

# Merge counts back to the points data frame
gdf_sites.set_geometry('geometry', inplace=True)
gdf_sites = gdf_sites.merge(counts_250, on='site_id', how='left').merge(counts_500, on='site_id', how='left')

# Fill missing values with 0
gdf_sites['historical_accidents_250m'] = gdf_sites['historical_accidents_250m'].fillna(0).astype(int)
gdf_sites['total_500m'] = gdf_sites['total_500m'].fillna(0).astype(int)

# --- DE-NESTING LOGIC (The Donut Ring) ---
# Subtract the inner 250m circle from the total 500m circle
gdf_sites['historical_accidents_250m_to_500m'] = gdf_sites['total_500m'] - gdf_sites['historical_accidents_250m']

# Drop the overlapping column
gdf_sites.drop(columns=['total_500m', 'geometry', 'buffer_250m', 'buffer_500m'], inplace=True)
df_final_spatial = pd.DataFrame(gdf_sites)

output_file = "site_spatial_features.parquet"
df_final_spatial.to_parquet(output_file)
print(f"🚀 Saved spatial baseline features to {output_file}!")