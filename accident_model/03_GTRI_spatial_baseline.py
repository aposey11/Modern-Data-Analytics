import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

# Spatial features for the GTRI model are computed here: KDE-based risk scores and
# historical accident counts within 250m and 500m of each AWV sensor site.
#
# All accident data is restricted to the training period (2019-2023) to prevent
# information from the 2024 test year leaking into the spatial baseline.

TRAIN_END_YEAR = 2023

print("Loading datasets...")
df_bikes = pd.read_parquet("gtri_bicycle_data_prepared.parquet")
df_accidents_all = pd.read_parquet("gtri_bike_accidents_filtered.parquet")

# Keep only accidents from the training period. The 2024 data is reserved for
# model evaluation and must not influence the spatial prior.
print(f"Filtering accidents to training period (up to {TRAIN_END_YEAR})...")
df_accidents = df_accidents_all[df_accidents_all['DT_YEAR_COLLISION'] <= TRAIN_END_YEAR].copy()
n_all   = len(df_accidents_all)
n_train = len(df_accidents)
print(f"Total accidents: {n_all:,}  |  Training-period: {n_train:,}  |  Held out: {n_all - n_train:,}")

print("Extracting unique AWV bike sensor sites...")
df_sites  = df_bikes[['site_id', 'long', 'lat', 'naam_site']].drop_duplicates().dropna(subset=['long', 'lat'])

# Site coordinates arrive as WGS84; reproject to Belgian Lambert 72 so all distance
# calculations are in metres.
geometry_sites = [Point(xy) for xy in zip(df_sites['long'], df_sites['lat'])]
gdf_sites      = gpd.GeoDataFrame(df_sites, geometry=geometry_sites, crs="EPSG:4326")
gdf_sites      = gdf_sites.to_crs("EPSG:31370")
gdf_sites['site_x_lambert'] = gdf_sites.geometry.x
gdf_sites['site_y_lambert'] = gdf_sites.geometry.y

# Accident coordinates are already in Lambert 72 (MS_X_COORD / MS_Y_COORD from Statbel).
print("Preparing accident geometries (training period only)...")
geometry_acc  = [Point(xy) for xy in zip(df_accidents['MS_X_COORD'], df_accidents['MS_Y_COORD'])]
gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_acc, crs="EPSG:31370")

# Fit a Gaussian KDE on accident locations. The 500m bandwidth reflects the spatial
# resolution of the AWV network — close enough to capture hotspot structure without
# smoothing across unrelated intersections.
print("Fitting KDE on training-period accidents (bandwidth=500m, Gaussian kernel)...")
kde             = KernelDensity(bandwidth=500.0, kernel='gaussian')
accident_coords = np.vstack([gdf_accidents.geometry.x, gdf_accidents.geometry.y]).T
kde.fit(accident_coords)

site_coords = np.vstack([gdf_sites['site_x_lambert'], gdf_sites['site_y_lambert']]).T
log_density = kde.score_samples(site_coords)
density     = np.exp(log_density)

scaler = MinMaxScaler()
gdf_sites['spatial_risk_score'] = scaler.fit_transform(density.reshape(-1, 1))

# Count accidents within two concentric buffer zones around each sensor.
# 250m captures the immediate intersection risk; subtracting it from the 500m total
# gives the donut-ring count for the surrounding neighbourhood.
print("Computing 250m and 500m hard buffer accident counts (training period only)...")
gdf_sites['buffer_250m'] = gdf_sites.geometry.buffer(250)
gdf_sites['buffer_500m'] = gdf_sites.geometry.buffer(500)

gdf_sites.set_geometry('buffer_250m', inplace=True)
sjoin_250  = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_250 = sjoin_250.groupby('site_id').size().reset_index(name='historical_accidents_250m')

gdf_sites.set_geometry('buffer_500m', inplace=True)
sjoin_500  = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
counts_500 = sjoin_500.groupby('site_id').size().reset_index(name='total_500m')

gdf_sites.set_geometry('geometry', inplace=True)
gdf_sites = gdf_sites.merge(counts_250, on='site_id', how='left').merge(counts_500, on='site_id', how='left')

gdf_sites['historical_accidents_250m']         = gdf_sites['historical_accidents_250m'].fillna(0).astype(int)
gdf_sites['total_500m']                         = gdf_sites['total_500m'].fillna(0).astype(int)
gdf_sites['historical_accidents_250m_to_500m'] = gdf_sites['total_500m'] - gdf_sites['historical_accidents_250m']

gdf_sites.drop(columns=['total_500m', 'geometry', 'buffer_250m', 'buffer_500m'], inplace=True)
df_final_spatial = pd.DataFrame(gdf_sites)

output_file = "gtri_site_spatial_features.parquet"
df_final_spatial.to_parquet(output_file)

print(f"\nSaved spatial baseline features to {output_file}")
print(f"Sites with >=1 accident within 250m  (train): {(df_final_spatial['historical_accidents_250m'] > 0).sum()}")
print(f"Sites with >=1 accident in donut ring (train): {(df_final_spatial['historical_accidents_250m_to_500m'] > 0).sum()}")
print(f"KDE spatial_risk_score range: [{df_final_spatial['spatial_risk_score'].min():.4f}, {df_final_spatial['spatial_risk_score'].max():.4f}]")
