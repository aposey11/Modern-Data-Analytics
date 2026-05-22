import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import itertools

print("🔄 Loading all prepared datasets...")
df_spatial = pd.read_parquet("site_spatial_features.parquet")
df_traffic = pd.read_parquet("bicycle_data_prepared.parquet")
df_weather = pd.read_parquet("raw_historical_weather.parquet")
df_accidents = pd.read_parquet("bike_accidents_filtered.parquet")

# ==========================================
# 1. CREATE THE MASTER TIME GRID
# ==========================================
print("📅 Building the Master Year-Month-Hour Grid...")
unique_sites = df_spatial['site_id'].unique()
years = range(2019, 2025)
months = range(1, 13)
hours = range(0, 24)

grid = list(itertools.product(unique_sites, years, months, hours))
df_master = pd.DataFrame(grid, columns=['site_id', 'Year', 'Month', 'Hour'])

# Cyclical Time Encoding
df_master['Hour_Sin'] = np.sin(2 * np.pi * df_master['Hour'] / 24)
df_master['Hour_Cos'] = np.cos(2 * np.pi * df_master['Hour'] / 24)
df_master['Month_Sin'] = np.sin(2 * np.pi * df_master['Month'] / 12)
df_master['Month_Cos'] = np.cos(2 * np.pi * df_master['Month'] / 12)

# ==========================================
# 2. PROCESS TARGETS (ACCIDENTS AS COUNTS)
# ==========================================
print("💥 Mapping accidents to targets (RAW COUNTS)...")
geometry_acc = [Point(xy) for xy in zip(df_accidents['MS_X_COORD'], df_accidents['MS_Y_COORD'])]
gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_acc, crs="EPSG:31370")

df_sites_geo = df_spatial[['site_id', 'site_x_lambert', 'site_y_lambert']].copy()
geometry_sites = [Point(xy) for xy in zip(df_sites_geo['site_x_lambert'], df_sites_geo['site_y_lambert'])]
gdf_sites = gpd.GeoDataFrame(df_sites_geo, geometry=geometry_sites, crs="EPSG:31370")
gdf_sites['geometry'] = gdf_sites.geometry.buffer(500)

acc_mapped = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")

# DT_TIME is already an integer (0-23)
acc_mapped['Hour'] = acc_mapped['DT_TIME']
acc_mapped = acc_mapped.dropna(subset=['DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'Hour'])

# Group by to get the RAW COUNT of accidents (No longer forcing to 1)
targets = acc_mapped.groupby(['site_id', 'DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'Hour']).size().reset_index()
targets = targets.rename(columns={'DT_YEAR_COLLISION': 'Year', 'DT_MONTH_COLLISION': 'Month', 0: 'Accident_Count'})

# ==========================================
# 3. PROCESS TRAFFIC FEATURES (EXTREMES)
# ==========================================
print("🚲 Aggregating Traffic features...")
df_traffic['van'] = pd.to_datetime(df_traffic['van'])
df_traffic['Year'] = df_traffic['van'].dt.year
df_traffic['Month'] = df_traffic['van'].dt.month
df_traffic['Day'] = df_traffic['van'].dt.day
df_traffic['Hour'] = df_traffic['van'].dt.hour

hourly_traffic = df_traffic.groupby(['site_id', 'Year', 'Month', 'Day', 'Hour']).agg(
    Traffic_Volume=('aantal', 'sum'),
    Traffic_Max_Spike=('aantal', 'max'),     
    Traffic_Volatility=('aantal', 'std')     
).reset_index()

hourly_traffic['Piek_Index'] = (hourly_traffic['Traffic_Max_Spike'] / (hourly_traffic['Traffic_Volume'] / 4)).fillna(0)
hourly_traffic['Traffic_Volatility'] = hourly_traffic['Traffic_Volatility'].fillna(0)

# AGGREGATE USING MAX FOR VOLATILITY AND SPIKES
monthly_traffic = hourly_traffic.groupby(['site_id', 'Year', 'Month', 'Hour']).agg(
    Avg_Traffic_Volume=('Traffic_Volume', 'mean'),
    Avg_Traffic_Max_Spike=('Traffic_Max_Spike', 'max'),   # <-- Changed to max
    Avg_Traffic_Volatility=('Traffic_Volatility', 'max'), # <-- Changed to max
    Avg_Piek_Index=('Piek_Index', 'max')                  # <-- Changed to max
).reset_index()

# ==========================================
# 4. PROCESS WEATHER FEATURES (EXTREMES)
# ==========================================
print("🌤️ Aggregating Weather features...")
df_weather = df_weather.sort_values(by=['site_id', 'timestamp'])
df_weather['Temp_Drop_1h'] = df_weather.groupby('site_id')['temperature_2m'].diff() * -1 
df_weather['Wind_Spike_1h'] = df_weather.groupby('site_id')['wind_speed_10m'].diff()

df_weather['Year'] = df_weather['timestamp'].dt.year
df_weather['Month'] = df_weather['timestamp'].dt.month
df_weather['Hour'] = df_weather['timestamp'].dt.hour

# AGGREGATE USING MAX FOR STORMS
monthly_weather = df_weather.groupby(['site_id', 'Year', 'Month', 'Hour']).agg(
    Avg_Temperature=('temperature_2m', 'mean'),
    Avg_Precipitation=('precipitation_1h', 'max'), # <-- Changed to max
    Avg_Wind_Speed=('wind_speed_10m', 'max'),      # <-- Changed to max
    Avg_Temp_Drop=('Temp_Drop_1h', 'max'),
    Avg_Wind_Spike=('Wind_Spike_1h', 'max')        # <-- Changed to max
).reset_index()

# ==========================================
# 5. MERGE EVERYTHING INTO MASTER DATASET
# ==========================================
print("🔗 Merging all pipelines into the Master DataFrame...")
df_master = df_master.merge(targets, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master['Accident_Count'] = df_master['Accident_Count'].fillna(0).astype(int)

df_master = df_master.merge(monthly_traffic, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master = df_master.merge(monthly_weather, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master = df_master.merge(df_spatial[['site_id', 'spatial_risk_score', 'historical_accidents_250m', 'historical_accidents_250m_to_500m']], 
                            on='site_id', how='left')

df_master = df_master.dropna(subset=['Avg_Traffic_Volume', 'Avg_Temperature'])

output_file = "master_training_data.parquet"
df_master.to_parquet(output_file)

print("\n🚀 SUCCESS! Feature Engineering Complete.")
print(f"Total Rows: {len(df_master):,}")
print(f"Total Accident Hours (Count > 0): {(df_master['Accident_Count'] > 0).sum():,}")
print(f"Saved to: {output_file}")