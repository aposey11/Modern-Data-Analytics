import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import itertools

# This script builds the master training dataset for the GTRI model.
# Features are computed in two stages: first at the actual day-hour level (to get
# genuine temporal lag signals), then aggregated to month-hour (to match the
# granularity of the Statbel accident target, which carries no day-of-month info).
#
# Weather:
#   Rain_Lag1h_Mean / Rain_Lag1h_Max  -- genuine previous-hour precipitation, obtained
#     by shifting the hourly weather series within each site before aggregating to
#     month-hour. This is the real "rain in the hour before" signal.
#   Rain_Surge -- log-difference formulation: log1p(current_precip) - log1p(lag1h_precip),
#     computed at each actual day-hour and aggregated (max) to month-hour.
#     Positive values mean rain is intensifying; negative means it is easing.
#     The log-difference is bounded by construction (roughly -3 to +3 for typical
#     precipitation ranges) and avoids the saturation issue of a raw ratio when the
#     previous hour is dry.
#   Rain_Surge_Ratio is retained alongside Rain_Surge. It is a cross-day measure
#     (max vs mean across days sharing the same month-hour slot), so it captures a
#     different aspect of rain character than the within-day temporal onset signal.
#
# Traffic:
#   Traffic_Volume_Delta_Mean -- mean of (current-hour volume minus previous-hour volume)
#     across all days in a given month-hour slot. Positive = traffic building into this
#     hour; negative = traffic declining.

print("Loading all prepared datasets...")
df_spatial   = pd.read_parquet("gtri_site_spatial_features.parquet")
df_traffic   = pd.read_parquet("gtri_bicycle_data_prepared.parquet")
df_weather   = pd.read_parquet("gtri_raw_historical_weather.parquet")
df_accidents = pd.read_parquet("gtri_bike_accidents_filtered.parquet")

# ==========================================
# 1. MASTER TIME GRID
# ==========================================
print("Building the master year-month-hour grid...")
unique_sites = df_spatial['site_id'].unique()
years  = range(2019, 2025)
months = range(1, 13)
hours  = range(0, 24)

grid      = list(itertools.product(unique_sites, years, months, hours))
df_master = pd.DataFrame(grid, columns=['site_id', 'Year', 'Month', 'Hour'])

# Cyclical encoding ensures the model sees hour 23 and hour 0 as adjacent,
# and December and January as adjacent. Both sin and cos are needed for a complete
# circular representation — dropping one breaks the pair.
df_master['Hour_Sin']  = np.sin(2 * np.pi * df_master['Hour']  / 24)
df_master['Hour_Cos']  = np.cos(2 * np.pi * df_master['Hour']  / 24)
df_master['Month_Sin'] = np.sin(2 * np.pi * df_master['Month'] / 12)
df_master['Month_Cos'] = np.cos(2 * np.pi * df_master['Month'] / 12)

# ==========================================
# 2. ACCIDENT TARGETS
# ==========================================
print("Mapping accidents to the site-year-month-hour grid...")
geometry_acc  = [Point(xy) for xy in zip(df_accidents['MS_X_COORD'], df_accidents['MS_Y_COORD'])]
gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_acc, crs="EPSG:31370")

df_sites_geo   = df_spatial[['site_id', 'site_x_lambert', 'site_y_lambert']].copy()
geometry_sites = [Point(xy) for xy in zip(df_sites_geo['site_x_lambert'], df_sites_geo['site_y_lambert'])]
gdf_sites      = gpd.GeoDataFrame(df_sites_geo, geometry=geometry_sites, crs="EPSG:31370")
gdf_sites['geometry'] = gdf_sites.geometry.buffer(500)

acc_mapped = gpd.sjoin(gdf_accidents, gdf_sites, how="inner", predicate="intersects")
acc_mapped['Hour'] = acc_mapped['DT_TIME']
acc_mapped = acc_mapped.dropna(subset=['DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'Hour'])

targets = acc_mapped.groupby(
    ['site_id', 'DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'Hour']
).size().reset_index()
targets = targets.rename(columns={
    'DT_YEAR_COLLISION': 'Year', 'DT_MONTH_COLLISION': 'Month', 0: 'Accident_Count'
})

# ==========================================
# 3. TRAFFIC FEATURES  (day-hour intermediate layer)
# ==========================================
print("Aggregating traffic features with volume delta...")
df_traffic['van']   = pd.to_datetime(df_traffic['van'])
df_traffic['Year']  = df_traffic['van'].dt.year
df_traffic['Month'] = df_traffic['van'].dt.month
df_traffic['Day']   = df_traffic['van'].dt.day
df_traffic['Hour']  = df_traffic['van'].dt.hour

# Aggregate to the actual day-hour level first, before computing deltas.
# This preserves the genuine temporal ordering that gets lost after month-hour aggregation.
hourly_by_day = df_traffic.groupby(['site_id', 'Year', 'Month', 'Day', 'Hour']).agg(
    Traffic_Volume=('aantal', 'sum'),
    Traffic_Max_Spike=('aantal', 'max'),
    Traffic_Volatility=('aantal', 'std')
).reset_index()

hourly_by_day['Piek_Index']         = (hourly_by_day['Traffic_Max_Spike'] / (hourly_by_day['Traffic_Volume'] / 4)).fillna(0)
hourly_by_day['Traffic_Volatility'] = hourly_by_day['Traffic_Volatility'].fillna(0)

hourly_by_day['_date']      = pd.to_datetime(hourly_by_day[['Year', 'Month', 'Day']].rename(
    columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}))
hourly_by_day['is_weekend'] = hourly_by_day['_date'].dt.dayofweek >= 5
hourly_by_day.drop(columns=['_date'], inplace=True)

# Traffic volume delta: how much did cycling volume change from the previous hour on the
# same calendar day? A positive mean delta signals traffic building into this hour
# (e.g. start of morning rush); negative signals it is declining.
hourly_by_day = hourly_by_day.sort_values(['site_id', 'Year', 'Month', 'Day', 'Hour'])
hourly_by_day['Traffic_Lag1h']        = hourly_by_day.groupby('site_id')['Traffic_Volume'].shift(1)
hourly_by_day['Traffic_Volume_Delta'] = hourly_by_day['Traffic_Volume'] - hourly_by_day['Traffic_Lag1h']
hourly_by_day['Traffic_Volume_Delta'] = hourly_by_day['Traffic_Volume_Delta'].fillna(0)

# Aggregate to month-hour. Using max for volatility and peak index preserves the
# worst-case character within each slot; mean volume is the representative load.
monthly_traffic = hourly_by_day.groupby(['site_id', 'Year', 'Month', 'Hour']).agg(
    Avg_Traffic_Volume=('Traffic_Volume', 'mean'),
    Avg_Traffic_Max_Spike=('Traffic_Max_Spike', 'max'),
    Avg_Traffic_Volatility=('Traffic_Volatility', 'max'),
    Avg_Piek_Index=('Piek_Index', 'max'),
    Traffic_Volume_Delta_Mean=('Traffic_Volume_Delta', 'mean'),
).reset_index()

# Weekend ratio: fraction of volume that falls on weekend days within each month-hour slot.
# This gives the model a day-type signal without requiring day-level accident targets.
weekend_agg = hourly_by_day.groupby(['site_id', 'Year', 'Month', 'Hour', 'is_weekend']).agg(
    Avg_Volume_DayType=('Traffic_Volume', 'mean')
).reset_index()

weekend_pivot = weekend_agg.pivot_table(
    index=['site_id', 'Year', 'Month', 'Hour'],
    columns='is_weekend',
    values='Avg_Volume_DayType',
    fill_value=0
).reset_index()
weekend_pivot.columns.name = None
weekend_pivot = weekend_pivot.rename(columns={False: 'Avg_Volume_Weekday', True: 'Avg_Volume_Weekend'})

monthly_traffic = monthly_traffic.merge(weekend_pivot, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
monthly_traffic['Avg_Volume_Weekend'] = monthly_traffic.get('Avg_Volume_Weekend', 0).fillna(0)
monthly_traffic['Avg_Volume_Weekday'] = monthly_traffic.get('Avg_Volume_Weekday', 0).fillna(0)

total_vol = monthly_traffic['Avg_Volume_Weekday'] + monthly_traffic['Avg_Volume_Weekend']
monthly_traffic['Traffic_Weekend_Ratio'] = (
    monthly_traffic['Avg_Volume_Weekend'] / total_vol.replace(0, np.nan)
).fillna(0.5)
monthly_traffic.drop(columns=['Avg_Volume_Weekday', 'Avg_Volume_Weekend'], inplace=True)

# ==========================================
# 4. WEATHER FEATURES  (day-hour intermediate layer)
# ==========================================
print("Aggregating weather features with genuine hourly lags...")

# Sort chronologically within each site before shifting, so lag(1) always refers to
# the actual preceding clock hour — including across midnight boundaries.
df_weather = df_weather.sort_values(by=['site_id', 'timestamp']).reset_index(drop=True)

df_weather['Temp_Drop_1h']  = df_weather.groupby('site_id')['temperature_2m'].diff() * -1
df_weather['Wind_Spike_1h'] = df_weather.groupby('site_id')['wind_speed_10m'].diff()

# Rain lag: the precipitation recorded in the hour immediately preceding this one.
# At midnight (hour 0) the shift correctly pulls from hour 23 of the previous day.
df_weather['Rain_Lag1h'] = df_weather.groupby('site_id')['precipitation_1h'].shift(1).fillna(0)
df_weather['Rain_Lag2h'] = df_weather.groupby('site_id')['precipitation_1h'].shift(2).fillna(0)

# Rain_Surge using log-difference rather than a raw ratio. The log-difference is
# naturally bounded (approximately -3 to +3 for precipitation values up to 20 mm/h)
# and avoids saturation when the previous hour is dry. Positive = intensifying;
# negative = easing; zero = steady.
df_weather['Rain_Surge_daily'] = (
    np.log1p(df_weather['precipitation_1h']) - np.log1p(df_weather['Rain_Lag1h'])
)

df_weather['Year']  = df_weather['timestamp'].dt.year
df_weather['Month'] = df_weather['timestamp'].dt.month
df_weather['Hour']  = df_weather['timestamp'].dt.hour

# Aggregate to month-hour. Max precipitation and max surge capture the worst-case
# event in a given slot; mean lag represents the typical prior-hour wetness.
monthly_weather = df_weather.groupby(['site_id', 'Year', 'Month', 'Hour']).agg(
    Avg_Temperature=('temperature_2m', 'mean'),
    Avg_Precipitation=('precipitation_1h', 'max'),
    Rain_Mean_Intensity=('precipitation_1h', 'mean'),
    Rain_Lag1h_Mean=('Rain_Lag1h', 'mean'),
    Rain_Lag1h_Max=('Rain_Lag1h', 'max'),
    Rain_Surge=('Rain_Surge_daily', 'max'),
    Avg_Wind_Speed=('wind_speed_10m', 'max'),
    Avg_Temp_Drop=('Temp_Drop_1h', 'max'),
    Avg_Wind_Spike=('Wind_Spike_1h', 'max')
).reset_index()

# Rain_Surge_Ratio is a cross-day measure: max precipitation in the slot relative to
# the mean. It captures the spikiness of rain events across different days sharing the
# same month-hour, and is a different signal from the within-day temporal onset above.
monthly_weather['Rain_Surge_Ratio'] = (
    monthly_weather['Avg_Precipitation'] / (monthly_weather['Rain_Mean_Intensity'] + 1e-6)
).clip(upper=20.0)

# ==========================================
# 5. MERGE INTO MASTER DATASET
# ==========================================
print("Merging all pipelines into the master dataframe...")
df_master = df_master.merge(targets,         on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master['Accident_Count'] = df_master['Accident_Count'].fillna(0).astype(int)

df_master = df_master.merge(monthly_traffic, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master = df_master.merge(monthly_weather, on=['site_id', 'Year', 'Month', 'Hour'], how='left')
df_master = df_master.merge(
    df_spatial[['site_id', 'spatial_risk_score',
                'historical_accidents_250m', 'historical_accidents_250m_to_500m']],
    on='site_id', how='left'
)

# Drop rows where the main feature sources are absent (sites or months outside coverage).
df_master = df_master.dropna(subset=['Avg_Traffic_Volume', 'Avg_Temperature'])

output_file = "gtri_master_training_data.parquet"
df_master.to_parquet(output_file)

print("\nFeature engineering complete.")
print(f"Total rows:                              {len(df_master):,}")
print(f"Accident hours (Accident_Count > 0):     {(df_master['Accident_Count'] > 0).sum():,}")
print(f"Traffic_Volume_Delta_Mean  -- mean:      {df_master['Traffic_Volume_Delta_Mean'].mean():.2f}")
print(f"Rain_Lag1h_Mean            -- mean:      {df_master['Rain_Lag1h_Mean'].mean():.4f} mm/h")
print(f"Rain_Surge              -- mean:      {df_master['Rain_Surge'].mean():.4f}")
print(f"Rain_Surge              -- range:     [{df_master['Rain_Surge'].min():.2f}, {df_master['Rain_Surge'].max():.2f}]")
print(f"Saved to: {output_file}")
