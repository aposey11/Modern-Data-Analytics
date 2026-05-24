import pandas as pd

# Filters the Statbel accident dataset (OPENDATA_MAP_2017-2024.xlsx) down to the
# subset relevant for the GTRI model: accidents that have valid Lambert 72 coordinates
# and involve at least one cyclist.
#
# The coordinate filter is a hard requirement — accidents without coordinates cannot
# be spatially matched to AWV sensor sites. The bike filter uses the road user type
# fields (TX_ROAD_USR_TYPE1_NL / TYPE2_NL), which Statbel populates for the primary
# and secondary road users involved in each collision.

print("Loading accident dataset...")
df = pd.read_excel('OPENDATA_MAP_2017-2024.xlsx')

# Drop rows with missing spatial coordinates. These cannot be assigned to a sensor
# site and are not usable for model training or evaluation.
df_filtered = df.dropna(subset=['MS_X_COORD', 'MS_Y_COORD'])

# Keep accidents where at least one road user type is recorded as a cyclist.
# Case-insensitive match handles the mixed capitalisation in the Statbel export.
condition_bike_type1 = df_filtered['TX_ROAD_USR_TYPE1_NL'].str.contains('Fiets', case=False, na=False)
condition_bike_type2 = df_filtered['TX_ROAD_USR_TYPE2_NL'].str.contains('Fiets', case=False, na=False)
df_bikes = df_filtered[condition_bike_type1 | condition_bike_type2]

print(f"Original accident records:                        {len(df):,}")
print(f"After coordinate filter:                          {len(df_filtered):,}")
print(f"After bike-involvement filter (final):            {len(df_bikes):,}")

output_file = 'gtri_bike_accidents_filtered.parquet'
df_bikes.to_parquet(output_file)
print(f"\nSaved filtered dataset to {output_file}")
