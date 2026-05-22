import pandas as pd


df = pd.read_excel('OPENDATA_MAP_2017-2024.xlsx', nrows=0)

print(df.columns.tolist())

import pandas as pd

print("🔄 Loading the dataset...")
# 1. Load the Excel file
df = pd.read_excel('OPENDATA_MAP_2017-2024.xlsx')

# 2. Filter out rows with missing coordinates
# dropna() removes rows where the specified columns have missing (NaN) values
df_filtered = df.dropna(subset=['MS_X_COORD', 'MS_Y_COORD'])

# 3. Filter to keep ONLY accidents involving a bike ("Fiets")
# We check if "Fiets" is present in either TYPE1 or TYPE2. 
# na=False ensures that if a cell is empty, it doesn't cause an error and is just treated as False.
# case=False makes the search case-insensitive (matches "fiets", "Fiets", "FIETS").
condition_bike_type1 = df_filtered['TX_ROAD_USR_TYPE1_NL'].str.contains('Fiets', case=False, na=False)
condition_bike_type2 = df_filtered['TX_ROAD_USR_TYPE2_NL'].str.contains('Fiets', case=False, na=False)

# Keep rows where condition 1 OR (|) condition 2 is True
df_bikes = df_filtered[condition_bike_type1 | condition_bike_type2]

print(f"✅ Filtering complete!")
print(f"Original number of accidents: {len(df)}")
print(f"Accidents with coordinates and involving bikes: {len(df_bikes)}")

# 4. Save the cleaned dataset to a new file
# Saving as parquet is much faster and uses less disk space, but you can also save as CSV or Excel
output_file = 'bike_accidents_filtered.parquet'
df_bikes.to_parquet(output_file)
# df_bikes.to_csv('bike_accidents_filtered.csv', index=False) # Uncomment to save as CSV

print(f"🚀 Saved the filtered dataset to {output_file}")