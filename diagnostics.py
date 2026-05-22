import pandas as pd

# Load the filtered accidents
df_accidents = pd.read_parquet("bike_accidents_filtered.parquet")

# Print the first 5 rows of the time columns and their exact data types
print("👀 DATA PREVIEW:")
print(df_accidents[['DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'DT_TIME']].head())

print("\n🧬 DATA TYPES:")
print(df_accidents[['DT_YEAR_COLLISION', 'DT_MONTH_COLLISION', 'DT_TIME']].dtypes)