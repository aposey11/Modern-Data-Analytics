import pandas as pd

# Load the parquet file into a DataFrame
df = pd.read_parquet('bike_accidents_filtered.parquet')

# Print the column headers as a list
print(df.columns.tolist())