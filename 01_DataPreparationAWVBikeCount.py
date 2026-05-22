import pandas as pd
import glob
from pathlib import Path

# 1. Settings and column definitions
data_path = Path("./") 

# Define column names based on the provided structures
# Renamed 'naam' to 'naam_site' in sites and 'naam_richting' in richtingen to avoid column name collisions
sites_columns = ['site_id', 'site_nr', 'long', 'lat', 'naam_site', 'domein', 'wegnr', 'district', 'gemeente', 'interval', 'datum_van']
richtingen_columns = ['site_id', 'richting', 'naam_richting']
data_columns = ['site_id', 'richting', 'type', 'van', 'tot', 'aantal']

print("🔄 Loading sites.csv and richtingen.csv (without headers)...")

# 2. Load sites and directions files
# Using header=None and assigning the specified columns manually
df_sites = pd.read_csv(data_path / "sites.csv", header=None, names=sites_columns)
df_richtingen = pd.read_csv(data_path / "richtingen.csv", header=None, names=richtingen_columns)

# 3. Find and process all monthly data files
data_files = sorted(list(data_path.glob("data-*.csv")))
data_list = []

print(f"🔄 Starting processing of {len(data_files)} data files...")

for f in data_files:
    # Load data without header and assign column names
    df = pd.read_csv(f, header=None, names=data_columns)
    data_list.append(df)
    print(f"✅ Loaded: {f.name}")

# Concatenate all monthly data files into one single DataFrame
if data_list:
    df_data_master = pd.concat(data_list, ignore_index=True)
else:
    df_data_master = pd.DataFrame(columns=data_columns)
    print("⚠️ No data files found. Please make sure they follow the 'data-<year>-<month>.csv' naming convention.")

# 4. Merge datasets
print("🔄 Merging datasets...")
# Merge the master data with sites on 'site_id'
df_merged = pd.merge(df_data_master, df_sites, on='site_id', how='left')

# Merge the result with richtingen on both 'site_id' and 'richting'
df_merged = pd.merge(df_merged, df_richtingen, on=['site_id', 'richting'], how='left')

# 5. Filter out faulty site
print("🔄 Filtering out faulty site ID 144...")
df_final = df_merged[df_merged['site_id'] != 144].reset_index(drop=True)

# 6. Save to Parquet
output_file = "bicycle_data_prepared.parquet"
df_final.to_parquet(output_file)
print(f"🚀 Done! Data preparation is complete. File saved as {output_file}")


