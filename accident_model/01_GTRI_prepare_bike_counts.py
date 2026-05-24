import pandas as pd
from pathlib import Path

# Loads and merges the AWV bike count source files into a single prepared dataset.
# The AWV export splits across three separate CSVs: sites (sensor metadata),
# richtingen (directional lane definitions), and monthly count files named
# data-<year>-<month>.csv. All three are joined on site_id so that downstream
# scripts have a single flat table to work from.

data_path = Path("./")

# Column definitions match the AWV export format, which ships without headers.
# 'naam' is disambiguated to 'naam_site' in sites and 'naam_richting' in richtingen
# to prevent collision when the two tables are merged.
sites_columns      = ['site_id', 'site_nr', 'long', 'lat', 'naam_site', 'domein',
                       'wegnr', 'district', 'gemeente', 'interval', 'datum_van']
richtingen_columns = ['site_id', 'richting', 'naam_richting']
data_columns       = ['site_id', 'richting', 'type', 'van', 'tot', 'aantal']

print("Loading sites.csv and richtingen.csv...")
df_sites     = pd.read_csv(data_path / "sites.csv",     header=None, names=sites_columns)
df_richtingen = pd.read_csv(data_path / "richtingen.csv", header=None, names=richtingen_columns)

# Collect all monthly data files in chronological order.
data_files = sorted(list(data_path.glob("data-*.csv")))
data_list  = []

print(f"Processing {len(data_files)} monthly count files...")
for f in data_files:
    df = pd.read_csv(f, header=None, names=data_columns)
    data_list.append(df)
    print(f"  Loaded: {f.name}")

if data_list:
    df_data_master = pd.concat(data_list, ignore_index=True)
else:
    df_data_master = pd.DataFrame(columns=data_columns)
    print("No data files found. Check that files follow the 'data-<year>-<month>.csv' naming convention.")

print("Merging count data with site metadata and direction labels...")
df_merged = pd.merge(df_data_master, df_sites,      on='site_id',                how='left')
df_merged = pd.merge(df_merged,      df_richtingen, on=['site_id', 'richting'],  how='left')

# Site 144 is faulty (known hardware issue in the AWV dataset) and is excluded from
# all downstream processing.
print("Filtering out faulty site ID 144...")
df_final = df_merged[df_merged['site_id'] != 144].reset_index(drop=True)

output_file = "gtri_bicycle_data_prepared.parquet"
df_final.to_parquet(output_file)
print(f"Done. Saved {len(df_final):,} rows to {output_file}")
