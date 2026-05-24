import pandas as pd
import requests
import time
import os
from pathlib import Path

# Fetches hourly historical weather from the Open-Meteo archive API for each AWV
# sensor site. Variables retrieved: temperature at 2m, hourly precipitation, and
# wind speed at 10m. Coverage: 2019-01-01 to 2024-12-31 (full model period).
#
# The script supports resuming an interrupted run: if the output file already exists,
# sites already present in it are skipped. This avoids re-downloading data for a
# large number of sites if the run is interrupted part-way through.

start_date  = "2019-01-01"
end_date    = "2024-12-31"
output_file = "gtri_raw_historical_weather.parquet"

print("Reading site coordinates from spatial features file...")
df_sites = pd.read_parquet("gtri_site_spatial_features.parquet")

weather_data_list = []
completed_sites   = set()

if os.path.exists(output_file):
    print("Existing weather file found — loading to resume from where the previous run stopped...")
    df_existing     = pd.read_parquet(output_file)
    completed_sites = set(df_existing['site_id'].unique())
    weather_data_list.append(df_existing)
    print(f"  Already completed: {len(completed_sites)} sites")

sites_to_process = df_sites[~df_sites['site_id'].isin(completed_sites)]
print(f"Sites remaining to download: {len(sites_to_process)}")

for idx, row in sites_to_process.iterrows():
    site_id = row['site_id']
    lat     = row['lat']
    lon     = row['long']

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     "temperature_2m,precipitation,wind_speed_10m",
        "timezone":   "Europe/Brussels"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data       = response.json()
                hourly_raw = data.get("hourly", {})

                df_weather_site = pd.DataFrame({
                    "timestamp":        pd.to_datetime(hourly_raw.get("time")),
                    "temperature_2m":   hourly_raw.get("temperature_2m"),
                    "precipitation_1h": hourly_raw.get("precipitation"),
                    "wind_speed_10m":   hourly_raw.get("wind_speed_10m")
                })
                df_weather_site["site_id"] = site_id
                weather_data_list.append(df_weather_site)
                print(f"  Downloaded: site {site_id} ({row['naam_site']})")

                # Respect the Open-Meteo free-tier rate limit.
                time.sleep(2)
                break

            elif response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limit (429) on site {site_id} — waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)

            else:
                print(f"  API error {response.status_code} for site {site_id} — skipping.")
                break

        except Exception as e:
            print(f"  Exception for site {site_id}: {e} — waiting 5s before retry...")
            time.sleep(5)

if weather_data_list:
    df_master_weather = pd.concat(weather_data_list, ignore_index=True)
    df_master_weather = df_master_weather[
        ["site_id", "timestamp", "temperature_2m", "precipitation_1h", "wind_speed_10m"]
    ]
    df_master_weather.to_parquet(output_file)
    print(f"\nComplete. {len(df_master_weather):,} hourly rows saved to {output_file}")
else:
    print("No weather records were processed.")
