import pandas as pd
import requests
import time
import os
from pathlib import Path

# 1. Config and Parameters
start_date = "2019-01-01"
end_date = "2024-12-31"
output_file = "raw_historical_weather.parquet"

print("🔄 Extracting coordinate list from site features...")
df_sites = pd.read_parquet("site_spatial_features.parquet")

# --- RESUME LOGIC ---
weather_data_list = []
completed_sites = set()

if os.path.exists(output_file):
    print("📂 Found existing weather data! Loading to resume...")
    df_existing = pd.read_parquet(output_file)
    completed_sites = set(df_existing['site_id'].unique())
    weather_data_list.append(df_existing)
    print(f"✅ Already completed {len(completed_sites)} sites. Resuming the rest...")

sites_to_process = df_sites[~df_sites['site_id'].isin(completed_sites)]
print(f"🌤️ Starting download for the remaining {len(sites_to_process)} sites...")

# 2. Iterate through remaining sites
for idx, row in sites_to_process.iterrows():
    site_id = row['site_id']
    lat = row['lat']
    lon = row['long']
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "Europe/Brussels"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                hourly_raw = data.get("hourly", {})
                
                df_weather_site = pd.DataFrame({
                    "timestamp": pd.to_datetime(hourly_raw.get("time")),
                    "temperature_2m": hourly_raw.get("temperature_2m"),
                    "precipitation_1h": hourly_raw.get("precipitation"),
                    "wind_speed_10m": hourly_raw.get("wind_speed_10m")
                })
                
                df_weather_site["site_id"] = site_id
                weather_data_list.append(df_weather_site)
                print(f"✅ Downloaded weather for site {site_id}: {row['naam_site']}")
                
                # Standard wait time to avoid triggering rate limit again
                time.sleep(2)
                break # Break out of the retry loop if successful
                
            elif response.status_code == 429:
                wait_time = 30 * (attempt + 1)
                print(f"⚠️ Rate limit hit (429) for site {site_id}. Pausing for {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
            else:
                print(f"❌ Failed API call for site {site_id}. Status Code: {response.status_code}")
                break # Don't retry on 404s or 400s
                
        except Exception as e:
            print(f"❌ Error fetching weather data for site {site_id}: {str(e)}")
            time.sleep(5)

# 3. Concatenate and overwrite with the full dataset
if weather_data_list:
    df_master_weather = pd.concat(weather_data_list, ignore_index=True)
    df_master_weather = df_master_weather[["site_id", "timestamp", "temperature_2m", "precipitation_1h", "wind_speed_10m"]]
    
    df_master_weather.to_parquet(output_file)
    print(f"\n🚀 Success! Weather dataset is now complete with {len(df_master_weather):,} total hourly rows.")
    print(f"Saved to: {output_file}")
else:
    print("❌ No weather records were processed.")