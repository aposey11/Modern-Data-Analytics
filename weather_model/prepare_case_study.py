"""
Pre-computation script for the Circulation Plan Analysis page.

Run this once before launching the Streamlit app, and re-run whenever
the underlying cycling data changes (e.g. new yearly parquet added).

What it does:
  1. Loads all processed yearly cycling parquets
  2. Trains a city-specific XGBoost model for Aalst and Kortrijk
  3. Fetches historical weather from Open-Meteo for training + prediction
  4. Generates daily predicted vs actual counts for every city sensor
  5. Saves everything to weather_model/cache/

Usage (from the project root or from weather_model/):
    python weather_model/prepare_case_study.py
    -- or --
    cd weather_model && python prepare_case_study.py

Output files:
    weather_model/cache/aalst_model.json
    weather_model/cache/kortrijk_model.json
    weather_model/cache/aalst_daily.parquet
    weather_model/cache/kortrijk_daily.parquet
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────

_DIR          = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(os.path.dirname(_DIR), "timelapse_tool", "Processed Data")
CACHE_DIR     = os.path.join(_DIR, "cache")

# ── Constants (keep in sync with case_study.py) ───────────────────────────────

AALST_INTERVENTION = pd.Timestamp("2021-08-16")
KORTRIJK_TRIAL     = pd.Timestamp("2022-07-01")
KORTRIJK_PERMANENT = pd.Timestamp("2023-10-01")

CITY_CONFIG = {
    "Aalst": {
        "sites":         {10: "Aalst 1", 11: "Aalst 2", 19: "Aalst 3"},
        "coords":        {10: (50.93433, 4.01471), 11: (50.93549, 4.01571), 19: (50.93385, 4.01647)},
        "weather_coords": (50.9344, 4.0159),
        "city_site_ids": [10, 11, 19],
        "pre_cutoff":    AALST_INTERVENTION,
    },
    "Kortrijk": {
        "sites":         {16: "Kortrijk 1", 17: "Kortrijk 2"},
        "coords":        {16: (50.83332, 3.27777), 17: (50.83339, 3.27773)},
        "weather_coords": (50.8276, 3.2647),
        "city_site_ids": [16, 17],
        "pre_cutoff":    KORTRIJK_TRIAL,   # sensors active since Aug 2019; pre_cutoff used for model training split
    },
}

MODEL_FEATURES = [
    "lat", "lon", "hour", "day_of_week", "month",
    "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def fetch_weather(start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
    print(f"    Fetching weather {start_date} → {end_date}  ({lat:.4f}, {lon:.4f})…", end="", flush=True)
    t0 = time.time()
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,wind_speed_10m,cloud_cover"
            ),
            "timezone": "Europe/Brussels",
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "hour_timestamp": pd.to_datetime(data["time"]),
        "temperature":    data["temperature_2m"],
        "humidity":       data["relative_humidity_2m"],
        "precipitation":  data["precipitation"],
        "wind_speed":     data["wind_speed_10m"],
        "cloud_cover":    data["cloud_cover"],
    })
    print(f" done ({time.time() - t0:.1f}s, {len(df):,} rows)")
    return df


def load_all_sites() -> pd.DataFrame:
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet"))
    df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, files[-1]),
        columns=["site_ID", "longitude", "latitude"],
    )
    sites = (
        df.groupby("site_ID")[["longitude", "latitude"]]
        .first()
        .reset_index()
        .rename(columns={"site_ID": "site_id", "longitude": "lon", "latitude": "lat"})
    )
    return sites[["site_id", "lat", "lon"]].copy()


def load_all_cycling_data() -> pd.DataFrame:
    print("  Loading all processed cycling parquets…", end="", flush=True)
    t0 = time.time()
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet"))
    dfs = [
        pd.read_parquet(
            os.path.join(PROCESSED_DIR, f),
            columns=["site_ID", "time_from", "count"],
        )
        for f in files
    ]
    df = pd.concat(dfs, ignore_index=True)
    df["time_from"] = pd.to_datetime(df["time_from"]).dt.floor("h")
    result = (
        df.groupby(["site_ID", "time_from"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"site_ID": "site_id", "time_from": "hour_timestamp", "count": "bike_count"})
    )
    print(f" done ({time.time() - t0:.1f}s, {len(result):,} rows)")
    return result


# ── Main routine ──────────────────────────────────────────────────────────────

def process_city(city: str, all_cycling: pd.DataFrame, all_sites: pd.DataFrame):
    cfg           = CITY_CONFIG[city]
    city_site_ids = cfg["city_site_ids"]
    cutoff        = cfg["pre_cutoff"]
    coord_map     = cfg["coords"]

    _banner(f"City: {city}")

    # ── 1. Build training dataset ──────────────────────────────────────────────
    non_city = all_cycling[~all_cycling["site_id"].isin(city_site_ids)]
    city_pre = all_cycling[
        all_cycling["site_id"].isin(city_site_ids) &
        (all_cycling["hour_timestamp"] < cutoff)
    ]
    train_df = pd.concat([non_city, city_pre], ignore_index=True)

    start_train = train_df["hour_timestamp"].min().strftime("%Y-%m-%d")
    end_train   = train_df["hour_timestamp"].max().strftime("%Y-%m-%d")
    print(f"  Training rows: {len(train_df):,}  ({start_train} → {end_train})")

    # Central Flanders weather approximation for training
    weather_train = fetch_weather(start_train, end_train, lat=51.05, lon=3.72)

    merged_train = (
        train_df
        .merge(weather_train, on="hour_timestamp", how="left")
        .merge(all_sites,     on="site_id",        how="left")
    )
    merged_train["hour"]        = merged_train["hour_timestamp"].dt.hour
    merged_train["day_of_week"] = merged_train["hour_timestamp"].dt.dayofweek
    merged_train["month"]       = merged_train["hour_timestamp"].dt.month
    merged_train = merged_train.dropna(subset=MODEL_FEATURES + ["bike_count"])

    # ── 2. Train model ─────────────────────────────────────────────────────────
    print(f"  Training XGBoost (150 estimators)…", end="", flush=True)
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=150, learning_rate=0.07, max_depth=6,
        subsample=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(merged_train[MODEL_FEATURES], merged_train["bike_count"])
    print(f" done ({time.time() - t0:.1f}s)")

    model_path = os.path.join(CACHE_DIR, f"{city.lower()}_model.json")
    model.save_model(model_path)
    print(f"  ✓ Model → {model_path}")

    # ── 3. Predict on city data ────────────────────────────────────────────────
    city_df    = all_cycling[all_cycling["site_id"].isin(city_site_ids)].copy()
    start_pred = city_df["hour_timestamp"].min().strftime("%Y-%m-%d")
    end_pred   = city_df["hour_timestamp"].max().strftime("%Y-%m-%d")
    print(f"  Prediction window: {start_pred} → {end_pred}")

    # City-specific weather for accurate predictions
    weather_pred = fetch_weather(start_pred, end_pred, *cfg["weather_coords"])

    merged_pred = city_df.merge(weather_pred, on="hour_timestamp", how="left")
    merged_pred["hour"]        = merged_pred["hour_timestamp"].dt.hour
    merged_pred["day_of_week"] = merged_pred["hour_timestamp"].dt.dayofweek
    merged_pred["month"]       = merged_pred["hour_timestamp"].dt.month
    merged_pred["lat"]         = merged_pred["site_id"].map(lambda x: coord_map[x][0])
    merged_pred["lon"]         = merged_pred["site_id"].map(lambda x: coord_map[x][1])
    merged_pred                = merged_pred.dropna(subset=MODEL_FEATURES)
    merged_pred["predicted"]   = model.predict(merged_pred[MODEL_FEATURES]).clip(min=0)

    # ── 4. Aggregate to daily and save ─────────────────────────────────────────
    print("  Aggregating to daily…", end="", flush=True)
    daily_parts = []
    for sid in city_site_ids:
        df_site = merged_pred[merged_pred["site_id"] == sid].copy()
        if df_site.empty:
            continue
        daily = (
            df_site.set_index("hour_timestamp")[["bike_count", "predicted"]]
            .resample("D").sum()
            .reset_index()
        )
        daily["site_id"] = sid
        daily_parts.append(daily)

    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all = daily_all[["site_id", "hour_timestamp", "bike_count", "predicted"]]

    parquet_path = os.path.join(CACHE_DIR, f"{city.lower()}_daily.parquet")
    daily_all.to_parquet(parquet_path, index=False)
    print(f" {len(daily_all):,} rows")
    print(f"  ✓ Daily predictions → {parquet_path}")


def main():
    print("=" * 60)
    print("  Circulation Plan Analysis — cache preparation")
    print("=" * 60)

    if not os.path.isdir(PROCESSED_DIR):
        print(f"\n❌  Processed data directory not found:\n    {PROCESSED_DIR}")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    t_total = time.time()
    all_cycling = load_all_cycling_data()
    all_sites   = load_all_sites()

    for city in CITY_CONFIG:
        process_city(city, all_cycling, all_sites)

    print(f"\n{'=' * 60}")
    print(f"  ✅  All done in {time.time() - t_total:.0f}s")
    print(f"      Cache written to: {CACHE_DIR}")
    print(f"      Restart your Streamlit app to pick up the new files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
