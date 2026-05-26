# Modern-Data-Analytics

*MDA Assignment — Group 12 (On-campus)*

Understanding our GitHub Repository:

## Overview

This repository contains a multi-tool Cycling Analytics Platform for Flanders, built on AWV bike sensor data, Statbel accident records, and Open-Meteo weather data covering the period 2019–2026. The platform is served as a single Streamlit application (`main_app.py`) that routes between four independent tools: an accident risk model, a weather simulation, a cycling timelapse, and a cluster analysis. Each tool lives in its own subfolder with its own requirements file.

## 1. Accident Risk — GTRI (`accident_model/`)

This is our primary predictive model. The pipeline runs across seven numbered scripts. Scripts 01 and 02 prepare the two main data sources: `01_GTRI_prepare_bike_counts.py` merges the AWV site metadata and monthly count CSVs into a single flat table, while `02_GTRI_filter_accidents.py` loads the Statbel Excel file and filters it to accidents with valid Lambert 72 coordinates that involve at least one cyclist.

Script 03 builds the spatial baseline. It fits a Gaussian KDE with 500m bandwidth on training-period accidents only (2019–2023, excluding 2024 to prevent leakage), and computes hard buffer counts at 250m and 250–500m around each sensor. These three spatial features turn out to be the dominant predictors in the final model. Script 04 fetches hourly temperature, precipitation, and wind speed from the Open-Meteo archive API for each sensor site. The fetcher is resumable: if interrupted, it picks up from the last completed site on the next run.

Script 05 constructs the master training dataset on a site × year × month × hour grid (122,053 rows). Weather lags and the rain onset signal (`Rain_Surge`) are computed at the actual day-hour level before aggregating to month-hour, which is the finest granularity the Statbel accident target supports — Statbel records carry no day-of-month information. Script 06 trains a LightGBM Tweedie regressor (variance power 1.5) on 2019–2023 data and evaluates on the 2024 hold-out. The Tweedie objective is appropriate here because accident counts are non-negative, zero-inflated (93% of site-hours record no accident), and right-skewed. The trained model, baseline risk value, feature names, and evaluation metrics are saved together in `GTRI_model_artifacts.pkl`. Script 07 is a Streamlit dashboard that loads the artifact and renders a live Folium map of relative risk scores across all 137 sensor sites, with sidebar controls for weather, traffic, and time-of-day inputs.

Key results on the 2024 test set: Spearman ρ = 0.10 (p < 10⁻⁹²), top-decile capture = 47.4% (4.7× lift over a random baseline of 10%).

## 2. Weather Simulation (`weather_model/`)

Predicts expected cyclist counts at Flemish monitoring sites given a set of weather conditions and a time of day. Users adjust temperature, precipitation, wind, and hour via sliders and see predicted counts per site updated in real time. The model is pre-trained and loaded from a saved artifact at startup to avoid re-fitting on each interaction.

## 3. Cycling Timelapse (`timelapse_tool/`)

Renders an animated map of real AWV traffic flow across Flanders. Each frame represents one hour of the day, cycling through a selected time period. The tool reads pre-processed monthly parquet files and aggregates them on the fly to show which sensor sites are busiest at each hour.

## 4. Cluster Analysis (`model_cluster/`)

Groups Flemish AWV monitoring sites by traffic patterns and site characteristics. The resulting clusters are visualised on an interactive map and in summary tables, allowing comparison of site types across the network.

## Running the platform

Install dependencies and launch the full app:

```bash
pip install -r requirements.txt
streamlit run main_app.py
```

To reproduce the GTRI model from scratch, run the seven scripts inside `accident_model/` in order (01 through 06), then launch script 07 or the main app. Raw data files are not tracked in this repository — see `.gitignore` for what needs to be obtained externally (AWV monthly CSVs, Statbel Excel file). The Open-Meteo weather data is fetched automatically by script 04.

## Link to Datasets

AWV bike sensor data is available through the Flemish open data portal. Statbel accident microdata (`OPENDATA_MAP_2017-2024.xlsx`) is available at [statbel.fgov.be](https://statbel.fgov.be). Weather data is fetched automatically from [archive-api.open-meteo.com](https://archive-api.open-meteo.com) (no API key required).
