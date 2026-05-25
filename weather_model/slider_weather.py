import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor
import numpy as np
import os
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))

# @st.cache_resource makes sure the model is only loaded once, not every time the user moves a slider
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(os.path.join(_DIR, "weather_bike_model.json"))
    return model

# @st.cache_data does the same for the CSV — only reads it once
@st.cache_data
def load_sites():
    # the csv has no header row so we define the column names manually
    cols = [
        "site_id", "site_nr", "long", "lat", "naam", "domein",
        "wegnr", "district", "gemeente", "interval", "datum_van"
    ]
    sites = pd.read_csv(
        os.path.join(_DIR, "Data", "sites.csv"),
        names=cols,
        header=None
    )
    # keep only the columns we actually need for the map and hover
    sites = sites[["site_id", "lat", "long", "naam", "gemeente", "district", "wegnr"]]
    # rename long to lon so plotly recognizes
    sites = sites.rename(columns={"long": "lon"})
    return sites


def show():
    model = load_model()
    sites = load_sites()

    st.title("🚴 Weather-Cycling Simulation Map")

    # -----Sidebar controls-----
    st.sidebar.header("⚙️ Simulation inputs")

    # date picker + hour slider
    dt   = st.sidebar.date_input("Date", value=datetime.today())
    hour = st.sidebar.slider("Hour of day", 0, 23, 8)

    # derive the remaining time features automatically from the chosen date
    day_of_week = pd.Timestamp(dt).dayofweek   # 0 = Monday, 6 = Sunday
    month       = pd.Timestamp(dt).month       # 1–12

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Weather conditions")

    # weather sliders — these are the remaining features the model expects
    temperature   = st.sidebar.slider("Temperature (°C)",  -5.0, 35.0, 18.0, step=0.5)
    humidity      = st.sidebar.slider("Humidity (%)",         0,  100,   60)
    precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 10.0,  0.0, step=0.1)
    wind_speed    = st.sidebar.slider("Wind speed (km/h)",  0.0, 50.0, 10.0, step=0.5)
    cloud_cover   = st.sidebar.slider("Cloud cover (%)",      0,  100,   30)

    # ------Build feature matrix-----
    features = ["lat", "lon", "hour", "day_of_week", "month",
                "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover"]

    X = sites[["lat", "lon"]].copy()
    X["hour"]          = hour
    X["day_of_week"]   = day_of_week
    X["month"]         = month
    X["temperature"]   = temperature
    X["humidity"]      = humidity
    X["precipitation"] = precipitation
    X["wind_speed"]    = wind_speed
    X["cloud_cover"]   = cloud_cover
    X = X[features]

    # ----------Predict--------
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    sites_plot = sites.copy()
    sites_plot["predicted_cyclists"] = preds.round(0).astype(int)

    # ------------Map------
    st.subheader(f"Predicted cyclist counts — {pd.Timestamp(dt).strftime('%A %d %b %Y')}, {hour:02d}:00")

    fig = px.scatter_map(
        sites_plot,
        lat="lat",
        lon="lon",
        size="predicted_cyclists",
        size_max=40,
        zoom=10,
        hover_name="naam",
        hover_data={
            "predicted_cyclists": True,
            "gemeente": True,
            "district": True,
            "wegnr": True,
            "site_id": True,
            "lat": False,
            "lon": False
        },
        labels={"predicted_cyclists": "Cyclists"},
        map_style="carto-positron"
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=650)
    st.plotly_chart(fig, width='stretch')

    # let's say if hour is 8am = all counts from 8:00 → 8:59 (4 intervals of 15 min each, summed up)
