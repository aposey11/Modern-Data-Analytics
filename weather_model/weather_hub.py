import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import slider_weather
import predict_weather

st.markdown("""
<style>
div[data-testid="stSegmentedControl"] button {
    font-size: 1rem !important;
    padding: 0.45rem 1.4rem !important;
}
</style>
""", unsafe_allow_html=True)

mode = st.segmented_control(
    "Mode",
    ["🎛️ Manual Simulation", "📅 7-Day Live Forecast"],
    default="🎛️ Manual Simulation",
    label_visibility="collapsed"
)

if mode == "🎛️ Manual Simulation":
    slider_weather.show()
else:
    predict_weather.show()
