import streamlit as st


def home():
    st.set_page_config(page_title="Cycling Analytics Platform", layout="wide", page_icon="🚴")

    st.title("🚴 Cycling Analytics Platform")
    st.markdown(
        "<p style='font-size:18px; color:grey;'>A suite of tools for understanding cycling patterns in Flanders.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌤️ Weather Simulation")
        st.write(
            "Predict cyclist counts at Flemish monitoring sites based on weather "
            "conditions and time of day. Adjust sliders to simulate different scenarios."
        )

    with col2:
        st.subheader("🚴 Cycling Timelapse")
        st.write(
            "Watch an animated map of real cyclist traffic flow across Flanders. "
            "See which sites are busiest hour by hour throughout the day."
        )

    with col3:
        st.subheader("⚠️ Accident Risk")
        st.write(
            "Explore the GTRI accident risk model. Identify high-risk sites based on "
            "weather, traffic volume, and temporal patterns across Flanders."
        )

    st.divider()
    st.caption("Use the sidebar to navigate between tools.")


pg = st.navigation([
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("weather_model/slider.py", title="Weather Simulation", icon="🌤️"),
    st.Page("timelapse_model/timelapse_app.py", title="Cycling Timelapse", icon="🚴"),
    st.Page("accident_model/07_GTRI_dashboard.py", title="Accident Risk", icon="⚠️"),
])
pg.run()
