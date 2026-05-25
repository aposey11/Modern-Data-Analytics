import streamlit as st


def home():
    st.set_page_config(page_title="Cycling Analytics Platform", layout="wide", page_icon="🚴")

    st.title("🚴 Cycling Analytics Platform")
    st.markdown(
        "<p style='font-size:18px; color:grey;'>A suite of tools for understanding cycling patterns in Flanders.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    cards = [
        {
            "icon": "🌤️",
            "title": "Weather Simulation",
            "desc": "Predict cyclist counts at Flemish monitoring sites based on weather conditions and time of day. Adjust sliders to simulate different scenarios.",
            "href": "/weather",
        },
        {
            "icon": "🚴",
            "title": "Cycling Timelapse",
            "desc": "Watch an animated map of real cyclist traffic flow across Flanders. See which sites are busiest hour by hour throughout the day.",
            "href": "/timelapse",
        },
        {
            "icon": "⚠️",
            "title": "Accident Risk",
            "desc": "Explore the GTRI accident risk model. Identify high-risk sites based on weather, traffic volume, and temporal patterns across Flanders.",
            "href": "/accident-risk",
        },
        {
            "icon": "🔵",
            "title": "Cluster Analysis",
            "desc": "Explore clustering of Flemish cycling monitoring sites based on traffic patterns and site characteristics.",
            "href": "/clusters",
        },
        {
            "icon": "📊",
            "title": "Circulation Plan Analysis",
            "desc": "Compare observed cyclist counts against a weather-normalised baseline before and after circulation plan changes in Aalst and Kortrijk.",
            "href": "/circulation",
        },
    ]

    card_items = "".join(
        f"""
        <a class="nav-card" href="{c['href']}">
            <div class="nav-card-icon">{c['icon']}</div>
            <div class="nav-card-title">{c['title']}</div>
            <div class="nav-card-desc">{c['desc']}</div>
        </a>
        """
        for c in cards
    )

    st.markdown(
        f"""
        <style>
        .nav-cards {{
            display: flex;
            gap: 1rem;
            align-items: stretch;
        }}
        .nav-card {{
            flex: 1 1 0;
            min-width: 0;
            text-decoration: none !important;
            color: inherit !important;
            border: 1px solid rgba(49,51,63,0.2);
            border-radius: 10px;
            padding: 1.3rem 1.2rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            cursor: pointer;
            transition: border-color .15s, box-shadow .15s;
        }}
        .nav-card:hover {{
            border-color: #1f77b4;
            box-shadow: 0 3px 14px rgba(31,119,180,0.18);
            text-decoration: none !important;
            color: inherit !important;
        }}
        .nav-card-icon  {{ font-size: 1.8rem; line-height: 1; }}
        .nav-card-title {{ font-size: 1rem; font-weight: 700; margin-top: 0.2rem; }}
        .nav-card-desc  {{ font-size: 0.87rem; color: grey; line-height: 1.55; flex: 1; }}
        </style>
        <div class="nav-cards">{card_items}</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Click any card above to open that tool, or use the sidebar to navigate.")


pg = st.navigation([
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("weather_model/weather_hub.py",           title="Weather Simulation",       icon="🌤️", url_path="weather"),
    st.Page("timelapse_tool/timelapse_app.py",        title="Cycling Timelapse",        icon="🚴",  url_path="timelapse"),
    st.Page("accident_model/07_GTRI_dashboard.py",    title="Accident Risk",            icon="⚠️",  url_path="accident-risk"),
    st.Page("model_cluster/app.py",                   title="Cluster Analysis",         icon="🔵",  url_path="clusters"),
    st.Page("weather_model/case_study.py",            title="Circulation Plan Analysis",icon="📊",  url_path="circulation"),
])
pg.run()
