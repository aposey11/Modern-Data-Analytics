import os
import pandas as pd
import pydeck as pdk
import streamlit as st

_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(layout='wide')
st.title('Cluster Analysis')
st.subheader('Commuting Patterns and Averages Through the Clusters')

# ---------- Load data ----------
@st.cache_data
def load_sites():
    return pd.read_csv(os.path.join(_DIR, 'sites_cluster.csv'))

sites = load_sites()

# ---------- Sidebar: cluster toggles ----------
st.sidebar.header("🔍 Filter clusters")
show = {
    0: st.sidebar.checkbox("🟦 Cluster 0", value=True),
    1: st.sidebar.checkbox("🟥 Cluster 1", value=True),
    2: st.sidebar.checkbox("🟩 Cluster 2", value=True),
    3: st.sidebar.checkbox("⬛ Cluster 3", value=True),
}
active_clusters = [c for c, on in show.items() if on]
filtered = sites[sites["cluster_agc_ward"].isin(active_clusters)]

# ---------- Map ----------
points = pdk.Layer(
    'ScatterplotLayer',
    data=filtered,
    id='datapoints',
    get_position=['longitude', 'latitude'],
    get_color='[color4, color5, color6]',
    pickable=True,
    auto_highlight=True,
    opacity=.3,
    get_radius=900,
)

view_point = pdk.ViewState(
    latitude=51, longitude=4.3, controller=True, zoom=7.6
)

chart = pdk.Deck(
    points,
    map_style=None,
    initial_view_state=view_point,
    tooltip={'text': 'Id: {siteid}\nCluster: {cluster_agc_ward}\nAverage Cyclists per Hour Count: {average_hour_count}'},
)

col1, col2 = st.columns([12, 2], gap='small')

with col1:
    st.pydeck_chart(chart, on_select='rerun', selection_mode='multi-object')
    st.write('**:blue[Cluster 0]** is the second lowest in terms of cycling counts, but the difference between it and cluster 1 is notable. These clusters are located at the entry corridors to large cities like Bruges and Ghent. From this it appears that the number of cyclists commuting into these towns is low, which could be due to alternative modes of transportation like busses or trains carrying the majority of commuters into the city.')
    st.write('**:red[Cluster 1]** contains the overall lowest cycling counts across the clusters. From the map, it is clear that these counters are located in rural areas where it is unlikely they will be commonly used for commuting or general daily use.')
    st.write('**:green[Cluster 2]** contains the largest counts across all clusters regardless of the time. These sites are not numerous, but they are near major entry points into large cycling cities like Leuven and Kortrijk. The cycling counts lower from noon to one pm, but the degree to which this happens is much less than both clusters 0 and 1. This could indicate that there is a general use of cycling to go to lunch or an overall high level of cycling infrastructure in the area since people are consistently cycling through the middle of the day.')
    st.write('**Cluster 3** is the second highest for cycling counts and is located near smaller cities like Hasselt and Mechelen. While having a lot less cyclists compared to cluster 2, these counts indicate that there is some cycling infrastructure within the city and that cycling is a reasonable alternative to driving for commuting purposes. Like cluster 2, the dropoff in counts after the morning commute is much less noticeable compared to clusters 0 and 1 and this could be due to people traveling through these corridors to grab lunch since the general cycling infrastructure is not poor in these areas.')
    st.write('**Overall**, there is a clear trend of morning and afternoon commuting indicating that the location of these counters are on commuting paths for workers.')

with col2:
    st.write('🟦 **Cluster 0**\n\n🟥 **Cluster 1**\n\n🟩 **Cluster 2**\n\n ⬛ **Cluster 3**')