import pandas as pd
import pydeck as pdk
import streamlit as st

sites = pd.read_csv('sites_cluster.csv')




#BELOW IS MY USE OF STREAMLIT INFO

points = pdk.Layer(
    'ScatterplotLayer',
    data=sites,
    id='',
    get_position=['longitude', 'latitude'],
    get_color='[color, color, 95]',
    pickable=True,
    auto_highlight=True,
    opacity=.5,
    get_radius=1750
)

view_point = pdk.ViewState(
    latitude=51, longitude=4.3, controller=True, zoom=7.3
)

chart = pdk.Deck(
    points,
    map_style=None,
    initial_view_state=view_point,
    tooltip={'Hi I am hoping that this works!'},
)

event = st.pydeck_chart(chart, selection_mode='multi-object')

#event.selection