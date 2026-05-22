import pandas as pd
import pydeck as pdk
import streamlit as st

sites = pd.read_csv('sites_cluster.csv')




#BELOW IS MY USE OF STREAMLIT INFO

points = pdk.Layer(
    'ScatterplotLayer',
    data=sites,
    id='datapoints',
    get_position=['longitude', 'latitude'],
    get_color='[color1, color2, color3]',
    pickable=True,
    auto_highlight=True,
    opacity=.3,
    get_radius=1500
)

view_point = pdk.ViewState(
    latitude=51, longitude=4.3, controller=True, zoom=7.1
)


chart = pdk.Deck(
    points,
    map_style=None,
    initial_view_state=view_point,
    tooltip={'text': 'Id: {siteid}\nCluster: {cluster}'},
    
)
st.title('Cluster Analysis')
#st.sidebar.selectbox('Try and select an option?', ['Yes', 'No', 'Maybe'])

col1, col2 = st.columns([12,2], gap='small')

with col1:
    st.pydeck_chart(chart, on_select='rerun', selection_mode='multi-object')
    st.text('Cluster 0 indicates high traffic counts particularly in the morning\nCluster 1 represents low overall traffic counts\nCluster 2 represents a combination')

with col2:
    st.text('Cluster 0\n\nCluster 1\n\nCluster2')
#event.selection