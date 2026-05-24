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
    tooltip={'text': 'Id: {siteid}\nCluster: {cluster_agc_ward}\nAverage Hour Count: {average_hour_count}'},
    
)
st.set_page_config(layout='wide')
st.title('Cluster Analysis')
st.subheader('Commuting Patterns and Averages Through the Clusters')
#st.sidebar.selectbox('Try and select an option?', ['Yes', 'No', 'Maybe'])

col1, col2 = st.columns([12,2], gap='small')

with col1:
    st.pydeck_chart(chart, on_select='rerun', selection_mode='multi-object')
    st.write('**Cluster 0** is the second lowest in terms of cycling counts but the difference between it and cluster 1 is notable. These clusters are located at the entry corridors to large cities like Bruges and Ghent. From this it appears that the number of cyclists commuting into these towns is low, which could be due to other alternative modes of transportation like busses or trains carrying the majority of commuters.')
    st.write('**Cluster 1** contains the overall lowest cycling counts across the clusters. From the map, it is clear that these counters are located in rural areas where it is unlikely they will be commonly used for commuting.')
    st.write('**Cluster 2** contains the largest counts across all clusters irregardless of the time. These points are not numerous but they are near major entry points into large cycling cities like Leuven and Kortrijk.')
    st.write('**Cluster 3** is the second highest for cycling counts and is located near smaller cities like Hasselt and Mechelen. While having a lot less cyclists compared to cluster 2, these counts indicate that there is some cycling infratructure within the city and that cycling is a reasonable alternative to driving for commuting purposes in these regions.')
    st.write('Overall, there is a clear trend in commuting patterns indicating that the location of these counters are on commuting paths for workers.')

with col2:
    st.write('🟦 **Cluster 0**\n\n🟥 **Cluster 1**\n\n🟩 **Cluster 2**\n\n ⬛ **Cluster 3**')
#event.selection