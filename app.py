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
    get_radius='average_hour_count*75'
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
st.subheader('Commuting patterns and averages through the clusters')
#st.sidebar.selectbox('Try and select an option?', ['Yes', 'No', 'Maybe'])

col1, col2 = st.columns([12,2], gap='small')

with col1:
    st.pydeck_chart(chart, on_select='rerun', selection_mode='multi-object')
    st.write('**Cluster 0** is representative of the lowest overall traffic counts. From analyzing the map, cluster 0 spots tend to be in more rural locations or areas where few cyclists bike into the city.')
    st.write('**Cluster 1** represents the highest traffic counts, it also indicates the highest commuting patterns with 8 am and 4 to 5 pm having by far the highest counts. These sites are located in cities/regions with heavy cycling usage or a defined entry points into a city like a bridge where traffic from the region is forced through a single point.')
    st.write('**Cluster 2** represents a combination of both low traffic and small commuting behavior. This cluster also has more consistent riders during the midday where the hourly dropoff is limited during the mid to early afternoon. These are also commonly located along roads leading into cities with relatively middle level cycling infrastructure or cycling commuting infrastructure, like Bruges and Ghent.')

with col2:
    st.write('🟦 **Cluster 0**\n\n🟥 **Cluster 1**\n\n🟩 **Cluster 2**')
#event.selection