import pandas as pd
#from sklearn import 
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geodatasets
import pydeck as pdk
import streamlit as st

sites = pd.read_csv('sites.csv', names=['siteid', 'site_nr', 'longitude', 'latitude', 'name', 'region', 'direction_number', 'district', 'municipality', 'interval', 'date'])



"""""
BELOW IS MY USE OF STREAMLIT INFO
"""""

points = pdk.Layer(
    'ScatterplotLayer',
    data=sites,
    id='',
    get_position=['longitude', 'latitude'],
    get_radius=1000,
    get_fill_color=[100,100,200,0],
    pickable=True
)

design = pdk.Deck(layers= [site_locations], initial_view_state = starting_view)
st.pydeck_chart(design)