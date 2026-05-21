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

starting_view = pdk.ViewState(latitude= 5, longitude= 50, zoom= 10, max_zoom= 16)

site_locations = pdk.Layer(
    'ScatterplotLayer',
    sites[['longitude', 'latitude']],
    auto_highlight=True,
    get_position=['longitude', 'latitude'],
    get_radius=1000,
    get_fill_color=[100,100,200,0],
    pickable=True
)

design = pdk.Deck(layers= [site_locations], initial_view_state = starting_view)
st.pydeck_chart(design)
