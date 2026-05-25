import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Cycling Traffic Flow", layout="wide")
st.markdown("<h1 style='text-align: center'>Live Cycling Traffic</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey; font-size: 16px;'><i>Hover over any sensor dot on the map to see its exact IN and OUT counts.</i></p>", unsafe_allow_html=True)

def build_processed_data():
    """Pre-aggregate raw parquet files into one file per year for fast loading.
    Runs automatically on first launch if Processed Data/ doesn't exist yet."""
    headers_sites = ["site_ID", "site_no", "longitude", "latitude", "name", "domain",
                     "road_no", "road_dist_no", "municipality", "interval_length", "installed_since"]
    data_folder = os.path.join(_DIR, "Data")
    processed_folder = os.path.join(_DIR, "Processed Data")
    os.makedirs(processed_folder, exist_ok=True)

    sites_df = pd.read_csv(os.path.join(_DIR, "sites.csv"), header=None, names=headers_sites)
    all_files = sorted(f for f in os.listdir(data_folder) if f.endswith(".parquet"))
    years = sorted({int(f.split("-")[1]) for f in all_files})

    for year in years:
        out_path = os.path.join(processed_folder, f"data-{year}.parquet")
        if os.path.exists(out_path):
            continue
        files = [f for f in all_files if int(f.split("-")[1]) == year]
        df = pd.concat(
            [pd.read_parquet(os.path.join(data_folder, f), columns=["site_ID", "direction", "type", "time_from", "count"]) for f in files],
            ignore_index=True
        )
        df = df.loc[df["type"] == "FIETSERS"]  # exclude non-cyclist counts
        df = pd.merge(df, sites_df[["site_ID", "longitude", "latitude", "name"]], on="site_ID").dropna(subset=["latitude", "longitude"])
        df["time_from"] = pd.to_datetime(df["time_from"]).dt.floor("h")
        df = df.groupby(["site_ID", "direction", "longitude", "latitude", "name", "time_from"])["count"].sum().reset_index()
        df["time_from"] = df["time_from"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_parquet(out_path, index=False)

# run once on first launch — skips years that are already processed
processed_folder = os.path.join(_DIR, "Processed Data")
if not os.path.isdir(processed_folder):
    with st.spinner("First launch: pre-processing data for faster future loads..."):
        build_processed_data()

@st.cache_data(show_spinner="Loading data...")
def load_data():
    processed_folder = os.path.join(_DIR, "Processed Data")
    data_files = [f for f in os.listdir(processed_folder) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(os.path.join(processed_folder, f)) for f in data_files], ignore_index=True)

df = load_data()
timestamps = sorted(df['time_from'].unique())

# these variables persist across reruns
if 'time_index' not in st.session_state:
    st.session_state.time_index = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = True

# list of all possible dates and times from the dataset, used for the controls and display
dt_timestamps = pd.to_datetime(timestamps)
unique_dates = pd.unique(dt_timestamps.date)
current_dt = dt_timestamps[st.session_state.time_index]

# sets a multiple-column layout; had to fiddle around with the numbers to get the right ratio
col_play, col_date, col_hour = st.columns([2, 3, 7])

with col_play:
    st.markdown("<br>", unsafe_allow_html=True) # otherwise the button goes above the other two controls and looks weird
    if st.button("Play / Pause", width='stretch'):
        st.session_state.is_playing = not st.session_state.is_playing
        st.rerun()
        
with col_date:
    target_date = st.date_input("Jump to Date", value=current_dt.date(), min_value=min(unique_dates), max_value=max(unique_dates), disabled=st.session_state.is_playing)
    
with col_hour:
    target_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=current_dt.hour, format="%02d:00", disabled=st.session_state.is_playing)

# the user can select any combination of date and hour only after pausing
if not st.session_state.is_playing:
    target_time_str = f"{target_date} {target_hour:02d}:00:00"
    new_index = timestamps.index(target_time_str)
    # starting the animation again, now from a different timestamp
    if new_index != st.session_state.time_index:
        st.session_state.time_index = new_index
        st.rerun()

current_time_str = timestamps[st.session_state.time_index]
current_data = df[df['time_from'] == current_time_str]

dash_col, map_col = st.columns([2, 7])

with dash_col:
    # we calculate the top 5 busiest sites each hour by adding 'IN' and 'OUT' counts
    st.markdown("### Five Busiest Sites →")
    
    top_5_names = current_data.groupby('name')['count'].sum().nlargest(5).index.tolist()

    top_sites_table = current_data[current_data['name'].isin(top_5_names)].pivot_table(index='name', columns='direction', values='count', aggfunc='sum').fillna(0)
    top_sites_table['Total'] = top_sites_table['IN'] + top_sites_table['OUT']
    top_sites_table = top_sites_table.reset_index().sort_values(by='Total', ascending=False)
    top_sites_table.columns = ['Site Name', 'IN', 'OUT', 'Total']

    st.dataframe(top_sites_table[['Site Name', 'Total']], width='stretch', hide_index=True)

with map_col:
    #calculating total for each site here too for the "1st", "2nd" etc labels on the map
    all_counts = current_data.pivot_table(index='name', columns='direction', values='count', aggfunc='sum').fillna(0).reset_index()
    all_counts['Total'] = all_counts['IN'] + all_counts['OUT']

    map_dots = current_data.drop_duplicates('site_ID').merge(all_counts, on='name')
    # the top 5 sites get big bright dots, the rest are small and faded out
    map_dots['dot_color'] = map_dots['name'].apply(lambda x: [181, 148, 16] if x in top_5_names else [211,211,211])
    map_dots['dot_radius'] = map_dots['name'].apply(lambda x: 800 if x in top_5_names else 100)
 
    site_layer = pdk.Layer("ScatterplotLayer",data=map_dots, get_position=["longitude", "latitude"], get_radius="dot_radius", radius_min_pixels=3, get_fill_color="dot_color", pickable=True)

    # dropping duplicates because some sites close to each other have the same name; sorting by longitude because [explained later]
    top_5_text = map_dots[map_dots['name'].isin(top_5_names)].drop_duplicates(subset=['name']).sort_values('longitude').reset_index(drop=True)
    top_5_text['rank_num'] = top_5_text['Total'].rank(ascending=False, method='min').astype(int)
    
    def get_ordinal(n):
        if n == 1: return "1st"
        if n == 2: return "2nd"
        if n == 3: return "3rd"
        return f"{n}th"
        
    top_5_text['rank_str'] = top_5_text['rank_num'].apply(get_ordinal)
    # some labels were clashing so I first sort them by longitude (latitude could also work) then give different offsets to alternating labels so that they don't clash
    # although now I see that this causes labels to clash in other cases, so may need to rework
    top_5_text['height_offset'] = top_5_text.index.map(lambda i: 0.03 if i % 2 == 0 else -0.03)
    top_5_text['text_lat'] = top_5_text['latitude'] + top_5_text['height_offset'] 
    top_5_text['display_label'] = top_5_text.apply(lambda row: f"{row['rank_str']}: {row['name']}\nIN: {int(row['IN'])} | OUT: {int(row['OUT'])}", axis=1)

    text_layer = pdk.Layer("TextLayer", data=top_5_text, get_position=["longitude", "text_lat"], get_text="display_label", get_size=16, get_color=[59, 59, 59])

    # these starting params should give the best view of Flanders to include all sites and exclude most other places nearby
    view_state = pdk.ViewState(latitude=51.1, longitude=4.15, zoom=8)
    final_map = pdk.Deck(layers=[site_layer, text_layer], initial_view_state=view_state, map_style="light", tooltip={"html": "<b style='font-size: 14px;'>{name}</b><br/>IN: <b>{IN}</b> | OUT: <b>{OUT}</b>"})
    st.pydeck_chart(final_map, height=700, width='stretch')

# actual animation loop
if st.session_state.is_playing:
    time.sleep(0.1)
    st.session_state.time_index = (st.session_state.time_index + 1) % len(timestamps) # basically resets to the beginning after reaching the last possible unique timestamp
    st.rerun()
