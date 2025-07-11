import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# -------------------- Functions --------------------

# --- Preprocess latitude and longitude ---
def convert_lat(lat_str):
    return float(lat_str[:-1]) * (1 if lat_str[-1] == 'N' else -1)

def convert_lon(lon_str):
    return float(lon_str[:-1]) * (-1 if lon_str[-1] == 'W' else 1)


# --- Identify storms that affected UF ---
def is_uf_hit(row):
    """
    In the data, maximum sustained wind radius is given in nautical miles.
    If this is -999, it means the radius is unknown/unrecorded, so we use a bounding box around UF.
    Otherwise, we calculate the geodesic distance from the storm center to UF
    and check if it is within the radius of maximum wind.
    """
    if row['radius_of_max_wind_nm'] == -999:
        # Use bounding box if no radius data
        return (
            UF_LAT - 0.75 <= row['Latitude'] <= UF_LAT + 0.75 and
            UF_LON - 0.75 <= row['Longitude'] <= UF_LON + 0.75
        )
    else:
        # Use radius-based geodesic distance
        storm_coord = (row['Latitude'], row['Longitude'])
        distance_km = geodesic(storm_coord, UF_COORDS).km
        max_radius_km = row['radius_of_max_wind_nm'] * 1.852  # convert to km
        return distance_km <= max_radius_km


def get_risk_color(prob):
    percent = prob * 100
    if percent < 0.5:
        return '#0000ff'
    elif percent < 1:
        return '#0033ff'
    elif percent < 2:
        return '#0066ff'
    elif percent < 3:
        return '#0099ff'
    elif percent < 4:
        return '#00ccff'
    elif percent < 6:
        return '#00ffcc'
    elif percent < 8:
        return '#00ff66'
    elif percent < 10:
        return '#66ff00'
    elif percent < 12:
        return '#aaff00'
    elif percent < 15:
        return '#e6ff00'
    elif percent < 18:
        return '#ffcc00'
    elif percent < 22:
        return '#ff9900'
    elif percent < 26:
        return '#ff6600'
    elif percent < 30:
        return '#ff3300'
    else:
        return '#ff0000'
    
# -------------------- Constants --------------------
GRID_SIZE = 1.0
UF_LAT, UF_LON = 29.643946, -82.355659
UF_COORDS = (UF_LAT, UF_LON)
MAX_TRACKS = 300 # Limit to avoid cluttering the map and performance issues, still going slow sometimes

# -------------------- Main Script --------------------
# --- Load raw hurricane data ---
storm_df = pd.read_csv("hurdat2.csv")

storm_df['Latitude'] = storm_df['latitude'].apply(convert_lat)
storm_df['Longitude'] = storm_df['longitude'].apply(convert_lon)

# --- Filter to hurricanes only ---
storm_df = storm_df[storm_df['maximum_sustained_wind_knots'] >= 64] # 64 knots = 74 mph = cat 1 hurricane strength

# --- Assign grid cells (1-degree resolution) ---
storm_df['lat_bin'] = storm_df['Latitude'].apply(lambda x: np.floor(x / GRID_SIZE) * GRID_SIZE)
storm_df['lon_bin'] = storm_df['Longitude'].apply(lambda x: np.floor(x / GRID_SIZE) * GRID_SIZE)
storm_df['grid_cell'] = list(zip(storm_df['lat_bin'], storm_df['lon_bin']))

uf_hits = storm_df[storm_df.apply(is_uf_hit, axis=1)]
uf_storm_ids = set(uf_hits['storm_id'])

# --- Compute total and UF-hit counts per grid cell ---
grid_stats = storm_df.groupby(['lat_bin', 'lon_bin'])['storm_id'].nunique().reset_index()
grid_stats.columns = ['lat_bin', 'lon_bin', 'total']

uf_subset = storm_df[storm_df['storm_id'].isin(uf_storm_ids)]
uf_grid_stats = uf_subset.groupby(['lat_bin', 'lon_bin'])['storm_id'].nunique().reset_index()
uf_grid_stats.columns = ['lat_bin', 'lon_bin', 'uf_hits']

# --- Merge counts and calculate probabilities ---
grid_df = pd.merge(grid_stats, uf_grid_stats, how='left', on=['lat_bin', 'lon_bin'])
grid_df['uf_hits'] = grid_df['uf_hits'].fillna(0)
grid_df['prob_to_uf'] = grid_df['uf_hits'] / grid_df['total']

storm_to_cells = storm_df.groupby('storm_id')['grid_cell'].apply(set)

# --- Streamlit layout ---
st.set_page_config(
    page_title = "UF Hurricane Risk",
    page_icon="🌀",
)
st.title("Hurricane Risk Zones Near UF")
st.markdown("""
This tool visualizes the probability that a storm **in a given region** will later pass within ~50 miles of the University of Florida (Gainesville).
Use the sidebar to adjust probability thresholds and toggle storm tracks.
""")
threshold = st.sidebar.slider("Minimum Probability Threshold", 0.0, 1.0, 0.05, 0.01)
show_tracks = st.sidebar.checkbox("Show storm tracks", value=False)

# --- Filter visible grid cells ---
filtered_grids = grid_df[grid_df['prob_to_uf'] >= threshold]
visible_cells = set(tuple(x) for x in filtered_grids[['lat_bin', 'lon_bin']].values)

# --- Identify storms passing through visible cells ---
visible_storms = [
    storm_id for storm_id, cells in storm_to_cells.items()
    if any(cell in visible_cells for cell in cells)
]

if len(visible_storms) > MAX_TRACKS:
    st.warning(f"Too many storms to display ({len(visible_storms)}). Showing only the first {MAX_TRACKS}.")
    visible_storms = visible_storms[:MAX_TRACKS]