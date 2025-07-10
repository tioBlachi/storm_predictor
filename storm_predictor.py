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
