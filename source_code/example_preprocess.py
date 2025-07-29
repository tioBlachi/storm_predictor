import pandas as pd
import numpy as np

GRID_SIZE = 1.0

def convert_lat(lat_str):
    return float(lat_str[:-1]) * (1 if lat_str[-1] == 'N' else -1)

def convert_lon(lon_str):
    return float(lon_str[:-1]) * (-1 if lon_str[-1] == 'W' else 1)

def preprocess_data(filepath):
    storm_df = pd.read_csv(filepath)
    storm_df['Latitude'] = storm_df['latitude'].apply(convert_lat)
    storm_df['Longitude'] = storm_df['longitude'].apply(convert_lon)

    # Assign grid cells
    storm_df['lat_bin'] = storm_df['Latitude'].apply(lambda x: np.floor(x / GRID_SIZE) * GRID_SIZE)
    storm_df['lon_bin'] = storm_df['Longitude'].apply(lambda x: np.floor(x / GRID_SIZE) * GRID_SIZE)
    storm_df['grid_cell'] = list(zip(storm_df['lat_bin'], storm_df['lon_bin']))

    # Extract year
    storm_df['year'] = pd.to_datetime(storm_df['date'], format='%Y%m%d', errors='coerce').dt.year
    return storm_df
