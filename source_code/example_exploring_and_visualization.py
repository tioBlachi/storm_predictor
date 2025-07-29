from geopy.distance import geodesic
import numpy as np

UF_LAT, UF_LON = 29.643946, -82.355659
UF_COORDS = (UF_LAT, UF_LON)

def is_uf_hit(row):
    if row['radius_of_max_wind_nm'] == -999:
        return geodesic((row['Latitude'], row['Longitude']), UF_COORDS).km <= 100
    else:
        distance_km = geodesic((row['Latitude'], row['Longitude']), UF_COORDS).km
        max_radius_km = row['radius_of_max_wind_nm'] * 1.852
        return distance_km <= max_radius_km

def get_risk_color(prob):
    percent = prob * 100
    if percent < 0.5: return '#0000ff'
    elif percent < 1: return '#0033ff'
    elif percent < 2: return '#0066ff'
    elif percent < 3: return '#0099ff'
    elif percent < 4: return '#00ccff'
    elif percent < 6: return '#00ffcc'
    elif percent < 8: return '#00ff66'
    elif percent < 10: return '#66ff00'
    elif percent < 12: return '#aaff00'
    elif percent < 15: return '#e6ff00'
    elif percent < 18: return '#ffcc00'
    elif percent < 22: return '#ff9900'
    elif percent < 26: return '#ff6600'
    elif percent < 30: return '#ff3300'
    else: return '#ff0000'

def compute_metrics(grid_df):
    mae = grid_df['abs_error'].mean()
    rmse = np.sqrt(grid_df['squared_error'].mean())
    return mae, rmse
