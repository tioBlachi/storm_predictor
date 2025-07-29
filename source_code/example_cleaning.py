def clean_data(df):
    # Example: filter to Cat 1+ hurricanes
    # df = df[df['maximum_sustained_wind_knots'] >= 64]
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df
