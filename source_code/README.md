# Hurricane Storm Risk Project
This project visualized the probability that a storm in a given Atlantic region will later pass within ~50 miles of the University of Florida.

It uses historical storm track data to compute the probabilities and evaluate prediction accuracy with MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

## Requirements
- Python 3.12+
- Dependencies listed in requirements.txt

## Installation
```bash
pip install -r requirements.txt
```

## Running the Main Interactive Map
```bash
streamlit run source_code/storm_predictor.py
```
This also serves as visualization.

## Exploring The Map
The global MAE and RMSE are displayed above the map. 
You can click colored grid sections to see the predicted probability, observed probability and MAE for that specific grid cell.

## Filtering Grid By Prediction Threshold
Adjust the slider in the left sidebar to show only grid cells at or above the chosen probability threshold.
Higher thresholds display fewer cells; lower thresholds display more.
### Note: Performance issues may occur while reloading the map when adjusting the slider.