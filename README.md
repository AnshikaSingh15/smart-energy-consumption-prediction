# Smart Energy Consumption Prediction

Predicts daily household electricity consumption and flags likely high-usage days, so a household or provider can act on it — e.g. shift laundry/dishwasher use to off-peak hours.

Built on the UCI household power consumption dataset + historical weather data, modeled with XGBoost.

## Approach

- Cleaned minute-level data, resampled to daily totals
- Engineered calendar, lag, rolling-average, and weather (heating-degree) features
- Trained XGBoost, tuned with `RandomizedSearchCV` using time-aware CV (no lookahead leakage)
- Compared against naive baselines instead of judging accuracy in isolation
- Used SHAP to understand what the model actually relies on
- Investigated the worst prediction errors against raw sub-metering data
- Built and validated a threshold-based high-usage alert system

## Results

| Model | MAE | RMSE |
|---|---|---|
| Naive (yesterday) | 286.29 | 405.35 |
| Naive (last week) | 387.99 | 532.19 |
| **Tuned XGBoost** | **247.80** | **338.54** |

~13–17% improvement over the best naive baseline.

**Key findings:**
- Recent consumption (yesterday, 7-day average) is by far the strongest predictor
- Weekends run ~17% higher than weekdays
- Weather helps, but less than expected — raw temperature slightly hurt accuracy; an engineered "heating degree" feature fixed that, and SHAP confirmed a real, physically sensible effect
- 8 of the 10 largest errors trace back to occupancy anomalies (travel, guests) confirmed via sub-metering data — not predictable from any available feature
- High-usage alerts (85th-percentile threshold): 67% recall, ~9% precision — chosen deliberately to favor catching real spikes over avoiding false alarms

## Limitations

- Single household — won't generalize without retraining
- No occupancy signal — can't predict "house is empty/busy" days, the biggest error source
- ~4 years of data — hasn't seen every seasonal extreme
- Alert system favors recall over precision by design

## Structure
Energy-prediction/
├── data/ # dataset (see link below)
├── models/ # saved model
├── notebooks/
│ └── exploration.ipynb
├── requirements.txt
└── README.md

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), place `household_power_consumption.txt` in `data/`, then open `notebooks/exploration.ipynb`. Weather data is pulled live from [Open-Meteo](https://open-meteo.com/) in the notebook.