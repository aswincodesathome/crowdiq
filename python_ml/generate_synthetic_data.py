"""
CrowdIQ — Synthetic Data Generator
======================================
Generates realistic crowd data for model training.
Simulates 2+ years of hourly transport data.

Usage:
    python generate_synthetic_data.py --city Chennai --years 2
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

ROUTE_CONFIGS = {
    'Chennai': [
        {'id': 'M1', 'name': 'Metro L1 Airport', 'type': 'metro', 'base_load': 72, 'capacity': 2000},
        {'id': 'M2', 'name': 'Metro L2 Harbour',  'type': 'metro', 'base_load': 58, 'capacity': 1800},
        {'id': 'M3', 'name': 'Metro L3 Suburban', 'type': 'metro', 'base_load': 65, 'capacity': 1800},
        {'id': 'B45','name': 'Bus 45C T.Nagar',   'type': 'bus',   'base_load': 84, 'capacity': 80},
        {'id': 'B12','name': 'Bus 12E Anna Nagar', 'type': 'bus',   'base_load': 51, 'capacity': 80},
        {'id': 'B7', 'name': 'Bus 7A Marina',      'type': 'bus',   'base_load': 41, 'capacity': 80},
        {'id': 'F1', 'name': 'Ferry Kovalam',       'type': 'ferry', 'base_load': 35, 'capacity': 200},
    ],
    'Mumbai': [
        {'id': 'WL', 'name': 'Western Line',  'type': 'metro', 'base_load': 91, 'capacity': 3000},
        {'id': 'CL', 'name': 'Central Line',  'type': 'metro', 'base_load': 88, 'capacity': 3000},
        {'id': 'HL', 'name': 'Harbour Line',  'type': 'metro', 'base_load': 76, 'capacity': 2500},
        {'id': 'B1', 'name': 'BEST Bus',       'type': 'bus',   'base_load': 79, 'capacity': 80},
    ],
    'Delhi': [
        {'id': 'YL', 'name': 'Yellow Line',  'type': 'metro', 'base_load': 85, 'capacity': 2400},
        {'id': 'BL', 'name': 'Blue Line',    'type': 'metro', 'base_load': 80, 'capacity': 2400},
        {'id': 'RL', 'name': 'Red Line',     'type': 'metro', 'base_load': 68, 'capacity': 2000},
        {'id': 'DTC','name': 'DTC Bus 470',  'type': 'bus',   'base_load': 72, 'capacity': 60},
    ],
}

HOUR_PATTERN = [
    0.12, 0.09, 0.07, 0.06, 0.08, 0.15,
    0.38, 0.85, 1.40, 1.20, 0.90, 0.78,
    0.95, 0.88, 0.72, 0.68, 0.82, 1.35,
    1.30, 1.10, 0.85, 0.60, 0.38, 0.20,
]

WEATHER_CODES = {'clear': 0, 'cloudy': 1, 'rain': 2, 'heavy_rain': 3, 'storm': 4}
WEATHER_IMPACT = {0: 1.0, 1: 1.02, 2: 1.18, 3: 1.30, 4: 0.70}
WEATHER_PROBS  = {0: 0.50, 1: 0.25, 2: 0.15, 3: 0.07, 4: 0.03}


def generate_data(city, years=2):
    rng = np.random.default_rng(seed=42)
    routes = ROUTE_CONFIGS.get(city, ROUTE_CONFIGS['Chennai'])

    start_date = datetime(2022, 1, 1)
    end_date   = start_date + timedelta(days=365 * years)
    timestamps = pd.date_range(start_date, end_date, freq='1h')

    records = []

    # Pre-generate weather (same for all routes at same time)
    weather_series = {}
    current_weather = 0
    for ts in timestamps:
        if rng.random() < 0.05:
            current_weather = rng.choice(list(WEATHER_CODES.values()),
                                         p=list(WEATHER_PROBS.values()))
        weather_series[ts] = current_weather

    # Public holidays (approximate)
    holidays = set()
    for year in range(2022, 2025):
        for m, d in [(1,1),(1,26),(3,8),(8,15),(10,2),(10,15),(11,1),(12,25)]:
            try: holidays.add(datetime(year, m, d).date())
            except: pass

    # IPL season (approx April-May)
    ipl_dates = set()
    for year in range(2022, 2025):
        for d in range(1, 60):
            try: ipl_dates.add((datetime(2022, 4, 1) + timedelta(days=d)).date())
            except: pass

    print(f"Generating {len(timestamps) * len(routes):,} records for {city}...")

    for route in routes:
        for ts in timestamps:
            hour     = ts.hour
            dow      = ts.dayofweek
            is_wknd  = int(dow >= 5)
            is_hol   = int(ts.date() in holidays)
            has_event= int(ts.date() in ipl_dates)
            wcode    = weather_series[ts]

            # Compute crowd
            dow_factor  = 0.65 if is_wknd else (1.08 if dow == 4 else 1.0)
            noise       = rng.normal(0, 6)
            crowd = (route['base_load']
                     * HOUR_PATTERN[hour]
                     * dow_factor
                     * WEATHER_IMPACT[wcode]
                     * (1.22 if has_event else 1.0)
                     * (0.55 if is_hol else 1.0))
            crowd = float(np.clip(crowd + noise, 2, 99))

            # Temperature (Chennai: 24-38°C, seasonal)
            month = ts.month
            base_temp = 30 + 4 * np.sin(2 * np.pi * (month - 4) / 12)
            temp = float(base_temp + rng.normal(0, 2))

            records.append({
                'timestamp':   ts,
                'route_id':    route['id'],
                'route_name':  route['name'],
                'route_type':  route['type'],
                'crowd_pct':   round(crowd, 1),
                'weather_code':wcode,
                'temperature': round(temp, 1),
                'has_event':   has_event,
                'is_holiday':  is_hol,
            })

    df = pd.DataFrame(records)
    os.makedirs('data', exist_ok=True)
    path = f"data/{city.lower()}_crowd_data.csv"
    df.to_csv(path, index=False)
    print(f"[✓] Saved {len(df):,} records to {path}")
    print(f"    Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"    Routes: {df['route_id'].nunique()}")
    print(f"    Mean crowd: {df['crowd_pct'].mean():.1f}%")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city',  type=str, default='Chennai')
    parser.add_argument('--years', type=int, default=2)
    args = parser.parse_args()
    generate_data(args.city, args.years)
