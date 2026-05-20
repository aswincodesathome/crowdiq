"""
CrowdIQ — XGBoost Training Script
====================================
Trains an XGBoost regressor for crowd prediction.
Works alongside LSTM for the ensemble.

Usage:
    python train_xgboost.py --city Chennai
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ─── FEATURE ENGINEERING ──────────────────────────────
def build_xgb_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)

    # Time
    df['hour']         = df['timestamp'].dt.hour
    df['dow']          = df['timestamp'].dt.dayofweek
    df['month']        = df['timestamp'].dt.month
    df['week']         = df['timestamp'].dt.isocalendar().week.astype(int)
    df['is_weekend']   = (df['dow'] >= 5).astype(int)
    df['is_peak_am']   = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_peak_pm']   = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_lunch']     = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
    df['is_night']     = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin']= np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']= np.cos(2 * np.pi * df['month'] / 12)

    # Route encoding
    le = LabelEncoder()
    df['route_enc'] = le.fit_transform(df['route_id'])

    # Lag features
    for route in df['route_id'].unique():
        m = df['route_id'] == route
        s = df.loc[m, 'crowd_pct']
        df.loc[m, 'lag_1h']          = s.shift(1)
        df.loc[m, 'lag_2h']          = s.shift(2)
        df.loc[m, 'lag_3h']          = s.shift(3)
        df.loc[m, 'lag_24h']         = s.shift(24)
        df.loc[m, 'lag_48h']         = s.shift(48)
        df.loc[m, 'lag_168h']        = s.shift(168)
        df.loc[m, 'rolling_3h']      = s.shift(1).rolling(3).mean()
        df.loc[m, 'rolling_6h']      = s.shift(1).rolling(6).mean()
        df.loc[m, 'rolling_24h']     = s.shift(1).rolling(24).mean()
        df.loc[m, 'rolling_7d']      = s.shift(1).rolling(168).mean()
        df.loc[m, 'rolling_3h_std']  = s.shift(1).rolling(3).std()
        df.loc[m, 'trend_1h']        = s.diff(1).shift(1)
        df.loc[m, 'trend_24h']       = s.diff(24).shift(1)

    df = df.dropna()
    return df, le


XGB_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_peak_am', 'is_peak_pm', 'is_lunch', 'is_night', 'week',
    'weather_code', 'temperature', 'has_event', 'is_holiday', 'route_enc',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_3h', 'rolling_6h', 'rolling_24h', 'rolling_7d',
    'rolling_3h_std', 'trend_1h', 'trend_24h',
]


# ─── TRAINING ─────────────────────────────────────────
def train(args):
    data_path = f"data/{args.city.lower()}_crowd_data.csv"
    if not os.path.exists(data_path):
        print(f"[!] Data file not found: {data_path}")
        return

    df_raw = pd.read_csv(data_path)
    df, le = build_xgb_features(df_raw)
    print(f"[✓] Loaded & engineered {len(df):,} records")

    X = df[XGB_FEATURES].values
    y = df['crowd_pct'].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    print(f"[✓] Train: {len(X_tr):,} | Val: {len(X_val):,}")

    model = xgb.XGBRegressor(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        objective         = 'reg:squarederror',
        eval_metric       = ['rmse', 'mae'],
        early_stopping_rounds = 20,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate
    y_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2   = r2_score(y_val, y_pred)
    print(f"\n[✓] Validation results:")
    print(f"    MAE:  {mae:.2f}%")
    print(f"    RMSE: {rmse:.2f}%")
    print(f"    R²:   {r2:.4f}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=XGB_FEATURES).sort_values(ascending=False)
    print(f"\n[✓] Top 10 important features:")
    print(fi.head(10).to_string())

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(model, f"checkpoints/xgb_{args.city.lower()}.pkl")
    joblib.dump(le,    f"checkpoints/route_encoder_{args.city.lower()}.pkl")
    print(f"\n[✓] Saved: checkpoints/xgb_{args.city.lower()}.pkl")


# ─── INFERENCE ────────────────────────────────────────
def predict_crowd(city, features_df):
    """
    features_df: DataFrame with XGB_FEATURES columns (1 row per prediction)
    Returns: array of predicted crowd percentages
    """
    model = joblib.load(f"checkpoints/xgb_{city.lower()}.pkl")
    preds = model.predict(features_df[XGB_FEATURES].values)
    return np.clip(preds, 0, 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Chennai')
    args = parser.parse_args()
    train(args)
