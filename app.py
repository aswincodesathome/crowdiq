"""
CrowdIQ Backend — Flask API with Real-Time Updates
====================================================
Serves predictions from trained ML models with live WebSocket updates.

Usage:
    python app.py
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import os
import json
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import AI integration
try:
    from api_integration import RecommendationAdvisor, GoogleMapsHelper, CommutePlanner
    AI_AVAILABLE = True
    print("✓ Google Maps + Crowd Intelligence loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ Google Maps integration not available: {e}")

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')
CORS(app)

# Load models for both cities
CITIES = ['Chennai', 'Bengaluru', 'Mumbai', 'Delhi']
MODELS = {}
SCALERS = {}

def load_models():
    """Load trained models and scalers"""
    for city in CITIES:
        try:
            # Load XGBoost
            xgb_path = f'checkpoints/xgb_{city.lower()}.pkl'
            if os.path.exists(xgb_path):
                MODELS[f'{city}_xgb'] = joblib.load(xgb_path)
            
            # Load LSTM
            lstm_path = f'checkpoints/lstm_{city.lower()}_best.pt'
            if os.path.exists(lstm_path):
                MODELS[f'{city}_lstm'] = torch.load(lstm_path, map_location='cpu')
            
            # Load route encoder
            encoder_path = f'checkpoints/route_encoder_{city.lower()}.pkl'
            if os.path.exists(encoder_path):
                MODELS[f'{city}_encoder'] = joblib.load(encoder_path)
                
            print(f'✓ Loaded models for {city}')
        except Exception as e:
            print(f'! Could not load models for {city}: {e}')

# Load models on startup
load_models()

# Real-time data storage
LIVE_DATA = {city: {} for city in CITIES}
UPDATE_RUNNING = False

def generate_live_predictions(city):
    """Generate real-time crowd predictions for a city"""
    try:
        # Load latest data
        data_path = f'data/{city.lower()}_crowd_data.csv'
        if not os.path.exists(data_path):
            return None
        
        df = pd.read_csv(data_path).tail(1000)  # Last 1000 records
        
        if len(df) == 0:
            return None
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Build XGBoost features (same as training)
        df['hour'] = df['timestamp'].dt.hour
        df['dow'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        df['is_peak_am'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_peak_pm'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_lunch'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 6, 24, 168]:
            df[f'lag_{lag}h'] = df['crowd_pct'].shift(lag)
        
        # Fill NaN values (forward fill then backward fill for missing)
        df = df.bfill().fillna(0)
        
        # Get latest predictions
        latest_row = df.iloc[-1]
        
        # Prepare features for XGBoost (30 features)
        feature_cols = [
            'hour', 'dow', 'month', 'week', 'is_weekend', 'is_peak_am', 'is_peak_pm',
            'is_lunch', 'is_night', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos', 'lag_1h', 'lag_6h', 'lag_24h', 'lag_168h'
        ]
        
        # Add additional features if they exist
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_features:
            if col not in feature_cols and col != 'crowd_pct':
                feature_cols.append(col)
        
        # Get data for features
        predictions = {
            'timestamp': str(latest_row['timestamp']),
            'overall_avg': float(df['crowd_pct'].mean()),
            'overall_peak': float(df['crowd_pct'].max()),
            'routes': {}
        }
        
        # Group by route
        for route_id in df['route_id'].unique():
            route_data = df[df['route_id'] == route_id]
            predictions['routes'][str(route_id)] = {
                'avg_crowd': float(route_data['crowd_pct'].mean()),
                'max_crowd': float(route_data['crowd_pct'].max()),
                'current': float(route_data['crowd_pct'].iloc[-1])
            }
        
        # XGBoost predictions
        try:
            if f'{city}_xgb' in MODELS:
                xgb_model = MODELS[f'{city}_xgb']
                X = df[feature_cols].fillna(0).astype(float)
                preds = xgb_model.predict(X)
                predictions['predictions'] = {
                    'xgboost': {
                        'current': float(preds[-1]),
                        'avg_24h': float(np.mean(preds[-24:])) if len(preds) >= 24 else float(np.mean(preds)),
                        'peak_hour': int(np.argmax(preds[-24:]) if len(preds) >= 24 else 0)
                    }
                }
        except Exception as e:
            predictions['predictions'] = {'error': str(e)}
        
        LIVE_DATA[city] = predictions
        return predictions
        
    except Exception as e:
        print(f'Error generating predictions for {city}: {e}')
        LIVE_DATA[city] = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        return None

    except Exception as e:
        print(f'Error generating predictions for {city}: {e}')
        return None

def background_updater():
    """Background thread for real-time updates"""
    global UPDATE_RUNNING
    UPDATE_RUNNING = True
    
    while UPDATE_RUNNING:
        for city in CITIES:
            try:
                generate_live_predictions(city)
            except Exception as e:
                print(f'Update error for {city}: {e}')
        
        time.sleep(10)  # Update every 10 seconds

# Start background updater
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# ─── API ENDPOINTS ───────────────────────────

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get available cities and their model status"""
    cities_data = []
    for city in CITIES:
        has_xgb = f'{city}_xgb' in MODELS
        has_lstm = f'{city}_lstm' in MODELS
        cities_data.append({
            'name': city,
            'xgboost': has_xgb,
            'lstm': has_lstm,
            'data': has_xgb or has_lstm
        })
    return jsonify(cities_data)

@app.route('/api/predictions/<city>', methods=['GET'])
def get_predictions(city):
    """Get current predictions for a city"""
    city = city.title()
    
    if city not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    if city not in LIVE_DATA or not LIVE_DATA[city]:
        generate_live_predictions(city)
    
    return jsonify(LIVE_DATA.get(city, {}))

@app.route('/api/predictions/<city>/route/<route>', methods=['GET'])
def get_route_prediction(city, route):
    """Get prediction for specific route in city"""
    city = city.title()
    
    if city not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    if city not in LIVE_DATA or not LIVE_DATA[city]:
        generate_live_predictions(city)
    
    live_data = LIVE_DATA.get(city, {})
    routes = live_data.get('routes', {})
    
    if route in routes:
        return jsonify(routes[route])
    
    return jsonify({'error': 'Route not found'}), 404

@app.route('/api/forecast/<city>', methods=['GET'])
def get_forecast(city):
    """Get hourly forecast for next 24 hours"""
    city = city.title()
    
    if city not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    try:
        data_path = f'data/{city.lower()}_crowd_data.csv'
        if not os.path.exists(data_path):
            return jsonify({'error': 'Data not found'}), 404
        
        df = pd.read_csv(data_path).tail(2000)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Average by hour
        hourly_avg = df.groupby('hour')['crowd_pct'].mean().to_dict()
        
        # Generate 24-hour forecast
        current_hour = datetime.now().hour
        forecast = []
        for i in range(24):
            hour = (current_hour + i) % 24
            base_crowd = hourly_avg.get(hour, 50)
            # Add slight randomness
            predicted = base_crowd + np.random.normal(0, 3)
            forecast.append({
                'hour': hour,
                'crowd': max(0, min(100, float(predicted))),
                'timestamp': (datetime.now() + timedelta(hours=i)).isoformat()
            })
        
        return jsonify(forecast)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cities_available': len([c for c in CITIES if LIVE_DATA[c]]),
        'total_cities': len(CITIES),
        'message': 'Backend operational'
    })

@app.route('/api/debug', methods=['GET'])
def debug():
    """Debug endpoint to check connection and CORS"""
    return jsonify({
        'status': 'ok',
        'api_base': '/api',
        'backend_url': 'http://localhost:5000',
        'cities': CITIES,
        'models_loaded': len(MODELS),
        'updater_running': UPDATE_RUNNING,
        'live_data_keys': list(LIVE_DATA.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get backend status and stats"""
    stats = {}
    for city in CITIES:
        data = LIVE_DATA.get(city, {})
        stats[city] = {
            'has_data': bool(data),
            'last_update': data.get('timestamp', None),
            'avg_crowd': data.get('overall_avg', None),
            'peak_crowd': data.get('overall_peak', None)
        }
    
    return jsonify({
        'backend_running': True,
        'timestamp': datetime.now().isoformat(),
        'cities': stats,
        'thread_active': updater_thread.is_alive()
    })

# ─── ERROR HANDLERS ───────────────────────────

# ─── AI ADVISOR ENDPOINTS ─────────────────────────

@app.route('/api/ai-advisor/<city>', methods=['POST'])
def ai_advisor(city):
    """Get recommendations using crowd data + Google Maps"""
    city = city.title()
    
    if city not in CITIES:
        return jsonify({'error': 'City not found'}), 404
    
    if not AI_AVAILABLE:
        return jsonify({
            'error': 'Google Maps integration not available',
            'message': 'Please ensure GOOGLE_API_KEY is set in .env file'
        }), 503
    
    try:
        # Get user query from request
        data = request.get_json() or {}
        user_query = data.get('query', None)
        
        # Get latest crowd data
        if city not in LIVE_DATA or not LIVE_DATA[city]:
            generate_live_predictions(city)
        
        crowd_data = LIVE_DATA.get(city, {})
        
        # Generate recommendation based on crowd data
        recommendation = RecommendationAdvisor.generate_recommendation(
            city, crowd_data, user_query
        )
        
        return jsonify({
            'city': city,
            'query': user_query or 'General commute advice',
            'recommendation': recommendation,
            'crowd_data': crowd_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': f'Advisor error: {str(e)}'}), 500


@app.route('/api/plan-commute', methods=['POST'])
def plan_commute():
    """Plan optimal commute with AI recommendations"""
    
    if not AI_AVAILABLE:
        return jsonify({'error': 'Commute planner not available'}), 503
    
    try:
        data = request.get_json() or {}
        origin = data.get('origin')
        destination = data.get('destination')
        city = data.get('city', 'Chennai').title()
        
        if not origin or not destination:
            return jsonify({'error': 'Origin and destination required'}), 400
        
        if city not in CITIES:
            return jsonify({'error': 'City not found'}), 404
        
        # Get latest crowd data
        if city not in LIVE_DATA or not LIVE_DATA[city]:
            generate_live_predictions(city)
        
        crowd_data = LIVE_DATA.get(city, {})
        
        # Plan commute
        plan = CommutePlanner.plan_commute(origin, destination, city, crowd_data)
        
        return jsonify(plan)
    
    except Exception as e:
        return jsonify({'error': f'Commute planning error: {str(e)}'}), 500


@app.route('/api/locations/<city>', methods=['GET'])
def nearby_locations(city):
    """Get nearby transit locations using Google Maps"""
    
    if not AI_AVAILABLE:
        return jsonify({'error': 'Location service not available'}), 503
    
    try:
        city = city.title()
        location = request.args.get('location', f'{city}, India')
        
        # Known city coordinates
        city_coords = {
            'Chennai': '13.0827,80.2707',
            'Bengaluru': '12.9716,77.5946',
            'Mumbai': '19.0760,72.8777',
            'Delhi': '28.7041,77.1025'
        }
        
        if city in city_coords:
            location = city_coords[city]
        
        locations = GoogleMapsHelper.get_nearby_locations(location)
        
        if locations:
            return jsonify(locations)
        else:
            return jsonify({'error': 'Could not fetch nearby locations'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Location error: {str(e)}'}), 500


# ─── ERROR HANDLERS ───────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print(f'🚀 CrowdIQ Backend Starting...')
    print(f'Available cities: {", ".join(CITIES)}')
    print(f'✓ Models loaded: {len(MODELS)}')
    print(f'✓ Real-time updates enabled')
    print(f'\n📡 Server running on http://localhost:5000')
    print(f'📊 Dashboard: http://localhost:5000')
    print(f'📈 API Status: http://localhost:5000/api/status')
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
