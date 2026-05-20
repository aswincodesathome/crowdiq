"""
CrowdIQ Google Maps Integration
=============================
Handles Google API integration for commute planning and crowd recommendations.
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class GoogleMapsHelper:
    """Helper class for Google Maps API calls"""
    
    @staticmethod
    def get_nearby_locations(location, radius=1000, place_type='transit_station'):
        """Get nearby transit locations using Google Places API"""
        try:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': location,
                'radius': radius,
                'type': place_type,
                'key': GOOGLE_API_KEY
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error fetching nearby locations: {e}")
            return None
    
    @staticmethod
    def get_directions(origin, destination, mode='transit'):
        """Get directions between two locations"""
        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'key': GOOGLE_API_KEY
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error fetching directions: {e}")
            return None
    
    @staticmethod
    def geocode(address):
        """Convert address to coordinates"""
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': GOOGLE_API_KEY
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error geocoding: {e}")
            return None


class RecommendationAdvisor:
    """Google-based recommendation system using crowd data and Google Maps"""
    
    @staticmethod
    def generate_recommendation(city, crowd_data, user_query=None):
        """Generate recommendations based on crowd data and Google Maps info"""
        try:
            # Extract crowd information
            overall_avg = crowd_data.get('overall_avg', 50)
            current_crowd = crowd_data.get('current_crowd', 60)
            peak_time = crowd_data.get('peak_hour', 'N/A')
            
            # Build recommendation logic
            if current_crowd < 40:
                recommendation = f"✓ Good time to travel! Current crowd level: {current_crowd}% (low). Safe to proceed."
            elif current_crowd < 70:
                recommendation = f"⚠ Moderate crowds detected: {current_crowd}%. Recommended: Travel after {peak_time}."
            else:
                recommendation = f"⛔ High congestion: {current_crowd}%. Recommended: Avoid this time, check again in 30 minutes."
            
            # Add route suggestions from crowd data
            routes_info = ""
            if crowd_data.get('routes'):
                routes_info = "Alternative routes by crowd level:\n"
                for route, stats in sorted(crowd_data.get('routes', {}).items(), key=lambda x: x[1].get('current', 100)):
                    crowd_level = stats.get('current', 'N/A')
                    routes_info += f"  • {route}: {crowd_level}%\n"
            
            response_text = f"""{recommendation}

{routes_info}
City: {city}
Analysis Time: {crowd_data.get('timestamp', 'Now')}"""
            
            return {
                'success': True,
                'response': response_text,
                'method': 'Google Maps + Crowd Data',
                'recommendation_type': 'crowd_based'
            }
        
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }


class CommutePlanner:
    """Integration of Google Maps + Crowd Data for complete commute planning"""
    
    @staticmethod
    def plan_commute(origin, destination, city, crowd_data):
        """Plan optimal commute considering crowd data and routes"""
        try:
            # Get directions from Google Maps
            directions = GoogleMapsHelper.get_directions(origin, destination, mode='transit')
            
            if not directions or directions.get('status') != 'OK':
                return {'error': 'Could not find route'}
            
            # Get first route info
            route = directions['routes'][0]
            distance = route['legs'][0]['distance']['text']
            duration = route['legs'][0]['duration']['text']
            
            # Apply crowd adjustment to estimated time
            current_crowd = crowd_data.get('current_crowd', 50)
            crowd_factor = 1 + (current_crowd - 50) / 100  # 50% crowd = 1x time
            
            return {
                'success': True,
                'origin': origin,
                'destination': destination,
                'city': city,
                'distance': distance,
                'estimated_time': duration,
                'crowd_adjusted_factor': f"{crowd_factor:.1f}x",
                'current_crowd_level': f"{current_crowd}%",
                'recommendation': f"Current crowd level: {current_crowd}%. Expect {crowd_factor:.1f}x normal travel time.",
                'route_overview': route.get('overview_polyline', {}).get('points', '')
            }
        except Exception as e:
            print(f"Error planning commute: {e}")
            return {'success': False, 'error': str(e)}


# Test function
if __name__ == '__main__':
    # Test Google Maps Helper
    print("🧪 Testing Google Maps Integration...")
    print(f"API Key Status: {'✓ Loaded' if GOOGLE_API_KEY else '✗ Missing'}")
    
    # Test Recommendation Advisor
    print("\n🧪 Testing Recommendation Advisor...")
    test_data = {
        'overall_avg': 45.3,
        'overall_peak': 89.2,
        'timestamp': 'Now',
        'current_crowd': 55,
        'routes': {'M1': {'current': 42, 'avg_crowd': 45}, 'B45': {'current': 38, 'avg_crowd': 40}}
    }
    response = RecommendationAdvisor.generate_recommendation('Chennai', test_data)
    print(f"Response Status: {'✓ Success' if response['success'] else '✗ Failed'}")
    if response['success']:
        print(f"Response: {response['response'][:100]}...")
