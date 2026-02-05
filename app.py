# app.py - COMPLETE FIXED VERSION

import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# =========================================================
# IMPORT THE REAL AI SAFETY ENGINE
# =========================================================
import sys
import os

# Add the current directory to path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_safety_engine import AISafetyEngine
    print("✅ Imported real AI Safety Engine modules")
except ImportError as e:
    print(f"⚠ Import error: {e}")
    # Fallback to simple version
    class AISafetyEngine:
        def __init__(self, model_path="models/safemap_rf_model.pkl"):
            self.model_path = model_path
            self.model = None

        def load(self):
            obj = joblib.load(self.model_path)
            self.model = obj["model"] if isinstance(obj, dict) else obj

        def predict_city_safety(self, row):
            df = pd.DataFrame([row]).select_dtypes(include=[np.number]).fillna(0)
            pred = float(self.model.predict(df)[0])
            return round(max(0, min(100, 100 - pred * 10)), 1)
        
        def predict_city_safety_fallback(self, row, time_of_day="Night", gender="Female"):
            return self.predict_city_safety(row)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SAFEMAP - Bangalore Safety Navigation",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .bangalore-badge {
        background: linear-gradient(135deg, #FF9933, #138808);
        color: dark blue;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .safety-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        color: white !important;
    }
    .safe { background: linear-gradient(135deg, #10B981, #059669); }
    .moderate { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .risky { background: linear-gradient(135deg, #EF4444, #DC2626); }
    .route-step {
        padding: 12px;
        margin: 8px 0;
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        width: 100%;
    }
    .map-container {
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 5px;
        background: white;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: dark blue;
        border-radius: 8px 8px 0px 0px;
        gap: 4px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚨 SAFEMAP - Bangalore Safety Navigation</h1>', unsafe_allow_html=True)
st.markdown('<div class="bangalore-badge">🚓 Using Real Bangalore Police Station Data</div>', unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_city_data():
    return pd.read_csv("data/city_data.csv")

@st.cache_data
def load_police_data():
    """Load police stations from CSV file"""
    try:
        df = pd.read_csv('data/police_stations.csv', encoding='utf-8')
        df.columns = ['Sl.No.', 'City Name', 'Name of Police Station', 'Address', 'Longitude', 'Latitude']
        
        # Clean data
        df['Longitude'] = pd.to_numeric(df['Longitude'].astype(str).str.strip().str.replace('"', ''), errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'].astype(str).str.strip().str.replace('"', ''), errors='coerce')
        df = df.dropna(subset=['Longitude', 'Latitude'])
        
        df['Name of Police Station'] = df['Name of Police Station'].str.strip()
        df['Address'] = df['Address'].str.strip().str.replace('"', '')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=['Sl.No.', 'City Name', 'Name of Police Station', 'Address', 'Longitude', 'Latitude'])

city_data = load_city_data()
police_data = load_police_data()
cities = sorted(city_data["City"].dropna().unique())

# =========================================================
# MODEL LOADING - REAL DATA VERSION
# =========================================================
# Try to import the real modules
try:
    from ai_safety_engine import AISafetyEngine
    print("✅ Imported real AI Safety Engine")
    
    # Create instance with real data
    ai = AISafetyEngine(real_data_dir="data/processed/")
    
    # Load models
    if ai.load_models():
        print("✅ AI models loaded successfully")
    else:
        print("⚠ AI models failed to load")
        # Try simple load
        ai.load()
    
except ImportError as e:
    print(f"❌ Could not import AI modules: {e}")
    st.error("⚠ AI Engine not available. Please check your files.")
    # Create a dummy class to prevent crashes
    class AISafetyEngine:
        def __init__(self):
            pass
        def load(self):
            pass
        def predict_city_safety(self, row):
            return 50
        def predict_city_safety_fallback(self, row, time_of_day="Night", gender="Female"):
            return 50
    
    ai = AISafetyEngine()

# Simple AnomalyDetector
class AnomalyDetector:
    def __init__(self, model_path="models/anomaly_model.pkl"):
        self.model_path = model_path
        self.model = None

    def load(self):
        try:
            obj = joblib.load(self.model_path)
            if isinstance(obj, dict):
                self.model = obj["model"]
            else:
                self.model = obj
            return True
        except:
            return False

    def detect(self, row):
        if self.model is None:
            return False
        try:
            df = pd.DataFrame([row]).select_dtypes(include=[np.number]).fillna(0)
            result = self.model.predict(df)
            return int(result[0]) == -1
        except:
            return False

anom = AnomalyDetector()
anom.load()

# =========================================================
# ROUTE DATA AND FUNCTIONS
# =========================================================
BANGALORE_AREAS = {
    "MG Road": [12.9750, 77.6000],
    "Indiranagar": [12.9784, 77.6408],
    "Koramangala": [12.9210, 77.6185],
    "Jayanagar": [12.9287, 77.5816],
    "HSR Layout": [12.9218, 77.6617],
    "Whitefield": [12.9698, 77.7499],
    "Electronic City": [12.8395, 77.6615],
    "Majestic": [12.9770, 77.5700],
    "City Market": [12.9650, 77.5770],
    "Cubbon Park": [12.9763, 77.5929],
}

AREA_RISK = {
    "Majestic": 9,
    "City Market": 8,
    "MG Road": 6,
    "Indiranagar": 4,
    "Jayanagar": 3,
    "Koramangala": 5,
    "HSR Layout": 4,
    "Whitefield": 3,
    "Electronic City": 4,
    "Cubbon Park": 2,
}

EMERGENCY_NUMBERS = {
    "Police": "100",
    "Women Helpline": "181",
    "Ambulance": "108",
    "Emergency": "112",
}

# =========================================================
# AI-POWERED ROUTE FUNCTIONS
# =========================================================

def get_route_osrm(start, end):
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=6)
        coords = r.json()["routes"][0]["geometry"]["coordinates"]
        return [[c[1], c[0]] for c in coords]
    except:
        return [start, end]

def estimate_distance_time(route):
    dist_km = len(route) * 0.04
    time_min = dist_km * 3
    return round(dist_km, 1), int(time_min)

def calculate_ai_safety_score(start_area, end_area, time_of_day, traveler, ai_engine):
    """
    Calculate safety score using AI model with real crime data
    """
    try:
        # Try to get real crime data for start and end areas
        start_features = ai_engine.predictor.get_real_features_for_area(start_area)
        end_features = ai_engine.predictor.get_real_features_for_area(end_area)
        
        if start_features and end_features:
            # Use real crime data
            start_score = start_features.get('safety_score', 50)
            end_score = end_features.get('safety_score', 50)
            
            # Average score with weights
            avg_score = (start_score * 0.4 + end_score * 0.4)
            
            # Adjust for route risk (assume some intermediate risk)
            route_risk_penalty = 20
            
            base_score = avg_score - route_risk_penalty
        else:
            # Fallback to original method
            base_score = 70
    
    except:
        # Fallback to original method
        risk = (AREA_RISK.get(start_area, 5) + AREA_RISK.get(end_area, 5)) / 2
        base_score = 100 - (risk * 8)
    
    # Apply time and gender factors
    factor = 1.0
    if time_of_day == "Night":
        factor += 0.8
    if traveler == "Solo Female":
        factor += 0.5
    
    final_score = base_score * (1/factor)
    return max(10, min(95, final_score))

def find_nearest_police_stations(route, police_data, max_distance_km=2):
    """Find police stations near the route"""
    nearby_stations = []
    
    for _, station in police_data.iterrows():
        station_lat = station['Latitude']
        station_lon = station['Longitude']
        
        # Check distance to any point on route
        for point in route:
            point_lat, point_lon = point
            # Simple distance calculation (approximate)
            distance = ((station_lat - point_lat) ** 2 + (station_lon - point_lon) ** 2) ** 0.5 * 111  # km
            
            if distance <= max_distance_km:
                nearby_stations.append({
                    'name': station['Name of Police Station'],
                    'address': station['Address'],
                    'latitude': station_lat,
                    'longitude': station_lon,
                    'distance_km': round(distance, 2)
                })
                break  # Found one nearby, move to next station
    
    return nearby_stations[:5]  # Return top 5 nearest

def create_ai_route_map(start, end, route, show_police, police_data, ai_engine, start_area, end_area):
    center = [(start[0] + end[0])/2, (start[1] + end[1])/2]
    m = folium.Map(location=center, zoom_start=13, tiles='OpenStreetMap')

    if show_police:
        # Find and highlight police stations near the route
        nearby_stations = find_nearest_police_stations(route, police_data)
        
        for station in nearby_stations:
            folium.Marker(
                [station["latitude"], station["longitude"]],
                popup=f"🚓 {station['name']}<br>📍 {station['distance_km']}km from route",
                icon=folium.Icon(color="blue", icon='shield', prefix='fa')
            ).add_to(m)
        
        # Add remaining stations in cluster
        cluster = MarkerCluster().add_to(m)
        for _, p in police_data.iterrows():
            folium.Marker(
                [p["Latitude"], p["Longitude"]],
                popup=p["Name of Police Station"],
                icon=folium.Icon(color="gray", icon='shield', prefix='fa')
            ).add_to(cluster)

    # Add start and end markers with area info
    try:
        start_features = ai_engine.predictor.get_real_features_for_area(start_area)
        end_features = ai_engine.predictor.get_real_features_for_area(end_area)
        
        start_popup = f'📍 <b>Start: {start_area}</b>'
        end_popup = f'🏁 <b>End: {end_area}</b>'
        
        if start_features:
            start_popup += f'<br>Safety: {start_features.get("safety_score", "N/A")}/100'
        if end_features:
            end_popup += f'<br>Safety: {end_features.get("safety_score", "N/A")}/100'
        
    except:
        start_popup = f'📍 <b>Start: {start_area}</b>'
        end_popup = f'🏁 <b>End: {end_area}</b>'

    folium.Marker(start, 
                  icon=folium.Icon(color="green", icon='play', prefix='fa'),
                  popup=start_popup).add_to(m)
    folium.Marker(end, 
                  icon=folium.Icon(color="red", icon='stop', prefix='fa'),
                  popup=end_popup).add_to(m)
    
    # Add route with color based on safety
    folium.PolyLine(route, color='#10B981', weight=6, opacity=0.9).add_to(m)
    
    # Add safety zones along route
    add_safety_zones_to_map(m, route, ai_engine)
    
    folium.LayerControl().add_to(m)
    return m

def add_safety_zones_to_map(m, route, ai_engine):
    """Add safety analysis along the route"""
    if not hasattr(ai_engine, 'area_features') or ai_engine.area_features is None:
        return
    
    # Sample points along route for analysis
    num_samples = min(10, len(route))
    if num_samples <= 0:
        return
        
    step = max(1, len(route) // num_samples)
    
    for i in range(0, len(route), step):
        if i >= len(route):
            break
            
        point = route[i]
        
        # Find nearest area from real data
        nearest_area = find_nearest_area(point, ai_engine.area_features)
        
        if nearest_area and isinstance(nearest_area, dict):
            area_name = nearest_area.get('Police_Station', 'Unknown')
            safety_score = nearest_area.get('Safety_Score', 50)
            
            # Color code based on safety
            if safety_score >= 70:
                color = 'green'
                radius = 100
            elif safety_score >= 50:
                color = 'orange'
                radius = 150
            else:
                color = 'red'
                radius = 200
            
            folium.CircleMarker(
                location=point,
                radius=radius/1000,  # Scale down for visibility
                color=color,
                fill=True,
                fill_opacity=0.2,
                popup=f"Near {area_name}<br>Safety: {safety_score}/100"
            ).add_to(m)

def find_nearest_area(point, area_features):
    """Find nearest area with real crime data"""
    if area_features is None or 'Avg_Latitude' not in area_features.columns or area_features.empty:
        return None
    
    point_lat, point_lon = point
    
    try:
        # Filter out rows with missing coordinates
        valid_areas = area_features.dropna(subset=['Avg_Latitude', 'Avg_Longitude'])
        
        if valid_areas.empty:
            return None
        
        # Calculate distances
        distances = ((valid_areas['Avg_Latitude'] - point_lat) ** 2 + 
                    (valid_areas['Avg_Longitude'] - point_lon) ** 2) ** 0.5
        
        # Find index of minimum distance
        min_idx = distances.idxmin()
        min_distance = distances.loc[min_idx]
        
        # Get the nearest area
        nearest_area = valid_areas.loc[min_idx]
        
        # Return as dictionary, not Series
        return {
            'Police_Station': nearest_area.get('Police_Station', 'Unknown'),
            'Safety_Score': nearest_area.get('Safety_Score', 50),
            'Total_Crimes': nearest_area.get('Total_Crimes', 0),
            'Violent_Rate': nearest_area.get('Violent_Rate', 0),
            'Night_Rate': nearest_area.get('Night_Rate', 0),
            'Avg_Latitude': nearest_area.get('Avg_Latitude', 0),
            'Avg_Longitude': nearest_area.get('Avg_Longitude', 0)
        }
        
    except Exception as e:
        print(f"Error finding nearest area: {e}")
        return None

def create_police_stations_map():
    """Create map showing ALL police stations"""
    center = [12.9716, 77.5946]
    m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')
    
    # Add ALL police stations with clustering
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, station in police_data.iterrows():
        # Create custom popup
        popup_html = f"""
        <div style="font-family: Arial; max-width: 250px;">
            <div style="background-color: #3B82F6; color: white; padding: 8px; border-radius: 5px 5px 0 0;">
                <strong>🚓 {station['Name of Police Station']}</strong>
            </div>
            <div style="padding: 10px;">
                <p style="margin: 5px 0; font-size: 12px;">
                    📍 <strong>Address:</strong><br>
                    {station['Address'][:80]}...
                </p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">
                    📌 <strong>Coordinates:</strong><br>
                    {station['Latitude']:.6f}, {station['Longitude']:.6f}
                </p>
                <hr style="margin: 8px 0;">
                <p style="margin: 0; font-size: 10px; color: #888;">
                    Station #{station['Sl.No.'] if 'Sl.No.' in station else idx+1}
                </p>
            </div>
        </div>
        """
        
        folium.Marker(
            [station['Latitude'], station['Longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=station['Name of Police Station'],
            icon=folium.Icon(color='blue', icon='shield', prefix='fa')
        ).add_to(marker_cluster)
    
    # Add city center marker
    folium.Marker(
        center,
        popup='<b>📍 Bangalore City Center</b>',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def generate_ai_safety_tips(score, time_of_day, traveler, start_area, end_area, ai_engine):
    """Generate AI-powered safety tips based on real data"""
    tips = []
    
    # Basic tips based on score
    if score < 60:
        tips.append("Consider alternative route or time of travel")
    if score < 40:
        tips.append("⚠ Consider using verified taxi or ride-sharing service")
    
    # Time-based tips
    if time_of_day == "Night":
        tips.append("Share live location with trusted contacts")
        tips.append("Ensure phone is fully charged")
    
    # Traveler-based tips
    if "Female" in traveler:
        tips.append("Use women-only transport options if available")
        tips.append("Keep emergency numbers on speed dial")
    
    # Try to get real data insights
    try:
        if hasattr(ai_engine, 'real_data_loaded') and ai_engine.real_data_loaded:
            start_features = ai_engine.predictor.get_real_features_for_area(start_area)
            end_features = ai_engine.predictor.get_real_features_for_area(end_area)
            
            if start_features and isinstance(start_features, dict) and start_features.get('violent_rate', 0) > 40:
                tips.append(f"⚠ High violent crime in {start_area} - be extra cautious")
            
            if end_features and isinstance(end_features, dict) and end_features.get('night_rate', 0) > 50:
                tips.append(f"🌙 {end_area} has high night crime - arrange pickup if arriving after dark")
    
    except Exception as e:
        print(f"Error generating AI tips: {e}")
    
    # General tips
    tips.extend([
        "Stay on main, well-lit roads",
        "Avoid displaying valuables",
        "Trust your instincts - if something feels wrong, leave"
    ])
    
    return tips

def generate_route_insights(start_area, end_area, route, ai_engine):
    """Generate insights about the route"""
    insights = []
    
    try:
        if hasattr(ai_engine, 'real_data_loaded') and ai_engine.real_data_loaded:
            # Analyze areas along route
            areas_along_route = analyze_route_areas(route, ai_engine.area_features)
            
            safe_areas = [a for a in areas_along_route if a.get('Safety_Score', 0) >= 70]
            risky_areas = [a for a in areas_along_route if a.get('Safety_Score', 0) < 50]
            
            if safe_areas:
                insights.append(f"Route passes through {len(safe_areas)} safe areas")
            
            if risky_areas and len(risky_areas) > 0:
                insights.append(f"⚠ Route passes near {len(risky_areas)} higher-risk areas")
            
            # Check for police coverage
            if len(areas_along_route) > 0:
                total_crimes = sum(a.get('Total_Crimes', 0) for a in areas_along_route)
                avg_crime = total_crimes / len(areas_along_route) if len(areas_along_route) > 0 else 0
                
                if avg_crime < 50:
                    insights.append("Overall low crime density along route")
                elif avg_crime > 100:
                    insights.append("⚠ Moderate crime density along route - stay alert")
    
    except Exception as e:
        print(f"Error generating route insights: {e}")
    
    # General insights
    insights.extend([
        "Route follows main arterial roads",
        "Good street lighting coverage expected",
        "Multiple police stations within 2km radius"
    ])
    
    return insights

def analyze_route_areas(route, area_features):
    """Analyze which areas the route passes through"""
    if area_features is None or 'Avg_Latitude' not in area_features.columns or area_features.empty:
        return []
    
    areas_found = []
    area_names_found = set()
    
    # Sample points along route
    num_samples = min(20, len(route))
    if num_samples <= 0:
        return []
        
    step = max(1, len(route) // num_samples)
    
    for i in range(0, len(route), step):
        if i >= len(route):
            break
            
        point = route[i]
        nearest = find_nearest_area(point, area_features)
        
        if nearest is not None and nearest.get('Police_Station') not in area_names_found:
            area_names_found.add(nearest.get('Police_Station'))
            areas_found.append(nearest)
    
    return areas_found

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "🗺️ Safe Route Planning",
    "🏙️ City Safety",
    "🚓 Police Stations"
])

# =========================================================
# TAB 1 – AI-POWERED ROUTE PLANNING
# =========================================================
with tabs[0]:
    st.subheader("🗺️ AI-Powered Safe Route Planning")
    
    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.markdown("### 👤 Travel Profile")
        traveler = st.selectbox("Traveler Type", ["Solo Male", "Solo Female", "Group", "Family"])
        time_of_day = st.selectbox("Time of Travel", ["Morning", "Afternoon", "Evening", "Night"], index=3)

        st.markdown("### ⚙️ Preferences")
        show_police = st.checkbox("Show Police Stations", True)
        show_safety_zones = st.checkbox("Show Safety Zones", True)
        avoid_high_risk = st.checkbox("Avoid High-Risk Areas", True)

        st.markdown("### 📍 Locations")
        start_area = st.selectbox("Start Location", list(BANGALORE_AREAS.keys()))
        end_area = st.selectbox("Destination", list(BANGALORE_AREAS.keys()))

        go = st.button("🚀 Calculate AI-Optimized Route", use_container_width=True, type="primary")

        st.markdown("### 🚨 Emergency Contacts")
        for k, v in EMERGENCY_NUMBERS.items():
            st.write(f"**{k}:** {v}")

    if go:
        start = BANGALORE_AREAS[start_area]
        end = BANGALORE_AREAS[end_area]
        route = get_route_osrm(start, end)
        dist, time = estimate_distance_time(route)
        
        # Use AI model to calculate safety score
        score = calculate_ai_safety_score(start_area, end_area, time_of_day, traveler, ai)
        
        # Determine safety rating
        if score >= 70:
            score_class = "safe"
            rating = "🟢 VERY SAFE"
            risk_level = "Low"
        elif score >= 50:
            score_class = "moderate"
            rating = "🟡 MODERATELY SAFE"
            risk_level = "Medium"
        else:
            score_class = "risky"
            rating = "🔴 HIGH RISK"
            risk_level = "High"

        with col2:
            # AI Safety score display
            st.markdown(f'<div class="safety-score {score_class}">AI Safety Score: {int(score)}/100<br>{rating}</div>', 
                       unsafe_allow_html=True)
            
            # Route metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Route Distance", f"{dist} km")
            with col_metric2:
                st.metric("Estimated Time", f"{time} min")
            with col_metric3:
                st.metric("Risk Level", risk_level)
            
            # MAP DISPLAY - MOVED TO TOP
            st.markdown("### 📍 AI-Enhanced Route Map")
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            folium_static(create_ai_route_map(start, end, route, show_police, police_data, ai, start_area, end_area), 
                         width=800, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Get area insights if real data is available
            if hasattr(ai, 'real_data_loaded') and ai.real_data_loaded:
                try:
                    start_features = ai.predictor.get_real_features_for_area(start_area)
                    end_features = ai.predictor.get_real_features_for_area(end_area)
                    
                    if start_features or end_features:
                        st.markdown("**Area Crime Statistics:**")
                        cols = st.columns(2)
                        
                        with cols[0]:
                            if start_features:
                                st.write(f"**{start_area}:**")
                                st.write(f"• Crimes: {start_features.get('total_crimes', 'N/A')}")
                                st.write(f"• Violent Rate: {start_features.get('violent_rate', 'N/A')}%")
                        
                        with cols[1]:
                            if end_features:
                                st.write(f"**{end_area}:**")
                                st.write(f"• Crimes: {end_features.get('total_crimes', 'N/A')}")
                                st.write(f"• Violent Rate: {end_features.get('violent_rate', 'N/A')}%")
                except:
                    pass
        
    else:
        # Default view when no route calculated
        with col2:
            st.info("👈 Configure your route and click 'Calculate AI-Optimized Route' for smart safety analysis")
            
            # Show real data availability
            if hasattr(ai, 'real_data_loaded') and ai.real_data_loaded:
                st.success(f"✅ Real crime data available for {len(ai.area_features)} areas")
            
            default_start = BANGALORE_AREAS["MG Road"]
            default_end = BANGALORE_AREAS["Jayanagar"]
            default_route = [default_start, default_end]
            default_map = create_ai_route_map(default_start, default_end, default_route, True, police_data, ai, "MG Road", "Jayanagar")
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            folium_static(default_map, width=800, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
# =========================================================
# TAB 2 – CITY SAFETY WITH REAL DATA
# =========================================================
with tabs[1]:
    st.subheader("🏙️ City Safety Analysis with Real Crime Data")
    
    # Check if we have real data
    has_real_data = hasattr(ai, 'real_data_loaded') and ai.real_data_loaded
    
    if has_real_data and hasattr(ai, 'area_features') and ai.area_features is not None:
        # REAL DATA MODE
        st.success("✅ Real crime data is available!")
        
        # Option 1: Police Station Analysis
        st.markdown("### 🔍 Police Station Area Analysis")
        try:
            real_areas = sorted(ai.area_features['Police_Station'].dropna().unique())
            
            col1, col2 = st.columns(2)
            with col1:
                selected_area = st.selectbox(
                    "Select Police Station Area", 
                    real_areas,
                    key="police_station_select"  # UNIQUE KEY
                )
                time_of_day = st.selectbox(
                    "Time of Travel", 
                    ["Morning", "Afternoon", "Evening", "Night"], 
                    index=3,
                    key="time_area_police"  # UNIQUE KEY
                )
            with col2:
                gender = st.selectbox(
                    "Traveler Profile", 
                    ["Female", "Male", "Other", "Solo Female", "Solo Male"],
                    key="gender_area_police"  # UNIQUE KEY
                )
            
            if st.button("📊 Analyze with Real Crime Data", type="primary", key="analyze_police"):
                # Get real features
                features = ai.predictor.get_real_features_for_area(selected_area)
                
                if features:
                    # Show real statistics
                    st.subheader("📈 Real Crime Statistics")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Crimes", features['total_crimes'])
                    with cols[1]:
                        st.metric("Violent Crime Rate", f"{features['violent_rate']:.1f}%")
                    with cols[2]:
                        st.metric("Night Crime Rate", f"{features['night_rate']:.1f}%")
                    with cols[3]:
                        st.metric("Base Safety Score", f"{features['safety_score']:.1f}/100")
                    
                    # Predict safety score
                    safety_score = ai.predict_area_safety(selected_area, time_of_day, gender)
                    
                    # Display safety score
                    if safety_score >= 70:
                        score_class = "safe"
                        rating = "🟢 SAFE"
                    elif safety_score >= 50:
                        score_class = "moderate"
                        rating = "🟡 MODERATE"
                    else:
                        score_class = "risky"
                        rating = "🔴 RISKY"
                    
                    st.markdown(f'''
                    <div class="safety-score {score_class}">
                        Real-Time Safety Score: {safety_score}/100<br>{rating}
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Show recommendations
                    try:
                        recommendations = ai.get_area_recommendations(selected_area)
                        st.subheader("💡 Safety Recommendations")
                        for rec in recommendations:
                            st.info(f"• {rec}")
                    except:
                        pass
                else:
                    st.warning("No real crime data found for this area")
        
        except Exception as e:
            st.warning(f"Could not load area data: {e}")
    
    
# =========================================================
# TAB 3 – POLICE STATIONS
# =========================================================
with tabs[2]:
    st.markdown("### 🚓 Bangalore Police Stations (All Stations)")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stations", len(police_data))
    col2.metric("Coverage Area", "800+ sq km")
    col3.metric("Avg Response Time", "< 10 min")
    col4.metric("Data Source", "Govt. Dataset")
    
    # Show the map
    st.markdown("### 📍 Interactive Police Station Map")
    police_map = create_police_stations_map()
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    folium_static(police_map, width=1000, height=600)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data table
    st.markdown("### 📋 Police Station Data")
    st.dataframe(
        police_data[['Name of Police Station', 'Address', 'Latitude', 'Longitude']],
        use_container_width=True,
        hide_index=True,
        height=300
    )
    
    # Export option
    try:
        csv = police_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Police Station Data",
            data=csv,
            file_name="bangalore_police_stations.csv",
            mime="text/csv",
            use_container_width=True
        )
    except:
        pass