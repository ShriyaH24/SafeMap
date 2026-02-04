import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SAFEMAP - Bangalore Safety Navigation",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS (from second version)
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
# AI ENGINE (VERSION 2 – UNCHANGED)
# =========================================================
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


class AnomalyDetector:
    def __init__(self, model_path="models/anomaly_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None

    def load(self):
        obj = joblib.load(self.model_path)
        if isinstance(obj, dict):
            self.model = obj["model"]
            self.feature_columns = obj.get("feature_columns")
        else:
            self.model = obj

    def detect(self, row):
        df = pd.DataFrame([row])
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        else:
            df = df.select_dtypes(include=[np.number])
        return int(self.model.predict(df.fillna(0))[0]) == -1


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
# MODEL LOADING
# =========================================================
ai = AISafetyEngine()
anom = AnomalyDetector()
ai.load()
anom.load()

# =========================================================
# VERSION 1 ROUTE DATA
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
# ROUTE FUNCTIONS (VERSION 1)
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


def calculate_safety_score(start, end, time_of_day, traveler):
    risk = (AREA_RISK.get(start, 5) + AREA_RISK.get(end, 5)) / 2
    factor = 1.0
    if time_of_day == "Night":
        factor += 0.8
    if traveler == "Solo Female":
        factor += 0.5
    score = 100 - (risk * factor * 8)
    return max(10, min(95, score))


def create_route_map(start, end, route, show_police):
    center = [(start[0] + end[0])/2, (start[1] + end[1])/2]
    m = folium.Map(location=center, zoom_start=13, tiles='OpenStreetMap')

    if show_police:
        cluster = MarkerCluster().add_to(m)
        for _, p in police_data.iterrows():
            folium.Marker(
                [p["Latitude"], p["Longitude"]],
                popup=p["Name of Police Station"],
                icon=folium.Icon(color="blue", icon='shield', prefix='fa')
            ).add_to(cluster)

    folium.Marker(start, 
                  icon=folium.Icon(color="green", icon='play', prefix='fa'),
                  popup='📍 <b>Start Point</b>').add_to(m)
    folium.Marker(end, 
                  icon=folium.Icon(color="red", icon='stop', prefix='fa'),
                  popup='🏁 <b>Destination</b>').add_to(m)
    folium.PolyLine(route, color='#10B981', weight=6, opacity=0.9).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m


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


# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "🗺️ Safe Route Planning",
    "City Safety",
    "🚓 Police Stations"
])

# =========================================================
# TAB 1 – ROUTE PLANNING (VERSION 1 UI)
# =========================================================
with tabs[0]:
    st.subheader("🗺️ Safe Route Planning")
    
    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.markdown("### Travel Profile")
        traveler = st.selectbox("Traveler Type", ["Solo Male", "Solo Female", "Group"])
        time_of_day = st.selectbox("Time of Travel", ["Day", "Evening", "Night"])

        st.markdown("### Preferences")
        show_police = st.checkbox("Show Police Stations", True)

        st.markdown("### Locations")
        start_area = st.selectbox("Start Location", BANGALORE_AREAS.keys())
        end_area = st.selectbox("Destination", BANGALORE_AREAS.keys())

        go = st.button("🚀 Calculate Safest Route", use_container_width=True, type="primary")

        st.markdown("### 🚨 Emergency Contacts")
        for k, v in EMERGENCY_NUMBERS.items():
            st.write(f"**{k}:** {v}")

    if go:
        start = BANGALORE_AREAS[start_area]
        end = BANGALORE_AREAS[end_area]
        route = get_route_osrm(start, end)
        dist, time = estimate_distance_time(route)
        score = calculate_safety_score(start_area, end_area, time_of_day, traveler)
        
        # Determine safety rating
        if score >= 70:
            score_class = "safe"
            rating = "🟢 VERY SAFE"
        elif score >= 50:
            score_class = "moderate"
            rating = "🟡 MODERATELY SAFE"
        else:
            score_class = "risky"
            rating = "🔴 HIGH RISK"

        with col2:
            # Safety score display
            st.markdown(f'<div class="safety-score {score_class}">{int(score)}/100<br>{rating}</div>', 
                       unsafe_allow_html=True)
            
            # Route metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Route Distance", f"{dist} km")
            with col_metric2:
                st.metric("Estimated Time", f"{time} min")
            
            # Safety tips
            st.markdown("#### 💡 Safety Tips")
            tips = []
            if score < 60:
                tips.append("Consider daytime travel")
            if time_of_day == "Night":
                tips.append("Share live location")
            if traveler == "Solo Female":
                tips.append("Use verified transport")
            
            for tip in tips:
                st.info(f"• {tip}")
            
            # Map display
            st.markdown("### 📍 Route Map")
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            folium_static(create_route_map(start, end, route, show_police), width=800, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
            
        
    else:
        # Default view when no route calculated
        with col2:
            st.info("👈 Configure your route in the sidebar and click 'Calculate Safest Route'")
            default_start = BANGALORE_AREAS["MG Road"]
            default_end = BANGALORE_AREAS["Jayanagar"]
            default_map = create_route_map(default_start, default_end, [default_start, default_end], True)
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            folium_static(default_map, width=800, height=500)
            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 – CITY SAFETY
# =========================================================
with tabs[1]:
    city = st.selectbox("City", cities)
    row = city_data[city_data["City"] == city].iloc[0].to_dict()

    st.metric("AI City Safety Score", ai.predict_city_safety(row))

    if anom.detect(row):
        st.error("⚠️ City profile significantly differs from average cities")
    else:
        st.success("✅ No anomaly detected")

# =========================================================
# TAB 4 – POLICE MAP
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
    csv = police_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Police Station Data",
        data=csv,
        file_name="bangalore_police_stations.csv",
        mime="text/csv",
        use_container_width=True
    )