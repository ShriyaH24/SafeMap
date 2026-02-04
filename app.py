import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="SafeMap AI",
    page_icon="🗺️",
    layout="wide"
)

# -----------------------------
# Utility: Safe model loading
# -----------------------------
def load_model(path):
    return joblib.load(path)

# -----------------------------
# AI Engine
# -----------------------------
class AISafetyEngine:
    """
    Uses trained ML model to predict safety score.
    """

    def __init__(self, model_path="models/safemap_rf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ML model not found: {self.model_path}")

        obj = load_model(self.model_path)

        # Support both formats:
        # 1) model only
        # 2) dict with {model, feature_columns}
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.feature_columns = obj.get("feature_columns", None)
        else:
            self.model = obj
            self.feature_columns = None

    def is_loaded(self):
        return self.model is not None

    def _encode_time(self, time_of_day):
        mapping = {
            "Early Morning": 0,
            "Morning": 1,
            "Day": 2,
            "Evening": 3,
            "Night": 4
        }
        return mapping.get(time_of_day, 2)

    def _encode_gender(self, gender):
        return 1 if gender.lower().strip() == "female" else 0

    def predict_city_safety(self, city_row: dict, time_of_day="Day", gender="Male"):
        """
        Returns safety score in range 0-100 (higher = safer).
        """

        row = dict(city_row)

        # Add encoded fields
        row["time_of_day_encoded"] = self._encode_time(time_of_day)
        row["gender_encoded"] = self._encode_gender(gender)

        # Ensure common infrastructure keys exist (if missing)
        defaults = {
            "police_count": 25,
            "lights_count": 2500,
            "cctv_count": 400,
            "emergency_phones_count": 10,
            "law_effectiveness": 75
        }
        for k, v in defaults.items():
            if k not in row or pd.isna(row.get(k)):
                row[k] = v

        # Convert to DataFrame
        df = pd.DataFrame([row])

        # If training stored feature columns, enforce same order
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        else:
            df = df.select_dtypes(include=[np.number])

        df = df.fillna(0)

        pred = float(self.model.predict(df)[0])

        # Convert to safety score
        if pred <= 10:
            risk_0_100 = pred * 10
        else:
            risk_0_100 = pred

        safety = 100 - risk_0_100
        safety = max(0, min(100, safety))
        return round(safety, 1)


class AnomalyDetector:
    """
    Uses trained anomaly model to flag abnormal high-risk patterns.
    """

    def __init__(self, model_path="models/anomaly_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Anomaly model not found: {self.model_path}")

        obj = load_model(self.model_path)

        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.feature_columns = obj.get("feature_columns", None)
        else:
            self.model = obj
            self.feature_columns = None

    def is_loaded(self):
        return self.model is not None

    def detect(self, city_row: dict):
        """
        Returns True if anomaly detected.
        IsolationForest-style:
        prediction = -1 means anomaly, 1 means normal
        """

        row = dict(city_row)

        defaults = {
            "police_count": 25,
            "lights_count": 2500,
            "cctv_count": 400,
            "emergency_phones_count": 10,
            "law_effectiveness": 75
        }
        for k, v in defaults.items():
            if k not in row or pd.isna(row.get(k)):
                row[k] = v

        df = pd.DataFrame([row])

        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        else:
            df = df.select_dtypes(include=[np.number])

        df = df.fillna(0)

        pred = int(self.model.predict(df)[0])
        return (pred == -1)

# -----------------------------
# Load City Data
# -----------------------------
@st.cache_data
def load_city_data(csv_path="data/city_data.csv"):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Normalize common column names
    if "City" in df.columns:
        df["City_Name"] = df["City"].astype(str)
    elif "City_Name" not in df.columns:
        return None

    # Clean city names: remove "(State)" part
    df["City_Name"] = (
        df["City_Name"]
        .astype(str)
        .str.replace(r"\s*\(.*?\)\s*", "", regex=True)
        .str.strip()
    )

    return df

# -----------------------------
# Load Police Data
# -----------------------------
@st.cache_data
def load_police_stations(csv_path="data/police_stations.csv"):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Try to normalize column names
    rename_map = {}

    for c in df.columns:
        low = c.lower().strip()
        if low in ["name of police station", "police station", "station", "name"]:
            rename_map[c] = "Name of Police Station"
        if low in ["address", "location"]:
            rename_map[c] = "Address"
        if low in ["latitude", "lat"]:
            rename_map[c] = "Latitude"
        if low in ["longitude", "lon", "lng"]:
            rename_map[c] = "Longitude"

    df = df.rename(columns=rename_map)

    required = ["Name of Police Station", "Address", "Latitude", "Longitude"]
    for col in required:
        if col not in df.columns:
            return None

    # Clean coordinates
    df["Latitude"] = pd.to_numeric(df["Latitude"].astype(str).str.replace('"', ''), errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"].astype(str).str.replace('"', ''), errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])

    df["Name of Police Station"] = df["Name of Police Station"].astype(str).str.strip()
    df["Address"] = df["Address"].astype(str).str.strip()

    return df

# -----------------------------
# Map Functions
# -----------------------------
def create_police_map(police_df):
    center = [12.9716, 77.5946]  # Bangalore
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    marker_cluster = MarkerCluster().add_to(m)

    for _, station in police_df.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; max-width: 250px;">
            <div style="background-color: #3B82F6; color: white; padding: 8px; border-radius: 5px 5px 0 0;">
                <strong>🚓 {station['Name of Police Station']}</strong>
            </div>
            <div style="padding: 10px;">
                <p style="margin: 5px 0; font-size: 12px;">
                    📍 <strong>Address:</strong><br>
                    {station['Address'][:90]}
                </p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">
                    📌 <strong>Coordinates:</strong><br>
                    {station['Latitude']:.6f}, {station['Longitude']:.6f}
                </p>
            </div>
        </div>
        """

        folium.Marker(
            [station["Latitude"], station["Longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=station["Name of Police Station"],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(marker_cluster)

    return m


def create_route_map(start_coords, end_coords, police_df=None, route_points=None):
    center = [
        (start_coords[0] + end_coords[0]) / 2,
        (start_coords[1] + end_coords[1]) / 2
    ]

    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # Optional: police markers
    if police_df is not None and len(police_df) > 0:
        marker_cluster = MarkerCluster().add_to(m)
        for _, station in police_df.iterrows():
            folium.Marker(
                [station["Latitude"], station["Longitude"]],
                popup=f"<b>🚓 {station['Name of Police Station']}</b>",
                tooltip=station["Name of Police Station"],
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)

    # Start marker
    folium.Marker(
        start_coords,
        popup="📍 <b>Start</b>",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # End marker
    folium.Marker(
        end_coords,
        popup="🏁 <b>Destination</b>",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Route polyline
    if route_points and len(route_points) > 1:
        folium.PolyLine(
            route_points,
            weight=6,
            opacity=0.9
        ).add_to(m)
    else:
        # fallback direct line
        folium.PolyLine(
            [start_coords, end_coords],
            weight=6,
            opacity=0.9
        ).add_to(m)

    return m


# -----------------------------
# Demo Bangalore Areas
# -----------------------------
BANGALORE_AREAS = {
    'MG Road': [12.9750, 77.6000],
    'Indiranagar': [12.9784, 77.6408],
    'Koramangala': [12.9210, 77.6185],
    'Jayanagar': [12.9287, 77.5816],
    'HSR Layout': [12.9218, 77.6617],
    'Whitefield': [12.9698, 77.7499],
    'Electronic City': [12.8395, 77.6615],
    'Majestic': [12.9770, 77.5700],
    'City Market': [12.9650, 77.5770],
    'Cubbon Park': [12.9763, 77.5929],
    'Commercial Street': [12.9833, 77.6070],
    'Brigade Road': [12.9747, 77.6080],
    'Shivajinagar': [12.9850, 77.6050],
    'Yeshwantpur': [13.0235, 77.5500],
    'Marathahalli': [12.9591, 77.7016],
    'Sarjapur Road': [12.9100, 77.6900],
    'Bellandur': [12.9304, 77.6784],
    'Silk Board': [12.9175, 77.6233]
}

# -----------------------------
# Load datasets
# -----------------------------
CITY_CSV = "data/city_data.csv"
POLICE_CSV = "data/police_stations.csv"

city_data = load_city_data(CITY_CSV)
police_data = load_police_stations(POLICE_CSV)

if city_data is None:
    st.error(f"City dataset not found OR missing City column. Please place CSV as `{CITY_CSV}`.")
    st.stop()

cities = sorted(city_data["City_Name"].dropna().unique().tolist())

# -----------------------------
# Sidebar / Tabs
# -----------------------------
st.title("🗺️ Safe Route Planning")
tabs = st.tabs(["🧭 Route Planning", "🏙️ City Safety", "📊 Analytics", "🚓 Police Stations Map"])

# -----------------------------
# Load AI Models (once)
# -----------------------------
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = AISafetyEngine()
    st.session_state.anomaly_detector = AnomalyDetector()
    st.session_state.ai_ok = False
    st.session_state.anomaly_ok = False

    try:
        st.session_state.ai_engine.load()
        st.session_state.ai_ok = True
    except Exception as e:
        st.session_state.ai_ok = False
        st.session_state.ai_err = str(e)

    try:
        st.session_state.anomaly_detector.load()
        st.session_state.anomaly_ok = True
    except Exception as e:
        st.session_state.anomaly_ok = False
        st.session_state.anomaly_err = str(e)

# Banner
if st.session_state.ai_ok:
    st.success("✅ AI Models Loaded: ML Safety Prediction Enabled")
else:
    st.warning("⚠️ AI Model not loaded. Using fallback scoring.")
    if "ai_err" in st.session_state:
        st.caption("Debug info (ML model load error):")
        st.code(st.session_state.ai_err)

if not st.session_state.anomaly_ok:
    st.warning("⚠️ Anomaly model not loaded.")
    if "anomaly_err" in st.session_state:
        st.caption("Debug info (Anomaly model load error):")
        st.code(st.session_state.anomaly_err)

# ============================================================
# TAB 1: Route Planning + ROUTE MAP
# ============================================================
with tabs[0]:

    colA, colB = st.columns([2.2, 1])

    with colA:
        st.subheader("Select City")
        selected_city = st.selectbox("City", cities, index=0, label_visibility="collapsed")

        st.subheader("Select Route (Bangalore Demo Areas)")
        area_names = list(BANGALORE_AREAS.keys())

        start_area = st.selectbox("Start Area", area_names, index=2)
        end_area = st.selectbox("Destination Area", area_names, index=4)

        st.markdown("---")
        st.subheader("Route Preferences")

        route_type = st.radio(
            "Route Type",
            ["🛡️ Safest Route", "⚖️ Balanced Route", "⚡ Fastest Route"],
            horizontal=True
        )

        time_of_travel = st.select_slider(
            "Time of Travel",
            options=["Early Morning", "Morning", "Day", "Evening", "Night"],
            value="Evening"
        )

        traveler_profile = st.selectbox(
            "Traveler Profile",
            ["👤 Solo Traveler", "👩 Women Traveling Alone", "👨‍👩‍👧 Family", "👥 Group"]
        )

        avoid_high_risk = st.checkbox("Avoid High-Risk Areas", value=True)
        prefer_lit = st.checkbox("Prefer Well-Lit Roads", value=True)
        show_police = st.checkbox("Show Police Stations on Map", value=True)

        calc_btn = st.button("🚀 Calculate Safest Route", width="stretch")

    with colB:
        st.subheader("Route Safety Analysis")

        city_row = city_data[city_data["City_Name"] == selected_city].iloc[0].to_dict()

        gender = "Female" if traveler_profile == "👩 Women Traveling Alone" else "Male"

        if st.session_state.ai_ok:
            ai_score = st.session_state.ai_engine.predict_city_safety(
                city_row,
                time_of_day=time_of_travel,
                gender=gender
            )
        else:
            ai_score = 60

        if route_type == "🛡️ Safest Route":
            ai_score = min(100, ai_score + 5)
        elif route_type == "⚡ Fastest Route":
            ai_score = max(0, ai_score - 5)

        if avoid_high_risk:
            ai_score = min(100, ai_score + 3)
        if prefer_lit:
            ai_score = min(100, ai_score + 2)

        # Fake distance/time (demo)
        base_dist = float(np.random.uniform(4, 12))
        travel_time_min = int(base_dist * 4)
        travel_time_max = int(base_dist * 5)

        if ai_score >= 80:
            risk_label = "🟢 Low Risk"
        elif ai_score >= 60:
            risk_label = "🟡 Moderate Risk"
        else:
            risk_label = "🔴 High Risk"

        st.metric("AI Route Safety Score", f"{int(ai_score)}/100")
        st.metric("Estimated Distance", f"{base_dist:.1f} km")
        st.metric("Travel Time", f"{travel_time_min}-{travel_time_max} min")

        st.markdown("### Risk Level")
        st.write(risk_label)

        st.markdown("### AI Safety Recommendations")

        recs = []
        if time_of_travel in ["Evening", "Night"]:
            recs.append("Avoid isolated roads; stay on main roads.")
        if traveler_profile == "👩 Women Traveling Alone":
            recs.append("Prefer routes with police stations and CCTV coverage.")
        if ai_score < 60:
            recs.append("Consider delaying travel or using trusted transport.")
        if prefer_lit:
            recs.append("Route prioritizes well-lit streets.")

        if not recs:
            recs.append("Route appears safe based on current AI risk model.")

        for r in recs:
            st.write("• " + r)

        if calc_btn:
            st.info("Route computed using ML-based safety prediction + route map.")

    # ------------------------------
    # ROUTE MAP SECTION (FULL)
    # ------------------------------
    st.markdown("---")
    st.subheader("🗺️ Route Map")

    start_coords = BANGALORE_AREAS.get(start_area, [12.9716, 77.5946])
    end_coords = BANGALORE_AREAS.get(end_area, [12.9287, 77.5816])

    # Create simple route points
    mid = [
        (start_coords[0] + end_coords[0]) / 2,
        (start_coords[1] + end_coords[1]) / 2
    ]
    route_points = [start_coords, mid, end_coords]

    # Add police on map if available
    police_df_to_use = police_data if (show_police and police_data is not None) else None

    route_map = create_route_map(
        start_coords=start_coords,
        end_coords=end_coords,
        police_df=police_df_to_use,
        route_points=route_points
    )

    folium_static(route_map, width=1200, height=550)


# ============================================================
# TAB 2: City Safety Dashboard (Anomaly Detection)
# ============================================================
with tabs[1]:
    st.title("🏙️ City Safety Overview")

    left, right = st.columns([1.4, 1])

    with left:
        selected_city2 = st.selectbox("Select City", cities, index=0, key="city2")
        city_info = city_data[city_data["City_Name"] == selected_city2].iloc[0].to_dict()

        st.subheader("City Data Snapshot")

        snapshot_df = pd.DataFrame(list(city_info.items()), columns=["Feature", "Value"])
        snapshot_df["Value"] = snapshot_df["Value"].astype(str)
        st.dataframe(snapshot_df, width="stretch")

    with right:
        st.subheader("AI Safety Alert")

        if st.session_state.anomaly_ok:
            try:
                is_anomaly = st.session_state.anomaly_detector.detect(city_info)

                if is_anomaly:
                    st.error("⚠️ AI Alert: This city shows abnormal high-risk patterns.")
                    st.write("Possible reasons: crime spike, weak enforcement, low infrastructure coverage.")
                else:
                    st.success("✅ AI Alert: City is within normal safety patterns.")
            except Exception as e:
                st.warning("Anomaly detection failed. Check feature mismatch.")
                st.code(str(e))
        else:
            st.warning("⚠️ Anomaly model not loaded. No AI alert available.")

        st.markdown("---")
        st.subheader("AI Predicted City Safety")

        if st.session_state.ai_ok:
            score_city = st.session_state.ai_engine.predict_city_safety(
                city_info,
                time_of_day="Day",
                gender="Male"
            )
            st.metric("AI City Safety Score", f"{int(score_city)}/100")
        else:
            st.metric("AI City Safety Score", "N/A")


# ============================================================
# TAB 3: Analytics
# ============================================================
with tabs[2]:
    st.title("📊 Analytics")

    st.write("This section shows dataset-level patterns (not AI).")

    numeric_cols = city_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in dataset.")
    else:
        col = st.selectbox("Select a numeric feature to view distribution", numeric_cols)
        st.bar_chart(city_data[col].fillna(0))

    st.markdown("---")
    st.write("AI Models:")
    st.write(f"• ML Model Loaded: {st.session_state.ai_ok}")
    st.write(f"• Anomaly Model Loaded: {st.session_state.anomaly_ok}")


# ============================================================
# TAB 4: Police Stations Full Map
# ============================================================
with tabs[3]:
    st.title("🚓 Police Stations Map")

    if police_data is None:
        st.warning(
            f"Police dataset not found.\n\n"
            f"Put your police CSV as `{POLICE_CSV}` inside the data folder."
        )
    else:
        st.metric("Total Police Stations", len(police_data))
        st.markdown("---")

        police_map = create_police_map(police_data)
        folium_static(police_map, width=1200, height=650)

        st.markdown("### 📋 Police Station Table")
        st.dataframe(
            police_data[["Name of Police Station", "Address", "Latitude", "Longitude"]],
            use_container_width=True,
            hide_index=True,
            height=350
        )
