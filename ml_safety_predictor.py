import joblib
import numpy as np
import os
import pandas as pd

MODEL_PATH = "models/safemap_rf_model.pkl"

class MLSafetyPredictor:
    def __init__(self, real_data_path="data/processed/"):
        self.model = None
        self.feature_columns = None
        self.loaded = False
        self.real_data_path = real_data_path
        self.police_station_features = None
        
        # Load real data
        self._load_real_data()

    def _load_real_data(self):
        """Load the real processed crime data"""
        try:
            # Load police station features
            police_file = os.path.join(self.real_data_path, "ml_features_police_stations.csv")
            if os.path.exists(police_file):
                self.police_station_features = pd.read_csv(police_file)
                print(f"✓ Loaded real police station features: {len(self.police_station_features)} stations")
            else:
                print(f"⚠ Real data file not found: {police_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not load real data: {e}")

    def load_model(self):
        try:
            obj = joblib.load(MODEL_PATH)
            if isinstance(obj, dict):
                self.model = obj["model"]
                self.feature_columns = obj.get("feature_columns", [])
            else:
                self.model = obj
            self.loaded = True
            print(f"✓ ML Model loaded")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
            return False

    def get_real_features_for_area(self, area_name):
        """Get real crime features for a specific area/police station"""
        if self.police_station_features is None:
            return None
        
        # Try to find the area
        area_name_lower = str(area_name).lower().strip()
        
        for col in ['Police_Station', 'area', 'Area', 'Police Station']:
            if col in self.police_station_features.columns:
                mask = self.police_station_features[col].astype(str).str.lower().str.strip() == area_name_lower
                if mask.any():
                    features = self.police_station_features[mask].iloc[0].to_dict()
                    
                    real_features = {
                        'total_crimes': features.get('Total_Crimes', 0),
                        'violent_rate': features.get('Violent_Rate', 0),
                        'night_rate': features.get('Night_Rate', 0),
                        'safety_score': features.get('Safety_Score', 50),
                        'avg_latitude': features.get('Avg_Latitude', 0),
                        'avg_longitude': features.get('Avg_Longitude', 0)
                    }
                    return real_features
        
        print(f"⚠ Area '{area_name}' not found in real data")
        return None

    def _get_time_factor(self, time_of_day):
        """Convert time of day to numerical factor"""
        factors = {
            'Morning': 1.0,
            'Afternoon': 1.1,
            'Evening': 1.3,
            'Night': 1.5,
            'Day': 1.0
        }
        return factors.get(time_of_day, 1.2)

    def _get_gender_factor(self, gender):
        """Convert gender to numerical factor"""
        factors = {
            'Female': 1.2,
            'Male': 1.0,
            'Other': 1.1,
            'Solo Female': 1.2,
            'Solo Male': 1.0
        }
        return factors.get(gender, 1.1)

    def predict_with_real_data(self, area_name, time_of_day="Night", gender="Female"):
        """
        Predict safety score using REAL crime data for an area
        """
        # Get real features for the area
        real_features = self.get_real_features_for_area(area_name)
        
        if real_features is None:
            # Fallback prediction
            return self._predict_fallback(time_of_day, gender)
        
        # Start with real safety score
        base_score = real_features['safety_score']
        
        # Adjust for time and gender
        time_factor = self._get_time_factor(time_of_day)
        gender_factor = self._get_gender_factor(gender)
        
        # Penalize for high crime rates
        penalty = 0
        if real_features['violent_rate'] > 30:
            penalty += 15
        if real_features['night_rate'] > 40:
            penalty += 10
        if real_features['total_crimes'] > 100:
            penalty += 5
        
        adjusted_score = base_score * (1/time_factor) * (1/gender_factor) - penalty
        return max(0, min(100, round(adjusted_score, 1)))

    def _predict_fallback(self, time_of_day="Night", gender="Female"):
        """Fallback prediction if no real data available"""
        base_score = 70
        time_factor = self._get_time_factor(time_of_day)
        gender_factor = self._get_gender_factor(gender)
        
        return max(0, min(100, base_score * (1/time_factor) * (1/gender_factor)))

    def predict_score(self, feature_dict):
        """Original predict method - works with feature dictionary"""
        if not self.loaded:
            return 50

        try:
            # Convert to DataFrame
            df = pd.DataFrame([feature_dict])
            df = df.select_dtypes(include=[np.number]).fillna(0)
            
            pred = float(self.model.predict(df)[0])
            # Convert to safety score (0-100, higher is safer)
            score = max(0, min(100, 100 - pred * 10))
            return round(score, 1)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 50