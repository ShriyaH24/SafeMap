import pandas as pd
import numpy as np
import os

from ml_safety_predictor import MLSafetyPredictor
from gemini_safety_advisor import GeminiSafetyAdvisor
from anomaly_detector import SafetyAnomalyDetector

class AISafetyEngine:
    """
    Real data version with real crime data integration
    """

    def __init__(self, real_data_dir="data/processed/", gemini_api_key=None):
        self.predictor = MLSafetyPredictor(real_data_dir)
        self.anomaly_detector = SafetyAnomalyDetector()
        self.real_data_loaded = False
        self.area_features = None
        self.real_data_dir = real_data_dir
        self.gemini_api_key = gemini_api_key
        
        # Load real data
        self._load_real_data()

        # Initialize Gemini advisor
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Gemini AI advisor"""
        if not self.gemini_api_key:
            print("⚠ Gemini API key not provided. Gemini features disabled.")
            self.has_gemini = False
            return
        
        try:
            self.gemini_advisor = GeminiSafetyAdvisor(self.gemini_api_key)
            if hasattr(self.gemini_advisor, 'available') and self.gemini_advisor.available:
                self.has_gemini = True
                print("✅ Gemini AI Advisor initialized")
            else:
                self.has_gemini = False
                print("⚠ Gemini AI failed to initialize")
        except Exception as e:
            print(f"⚠ Error initializing Gemini: {e}")
            self.has_gemini = False

    def _load_real_data(self):
        """Load real crime data"""
        try:
            police_file = os.path.join(self.real_data_dir, "ml_features_police_stations.csv")
            if os.path.exists(police_file):
                self.area_features = pd.read_csv(police_file)
                self.real_data_loaded = True
                print(f"✓ Loaded real crime data for {len(self.area_features)} areas")
            else:
                print(f"⚠ Real data file not found: {police_file}")
        except Exception as e:
            print(f"⚠ Could not load real data: {e}")
            self.real_data_loaded = False

    def load_models(self):
        """Load ML models"""
        return self.predictor.load_model()

    def load(self):
        """Alias for load_models for backward compatibility"""
        return self.load_models()

    def predict_area_safety(self, area_name, time_of_day="Night", gender="Female"):
        """
        Predict safety for a specific area using REAL crime data
        """
        return self.predictor.predict_with_real_data(area_name, time_of_day, gender)

    def predict_city_safety(self, city_row: dict, time_of_day="Night", gender="Female"):
        """
        Main prediction method - uses real data if available
        """
        # First try to use real data
        if self.real_data_loaded and self.area_features is not None:
            city_name = city_row.get("City", "Unknown")
            
            # Try to find areas for this city
            try:
                city_areas = self.area_features[
                    self.area_features['Police_Station'].str.contains(city_name, case=False, na=False)
                ]
                
                if not city_areas.empty:
                    # Average safety score for areas in this city
                    avg_safety = city_areas['Safety_Score'].mean()
                    
                    # Adjust for time and gender
                    time_factor = self.predictor._get_time_factor(time_of_day)
                    gender_factor = self.predictor._get_gender_factor(gender)
                    
                    adjusted_score = avg_safety * (1/time_factor) * (1/gender_factor)
                    return max(0, min(100, round(adjusted_score, 1)))
            except:
                pass
        
        # Fallback to original method
        return self.predict_city_safety_fallback(city_row, time_of_day, gender)

    def predict_city_safety_fallback(self, city_row, time_of_day="Night", gender="Female"):
        """
        Fallback method using original features
        """
        input_data = {
            "safemap_risk": float(city_row.get("safemap_risk", 0.2)),
            "women_crime_ratio": float(city_row.get("women_crime_ratio", 15)),
            "law_effectiveness": float(city_row.get("law_effectiveness", 75)),
            "police_count": float(city_row.get("police_count", 25)),
            "lights_count": float(city_row.get("lights_count", 2500)),
            "cctv_count": float(city_row.get("cctv_count", 400)),
            "emergency_phones_count": float(city_row.get("emergency_phones_count", 10)),
            "time_of_day": time_of_day,
            "gender": gender
        }

        return self.predictor.predict_score(input_data)

    def get_area_recommendations(self, area_name):
        """
        Get safety recommendations based on real crime data
        """
        if not self.real_data_loaded:
            return ["Use verified transport", "Avoid isolated areas"]
        
        features = self.predictor.get_real_features_for_area(area_name)
        if features is None:
            return ["Data not available for this area"]
        
        recommendations = []
        
        # Crime-based recommendations
        if features.get('violent_rate', 0) > 30:
            recommendations.append("⚠ High violent crime area - travel in groups")
        
        if features.get('night_rate', 0) > 40:
            recommendations.append("🌙 High night crime area - avoid after dark")
        
        if features.get('total_crimes', 0) > 100:
            recommendations.append("🚨 High crime density area - stay alert")
        
        # Add general recommendations
        recommendations.extend([
            "Share live location with trusted contacts",
            "Use well-lit main roads",
            "Keep emergency numbers handy"
        ])
        
        return recommendations
    
    def get_ai_safety_analysis(self, start_area, end_area, time_of_day, traveler_type):
        """
        Get comprehensive safety analysis from Gemini AI
        """
        if not self.has_gemini:
            return self._get_basic_analysis(start_area, end_area, time_of_day, traveler_type)
        
        # Prepare crime stats
        crime_stats = {}
        if self.real_data_loaded:
            start_features = self.predictor.get_real_features_for_area(start_area)
            end_features = self.predictor.get_real_features_for_area(end_area)
            
            crime_stats = {
                'start_crimes': start_features.get('total_crimes', 0) if start_features else 0,
                'start_violent_rate': start_features.get('violent_rate', 0) if start_features else 0,
                'end_crimes': end_features.get('total_crimes', 0) if end_features else 0,
                'end_violent_rate': end_features.get('violent_rate', 0) if end_features else 0,
            }
        
        # Get analysis from Gemini
        return self.gemini_advisor.generate_safety_analysis(
            start_area, end_area, time_of_day, traveler_type, crime_stats
        )
    
    def _get_basic_analysis(self, start_area, end_area, time_of_day, traveler_type):
        """Fallback basic analysis when Gemini is not available"""
        return {
            "safety_score": 70,
            "score_explanation": "Standard safety assessment (Gemini AI not available)",
            "recommendations": [
                "Share your live location",
                "Use well-lit main roads",
                "Keep emergency numbers handy"
            ],
            "alternative_suggestions": [
                "Consider daytime travel",
                "Use verified transport services"
            ],
            "emergency_tips": [
                "Police: 100",
                "Women Helpline: 181"
            ]
        }