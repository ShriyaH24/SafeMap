import pandas as pd
import numpy as np

from ml_safety_predictor import MLSafetyPredictor
from anomaly_detector import SafetyAnomalyDetector


class AISafetyEngine:
    """
    Combines:
    - ML Safety Prediction (Option A)
    - Anomaly Detection (Option B)
    """

    def __init__(self):
        self.predictor = MLSafetyPredictor()
        self.anomaly_detector = SafetyAnomalyDetector()

    def load_models(self):
        self.predictor.load_model()

    def predict_city_safety(self, city_row: dict, time_of_day="Night", gender="Female"):
        """
        city_row should contain safemap_risk, women_crime_ratio, law_effectiveness
        We will add infrastructure values.
        """

        # If infra missing, use safe defaults
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

        return self.predictor.predict_safety_score(input_data)

    def train_anomaly_detector(self, city_df):
        """
        city_df must contain required columns.
        """
        self.anomaly_detector.train(city_df)

    def detect_risky_pockets(self, city_df):
        return self.anomaly_detector.detect(city_df)
