import joblib
from anomaly_detector import SafetyAnomalyDetector


def load_anomaly_model(path="models/anomaly_model.pkl"):
    detector = SafetyAnomalyDetector()
    detector.model = joblib.load(path)
    return detector
