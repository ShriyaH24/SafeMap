import joblib
import numpy as np
import os


ANOMALY_MODEL_PATH = "models/anomaly_model.pkl"


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.loaded = False
        self.load_model()

    def load_model(self):
        if not os.path.exists(ANOMALY_MODEL_PATH):
            self.loaded = False
            return

        try:
            obj = joblib.load(ANOMALY_MODEL_PATH)
            self.model = obj["model"]
            self.feature_columns = obj["feature_columns"]
            self.loaded = True
        except Exception:
            self.loaded = False

    def detect(self, feature_dict):
        if not self.loaded:
            return None

        X = []
        for col in self.feature_columns:
            X.append(float(feature_dict.get(col, 0)))

        X = np.array(X).reshape(1, -1)

        # -1 = anomaly, 1 = normal
        result = self.model.predict(X)[0]
        return result
