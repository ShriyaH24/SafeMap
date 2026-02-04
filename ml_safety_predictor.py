import joblib
import numpy as np
import os


MODEL_PATH = "models/safemap_rf_model.pkl"


class MLSafetyPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.loaded = False

        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.loaded = False
            return

        try:
            obj = joblib.load(MODEL_PATH)
            self.model = obj["model"]
            self.feature_columns = obj["feature_columns"]
            self.loaded = True
        except Exception:
            self.loaded = False

    def predict_score(self, feature_dict):
        if not self.loaded:
            return None

        X = []
        for col in self.feature_columns:
            X.append(float(feature_dict.get(col, 0)))

        X = np.array(X).reshape(1, -1)

        pred = self.model.predict(X)[0]

        # Clamp prediction between 0 and 100
        pred = max(0, min(100, pred))
        return pred
