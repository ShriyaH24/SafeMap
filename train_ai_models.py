import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


DATA_PATH = "data/city_data.csv"
MODELS_DIR = "models"

ML_MODEL_PATH = os.path.join(MODELS_DIR, "safemap_rf_model.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")


def build_training_data(df):
    """
    Builds training data (X, y).
    If dataset has a target column, use it.
    Otherwise, generate a synthetic risk score.
    """

    # Keep numeric only
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(0)

    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found in dataset.")

    # If any known target exists, use it
    possible_targets = ["risk_score", "crime_index", "risk", "danger_score"]

    for t in possible_targets:
        if t in numeric.columns:
            y = numeric[t].astype(float)
            X = numeric.drop(columns=[t])
            return X, y

    # Otherwise synthetic risk target
    row_sum = numeric.sum(axis=1)
    y = 100 * (row_sum - row_sum.min()) / (row_sum.max() - row_sum.min() + 1e-9)

    X = numeric
    return X, y


def train_and_save():
    print("🚀 Training SafeMap AI Models...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Remove text columns if present
    for col in ["City", "City_Name", "State"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    X, y = build_training_data(df)

    feature_columns = X.columns.tolist()

    # ---------------- ML Model ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ ML Training Complete")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")

    joblib.dump(
        {"model": rf, "feature_columns": feature_columns},
        ML_MODEL_PATH
    )

    print(f"✅ Model saved to {ML_MODEL_PATH}")

    # ---------------- Anomaly Model ----------------
    iso = IsolationForest(
        n_estimators=250,
        contamination=0.10,
        random_state=42
    )
    iso.fit(X)

    joblib.dump(
        {"model": iso, "feature_columns": feature_columns},
        ANOMALY_MODEL_PATH
    )

    print("✅ Anomaly model trained.")
    print(f"✅ Anomaly model saved to {ANOMALY_MODEL_PATH}")

    print("🎉 All models trained and saved successfully")


if __name__ == "__main__":
    train_and_save()
