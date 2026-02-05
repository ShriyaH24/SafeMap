# train_with_real_data.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def train_real_model():
    """Train ML model with real crime data"""
    
    # Load your real features
    real_features_path = "data/processed/ml_features_police_stations.csv"
    
    if not os.path.exists(real_features_path):
        print("❌ Real features file not found. Run preprocessing first.")
        return
    
    df = pd.read_csv(real_features_path)
    print(f"✓ Loaded {len(df)} real crime features")
    
    # Prepare features and target
    feature_cols = ['Total_Crimes', 'Violent_Rate', 'Night_Rate']
    
    # Make sure we have all required columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠ Missing columns: {missing_cols}")
        # Use available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    if 'Safety_Score' not in df.columns:
        print("❌ Safety_Score column not found")
        return
    
    X = df[feature_cols].fillna(0)
    y = df['Safety_Score']
    
    print(f"Features: {feature_cols}")
    print(f"Target: Safety_Score")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n📊 Model Performance:")
    print(f"Training R²: {train_score:.3f}")
    print(f"Testing R²: {test_score:.3f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    
    model_data = {
        "model": model,
        "feature_columns": feature_cols,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "data_source": "Real Crime Data"
    }
    
    joblib.dump(model_data, "models/safemap_real_model.pkl")
    print(f"✅ Model saved to models/safemap_real_model.pkl")
    
    # Feature importance
    print(f"\n🔍 Feature Importance:")
    for name, importance in zip(feature_cols, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    
    return model

if __name__ == "__main__":
    train_real_model()