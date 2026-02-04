import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self):
        self.data_path = "data"

    def load_city_data(self):
        """Load city data from data/city_data.csv"""

        file_path = os.path.join(self.data_path, "city_data.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"City dataset not found: {file_path}")

        df = pd.read_csv(file_path)

        # ----------------------------
        # Ensure City column exists
        # ----------------------------
        if "City" not in df.columns:
            raise ValueError("Dataset must contain a 'City' column.")

        # ----------------------------
        # Create City_Name and State
        # ----------------------------
        df["City_Name"] = (
            df["City"]
            .astype(str)
            .str.replace(r"\s*\(.*?\)\s*", "", regex=True)
            .str.strip()
        )

        df["State"] = df["City"].astype(str).str.extract(r"\((.*?)\)")

        # ----------------------------
        # Risk score normalization
        # ----------------------------
        # If safemap_risk exists and is 0-1 scale, convert to percentage.
        # If it's already 0-100, keep it.
        if "safemap_risk" in df.columns:
            # Convert safely
            df["safemap_risk"] = pd.to_numeric(df["safemap_risk"], errors="coerce")

            # If max is <= 1.5, assume 0-1 range
            if df["safemap_risk"].max(skipna=True) <= 1.5:
                df["safemap_risk_pct"] = df["safemap_risk"] * 100
            else:
                df["safemap_risk_pct"] = df["safemap_risk"]
        else:
            # If column doesn't exist, create a fallback
            df["safemap_risk"] = np.nan
            df["safemap_risk_pct"] = np.nan

        return df

    def load_state_data(self):
        """Load state preprocessed data (static for demo)"""

        state_data = {
            "State/UT": [
                "Karnataka", "Delhi", "Maharashtra", "Tamil Nadu",
                "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan",
                "Madhya Pradesh", "Bihar", "Kerala", "Telangana"
            ],
            "state_risk": [
                0.024, 0.368, 0.112, 0.007, 0.026, 0.023,
                0.020, 0.077, 0.045, 0.036, 0.009, 0.036
            ],
            "women_crime": [
                6.33, 1.72, 3.97, 23.31, 4.15, 56.01,
                1.10, 4.75, 8.01, 4.76, 32.82, 1.84
            ],
            "law_effectiveness": [
                78.3, 65.0, 72.1, 84.5, 58.9, 89.8,
                73.2, 66.7, 71.5, 62.4, 96.0, 75.8
            ]
        }

        return pd.DataFrame(state_data)

    def merge_data(self):
        """Merge city and state data"""

        city_df = self.load_city_data()
        state_df = self.load_state_data()

        merged = pd.merge(
            city_df,
            state_df,
            left_on="State",
            right_on="State/UT",
            how="left"
        )

        # Fill missing values
        if "law_effectiveness" in merged.columns:
            merged["law_effectiveness"] = merged["law_effectiveness"].fillna(75)

        return merged

    def get_city_ranking(self):
        """Rank cities by safety (lower risk = safer)"""

        df = self.load_city_data()

        if "safemap_risk" not in df.columns:
            raise ValueError("Cannot rank cities: 'safemap_risk' column missing.")

        df["safemap_risk"] = pd.to_numeric(df["safemap_risk"], errors="coerce")
        df["safety_rank"] = df["safemap_risk"].rank(ascending=True)

        return df.sort_values("safety_rank")
