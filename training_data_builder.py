import pandas as pd
import numpy as np
import random


def build_training_dataset(merged_df, samples_per_city=8):
    """
    Creates a synthetic dataset for ML training.
    It uses your existing merged_df and generates variations.
    """

    time_options = ["Day", "Evening", "Night", "Early Morning"]
    gender_options = ["Female", "Male", "Other"]

    rows = []

    for _, row in merged_df.iterrows():
        for _ in range(samples_per_city):
            time_of_day = random.choice(time_options)
            gender = random.choice(gender_options)

            # Create simulated infrastructure counts
            police_count = random.randint(10, 60)
            lights_count = random.randint(800, 6000)
            cctv_count = random.randint(50, 1200)
            emergency_phones_count = random.randint(0, 40)

            # Create a target safety score (rule-based)
            base_risk = float(row["safemap_risk"]) * 100
            women_crime_ratio = float(row.get("women_crime_ratio", 15))
            law_effectiveness = float(row.get("law_effectiveness", 75))

            # time effect
            time_factor = {
                "Day": 1.0,
                "Evening": 1.2,
                "Night": 1.5,
                "Early Morning": 1.3
            }[time_of_day]

            # gender effect
            gender_factor = {
                "Female": 1.15,
                "Male": 1.0,
                "Other": 1.1
            }[gender]

            infra_bonus = (
                min(police_count / 60, 1.0) * 15 +
                min(lights_count / 6000, 1.0) * 20 +
                min(cctv_count / 1200, 1.0) * 15 +
                min(emergency_phones_count / 40, 1.0) * 10
            )

            # Risk penalty
            risk_penalty = base_risk * time_factor * gender_factor

            # Final safety score
            target = 100 - risk_penalty + infra_bonus
            target = target * 0.65 + law_effectiveness * 0.35

            target = max(0, min(100, target))

            rows.append({
                "City_Name": row.get("City_Name", row.get("City", "Unknown")),
                "State": row.get("State", "Unknown"),
                "safemap_risk": float(row["safemap_risk"]),
                "women_crime_ratio": women_crime_ratio,
                "law_effectiveness": law_effectiveness,
                "police_count": police_count,
                "lights_count": lights_count,
                "cctv_count": cctv_count,
                "emergency_phones_count": emergency_phones_count,
                "time_of_day": time_of_day,
                "gender": gender,
                "target_safety_score": target
            })

    return pd.DataFrame(rows)
