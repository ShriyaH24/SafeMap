import pandas as pd
import numpy as np
from datetime import datetime
import random

class SafetyCalculator:
    def __init__(self):
        self.crime_weights = {
            'Night': 1.5,
            'Evening': 1.2,
            'Day': 1.0,
            'Early Morning': 1.3
        }
        
        self.gender_factors = {
            'Female': 1.3,
            'Male': 1.0,
            'Other': 1.2
        }
        
        self.age_factors = {
            'young': (18, 25, 1.2),
            'adult': (26, 45, 1.0),
            'middle': (46, 60, 0.9),
            'senior': (61, 100, 1.1)
        }
    
    def calculate_city_score(self, city_data, time_of_day):
        """Calculate comprehensive safety score (0-100)"""
        # Base score from crime data
        base_risk = city_data['safemap_risk']
        women_safety = city_data['women_crime_ratio']
        law_effectiveness = city_data.get('law_effectiveness', 75)
        
        # Normalize women safety (higher is worse in your data)
        women_factor = max(0, 100 - (women_safety * 2))
        
        # Time factor
        time_factor = self.crime_weights.get(time_of_day.split('(')[0].strip(), 1.0)
        
        # Calculate final score
        score = 100 - (base_risk * 100 * time_factor)
        score = score * 0.6 + women_factor * 0.2 + law_effectiveness * 0.2
        
        # Ensure bounds
        return max(0, min(100, score))
    
    def calculate_safe_route(self, city, start, end, time_of_day, preference, gender):
        """Calculate safe route between two points"""
        # Mock route data - in real implementation, use OSM routing
        routes = {
            'Safest': {
                'distance': 4.2,
                'time': 25,
                'safety_score': 85,
                'risk_level': 'Low',
                'steps': [
                    f"Start at {start}",
                    "Head northeast on MG Road (well-lit)",
                    "Turn right onto Brigade Road (police patrol area)",
                    "Continue straight for 2km (CCTV coverage)",
                    f"Arrive at {end}"
                ]
            },
            'Balanced': {
                'distance': 3.8,
                'time': 21,
                'safety_score': 65,
                'risk_level': 'Medium',
                'steps': [
                    f"Start at {start}",
                    "Take shortcut through market area",
                    "Cross park (moderate lighting)",
                    f"Arrive at {end}"
                ]
            },
            'Fastest': {
                'distance': 3.5,
                'time': 18,
                'safety_score': 45,
                'risk_level': 'High',
                'steps': [
                    f"Start at {start}",
                    "Take alley shortcut (poor lighting)",
                    "Pass through industrial area (isolated)",
                    f"Arrive at {end}"
                ]
            }
        }
        
        return routes.get(preference, routes['Balanced'])
    
    def identify_safety_zones(self, city):
        """Identify safe and risky zones in city"""
        zones = [
            {
                'name': 'Commercial Districts',
                'score': 85,
                'description': 'Well-lit with police presence',
                'type': 'safe'
            },
            {
                'name': 'Residential Areas',
                'score': 75,
                'description': 'Moderate safety, community watch',
                'type': 'safe'
            },
            {
                'name': 'Industrial Zones',
                'score': 40,
                'description': 'Poor lighting, isolated at night',
                'type': 'risky'
            },
            {
                'name': 'Market Areas',
                'score': 60,
                'description': 'Crowded but pickpocket risk',
                'type': 'moderate'
            }
        ]
        
        return zones
    
    def get_risk_areas(self, city):
        """Get high-risk areas to avoid"""
        risk_areas = [
            {
                'name': 'Old Market Alley',
                'risk_reason': 'Poor lighting, history of theft',
                'risk_level': 'High',
                'avoid_times': ['Night', 'Evening']
            },
            {
                'name': 'Riverside Park',
                'risk_reason': 'Isolated, limited CCTV',
                'risk_level': 'Medium',
                'avoid_times': ['Night']
            },
            {
                'name': 'Industrial Estate Road',
                'risk_reason': 'Heavy vehicle traffic, no sidewalks',
                'risk_level': 'Medium',
                'avoid_times': ['All']
            }
        ]
        
        return risk_areas
    
    def generate_time_patterns(self, city):
        """Generate hourly risk patterns"""
        hours = list(range(24))
        # Simulate pattern - highest risk at night
        risks = [max(0, min(100, 60 + 20 * np.sin(h/24 * 2*np.pi) + random.uniform(-5, 5))) 
                for h in hours]
        
        return pd.DataFrame({'Hour': hours, 'Risk_Score': risks})
    
    def analyze_infrastructure(self, city):
        """Analyze safety infrastructure"""
        # Mock data - in real app, fetch from OSM
        return {
            'police_count': random.randint(15, 50),
            'lights_count': random.randint(1000, 5000),
            'cctv_count': random.randint(200, 1000),
            'hospitals': random.randint(5, 20),
            'help_points': random.randint(10, 30)
        }