import google.generativeai as genai
import os
from typing import Dict, List
import json
from config import Config

class GeminiSafetyAdvisor:
    def __init__(self, api_key=None):
        # Priority: 1. Provided key, 2. Environment, 3. Config
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or Config.GEMINI_API_KEY
        
        if not self.api_key:
            print("⚠ Gemini API key not found. Gemini features disabled.")
            self.available = False
            return
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            self.available = True
            print("✅ Gemini AI Advisor initialized successfully")
        except Exception as e:
            print(f"⚠ Failed to initialize Gemini: {e}")
            self.available = False
    
    def is_available(self):
        return self.available
        
    def generate_safety_analysis(self, 
                                 start_area: str,
                                 end_area: str,
                                 time_of_day: str,
                                 traveler_type: str,
                                 crime_stats: Dict = None,
                                 route_info: Dict = None) -> Dict:
        """
        Generate AI-powered safety analysis using Gemini
        """
        
        # Prepare crime statistics text
        crime_text = ""
        if crime_stats:
            crime_text = f"""
            Crime Statistics:
            - Start Area ({start_area}): {crime_stats.get('start_crimes', 0)} crimes, 
              {crime_stats.get('start_violent_rate', 0)}% violent crime rate
            - End Area ({end_area}): {crime_stats.get('end_crimes', 0)} crimes,
              {crime_stats.get('end_violent_rate', 0)}% violent crime rate
            - Route passes through {crime_stats.get('areas_along_route', 0)} police station areas
            """
        
        # Prepare route info text
        route_text = ""
        if route_info:
            route_text = f"""
            Route Information:
            - Distance: {route_info.get('distance', 0)} km
            - Estimated Time: {route_info.get('time', 0)} minutes
            - Main roads: {route_info.get('main_roads', True)}
            - Well-lit: {route_info.get('well_lit', True)}
            """
        
        prompt = f"""
        You are a safety advisor for SafeMap, a personal safety navigation app.
        
        Analyze this journey for safety and provide specific, actionable advice:
        
        Journey Details:
        - From: {start_area}
        - To: {end_area}
        - Time: {time_of_day}
        - Traveler: {traveler_type}
        
        {crime_text}
        {route_text}
        
        Please provide:
        1. A safety score from 0-100 (with brief justification)
        2. Top 5 specific safety recommendations for THIS specific journey
        3. Any alternative suggestions if the risk is high
        4. Emergency preparedness tips
        
        Format your response as JSON:
        {{
            "safety_score": 85,
            "score_explanation": "Brief explanation here...",
            "recommendations": ["Tip 1", "Tip 2", ...],
            "alternative_suggestions": ["Alternative 1", ...],
            "emergency_tips": ["Tip 1", "Tip 2"]
        }}
        
        Keep recommendations specific to Bangalore, India context.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Sometimes Gemini adds markdown, clean it
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._get_fallback_response(start_area, end_area, time_of_day, traveler_type)
    
    def _get_fallback_response(self, start_area, end_area, time_of_day, traveler_type):
        """Fallback if Gemini fails"""
        return {
            "safety_score": 70,
            "score_explanation": "Standard safety assessment",
            "recommendations": [
                "Share your live location",
                "Use well-lit main roads",
                "Keep emergency numbers handy",
                "Stay alert in crowded areas",
                "Trust your instincts"
            ],
            "alternative_suggestions": [
                "Consider daytime travel",
                "Use verified transport services"
            ],
            "emergency_tips": [
                "Police: 100",
                "Women Helpline: 181",
                "Save local police station number"
            ]
        }
    
    def generate_route_description(self, start_area, end_area, route_details):
        """Generate natural language route description"""
        prompt = f"""
        Describe this route in a helpful, safety-conscious way:
        
        Route: {start_area} to {end_area}
        Details: {route_details}
        
        Include:
        - Key landmarks along the way
        - Safety considerations
        - Estimated travel experience
        - Any known issues or cautions
        
        Keep it concise and helpful for navigation.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except:
            return f"Route from {start_area} to {end_area}. Follow the mapped route for safe navigation."