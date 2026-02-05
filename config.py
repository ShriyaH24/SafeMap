# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_PATH = "models/safemap_rf_model.pkl"
    DATA_PATH = "data/processed/"
    
    @classmethod
    def validate(cls):
        """Validate that required environment variables are set"""
        if not cls.GEMINI_API_KEY:
            print("⚠ WARNING: GEMINI_API_KEY not found in environment variables")
            print("   Gemini AI features will be disabled")
            print("   Add your key to .env file or set GEMINI_API_KEY environment variable")
        return cls.GEMINI_API_KEY is not None