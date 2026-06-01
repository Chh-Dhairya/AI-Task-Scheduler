"""
Configuration file for Hybrid Scheduling System
Contains all constants, weights, and API settings
"""
import os

API_KEY = os.getenv("API_KEY")
# ============================================================================
# SCHEDULING WEIGHTS
# ============================================================================
DEFAULT_MODEL_WEIGHT = 0.8  # ML System weight (70-90%)
DEFAULT_API_WEIGHT = 0.2    # API System weight (10-30%)

# Weight adjustment thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5  # If habit confidence < this, increase API weight
HIGH_ADHERENCE_THRESHOLD = 0.8  # If user follows schedule well, reduce API weight

# ============================================================================
# TIME CONSTRAINTS
# ============================================================================
DAY_START_HOUR = 6   # 6 AM
DAY_END_HOUR = 23    # 11 PM
MIN_TASK_DURATION = 15  # minutes
MAX_TASK_DURATION = 480  # 8 hours

# ============================================================================
# ACTIVITY RULES (For API Scoring)
# ============================================================================
ACTIVITY_TIME_RULES = {
    'Breakfast': {'ideal_start': 7, 'ideal_end': 9, 'penalty_outside': 0.3},
    'Lunch': {'ideal_start': 12, 'ideal_end': 14, 'penalty_outside': 0.3},
    'Dinner': {'ideal_start': 18, 'ideal_end': 21, 'penalty_outside': 0.3},
    'Sleep': {'ideal_start': 22, 'ideal_end': 7, 'penalty_outside': 0.5},
    'Exercise': {'ideal_start': 6, 'ideal_end': 10, 'bonus': 0.2},
    'Work': {'ideal_start': 9, 'ideal_end': 18, 'penalty_outside': 0.2},
    'Study': {'ideal_start': 9, 'ideal_end': 22, 'penalty_outside': 0.1},
}

# Unhealthy patterns
UNHEALTHY_PATTERNS = {
    'late_eating': {'activity': ['Breakfast', 'Lunch', 'Dinner'], 'after_hour': 22, 'penalty': 0.4},
    'late_sleep': {'activity': ['Sleep'], 'after_hour': 1, 'penalty': 0.5},
    'early_sleep': {'activity': ['Sleep'], 'before_hour': 20, 'penalty': 0.2},
    'night_exercise': {'activity': ['Exercise'], 'after_hour': 21, 'penalty': 0.3},
}

# ============================================================================
# SCORING PARAMETERS
# ============================================================================
# ML Score components (should sum to 1.0)
ML_WEIGHTS = {
    'rl_reward': 0.4,
    'habit_alignment': 0.3,
    'deadline_satisfaction': 0.2,
    'priority_score': 0.1
}

# API Score components (should sum to 1.0)
API_WEIGHTS = {
    'time_appropriateness': 0.5,
    'health_routine': 0.3,
    'human_behavior': 0.2
}

# ============================================================================
# API CONFIGURATION (Mock API for now)
# ============================================================================
USE_MOCK_API = False  # Set to False when real API is available
API_ENDPOINT = "http://localhost:8000/schedule"  # Replace with real API
API_TIMEOUT = 5  # seconds

# ============================================================================
# UI CONFIGURATION
# ============================================================================
STREAMLIT_THEME = {
    'primaryColor': '#FF6B6B',
    'backgroundColor': '#FFFFFF',
    'secondaryBackgroundColor': '#F0F2F6',
    'textColor': '#262730',
    'font': 'sans serif'
}

# Task priorities
PRIORITY_LEVELS = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Critical': 4
}

# ============================================================================
# FALLBACK SETTINGS
# ============================================================================
ENABLE_ADAPTIVE_WEIGHTING = True
ENABLE_API_FALLBACK = True
MAX_RETRIES_API = 3