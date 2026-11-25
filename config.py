"""
Configuration file for Hybrid Scheduling System
Contains all constants, weights, and API settings
"""
import os


def _load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=VALUE lines from .env (gitignored) into the environment.
    A variable already set in the shell wins. No extra package needed."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Secrets come from the environment or .env only. Never hard-code keys here.
#   put  GEMINI_API_KEY="your-key"  in task_scheduler/.env   (gitignored)
#   or   export GEMINI_API_KEY="your-key"
#   export GEMINI_MODEL="gemini-3.6-flash"   # optional override
API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-3.6-flash: stable, no shutdown date (checked deprecations page 2026-11-18).
# Not gemini-3.5-flash: that id is silently re-routed to 3.6, so the report would name the wrong model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
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
# Keys are the exact activity names used in the dataset ('Activity Type').
# Windows are [ideal_start, ideal_end) in hours; if start > end the window
# wraps past midnight (e.g. Sleeping 21 → 1 means 21:00-01:00).
ACTIVITY_TIME_RULES = {
    'Eating':     {'ideal_start': 7,  'ideal_end': 21, 'penalty_outside': 0.3},
    'Cooking':    {'ideal_start': 7,  'ideal_end': 21, 'penalty_outside': 0.2},
    'Sleeping':   {'ideal_start': 21, 'ideal_end': 1,  'penalty_outside': 0.5},  # wraps midnight
    'Exercising': {'ideal_start': 6,  'ideal_end': 10, 'bonus': 0.1, 'penalty_outside': 0.2},
    'Working':    {'ideal_start': 9,  'ideal_end': 18, 'penalty_outside': 0.2},
    'Walking':    {'ideal_start': 6,  'ideal_end': 20, 'penalty_outside': 0.1},
    'Shopping':   {'ideal_start': 9,  'ideal_end': 21, 'penalty_outside': 0.1},
}

# Unhealthy patterns: a start hour inside 'window' [start, end) is penalised.
UNHEALTHY_PATTERNS = {
    'late_eating':    {'activity': ['Eating'],     'window': (22, 5),  'penalty': 0.4},
    'late_sleep':     {'activity': ['Sleeping'],   'window': (1, 12),  'penalty': 0.5},
    'early_sleep':    {'activity': ['Sleeping'],   'window': (12, 20), 'penalty': 0.2},
    'night_exercise': {'activity': ['Exercising'], 'window': (21, 5),  'penalty': 0.3},
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