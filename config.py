"""
Configuration - edit ONLY these fields for each new brand.
"""

# ── Brand Settings ──────────────────────────────────────────────────────────
BRAND_NAME   = "Avis Budget Group"
APP_NAME     = BRAND_NAME
KEYWORDS     = [
    "Avis car rental",
    "Budget car rental",
    "Zipcar",
]
# For Avis
APP_STORE_ID = "308342527"
GOOGLE_SEARCH_QUERY = None  # not needed anymore

# For Budget
APP_STORE_ID = "538787758"

# For Zipcar
APP_STORE_ID = "329384702"
APP_COUNTRY  = "us"

PLATFORM_TITLE    = "ABG Intelligence Platform"
PLATFORM_SUBTITLE = "Customer Experience & Operations Analytics"
PLATFORM_ICON     = "🚗"

# ── AI Model ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Scraper Settings ──────────────────────────────────────────────────────────
MAX_REVIEW_PAGES = 10

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
REVIEWS_CSV    = "data/reviews.csv"
BUSINESSES_CSV = "data/businesses.csv"

# ── Analytics Settings ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD_STARS = 0.4
SIGNIFICANT_DELTA_STARS = 0.15
