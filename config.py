"""
Configuration - edit ONLY these fields for each new brand.
How it works automatically:
  1. If APP_STORE_ID is set -> scrapes Apple App Store reviews
  2. If APP_STORE_ID is empty -> scrapes Google Reviews via SerpAPI
  3. GitHub Actions runs this on every push to config.py
"""

# ── Brand Settings (only thing you change) ────────────────────────────────────
BRAND_NAME   = "Tapestry"
APP_NAME     = BRAND_NAME
KEYWORDS     = [
    "Coach outlet store",
    "Coach New York store",
    "Kate Spade outlet store",
    "Coach handbag store",
]

# ── App Store (leave blank if no app) ────────────────────────────────────────
APP_STORE_ID = ""
APP_COUNTRY  = "us"

# ── Platform Branding ─────────────────────────────────────────────────────────
PLATFORM_TITLE    = "Tapestry  Intelligence Platform"
PLATFORM_SUBTITLE = "Customer Insights & Operations"
PLATFORM_ICON     = "👜"

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
