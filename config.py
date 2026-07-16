"""
Configuration — edit this file to target any retail/service brand.

To use with a different company:
1. Update PLATFORM_TITLE, PLATFORM_SUBTITLE, PLATFORM_ICON
2. Set APP_STORE_ID (or None to skip)
3. Set GOOGLE_SEARCH_QUERY (or None to skip)
4. Add GOOGLE_PLACES_API_KEY to your .env file
5. Re-run: python module1_voice_of_customer/01_extract_reviews.py
"""

PLATFORM_TITLE    = "Retail Intelligence Platform"
PLATFORM_SUBTITLE = "Customer Insights & Store Operations"
PLATFORM_ICON     = "🏪"

GROQ_MODEL = "llama-3.3-70b-versatile"

DATA_DIR       = "data"
BUSINESSES_CSV = "data/businesses.csv"
REVIEWS_CSV    = "data/reviews.csv"

# ── App Store ─────────────────────────────────────────────────────────────────
# No API key needed. Set to None to skip.
APP_STORE_ID          = None    # e.g. "308342527" for Avis
APP_STORE_COUNTRY     = "us"
APP_STORE_MAX_REVIEWS = 2000

# ── Google Maps (Places API) ──────────────────────────────────────────────────
# Free $200/month credit → console.cloud.google.com → enable Places API
GOOGLE_SEARCH_QUERY    = None   # e.g. "Avis car rental"
GOOGLE_MAX_LOCATIONS   = 200
GOOGLE_REVIEWS_PER_LOC = 5

# ── Analysis Thresholds ───────────────────────────────────────────────────────
ANOMALY_THRESHOLD_STARS  = 0.4
PEER_GROUP_COLUMN        = "state"
SIGNIFICANT_DELTA_STARS  = 0.3
