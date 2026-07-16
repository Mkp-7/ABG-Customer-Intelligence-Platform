"""
Configuration - edit this file to target any retail/service brand(s).

To use with a different company:
1. Update PLATFORM_TITLE, PLATFORM_SUBTITLE, PLATFORM_ICON
2. Add entries to BRANDS list
3. Re-run: python module1_voice_of_customer/01_extract_reviews.py
"""

PLATFORM_TITLE    = "Avis Budget Group Intelligence"
PLATFORM_SUBTITLE = "Customer Insights & Store Operations"
PLATFORM_ICON     = "🚗"

GROQ_MODEL = "llama-3.3-70b-versatile"

DATA_DIR       = "data"
BUSINESSES_CSV = "data/businesses.csv"
REVIEWS_CSV    = "data/reviews.csv"

# ── Brands ────────────────────────────────────────────────────────────────────
# Add/remove brands here. Set app_store_id or google_query to None to skip that source.
BRANDS = [
    {
        "brand_id":      "avis",
        "name":          "Avis",
        "app_store_id":  "308342527",
        "google_query":  "Avis car rental",
    },
    {
        "brand_id":      "budget",
        "name":          "Budget",
        "app_store_id":  "538787758",
        "google_query":  "Budget car rental",
    },
    {
        "brand_id":      "zipcar",
        "name":          "Zipcar",
        "app_store_id":  "329384702",
        "google_query":  None,          # no physical locations
    },
]

# ── App Store ─────────────────────────────────────────────────────────────────
APP_STORE_COUNTRY     = "us"
APP_STORE_MAX_REVIEWS = 2000           # per brand

# ── Google Maps (Places API) ──────────────────────────────────────────────────
# Key stored in GitHub Secrets / Streamlit Secrets as GOOGLE_PLACES_API_KEY
GOOGLE_MAX_LOCATIONS   = 200           # per brand
GOOGLE_REVIEWS_PER_LOC = 5

# ── Analysis Thresholds ───────────────────────────────────────────────────────
ANOMALY_THRESHOLD_STARS  = 0.4
PEER_GROUP_COLUMN        = "state"
SIGNIFICANT_DELTA_STARS  = 0.3
