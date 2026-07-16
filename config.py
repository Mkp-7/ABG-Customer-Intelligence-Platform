"""
Configuration — edit this file to target any retail/service brand(s).

To use with a different company:
1. Update PLATFORM_TITLE, PLATFORM_SUBTITLE, PLATFORM_ICON
2. Add/update entries in BRANDS list
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
# app_store_id : iOS App Store ID — set to "" to skip App Store for that brand
# keywords     : used for Google Maps SerpAPI search queries
BRANDS = [
    {
        "brand_id":     "avis",
        "name":         "Avis",
        "app_store_id": "308342527",
        "keywords":     ["Avis car rental"],
    },
    {
        "brand_id":     "budget",
        "name":         "Budget",
        "app_store_id": "538787758",
        "keywords":     ["Budget car rental"],
    },
    {
        "brand_id":     "zipcar",
        "name":         "Zipcar",
        "app_store_id": "329384702",
        "keywords":     [],              # no physical locations
    },
]

# ── App Store ─────────────────────────────────────────────────────────────────
APP_COUNTRY        = "us"
MAX_REVIEW_PAGES   = 10               # Apple RSS gives up to 10 pages × ~50 reviews

# ── Google Maps (Places API) ──────────────────────────────────────────────────
# GOOGLE_PLACES_API_KEY stored in GitHub Secrets / Streamlit Secrets
# Free $200/month credit → console.cloud.google.com → enable Places API
GOOGLE_MAX_LOCATIONS   = 200          # per brand
GOOGLE_REVIEWS_PER_LOC = 5           # Places API returns max 5 reviews per location

# ── Analysis Thresholds ───────────────────────────────────────────────────────
ANOMALY_THRESHOLD_STARS  = 0.4
PEER_GROUP_COLUMN        = "state"
SIGNIFICANT_DELTA_STARS  = 0.3
