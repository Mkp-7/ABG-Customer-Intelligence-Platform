TARGET_BUSINESS_NAMES = [
    "Avis", "Budget", "Zipcar"
]

PLATFORM_TITLE    = "ABG Customer Intelligence Platform"
PLATFORM_SUBTITLE = "Avis · Budget · Zipcar"
PLATFORM_ICON     = "🚗"

HF_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_DIR         = "data"
BUSINESSES_CSV   = "data/businesses.csv"
REVIEWS_CSV      = "data/reviews.csv"

ANOMALY_THRESHOLD_STARS  = 0.4
PEER_GROUP_COLUMN        = "state"
SIGNIFICANT_DELTA_STARS  = 0.3
