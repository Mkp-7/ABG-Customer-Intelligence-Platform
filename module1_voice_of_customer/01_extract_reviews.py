"""
Step 1 - Extract reviews from App Store and/or Google Maps (US only).

Data sources used (both free):
  • App Store  → app-store-scraper  (no API key needed)
  • Google Maps → Google Places API  (free $200/month credit)
                  Get key: console.cloud.google.com → enable Places API

Configure in config.py:
  APP_STORE_ID          = "308342527"       # set to None to skip
  GOOGLE_PLACES_API_KEY = "AIza..."         # set to None to skip
  GOOGLE_SEARCH_QUERY   = "Avis car rental" # what to search on Google Maps

Usage:
    python module1_voice_of_customer/01_extract_reviews.py

Output:
    data/businesses.csv
    data/reviews.csv
"""

import os
import sys
import uuid
import hashlib
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    APP_STORE_ID,
    APP_STORE_COUNTRY,
    APP_STORE_MAX_REVIEWS,
    GOOGLE_PLACES_API_KEY,
    GOOGLE_SEARCH_QUERY,
    GOOGLE_MAX_LOCATIONS,
    GOOGLE_REVIEWS_PER_LOC,
    BUSINESSES_CSV,
    REVIEWS_CSV,
    DATA_DIR,
    PLATFORM_TITLE,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_id(*parts) -> str:
    """Deterministic short ID from input parts."""
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Source 1: App Store ───────────────────────────────────────────────────────

def extract_app_store() -> tuple[list[dict], list[dict]]:
    """Pull reviews from the iOS App Store. Returns (businesses, reviews)."""
    if not APP_STORE_ID:
        print("   App Store: skipped (APP_STORE_ID not set in config.py)")
        return [], []

    try:
        from app_store_scraper import AppStore
    except ImportError:
        print("   ERROR: run  pip install app-store-scraper")
        return [], []

    print(f"   Fetching up to {APP_STORE_MAX_REVIEWS} App Store reviews (ID: {APP_STORE_ID})...")

    scraper = AppStore(
        country=APP_STORE_COUNTRY,
        app_name=PLATFORM_TITLE.lower().replace(" ", "-"),
        app_id=APP_STORE_ID,
    )
    scraper.review(how_many=APP_STORE_MAX_REVIEWS)

    raw = scraper.reviews or []
    if not raw:
        print("   App Store: 0 reviews returned.")
        return [], []

    biz_id = f"appstore_{APP_STORE_ID}"

    business = [{
        "business_id":  biz_id,
        "name":         f"{PLATFORM_TITLE} (App Store)",
        "address":      "",
        "city":         "",
        "state":        "APP",          # sentinel so modules can filter
        "postal_code":  "",
        "latitude":     "",
        "longitude":    "",
        "stars":        round(sum(r.get("rating", 0) for r in raw) / len(raw), 2),
        "review_count": len(raw),
        "is_open":      1,
        "source":       "appstore",
    }]

    reviews = []
    for r in raw:
        date_val = r.get("date", "")
        if isinstance(date_val, datetime):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)[:10]

        reviews.append({
            "review_id":   make_id(biz_id, r.get("userName", ""), date_str),
            "business_id": biz_id,
            "user_id":     r.get("userName", "anonymous"),
            "stars":       r.get("rating", ""),
            "date":        date_str,
            "text":        (r.get("review") or r.get("title") or "").replace("\n", " ").strip(),
            "useful":      0,
            "funny":       0,
            "cool":        0,
            "source":      "appstore",
            "app_version": r.get("version", ""),
        })

    print(f"   App Store: {len(reviews)} reviews collected ✅")
    return business, reviews


# ── Source 2: Google Maps ─────────────────────────────────────────────────────

# US states for targeted search
US_STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
]


def extract_google_maps() -> tuple[list[dict], list[dict]]:
    """Pull reviews from Google Maps via Places API. Returns (businesses, reviews)."""
    api_key = GOOGLE_PLACES_API_KEY or os.environ.get("GOOGLE_PLACES_API_KEY")

    if not api_key:
        print("   Google Maps: skipped (GOOGLE_PLACES_API_KEY not set)")
        return [], []

    if not GOOGLE_SEARCH_QUERY:
        print("   Google Maps: skipped (GOOGLE_SEARCH_QUERY not set in config.py)")
        return [], []

    try:
        import googlemaps
    except ImportError:
        print("   ERROR: run  pip install googlemaps")
        return [], []

    gmaps = googlemaps.Client(key=api_key)

    print(f"   Google Maps: searching '{GOOGLE_SEARCH_QUERY}' across US states...")
    print(f"   Cap: {GOOGLE_MAX_LOCATIONS} locations, {GOOGLE_REVIEWS_PER_LOC} reviews each")

    seen_place_ids = set()
    businesses = []
    reviews = []

    for state in tqdm(US_STATES, desc="   States"):
        if len(seen_place_ids) >= GOOGLE_MAX_LOCATIONS:
            break

        query = f"{GOOGLE_SEARCH_QUERY} {state} USA"
        try:
            result = gmaps.places(query=query, region="us")
        except Exception as e:
            print(f"   Warning: search failed for {state}: {e}")
            continue

        for place in result.get("results", []):
            if len(seen_place_ids) >= GOOGLE_MAX_LOCATIONS:
                break

            place_id = place.get("place_id")
            if not place_id or place_id in seen_place_ids:
                continue

            seen_place_ids.add(place_id)

            # Get full details including reviews
            try:
                details = gmaps.place(
                    place_id=place_id,
                    fields=[
                        "name", "formatted_address", "geometry",
                        "rating", "user_ratings_total", "reviews",
                        "address_components", "business_status",
                    ]
                ).get("result", {})
            except Exception as e:
                print(f"   Warning: details failed for {place_id}: {e}")
                continue

            # Extract state from address components (US only filter)
            addr_components = details.get("address_components", [])
            loc_state = ""
            loc_city  = ""
            loc_zip   = ""
            country   = ""
            for comp in addr_components:
                types = comp.get("types", [])
                if "administrative_area_level_1" in types:
                    loc_state = comp.get("short_name", "")
                if "locality" in types:
                    loc_city = comp.get("long_name", "")
                if "postal_code" in types:
                    loc_zip = comp.get("long_name", "")
                if "country" in types:
                    country = comp.get("short_name", "")

            # Skip non-US results
            if country and country != "US":
                continue

            geo = details.get("geometry", {}).get("location", {})
            biz_id = make_id("google", place_id)

            businesses.append({
                "business_id":  biz_id,
                "name":         details.get("name", ""),
                "address":      details.get("formatted_address", ""),
                "city":         loc_city,
                "state":        loc_state,
                "postal_code":  loc_zip,
                "latitude":     geo.get("lat", ""),
                "longitude":    geo.get("lng", ""),
                "stars":        details.get("rating", ""),
                "review_count": details.get("user_ratings_total", 0),
                "is_open":      1 if details.get("business_status") == "OPERATIONAL" else 0,
                "source":       "google",
            })

            # Reviews (Google Places returns up to 5)
            for r in details.get("reviews", [])[:GOOGLE_REVIEWS_PER_LOC]:
                date_str = datetime.utcfromtimestamp(
                    r.get("time", 0)
                ).strftime("%Y-%m-%d")

                reviews.append({
                    "review_id":   make_id("google", place_id, r.get("author_name", ""), date_str),
                    "business_id": biz_id,
                    "user_id":     r.get("author_name", "anonymous"),
                    "stars":       r.get("rating", ""),
                    "date":        date_str,
                    "text":        r.get("text", "").replace("\n", " ").strip(),
                    "useful":      0,
                    "funny":       0,
                    "cool":        0,
                    "source":      "google",
                    "app_version": "",
                })

    print(f"   Google Maps: {len(businesses)} locations, {len(reviews)} reviews collected ✅")
    return businesses, reviews


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Smart Data Extractor")
    print(f"  {PLATFORM_TITLE}")
    print("  Combines App Store + Google Maps (US only)")
    print("=" * 55)

    os.makedirs(DATA_DIR, exist_ok=True)

    all_businesses = []
    all_reviews    = []

    # --- App Store ---
    print("\n📱 App Store")
    biz, rev = extract_app_store()
    all_businesses.extend(biz)
    all_reviews.extend(rev)

    # --- Google Maps ---
    print("\n🌍 Google Maps")
    biz, rev = extract_google_maps()
    all_businesses.extend(biz)
    all_reviews.extend(rev)

    # --- Save ---
    if not all_reviews:
        print("\n⚠️  No reviews collected from any source.")
        print("   Check APP_STORE_ID and GOOGLE_PLACES_API_KEY in config.py / .env")
        sys.exit(1)

    biz_df = pd.DataFrame(all_businesses).drop_duplicates(subset=["business_id"])
    rev_df = pd.DataFrame(all_reviews).drop_duplicates(subset=["review_id"])

    biz_df.to_csv(BUSINESSES_CSV, index=False)
    rev_df.to_csv(REVIEWS_CSV, index=False)

    print(f"\n✅ Saved {len(biz_df)} locations → {BUSINESSES_CSV}")
    print(f"✅ Saved {len(rev_df):,} reviews  → {REVIEWS_CSV}")

    # Summary
    avg = rev_df["stars"].apply(pd.to_numeric, errors="coerce").mean()
    sources = rev_df["source"].value_counts().to_dict()
    print(f"\n   Avg rating : {avg:.2f} ⭐")
    print(f"   By source  : {sources}")
    print("\n" + "=" * 55)
    print("  Done. Run: streamlit run main_app.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
