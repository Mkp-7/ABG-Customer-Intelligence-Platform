"""
Step 1 - Extract reviews from App Store and/or Google Maps (US only).

Data sources used (both free):
  • App Store
  • Google Maps
Loops through all brands defined in config.py → BRANDS list.

Usage:
    python module1_voice_of_customer/01_extract_reviews.py

Output:
    data/businesses.csv   (all brands combined, tagged with brand_id)
    data/reviews.csv      (all brands combined, tagged with brand_id)
"""

import os
import sys
import hashlib
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BRANDS,
    APP_STORE_COUNTRY,
    APP_STORE_MAX_REVIEWS,
    GOOGLE_MAX_LOCATIONS,
    GOOGLE_REVIEWS_PER_LOC,
    BUSINESSES_CSV,
    REVIEWS_CSV,
    DATA_DIR,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_id(*parts) -> str:
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Source 1: App Store ───────────────────────────────────────────────────────

def extract_app_store(brand: dict) -> tuple[list[dict], list[dict]]:
    """Pull iOS App Store reviews for one brand."""
    app_id = brand.get("app_store_id")
    if not app_id:
        print(f"   [{brand['name']}] App Store: skipped (no app_store_id)")
        return [], []

    try:
        from app_store_scraper import AppStore
    except ImportError:
        print("   ERROR: run  pip install app-store-scraper")
        return [], []

    print(f"   [{brand['name']}] App Store: fetching up to {APP_STORE_MAX_REVIEWS} reviews (ID: {app_id})...")

    scraper = AppStore(
        country=APP_STORE_COUNTRY,
        app_name=brand["name"].lower().replace(" ", "-"),
        app_id=app_id,
    )
    scraper.review(how_many=APP_STORE_MAX_REVIEWS)

    raw = scraper.reviews or []
    if not raw:
        print(f"   [{brand['name']}] App Store: 0 reviews returned.")
        return [], []

    biz_id = f"appstore_{app_id}"

    businesses = [{
        "business_id":  biz_id,
        "brand_id":     brand["brand_id"],
        "brand_name":   brand["name"],
        "name":         f"{brand['name']} (App Store)",
        "address":      "",
        "city":         "",
        "state":        "APP",
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
        date_str = date_val.strftime("%Y-%m-%d") if isinstance(date_val, datetime) else str(date_val)[:10]

        reviews.append({
            "review_id":   make_id(biz_id, r.get("userName", ""), date_str),
            "business_id": biz_id,
            "brand_id":    brand["brand_id"],
            "brand_name":  brand["name"],
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

    print(f"   [{brand['name']}] App Store: {len(reviews)} reviews ✅")
    return businesses, reviews


# ── Source 2: Google Maps ─────────────────────────────────────────────────────

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


def extract_google_maps(brand: dict, gmaps) -> tuple[list[dict], list[dict]]:
    """Pull Google Maps reviews for one brand."""
    query = brand.get("google_query")
    if not query:
        print(f"   [{brand['name']}] Google Maps: skipped (no google_query)")
        return [], []

    print(f"   [{brand['name']}] Google Maps: searching '{query}' across US states...")
    print(f"   Cap: {GOOGLE_MAX_LOCATIONS} locations, {GOOGLE_REVIEWS_PER_LOC} reviews each")

    seen_place_ids = set()
    businesses = []
    reviews = []

    for state in tqdm(US_STATES, desc=f"   {brand['name']} states"):
        if len(seen_place_ids) >= GOOGLE_MAX_LOCATIONS:
            break

        try:
            result = gmaps.places(query=f"{query} {state} USA", region="us")
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

            loc_state = loc_city = loc_zip = country = ""
            for comp in details.get("address_components", []):
                types = comp.get("types", [])
                if "administrative_area_level_1" in types:
                    loc_state = comp.get("short_name", "")
                if "locality" in types:
                    loc_city = comp.get("long_name", "")
                if "postal_code" in types:
                    loc_zip = comp.get("long_name", "")
                if "country" in types:
                    country = comp.get("short_name", "")

            if country and country != "US":
                continue

            geo    = details.get("geometry", {}).get("location", {})
            biz_id = make_id("google", place_id)

            businesses.append({
                "business_id":  biz_id,
                "brand_id":     brand["brand_id"],
                "brand_name":   brand["name"],
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
                "app_version":  "",
            })

            for r in details.get("reviews", [])[:GOOGLE_REVIEWS_PER_LOC]:
                date_str = datetime.utcfromtimestamp(r.get("time", 0)).strftime("%Y-%m-%d")
                reviews.append({
                    "review_id":   make_id("google", place_id, r.get("author_name", ""), date_str),
                    "business_id": biz_id,
                    "brand_id":    brand["brand_id"],
                    "brand_name":  brand["name"],
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

    print(f"   [{brand['name']}] Google Maps: {len(businesses)} locations, {len(reviews)} reviews ✅")
    return businesses, reviews


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Smart Data Extractor - Multi-Brand")
    print(f"  Brands: {', '.join(b['name'] for b in BRANDS)}")
    print("  Sources: App Store + Google Maps (US only)")
    print("=" * 55)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Init Google Maps client once (shared across brands)
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    gmaps = None
    if api_key:
        try:
            import googlemaps
            gmaps = googlemaps.Client(key=api_key)
        except ImportError:
            print("⚠️  googlemaps not installed. Run: pip install googlemaps")
    else:
        print("⚠️  GOOGLE_PLACES_API_KEY not set - Google Maps will be skipped for all brands.")

    all_businesses = []
    all_reviews    = []

    for brand in BRANDS:
        print(f"\n{'─'*55}")
        print(f"  🏷  {brand['name'].upper()}")
        print(f"{'─'*55}")

        # App Store
        biz, rev = extract_app_store(brand)
        all_businesses.extend(biz)
        all_reviews.extend(rev)

        # Google Maps
        if gmaps:
            biz, rev = extract_google_maps(brand, gmaps)
            all_businesses.extend(biz)
            all_reviews.extend(rev)

    # ── Save ──────────────────────────────────────────────────────────────────
    if not all_reviews:
        print("\n⚠️  No reviews collected from any source.")
        print("   Check app_store_id values in BRANDS and GOOGLE_PLACES_API_KEY secret.")
        sys.exit(1)

    biz_df = pd.DataFrame(all_businesses).drop_duplicates(subset=["business_id"])
    rev_df = pd.DataFrame(all_reviews).drop_duplicates(subset=["review_id"])

    biz_df.to_csv(BUSINESSES_CSV, index=False)
    rev_df.to_csv(REVIEWS_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"✅ {len(biz_df)} locations  → {BUSINESSES_CSV}")
    print(f"✅ {len(rev_df):,} reviews   → {REVIEWS_CSV}")

    for brand in BRANDS:
        br = rev_df[rev_df["brand_id"] == brand["brand_id"]]
        if len(br):
            avg = pd.to_numeric(br["stars"], errors="coerce").mean()
            src = br["source"].value_counts().to_dict()
            print(f"   {brand['name']}: {len(br):,} reviews | avg {avg:.2f}⭐ | {src}")

    print(f"\n  Run: streamlit run main_app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
