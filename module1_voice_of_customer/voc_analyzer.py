"""
Voice of Customer AI engine - Groq/LLaMA
"""

import os
import json
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "1. Get a free key at https://console.groq.com\n"
            "2. Add it to Streamlit Cloud secrets or a local .env file."
        )
    return Groq(api_key=api_key)


def cluster_themes(reviews_sample: list, client: Groq, industry: str = "car rental") -> dict:
    numbered = "\n".join([f"[{i+1}] {r[:300]}" for i, r in enumerate(reviews_sample)])

    prompt = f"""You are analyzing customer reviews for a {industry} company.

Here are {len(reviews_sample)} customer reviews:

{numbered}

Identify the TOP 6 recurring themes. For each theme respond with:
- name: short label (3-5 words)
- description: what customers say (1 sentence)
- percent: estimated % of reviews mentioning it (integer)
- sentiment: exactly one of: positive, negative, mixed
- example_quote: one representative phrase under 15 words

Respond ONLY in this JSON format, no other text:
{{
  "themes": [
    {{
      "name": "...",
      "description": "...",
      "percent": 0,
      "sentiment": "positive",
      "example_quote": "..."
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"themes": [], "error": "Parse error", "raw": raw}


def detect_anomalies(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """
    Find locations performing significantly below their brand average rating.

    Compares each location's avg rating against the overall brand average
    (not time-based) so it works reliably regardless of review age.
    Only uses reviews that have a real location (Google Maps reviews).
    """
    df = df.copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])

    if df.empty:
        return pd.DataFrame()

    loc_col = "place_name" if "place_name" in df.columns else (
              "business_id" if "business_id" in df.columns else None)
    if loc_col is None:
        return pd.DataFrame()

    # Only use reviews that have a real location (drops App Store reviews)
    df = df[df[loc_col].fillna("").str.strip() != ""]
    if df.empty:
        return pd.DataFrame()

    # Per-location average
    group_cols = [loc_col]
    if "brand_id" in df.columns:
        group_cols = ["brand_id", loc_col]

    loc_agg = (
        df.groupby(group_cols)["stars"]
        .agg(avg_rating="mean", total_reviews="count")
        .reset_index()
        .rename(columns={loc_col: "business_id"})
    )

    # Brand average computed PER brand so each brand's locations are compared
    # against their own brand's standard (not a blended cross-brand average)
    if "brand_id" in df.columns:
        brand_avgs = df.groupby("brand_id")["stars"].mean().rename("brand_avg")
        loc_agg = loc_agg.merge(brand_avgs, on="brand_id", how="left")
    else:
        loc_agg["brand_avg"] = df["stars"].mean()

    loc_agg["brand_avg"]      = loc_agg["brand_avg"].round(2)
    loc_agg["rating_drop"]    = (loc_agg["brand_avg"] - loc_agg["avg_rating"]).round(2)
    loc_agg["historical_avg"] = loc_agg["brand_avg"]   # brand avg = baseline
    loc_agg["recent_avg"]     = loc_agg["avg_rating"].round(2)
    loc_agg["recent_reviews"] = loc_agg["total_reviews"]

    # Flag locations below their own brand average by >= threshold with at least 2 reviews
    anomalies = loc_agg[
        (loc_agg["rating_drop"] >= threshold) &
        (loc_agg["total_reviews"] >= 2)
    ].copy()

    return anomalies.sort_values("rating_drop", ascending=False)


def write_exec_summary(
    themes: list,
    anomaly_stores: pd.DataFrame,
    total_reviews: int,
    avg_rating: float,
    date_range: str,
    client: Groq,
    brand_name: str = "the brand",
) -> str:
    themes_text = ""
    for t in themes[:5]:
        themes_text += f"- {t['name']} ({t['percent']}%, {t['sentiment']}): {t['description']}\n"

    anomaly_text = ""
    if not anomaly_stores.empty:
        for _, row in anomaly_stores.head(3).iterrows():
            loc = row.get("city", row.get("business_id", "Unknown"))
            anomaly_text += (
                f"- {loc}: {row['recent_avg']:.1f}⭐ "
                f"(brand avg: {row['historical_avg']:.1f}⭐, "
                f"gap: -{row['rating_drop']:.1f})\n"
            )
    else:
        anomaly_text = "No locations significantly below brand average.\n"

    prompt = f"""You are writing a weekly executive summary for the VP of Store Operations at {brand_name}.

DATA:
- Period: {date_range}
- Reviews analyzed: {total_reviews:,}
- Average rating: {avg_rating:.2f} / 5.0

TOP THEMES:
{themes_text}

LOCATIONS NEEDING ATTENTION (below brand average by 0.4+ stars):
{anomaly_text}

Write a concise executive summary (3-4 short paragraphs) covering:
1. Overall customer experience headline
2. Most important theme finding
3. Location anomalies as action items for Field Leaders
4. One specific recommendation

Plain business English. No bullet points. No headers. Under 200 words."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def score_sentiment_batch(texts: list, client: Groq) -> list:
    numbered = "\n".join([f"[{i+1}] {t[:200]}" for i, t in enumerate(texts)])

    prompt = f"""Rate the sentiment of each review.
Respond ONLY with a JSON array of strings.
Each string must be exactly: "positive", "neutral", or "negative"
Example for 3 reviews: ["positive", "negative", "neutral"]

Reviews:
{numbered}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        labels = json.loads(raw)
        valid = {"positive", "neutral", "negative"}
        return [l if l in valid else "neutral" for l in labels]
    except Exception:
        return ["neutral"] * len(texts)
