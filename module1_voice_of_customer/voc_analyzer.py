"""
Voice of Customer AI engine.
Uses Groq (free LLM) for theme clustering, anomaly detection,
sentiment scoring, and executive summary generation.
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
            "2. Copy .env.example to .env and paste your key."
        )
    return Groq(api_key=api_key)


def cluster_themes(reviews_sample: list, client: Groq, industry: str = "retail") -> dict:
    """
    Identify top recurring themes across a sample of review texts.
    Returns structured JSON with theme names, sentiment, percentage, and example quote.
    """
    numbered = "\n".join([f"[{i+1}] {r[:300]}" for i, r in enumerate(reviews_sample)])

    prompt = f"""You are analyzing customer reviews for a {industry} chain.

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
    Find locations where the recent 30-day average rating has dropped
    significantly compared to their all-time average.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return pd.DataFrame()

    cutoff = df["date"].max() - pd.Timedelta(days=30)

    loc_col = "business_id" if "business_id" in df.columns else "place_name"

    # Drop rows with no location - App Store reviews have empty place_name
    df = df[df[loc_col].fillna("").str.strip() != ""]
    if df.empty:
        return pd.DataFrame()

    historical = (
        df.groupby(loc_col)["stars"]
        .agg(historical_avg="mean", total_reviews="count")
        .reset_index()
        .rename(columns={loc_col: "business_id"})
    )

    recent_df = df[df["date"] >= cutoff]
    if recent_df.empty:
        q80 = df["date"].quantile(0.80)
        recent_df = df[df["date"] >= q80]

    recent = (
        recent_df.groupby(loc_col)["stars"]
        .agg(recent_avg="mean", recent_reviews="count")
        .reset_index()
        .rename(columns={loc_col: "business_id"})
    )

    merged = historical.merge(recent, on="business_id", how="inner")
    merged["rating_drop"] = merged["historical_avg"] - merged["recent_avg"]
    anomalies = merged[merged["rating_drop"] >= threshold].copy()
    return anomalies.sort_values("rating_drop", ascending=False)


def write_exec_summary(
    themes: list,
    anomaly_stores: pd.DataFrame,
    total_reviews: int,
    avg_rating: float,
    date_range: str,
    client: Groq,
    brand_name: str = "the chain",
) -> str:
    """
    Generate a ready-to-send executive summary paragraph for leadership.
    """
    themes_text = ""
    for t in themes[:5]:
        themes_text += f"- {t['name']} ({t['percent']}%, {t['sentiment']}): {t['description']}\n"

    anomaly_text = ""
    if not anomaly_stores.empty:
        for _, row in anomaly_stores.head(3).iterrows():
            loc = f"{row.get('city', row['business_id'])}"
            anomaly_text += (
                f"- {loc}: rating dropped {row['rating_drop']:.1f} stars "
                f"(from {row['historical_avg']:.1f} to {row['recent_avg']:.1f})\n"
            )
    else:
        anomaly_text = "No significant anomaly locations detected this period.\n"

    prompt = f"""You are writing a weekly executive summary for the VP of Store Operations at {brand_name}.

DATA:
- Period: {date_range}
- Reviews analyzed: {total_reviews:,}
- Average rating: {avg_rating:.2f} / 5.0

TOP THEMES:
{themes_text}

LOCATIONS NEEDING ATTENTION:
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
    """
    Score a batch of review texts as positive / neutral / negative.
    Returns a list of labels in the same order as input.
    """
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
