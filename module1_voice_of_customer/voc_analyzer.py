"""
Voice of Customer AI engine — Hugging Face (huggingface_hub InferenceClient)
"""
import os, json
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def get_client():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            import streamlit as st
            token = st.secrets.get("HF_TOKEN", "")
        except Exception:
            pass
    if not token:
        raise ValueError(
            "HF_TOKEN not set.\n"
            "Get a free token at https://huggingface.co/settings/tokens\n"
            "Add it to Streamlit Secrets: HF_TOKEN = 'hf_...'"
        )
    return InferenceClient(token=token)

def get_groq_client():
    return get_client()


def _generate(client: InferenceClient, prompt: str, max_tokens: int = 800) -> str:
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def cluster_themes(reviews_sample: list, client, industry: str = "retail") -> dict:
    numbered = "\n".join([f"[{i+1}] {r[:300]}" for i, r in enumerate(reviews_sample)])

    prompt = f"""You are analyzing customer reviews for a {industry} company.

Here are {len(reviews_sample)} customer reviews:
{numbered}

Identify the TOP 6 recurring themes. Respond ONLY in this exact JSON format, no other text:
{{
  "themes": [
    {{
      "name": "short label 3-5 words",
      "description": "one sentence",
      "percent": 0,
      "sentiment": "positive",
      "example_quote": "under 15 words"
    }}
  ]
}}"""

    raw = _generate(client, prompt, max_tokens=1000)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except:
        return {"themes": [], "error": "parse error", "raw": raw}


def detect_anomalies(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    df = df.copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    if df.empty:
        return pd.DataFrame()

    loc_col = "place_name" if "place_name" in df.columns else (
              "business_id" if "business_id" in df.columns else None)
    if loc_col is None:
        return pd.DataFrame()

    df = df[df[loc_col].fillna("").str.strip() != ""]
    if df.empty:
        return pd.DataFrame()

    loc_agg = (
        df.groupby(loc_col)["stars"]
        .agg(avg_rating="mean", total_reviews="count")
        .reset_index()
        .rename(columns={loc_col: "business_id"})
    )

    brand_avg = df["stars"].mean()
    loc_agg["brand_avg"]      = round(brand_avg, 2)
    loc_agg["rating_drop"]    = (loc_agg["brand_avg"] - loc_agg["avg_rating"]).round(2)
    loc_agg["historical_avg"] = loc_agg["brand_avg"]
    loc_agg["recent_avg"]     = loc_agg["avg_rating"].round(2)
    loc_agg["recent_reviews"] = loc_agg["total_reviews"]

    anomalies = loc_agg[
        (loc_agg["rating_drop"] >= threshold) &
        (loc_agg["total_reviews"] >= 2)
    ].copy()
    return anomalies.sort_values("rating_drop", ascending=False)


def write_exec_summary(themes, anomaly_stores, total_reviews,
                       avg_rating, date_range, client, brand_name="ABG") -> str:
    themes_text = ""
    for t in themes[:5]:
        themes_text += f"- {t['name']} ({t['percent']}%, {t['sentiment']}): {t['description']}\n"

    anomaly_text = ""
    if not anomaly_stores.empty:
        for _, row in anomaly_stores.head(3).iterrows():
            loc = row.get("city", row.get("business_id", "Unknown"))
            anomaly_text += f"- {loc}: {row['recent_avg']:.1f}⭐ (avg: {row['historical_avg']:.1f}⭐)\n"
    else:
        anomaly_text = "No significant anomalies detected.\n"

    prompt = f"""Write a concise executive summary (3 short paragraphs, under 200 words) for {brand_name} leadership.

Data:
- Period: {date_range}
- Reviews: {total_reviews:,}
- Avg rating: {avg_rating:.2f}/5.0

Top themes:
{themes_text}

Locations needing attention:
{anomaly_text}

Plain business English. No bullet points. No headers."""

    return _generate(client, prompt, max_tokens=400)


def score_sentiment_batch(texts: list, client) -> list:
    numbered = "\n".join([f"[{i+1}] {t[:200]}" for i, t in enumerate(texts)])

    prompt = f"""Rate the sentiment of each review.
Respond ONLY with a JSON array of strings, nothing else.
Each string must be exactly: "positive", "neutral", or "negative"
Example for 3 reviews: ["positive", "negative", "neutral"]

Reviews:
{numbered}"""

    raw = _generate(client, prompt, max_tokens=200)
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        labels = json.loads(raw)
        valid = {"positive", "neutral", "negative"}
        return [l if l in valid else "neutral" for l in labels]
    except:
        return ["neutral"] * len(texts)
