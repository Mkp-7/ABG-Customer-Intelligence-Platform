"""
Voice of Customer AI engine — pure NLP (TF-IDF + VADER)
No API keys, no rate limits, works on Streamlit Cloud forever.
"""
import os, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

STOPWORDS = {
    "the","a","an","and","was","is","to","i","my","it","of","in","for","that",
    "they","we","very","good","great","this","with","have","not","but","are",
    "so","be","had","from","at","he","she","our","their","were","just","been",
    "would","there","when","all","also","you","your","will","more","than",
    "could","has","its","me","us","get","got","went","back","like","what",
    "which","then","them","said","time","one","out","up","do","did","no",
    "car","rental","rent","avis","budget","zipcar","vehicle","location","place",
}

analyzer = SentimentIntensityAnalyzer()


# ── Keep these so existing callers don't break ────────────────────────────────
def get_client():
    return None

def get_groq_client():
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    if score >= 0.15:  return "positive"
    if score <= -0.15: return "negative"
    return "mixed"

def _clean(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text.lower()).strip()

def _top_words(texts: list, n: int = 6) -> list:
    words = " ".join([_clean(t) for t in texts]).split()
    freq = {}
    for w in words:
        if len(w) > 4 and w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]]

def _best_quote(texts: list, keywords: list) -> str:
    """Pick the shortest review that contains at least one theme keyword."""
    kw_set = set(keywords)
    candidates = [
        t for t in texts
        if any(k in t.lower() for k in kw_set) and 20 < len(t) < 180
    ]
    if candidates:
        return min(candidates, key=len)
    return texts[0][:120] if texts else ""


# ── Theme descriptions — maps top TF-IDF terms to human-readable labels ──────

THEME_LABELS = {
    frozenset(["wait", "line", "long", "hour", "minute", "slow"]):
        ("Long Wait Times", "negative",
         "Customers frequently report excessive wait times at pickup, even with reservations."),
    frozenset(["staff", "rude", "employee", "agent", "helpful", "friendly", "unprofessional"]):
        ("Staff & Service Quality", "mixed",
         "Staff interactions are a major differentiator — praised when helpful, criticized when dismissive."),
    frozenset(["clean", "dirty", "smell", "condition", "damage", "scratch", "dent"]):
        ("Vehicle Condition & Cleanliness", "negative",
         "Customers commonly flag dirty interiors, undisclosed damage, and poorly maintained vehicles."),
    frozenset(["charge", "bill", "fee", "price", "refund", "overcharg", "hidden", "extra"]):
        ("Unexpected Fees & Billing", "negative",
         "Hidden charges, credit card holds, and difficulty getting refunds are recurring frustrations."),
    frozenset(["reservation", "book", "confirm", "cancel", "available", "upgrade"]):
        ("Reservation & Booking Issues", "negative",
         "Customers report reserved vehicles not being available or bookings being incorrectly handled."),
    frozenset(["easy", "smooth", "quick", "fast", "great", "excellent", "perfect", "love"]):
        ("Smooth & Easy Experience", "positive",
         "Many customers praise fast pickups, friendly staff, and hassle-free returns when things go right."),
    frozenset(["app", "online", "website", "digital", "check", "mobile"]):
        ("App & Digital Experience", "mixed",
         "The mobile app and online check-in process receives mixed reviews for reliability."),
    frozenset(["return", "drop", "office", "close", "park", "shuttle"]):
        ("Return & Drop-off Process", "mixed",
         "Drop-off experiences vary — some find it seamless while others struggle with unclear processes."),
}

def _match_label(keywords: list):
    kw_set = set(keywords)
    best_match, best_score = None, 0
    for label_keys, label_data in THEME_LABELS.items():
        score = len(kw_set & label_keys)
        if score > best_score:
            best_score = score
            best_match = label_data
    return best_match


# ── Main functions ────────────────────────────────────────────────────────────

def cluster_themes(reviews_sample: list, client=None, industry: str = "retail") -> dict:
    """
    TF-IDF + KMeans clustering — finds top 6 themes from review text.
    No API needed.
    """
    texts = [t for t in reviews_sample if isinstance(t, str) and len(t.strip()) > 10]
    if len(texts) < 6:
        return {"themes": [], "error": "Not enough reviews to cluster."}

    n_clusters = min(6, len(texts) // 5)

    try:
        vec = TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        X = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        themes = []
        for cluster_id in range(n_clusters):
            cluster_texts = [texts[i] for i, l in enumerate(labels) if l == cluster_id]
            if not cluster_texts:
                continue

            # Top TF-IDF keywords for this cluster
            center     = km.cluster_centers_[cluster_id]
            top_idx    = center.argsort()[-8:][::-1]
            keywords   = [terms[i] for i in top_idx]
            top_words  = _top_words(cluster_texts, n=5)

            # Sentiment via VADER
            scores     = [analyzer.polarity_scores(t)["compound"] for t in cluster_texts[:50]]
            avg_score  = np.mean(scores)
            sentiment  = _sentiment_label(avg_score)

            # Match to human-readable label
            matched = _match_label(keywords + top_words)
            if matched:
                name, _, description = matched
            else:
                name        = " ".join(top_words[:3]).title()
                description = f"Customers frequently discuss {', '.join(top_words[:4])} in their reviews."

            # Override sentiment from matched label only if cluster is strongly one-way
            if matched and abs(avg_score) < 0.1:
                sentiment = matched[1]

            percent = round(len(cluster_texts) / len(texts) * 100)
            quote   = _best_quote(cluster_texts, top_words)

            themes.append({
                "name":          name,
                "description":   description,
                "percent":       percent,
                "sentiment":     sentiment,
                "example_quote": quote,
            })

        # Sort: negatives first (most actionable)
        themes.sort(key=lambda t: (t["sentiment"] != "negative", -t["percent"]))
        return {"themes": themes}

    except Exception as e:
        return {"themes": [], "error": str(e)}


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

    brand_avg                 = df["stars"].mean()
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
                       avg_rating, date_range, client=None, brand_name="ABG") -> str:
    """
    Generates a human-readable executive summary from computed theme data.
    No API needed — pure template + data logic.
    """
    if not themes:
        return "Insufficient data to generate an executive summary."

    neg_themes = [t for t in themes if t["sentiment"] == "negative"]
    pos_themes = [t for t in themes if t["sentiment"] == "positive"]

    # Headline
    if avg_rating >= 4.0:
        headline = f"Customer sentiment for {brand_name} is strong, with an average rating of {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}."
    elif avg_rating >= 3.0:
        headline = f"{brand_name} is delivering a mixed customer experience, averaging {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}. There is clear room for improvement."
    else:
        headline = f"Customer experience at {brand_name} requires immediate attention — the average rating is {avg_rating:.2f}⭐ across {total_reviews:,} reviews during {date_range}."

    # Top issue
    if neg_themes:
        top = neg_themes[0]
        issue_para = (
            f"The most urgent theme is **{top['name']}** ({top['percent']}% of reviews): "
            f"{top['description']} "
            f"One customer noted: \"{top['example_quote']}\""
        )
    elif themes:
        top = themes[0]
        issue_para = (
            f"The dominant theme is **{top['name']}** ({top['percent']}% of reviews): "
            f"{top['description']}"
        )
    else:
        issue_para = "No dominant theme could be identified from the current dataset."

    # Positive signal
    if pos_themes:
        pos = pos_themes[0]
        pos_para = f"On the positive side, **{pos['name']}** ({pos['percent']}% of reviews) shows what's working: {pos['description']}"
    else:
        pos_para = "Positive signals are limited in this period — a focus on resolving the top complaints would likely move the rating meaningfully."

    # Anomaly callout
    if not anomaly_stores.empty:
        top_loc = anomaly_stores.iloc[0]
        loc_name = top_loc.get("city", top_loc.get("business_id", "One location"))
        anomaly_para = (
            f"Field Leaders should prioritize **{loc_name}**, which is rated "
            f"{top_loc['recent_avg']:.1f}⭐ against a brand average of {top_loc['historical_avg']:.1f}⭐ "
            f"— a gap of {top_loc['rating_drop']:.1f} stars that warrants an immediate operational review."
        )
    else:
        anomaly_para = "No locations are significantly below the brand average at this time."

    return f"{headline}\n\n{issue_para}\n\n{pos_para}\n\n{anomaly_para}"


def score_sentiment_batch(texts: list, client=None) -> list:
    """VADER sentiment scoring — no API needed."""
    results = []
    for t in texts:
        score = analyzer.polarity_scores(str(t))["compound"]
        results.append(_sentiment_label(score))
    return results
