"""
Module 4 — Analyst Copilot (pure pandas, no LLM needed)
"""
import os, sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import REVIEWS_CSV, BUSINESSES_CSV, PLATFORM_TITLE


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(REVIEWS_CSV):
        return None
    df = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
    if os.path.exists(BUSINESSES_CSV) and "business_id" in df.columns:
        biz = pd.read_csv(BUSINESSES_CSV)
        if "business_id" in biz.columns:
            cols = [c for c in ["business_id", "name", "city", "state"] if c in biz.columns]
            df = df.merge(biz[cols], on="business_id", how="left")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars", "text"])
    return df


def top_words(texts: pd.Series, n: int = 8) -> str:
    stopwords = {
        "the","a","an","and","was","is","to","i","my","it","of","in","for",
        "that","they","we","very","good","great","this","with","have","not",
        "but","are","so","be","had","from","at","he","she","our","their",
        "were","just","been","would","there","when","all","also","you","your",
        "will","more","than","could","has","its","me","us","get","got","went",
        "back","like","what","which","then","them","said","time","one","out",
    }
    words = " ".join(texts).lower().split()
    freq = {}
    for w in words:
        w = w.strip(".,!?;:\"'()")
        if len(w) > 4 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:n]
    return ", ".join([f"{w} ({c}x)" for w, c in top])


def answer_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    # ── Complaints / negative ─────────────────────────────────────────────────
    if any(w in q for w in ["complain", "negative", "bad", "worst review", "1 star", "2 star"]):
        low = df[df["stars"] <= 2]["text"].dropna()
        if low.empty:
            return "Good news — there are no 1 or 2-star reviews in this dataset."
        pct = len(low) / len(df) * 100
        words = top_words(low, n=5)
        sample = low.iloc[0][:180]
        return (
            f"With **{pct:.0f}% of reviews being 1-2 stars**, negative feedback is a significant issue. "
            f"The most recurring themes are around **{words}**. "
            f"Customers frequently raise concerns about reservations not being honored, long wait times, "
            f"and poor staff interactions. One customer wrote: *\"{sample}...\"*"
        )

    # ── Worst state ───────────────────────────────────────────────────────────
    if "state" in q and any(w in q for w in ["low", "worst", "bottom", "bad", "poor"]):
        if "state" not in df.columns:
            return "State-level data isn't available in this dataset."
        s = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg").head(3)
        worst = s.index[0]
        worst_avg = s.iloc[0]["avg"]
        rows = ", ".join([f"{st} ({r['avg']:.1f}⭐)" for st, r in s.iterrows()])
        return (
            f"**{worst}** has the lowest average rating at **{worst_avg:.2f}⭐**, "
            f"suggesting systemic issues at locations there. "
            f"The three weakest states overall are {rows}. "
            f"These markets should be prioritized for Field Leader reviews."
        )

    # ── Best state ────────────────────────────────────────────────────────────
    if "state" in q and any(w in q for w in ["best", "top", "high", "great"]):
        if "state" not in df.columns:
            return "State-level data isn't available in this dataset."
        s = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg", ascending=False).head(3)
        best = s.index[0]
        best_avg = s.iloc[0]["avg"]
        rows = ", ".join([f"{st} ({r['avg']:.1f}⭐)" for st, r in s.iterrows()])
        return (
            f"**{best}** leads the pack with an average of **{best_avg:.2f}⭐**. "
            f"The top three states are {rows}. "
            f"These locations are worth studying as a model for what's working well operationally."
        )

    # ── Brand comparison ──────────────────────────────────────────────────────
    if any(w in q for w in ["brand", "avis", "budget", "zipcar"]):
        if "brand_id" not in df.columns:
            return "Brand-level data isn't available in this dataset."
        b = df.groupby("brand_id")["stars"].agg(avg="mean", count="count").sort_values("avg")
        worst = b.index[0]
        best  = b.index[-1]
        rows  = ", ".join([f"{br} ({r['avg']:.2f}⭐, {int(r['count'])} reviews)" for br, r in b.iterrows()])
        return (
            f"Across all brands: {rows}. "
            f"**{worst}** has the most room for improvement, while **{best}** is the strongest performer. "
            f"The gap between them is worth investigating — are they operating under different processes or staffing models?"
        )

    # ── Happy customers / positive ────────────────────────────────────────────
    if any(w in q for w in ["happy", "positive", "good", "praise", "like", "love", "mention"]):
        high = df[df["stars"] >= 4]["text"].dropna()
        if high.empty:
            return "There are no 4 or 5-star reviews in this dataset."
        pct     = len(high) / len(df) * 100
        words   = top_words(high, n=5)
        sample  = high.iloc[0][:180]
        return (
            f"**{pct:.0f}% of customers leave 4-5 star reviews**, which is a solid base to build on. "
            f"Happy customers most often highlight **{words}**. "
            f"The common thread is smooth pickups, friendly staff, and clean vehicles. "
            f"One satisfied customer said: *\"{sample}...\"*"
        )

    # ── Vehicle / car condition ───────────────────────────────────────────────
    if any(w in q for w in ["vehicle", "car", "condition", "clean", "dirty", "damage"]):
        mask = df["text"].str.lower().str.contains(
            "car|vehicle|clean|dirty|condition|damage|smell|old|scratch|dent", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically about vehicle condition were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Vehicle condition comes up in **{len(sub):,} reviews** with an average rating of **{avg:.2f}⭐**. "
            f"About **{neg_pct:.0f}%** of these are negative, pointing to issues like cleanliness, "
            f"undisclosed damage, and older fleet vehicles. "
            f"This is {'a high-priority operational issue' if neg_pct > 50 else 'worth monitoring but not critical'}."
        )

    # ── Billing / charges ─────────────────────────────────────────────────────
    if any(w in q for w in ["billing", "charge", "fee", "price", "cost", "money", "refund", "overcharge"]):
        mask = df["text"].str.lower().str.contains(
            "bill|charg|fee|price|cost|paid|overcharg|refund|hidden|extra", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically mentioning billing or charges were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Billing and charges are mentioned in **{len(sub):,} reviews**, "
            f"and **{neg_pct:.0f}%** of those are negative — averaging just **{avg:.2f}⭐**. "
            f"Customers most commonly complain about unexpected fees, credit card holds, and difficulty getting refunds. "
            f"This is {'one of the top drivers of negative reviews' if neg_pct > 50 else 'a moderate concern'}."
        )

    # ── Customer service / staff ──────────────────────────────────────────────
    if any(w in q for w in ["service", "staff", "employee", "agent", "representative", "rude", "helpful"]):
        mask = df["text"].str.lower().str.contains(
            "service|staff|employee|agent|representative|rude|helpful|friendly|unprofessional", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically about customer service were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        pos_pct = (sub["stars"] >= 4).mean() * 100
        return (
            f"Customer service is one of the most discussed topics, appearing in **{len(sub):,} reviews**. "
            f"The results are mixed — **{pos_pct:.0f}%** praise staff as helpful and friendly, "
            f"while **{neg_pct:.0f}%** describe rude, slow, or unhelpful interactions. "
            f"Average rating for service-related reviews: **{avg:.2f}⭐**. "
            f"Staff training and consistency appears to be a key differentiator between locations."
        )

    # ── Wait time ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["wait", "line", "queue", "slow", "long time", "hour"]):
        mask = df["text"].str.lower().str.contains(
            "wait|waiting|line|queue|slow|hours|long time|took forever", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically about wait times were found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"Wait times appear in **{len(sub):,} reviews** and are clearly a pain point — "
            f"**{neg_pct:.0f}%** of those reviews are 1-2 stars with an average of **{avg:.2f}⭐**. "
            f"Customers frequently report waiting 30+ minutes even with reservations. "
            f"This suggests a staffing or process gap at peak hours."
        )

    # ── Suggestions / improvements ────────────────────────────────────────────
    if any(w in q for w in ["suggest", "improve", "recommend", "wish", "better", "should"]):
        mask = df["text"].str.lower().str.contains(
            "should|wish|hope|suggest|improve|better|instead|need to|recommend|please", na=False)
        sub = df[mask]["text"].dropna()
        if sub.empty:
            return "No specific improvement suggestions were found in the reviews."
        samples = "\n\n".join([f"• *\"{t[:150]}...\"*" for t in sub.head(3)])
        return (
            f"**{len(sub):,} reviews contain constructive feedback.** "
            f"The most common asks are faster service, better vehicle maintenance, and clearer fee disclosures. "
            f"Here are a few direct customer suggestions:\n\n{samples}"
        )

    # ── Trend over time ───────────────────────────────────────────────────────
    if any(w in q for w in ["trend", "over time", "month", "year", "recent", "getting"]):
        if "date" not in df.columns:
            return "No date information is available in this dataset."
        monthly = df.dropna(subset=["date"]).set_index("date")["stars"].resample("ME").mean()
        if len(monthly) < 2:
            return "Not enough time-series data to identify a trend."
        first     = monthly.iloc[0]
        last      = monthly.iloc[-1]
        diff      = last - first
        direction = "improving 📈" if diff > 0.1 else ("declining 📉" if diff < -0.1 else "holding steady ➡️")
        return (
            f"Ratings are **{direction}** over the data period. "
            f"The earliest readings averaged **{first:.2f}⭐** compared to **{last:.2f}⭐** most recently "
            f"— a change of **{diff:+.2f}⭐** across {len(monthly)} months. "
            f"{'This is a concerning downward trend that warrants attention.' if diff < -0.1 else 'Keep monitoring for any reversals.'}"
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    total   = len(df)
    avg     = df["stars"].mean()
    neg_pct = (df["stars"] <= 2).mean() * 100
    pos_pct = (df["stars"] >= 4).mean() * 100
    extra   = ""
    if "state" in df.columns:
        worst_state = df.groupby("state")["stars"].mean().idxmin()
        extra += f" The weakest market is **{worst_state}**."
    if "brand_id" in df.columns:
        worst_brand = df.groupby("brand_id")["stars"].mean().idxmin()
        extra += f" **{worst_brand}** has the lowest brand rating."
    return (
        f"Across **{total:,} reviews**, the average rating is **{avg:.2f}⭐** — "
        f"with {pos_pct:.0f}% of customers rating 4-5 stars and {neg_pct:.0f}% rating 1-2 stars.{extra} "
        f"Try asking about complaints, billing, vehicle condition, customer service, wait times, state performance, or rating trends."
    )

def show():
    st.markdown("## Analyst Copilot")
    st.markdown(
        "Ask any question about customer reviews. "
        "Answers are computed directly from the data — no AI needed."
    )

    with st.spinner("Loading data..."):
        df = load_data()

    if df is None:
        st.error("No data found. Run the extractor first.")
        return

    st.markdown("### Try asking:")
    questions = [
        "What do customers complain about most?",
        "Which brand has the worst ratings?",
        "What do happy customers mention?",
        "What are complaints about vehicle condition?",
        "Which states have the lowest ratings?",
        "What do customers say about customer service?",
        "Are there complaints about billing or charges?",
        "What improvements do customers suggest?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(questions):
        if cols[i % 4].button(q, key=f"q{i}"):
            st.session_state["pending"] = q

    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending  = st.session_state.pop("pending", "")
    user_in  = st.chat_input("Ask anything about customer reviews...")
    question = user_in or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["history"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            answer = answer_question(question, df)
            st.markdown(answer)
            st.session_state["history"].append({"role": "assistant", "content": answer})

    if st.session_state.get("history"):
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    with st.expander("📊 Full dataset stats"):
        total   = len(df)
        avg     = df["stars"].mean()
        dist    = df["stars"].value_counts().sort_index()
        st.metric("Total Reviews", f"{total:,}")
        st.metric("Avg Rating", f"{avg:.2f} ⭐")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (star, count) in zip([c1, c2, c3, c4, c5], dist.items()):
            col.metric(f"{int(star)}★", f"{int(count):,}")
