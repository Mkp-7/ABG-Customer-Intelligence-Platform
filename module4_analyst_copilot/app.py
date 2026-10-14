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
            return "No 1-2 star reviews found in the dataset."
        keywords = top_words(low)
        pct = len(low) / len(df) * 100
        return (
            f"**{len(low):,} negative reviews ({pct:.0f}% of total)**\n\n"
            f"Most frequent words in 1-2★ reviews:\n{keywords}\n\n"
            f"Sample review: *\"{low.iloc[0][:200]}...\"*"
        )

    # ── Worst state ───────────────────────────────────────────────────────────
    if "state" in q and any(w in q for w in ["low", "worst", "bottom", "bad", "poor"]):
        if "state" not in df.columns:
            return "No state data available in the dataset."
        s = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg").head(5)
        rows = "\n".join([f"  {st}: {r['avg']:.2f}⭐ ({int(r['count'])} reviews)" for st, r in s.iterrows()])
        return f"**5 lowest rated states:**\n{rows}"

    # ── Best state ────────────────────────────────────────────────────────────
    if "state" in q and any(w in q for w in ["best", "top", "high", "great"]):
        if "state" not in df.columns:
            return "No state data available in the dataset."
        s = df.groupby("state")["stars"].agg(avg="mean", count="count").sort_values("avg", ascending=False).head(5)
        rows = "\n".join([f"  {st}: {r['avg']:.2f}⭐ ({int(r['count'])} reviews)" for st, r in s.iterrows()])
        return f"**5 highest rated states:**\n{rows}"

    # ── Brand comparison ──────────────────────────────────────────────────────
    if any(w in q for w in ["brand", "avis", "budget", "zipcar"]):
        if "brand_id" not in df.columns:
            return "No brand data available in the dataset."
        b = df.groupby("brand_id")["stars"].agg(avg="mean", count="count").sort_values("avg")
        rows = "\n".join([f"  {br}: {r['avg']:.2f}⭐ ({int(r['count'])} reviews)" for br, r in b.iterrows()])
        worst = b.index[0]
        best  = b.index[-1]
        return f"**Brand ratings (lowest to highest):**\n{rows}\n\n⚠️ Worst: **{worst}** · ✅ Best: **{best}**"

    # ── Happy customers / positive ────────────────────────────────────────────
    if any(w in q for w in ["happy", "positive", "good", "praise", "like", "love", "mention"]):
        high = df[df["stars"] >= 4]["text"].dropna()
        if high.empty:
            return "No 4-5 star reviews found."
        keywords = top_words(high)
        pct = len(high) / len(df) * 100
        return (
            f"**{len(high):,} positive reviews ({pct:.0f}% of total)**\n\n"
            f"Most frequent words in 4-5★ reviews:\n{keywords}\n\n"
            f"Sample review: *\"{high.iloc[0][:200]}...\"*"
        )

    # ── Vehicle / car condition ───────────────────────────────────────────────
    if any(w in q for w in ["vehicle", "car", "condition", "clean", "dirty", "damage"]):
        mask = df["text"].str.lower().str.contains(
            "car|vehicle|clean|dirty|condition|damage|smell|old|scratch|dent", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically mentioning vehicle condition found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        words   = top_words(sub["text"].dropna())
        return (
            f"**{len(sub):,} reviews mention vehicle condition**\n\n"
            f"Avg rating: {avg:.2f}⭐ · {neg_pct:.0f}% negative\n\n"
            f"Top words: {words}"
        )

    # ── Billing / charges ─────────────────────────────────────────────────────
    if any(w in q for w in ["billing", "charge", "fee", "price", "cost", "money", "refund", "overcharge"]):
        mask = df["text"].str.lower().str.contains(
            "bill|charg|fee|price|cost|paid|overcharg|refund|hidden|extra", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews mentioning billing or charges found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"**{len(sub):,} reviews mention billing/charges**\n\n"
            f"Avg rating: {avg:.2f}⭐ · {neg_pct:.0f}% are negative\n\n"
            f"This is {'a major pain point' if neg_pct > 50 else 'a moderate concern'} "
            f"({neg_pct:.0f}% of billing-related reviews are 1-2★)."
        )

    # ── Customer service / staff ──────────────────────────────────────────────
    if any(w in q for w in ["service", "staff", "employee", "agent", "representative", "rude", "helpful"]):
        mask = df["text"].str.lower().str.contains(
            "service|staff|employee|agent|representative|rude|helpful|friendly|unprofessional", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically mentioning customer service found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        words   = top_words(sub["text"].dropna())
        return (
            f"**{len(sub):,} reviews mention customer service**\n\n"
            f"Avg rating: {avg:.2f}⭐ · {neg_pct:.0f}% negative\n\n"
            f"Top words: {words}"
        )

    # ── Wait time / line ──────────────────────────────────────────────────────
    if any(w in q for w in ["wait", "line", "queue", "slow", "long time", "hour"]):
        mask = df["text"].str.lower().str.contains(
            "wait|waiting|line|queue|slow|hours|long time|took forever", na=False)
        sub = df[mask]
        if sub.empty:
            return "No reviews specifically mentioning wait times found."
        avg     = sub["stars"].mean()
        neg_pct = (sub["stars"] <= 2).mean() * 100
        return (
            f"**{len(sub):,} reviews mention wait times / lines**\n\n"
            f"Avg rating: {avg:.2f}⭐ · {neg_pct:.0f}% negative\n\n"
            f"Wait times are {'a serious issue' if neg_pct > 60 else 'a moderate concern'}."
        )

    # ── Suggestions / improvements ────────────────────────────────────────────
    if any(w in q for w in ["suggest", "improve", "recommend", "wish", "better", "should"]):
        mask = df["text"].str.lower().str.contains(
            "should|wish|hope|suggest|improve|better|instead|need to|recommend|please", na=False)
        sub = df[mask]["text"].dropna()
        if sub.empty:
            return "No improvement suggestions found in reviews."
        samples = "\n\n".join([f"• *\"{t[:150]}...\"*" for t in sub.head(4)])
        return f"**{len(sub):,} reviews contain suggestions:**\n\n{samples}"

    # ── Rating / overall stats ────────────────────────────────────────────────
    if any(w in q for w in ["rating", "average", "overall", "score", "stars", "how many", "total"]):
        total   = len(df)
        avg     = df["stars"].mean()
        neg_pct = (df["stars"] <= 2).mean() * 100
        pos_pct = (df["stars"] >= 4).mean() * 100
        dist    = df["stars"].value_counts().sort_index()
        dist_t  = " · ".join([f"{int(k)}★: {int(v)}" for k, v in dist.items()])
        return (
            f"**Overall stats: {total:,} reviews**\n\n"
            f"Average rating: **{avg:.2f}⭐**\n"
            f"Positive (4-5★): {pos_pct:.0f}% · Negative (1-2★): {neg_pct:.0f}%\n\n"
            f"Distribution: {dist_t}"
        )

    # ── Trend over time ───────────────────────────────────────────────────────
    if any(w in q for w in ["trend", "over time", "month", "year", "recent", "getting"]):
        if "date" not in df.columns:
            return "No date data available."
        monthly = df.dropna(subset=["date"]).set_index("date")["stars"].resample("ME").mean()
        if len(monthly) < 2:
            return "Not enough data to show a trend."
        first = monthly.iloc[0]
        last  = monthly.iloc[-1]
        diff  = last - first
        direction = "improving 📈" if diff > 0.1 else ("declining 📉" if diff < -0.1 else "stable ➡️")
        return (
            f"**Rating trend: {direction}**\n\n"
            f"Earliest period avg: {first:.2f}⭐\n"
            f"Most recent period avg: {last:.2f}⭐\n"
            f"Change: {diff:+.2f}⭐ over {len(monthly)} months of data"
        )

    # ── Fallback — general stats ──────────────────────────────────────────────
    total   = len(df)
    avg     = df["stars"].mean()
    neg_pct = (df["stars"] <= 2).mean() * 100
    pos_pct = (df["stars"] >= 4).mean() * 100
    extra   = ""
    if "state" in df.columns:
        worst_state = df.groupby("state")["stars"].mean().idxmin()
        extra = f"\nWorst performing state: **{worst_state}**"
    if "brand_id" in df.columns:
        worst_brand = df.groupby("brand_id")["stars"].mean().idxmin()
        extra += f"\nLowest rated brand: **{worst_brand}**"
    return (
        f"**Dataset summary: {total:,} reviews · avg {avg:.2f}⭐**\n\n"
        f"Positive (4-5★): {pos_pct:.0f}% · Negative (1-2★): {neg_pct:.0f}%"
        f"{extra}\n\n"
        f"Try asking about: complaints, billing, vehicle condition, customer service, wait times, trends by state, or brand comparisons."
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
