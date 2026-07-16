"""
Module 4 - Analyst Copilot
"""

import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import REVIEWS_CSV, BUSINESSES_CSV, PLATFORM_TITLE, GROQ_MODEL
from module1_voice_of_customer.voc_analyzer import get_groq_client


@st.cache_data(show_spinner=False)
def build_context(brand_id: str = None, brand_name: str = "All Brands"):
    if not os.path.exists(REVIEWS_CSV):
        return None, None

    reviews = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])

    if os.path.exists(BUSINESSES_CSV):
        biz = pd.read_csv(BUSINESSES_CSV)
        df  = reviews.merge(biz[["business_id","name","city","state"]], on="business_id", how="left")
    else:
        df = reviews

    if brand_id and "brand_id" in df.columns:
        df = df[df["brand_id"] == brand_id]

    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])

    total   = len(df)
    avg     = df["stars"].mean()
    if "date" in df.columns:
        valid_dates = df["date"].dropna()
        d_min = valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) else "N/A"
        d_max = valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) else "N/A"
    else:
        d_min = d_max = "N/A"
    loc_col = "business_id" if "business_id" in df.columns else "place_name"
    locs    = df[loc_col].nunique() if loc_col in df.columns else "-"

    dist = df["stars"].value_counts().sort_index()
    dist_text = ", ".join([f"{int(k)} star: {int(v)} ({v/total*100:.1f}%)" for k,v in dist.items()])

    state_stats = df.groupby("state")["stars"].agg(avg="mean",count="count").sort_values("avg").reset_index()
    state_text  = "\n".join([f"  {r['state']}: avg={r['avg']:.2f}, n={r['count']}" for _,r in state_stats.iterrows()])

    store_agg = df.groupby("business_id").agg(
        avg_stars=("stars","mean"), n=("stars","count"),
        city=("city","first"), state=("state","first")
    ).reset_index()

    worst = store_agg.nsmallest(5,"avg_stars")[["city","state","avg_stars","n"]]
    worst_text = "\n".join([f"  {r['city']}, {r['state']}: {r['avg_stars']:.2f}⭐ ({r['n']} reviews)" for _,r in worst.iterrows()])

    best = store_agg.nlargest(5,"avg_stars")[["city","state","avg_stars","n"]]
    best_text = "\n".join([f"  {r['city']}, {r['state']}: {r['avg_stars']:.2f}⭐ ({r['n']} reviews)" for _,r in best.iterrows()])

    low_reviews = df[df["stars"]<=2]["text"].dropna().sample(min(10,len(df[df["stars"]<=2])),random_state=42).tolist()
    low_sample  = "\n".join([f"- {r[:200]}" for r in low_reviews])

    context = f"""STORE PERFORMANCE DATA SUMMARY - {brand_name}
================================
Total reviews: {total:,}
Date range: {d_min} to {d_max}
Average rating: {avg:.2f} / 5.0
Locations covered: {locs}

RATING DISTRIBUTION:
{dist_text}

PERFORMANCE BY STATE:
{state_text}

WORST 5 LOCATIONS:
{worst_text}

BEST 5 LOCATIONS:
{best_text}

SAMPLE LOW-RATING REVIEWS (1-2 stars):
{low_sample}
"""
    return context, df


def show():
    brand_id   = st.session_state.get("selected_brand_id")
    brand_name = st.session_state.get("selected_brand_name", "All Brands")

    st.markdown(f"## Analyst Copilot - {brand_name}")
    st.markdown(
        "Ask any question about store performance in plain English. "
        "The AI has full context of the dataset and responds with real numbers."
    )

    with st.spinner("Preparing data context..."):
        context, df = build_context(brand_id, brand_name)

    if context is None:
        st.error("No data found. Run the extractor first:\n```bash\npython module1_voice_of_customer/01_extract_reviews.py\n```")
        return

    try:
        client = get_groq_client()
    except ValueError as e:
        st.error(str(e))
        return

    st.markdown("### Try asking:")
    questions = [
        "Which states have the lowest ratings?",
        "What do customers complain about most?",
        "How many locations are below 3 stars?",
        "Which cities have the best performance?",
        "What percentage of reviews are 1 or 2 stars?",
        "What do happy customers mention most?",
        "Which state has the most reviews?",
        "Is there a trend in ratings over time?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(questions):
        if cols[i%4].button(q, key=f"q{i}"):
            st.session_state["pending"] = q

    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending  = st.session_state.pop("pending", "")
    user_in  = st.chat_input("Ask anything about store performance...")
    question = user_in or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["history"].append({"role":"user","content":question})

        system = f"""You are an expert retail data analyst with access to the following data:

{context}

Answer using ONLY the data above. Be direct and specific. Use numbers. Under 150 words unless asked for detail."""

        msgs = [{"role":"system","content":system}]
        msgs += [{"role":m["role"],"content":m["content"]} for m in st.session_state["history"][-6:]]

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=msgs,
                    temperature=0.3,
                    max_tokens=500,
                )
                answer = resp.choices[0].message.content.strip()
                st.markdown(answer)

        st.session_state["history"].append({"role":"assistant","content":answer})

    if st.session_state.get("history"):
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    with st.expander("View data context the AI uses"):
        st.code(context, language="text")
