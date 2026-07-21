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
MOD_DIR  = os.path.join(BASE_DIR, "module1_voice_of_customer")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MOD_DIR)

from config import REVIEWS_CSV, GROQ_MODEL
from voc_analyzer import get_groq_client


@st.cache_data(show_spinner=False)
def build_context(brand_ids_key: str, brand_label: str):
    """brand_ids_key is comma-joined sorted brand_ids for cache invalidation."""
    if not os.path.exists(REVIEWS_CSV):
        return None, None

    df = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])

    brand_ids = [b for b in brand_ids_key.split(",") if b]
    if brand_ids and "brand_id" in df.columns:
        df = df[df["brand_id"].isin(brand_ids)]

    if df.empty:
        return None, None

    total = len(df)
    avg   = df["stars"].mean()

    valid_dates = df["date"].dropna()
    d_min = valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) else "N/A"
    d_max = valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) else "N/A"

    loc_col = "place_name" if "place_name" in df.columns else (
              "business_id" if "business_id" in df.columns else None)
    locs = df[loc_col].nunique() if loc_col else "-"

    # Rating distribution
    dist = df["stars"].value_counts().sort_index()
    dist_text = ", ".join([f"{int(k)} star: {int(v)} ({v/total*100:.1f}%)"
                           for k, v in dist.items()])

    # Per-brand summary
    brand_text = ""
    if "brand_name" in df.columns and df["brand_name"].nunique() > 1:
        for brand, grp in df.groupby("brand_name"):
            b_avg = grp["stars"].mean()
            b_neg = (grp["stars"] <= 2).mean() * 100
            b_pos = (grp["stars"] >= 4).mean() * 100
            brand_text += (f"  {brand}: {len(grp):,} reviews, avg {b_avg:.2f}⭐, "
                           f"{b_neg:.1f}% negative, {b_pos:.1f}% positive\n")
    else:
        brand_text = "  (single brand selected)\n"

    # State performance - only for reviews with state data
    df_loc = df[df["state"].fillna("").str.strip() != ""] if "state" in df.columns else df
    if not df_loc.empty and "state" in df_loc.columns:
        state_stats = (df_loc.groupby("state")["stars"]
                       .agg(avg="mean", count="count")
                       .sort_values("avg")
                       .reset_index())
        state_text = "\n".join([f"  {r['state']}: avg={r['avg']:.2f}, n={r['count']}"
                                 for _, r in state_stats.iterrows()])
    else:
        state_text = "  No state data available (App Store reviews have no location)"

    # Worst / best locations (Google Maps only)
    if loc_col and not df_loc.empty:
        store_agg = (df_loc.groupby(loc_col)
                     .agg(avg_stars=("stars", "mean"), n=("stars", "count"),
                          city=("city", "first"), state=("state", "first"),
                          brand=("brand_name", "first"))
                     .reset_index())
        store_agg = store_agg[store_agg["n"] >= 2]

        worst = store_agg.nsmallest(5, "avg_stars")
        worst_text = "\n".join([
            f"  {r['brand']} - {r['city']}, {r['state']}: {r['avg_stars']:.2f}⭐ ({r['n']} reviews)"
            for _, r in worst.iterrows()
        ])

        best = store_agg.nlargest(5, "avg_stars")
        best_text = "\n".join([
            f"  {r['brand']} - {r['city']}, {r['state']}: {r['avg_stars']:.2f}⭐ ({r['n']} reviews)"
            for _, r in best.iterrows()
        ])
    else:
        worst_text = best_text = "  No location data available"

    # Sample low-rating reviews
    low_df = df[df["stars"] <= 2]["text"].dropna()
    low_sample = "\n".join([
        f"- [{row.get('brand_name','') if 'brand_name' in df.columns else ''}] {str(t)[:200]}"
        for t, row in zip(
            low_df.sample(min(10, len(low_df)), random_state=42),
            df.loc[low_df.sample(min(10, len(low_df)), random_state=42).index].itertuples()
        )
    ]) if len(low_df) else "No low-rating reviews found."

    context = f"""CUSTOMER INTELLIGENCE DATA - {brand_label}
================================
Total reviews: {total:,}
Date range: {d_min} to {d_max}
Average rating: {avg:.2f} / 5.0
Locations covered: {locs}

RATING DISTRIBUTION:
{dist_text}

PER-BRAND BREAKDOWN:
{brand_text}
PERFORMANCE BY STATE (Google Maps reviews):
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
    from config import BRANDS as _BRANDS
    brand_ids   = st.session_state.get("selected_brand_ids", [])
    brand_names = st.session_state.get("selected_brand_names", ["All Brands"])
    _order      = {b["name"]: i for i, b in enumerate(_BRANDS)}
    brand_names = sorted(brand_names, key=lambda n: _order.get(n, 99))
    brand_label = ", ".join(brand_names) if brand_names else "All Brands"
    cache_key   = ",".join(sorted(brand_ids))

    st.markdown(f"## 🤖 Analyst Copilot - {brand_label}")
    st.markdown(
        "Ask any question about store performance in plain English. "
        "The AI has full context of the dataset and responds with real numbers."
    )

    with st.spinner("Preparing data context..."):
        context, df = build_context(cache_key, brand_label)

    if context is None:
        st.error("No data found. Run the extractor first:\n"
                 "```bash\npython module1_voice_of_customer/01_extract_reviews.py\n```")
        return

    try:
        client = get_groq_client()
    except ValueError as e:
        st.error(str(e))
        return

    st.markdown("### Try asking:")
    questions = [
        "Which brand has the lowest ratings?",
        "What do customers complain about most?",
        "Which states have the worst performance?",
        "Compare Avis vs Budget ratings",
        "How many locations are below 3 stars?",
        "What do happy customers mention most?",
        "Which cities have the best performance?",
        "Is there a trend in ratings over time?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(questions):
        if cols[i % 4].button(q, key=f"q{i}"):
            st.session_state["copilot_pending"] = q

    st.markdown("---")

    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []

    for msg in st.session_state["copilot_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending  = st.session_state.pop("copilot_pending", "")
    user_in  = st.chat_input("Ask anything about store performance...")
    question = user_in or pending

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["copilot_history"].append({"role": "user", "content": question})

        system = f"""You are an expert retail data analyst with access to the following data:

{context}

Answer using ONLY the data above. Be direct and specific. Use numbers.
When multiple brands are present, compare them. Under 150 words unless asked for detail."""

        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state["copilot_history"][-6:]]

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

        st.session_state["copilot_history"].append({"role": "assistant", "content": answer})

    if st.session_state.get("copilot_history"):
        if st.button("Clear conversation"):
            st.session_state["copilot_history"] = []
            st.rerun()

    with st.expander("View data context the AI uses"):
        st.code(context, language="text")
