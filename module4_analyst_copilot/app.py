    """
Module 4 — Analyst Copilot with RAG (HuggingFace + GitHub CSV)
"""
import os, sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import REVIEWS_CSV, BUSINESSES_CSV, PLATFORM_TITLE

HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def get_token():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN", "")
        except Exception:
            pass
    if not token:
        raise ValueError("HF_TOKEN not set in Streamlit Secrets.")
    return InferenceClient(token=token)


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(REVIEWS_CSV):
        return None
    df = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
    if os.path.exists(BUSINESSES_CSV) and "business_id" in df.columns:
        biz = pd.read_csv(BUSINESSES_CSV)
        if "business_id" in biz.columns:
            cols = [c for c in ["business_id","name","city","state"] if c in biz.columns]
            df = df.merge(biz[cols], on="business_id", how="left")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars", "text"])
    return df


def retrieve_relevant(df: pd.DataFrame, question: str, n: int = 150) -> pd.DataFrame:
    """Keyword-based retrieval — finds reviews most relevant to the question."""
    q_lower = question.lower()
    keywords = [w for w in q_lower.split() if len(w) > 3
                and w not in {"what","which","where","when","how","are","the","for","does","have","been","that","with","from"}]

    if not keywords:
        return df.sample(min(n, len(df)), random_state=42)

    scores = pd.Series(0, index=df.index)
    for kw in keywords:
        scores += df["text"].str.lower().str.contains(kw, na=False).astype(int)

    # Mix: top scored + random low-scorers for diversity
    top = df[scores > 0].nlargest(min(n, len(df)), scores.reindex(df[scores > 0].index))
    if len(top) < n:
        remaining = df[scores == 0].sample(min(n - len(top), len(df[scores == 0])), random_state=42)
        top = pd.concat([top, remaining])

    return top.head(n)


def build_stats(df: pd.DataFrame) -> str:
    total  = len(df)
    avg    = df["stars"].mean()
    dist   = df["stars"].value_counts().sort_index()
    dist_t = ", ".join([f"{int(k)}★: {int(v)} ({v/total*100:.0f}%)" for k,v in dist.items()])

    lines = [
        f"Total reviews: {total:,}",
        f"Average rating: {avg:.2f}/5.0",
        f"Rating distribution: {dist_t}",
    ]

    if "state" in df.columns:
        state_avg = df.groupby("state")["stars"].mean().sort_values()
        lines.append(f"Worst state: {state_avg.index[0]} ({state_avg.iloc[0]:.2f}★)")
        lines.append(f"Best state: {state_avg.index[-1]} ({state_avg.iloc[-1]:.2f}★)")

    if "brand_id" in df.columns:
        brand_avg = df.groupby("brand_id")["stars"].mean().sort_values()
        lines.append("Brand ratings: " + ", ".join([f"{b}: {v:.2f}★" for b,v in brand_avg.items()]))

    return "\n".join(lines)


def ask_hf(question: str, reviews_df: pd.DataFrame, stats: str, client) -> str:
    relevant = retrieve_relevant(reviews_df, question, n=150)
    sample   = "\n".join([
        f"[{i+1}] ({int(r['stars'])}★ {str(r.get('date',''))[:7]}) {str(r['text'])[:250]}"
        for i, (_, r) in enumerate(relevant.iterrows())
    ])

    prompt = f"""You are a retail analytics expert analyzing customer reviews for a car rental company (Avis, Budget, Zipcar).

DATASET STATS:
{stats}

RELEVANT REVIEWS ({len(relevant)} retrieved for this question):
{sample}

QUESTION: {question}

Answer using the data above. Be specific, use numbers, keep it under 200 words."""

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def show():
    st.markdown("## Analyst Copilot")
    st.markdown("Ask any question about customer reviews in plain English. AI retrieves the most relevant reviews and answers with real data.")

    with st.spinner("Loading data..."):
        df = load_data()

    if df is None:
        st.error("No data found. Run the extractor first.")
        return

    try:
        token = get_token()
    except ValueError as e:
        st.error(str(e))
        return

    stats = build_stats(df)

    st.markdown("### Try asking:")
    questions = [
        "What do customers complain about most?",
        "Which brand has the worst ratings?",
        "What do happy customers mention?",
        "What are common complaints about vehicle condition?",
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
            with st.spinner("Retrieving relevant reviews and analyzing..."):
                try:
                    answer = ask_hf(question, df, stats, token)
                    st.markdown(answer)
                    st.session_state["history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.get("history"):
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    with st.expander("Dataset stats the AI uses"):
        st.code(stats, language="text")
