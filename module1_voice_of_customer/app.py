"""
Module 1 - Voice of Customer AI
"""

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, MOD_DIR)   # ← fixes ImportError on Streamlit Cloud

from config import REVIEWS_CSV, PLATFORM_TITLE
from voc_analyzer import (       # ← direct import, no package prefix needed
    get_groq_client,
    cluster_themes,
    detect_anomalies,
    write_exec_summary,
)


@st.cache_data(show_spinner=False)
def load_data(brand_id: str = None):
    if not os.path.exists(REVIEWS_CSV):
        return None
    df = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    # Filter by brand - works for Avis / Budget / Zipcar
    if brand_id and "brand_id" in df.columns:
        df = df[df["brand_id"] == brand_id]
    return df


def show():
    brand_id   = st.session_state.get("selected_brand_id")
    brand_name = st.session_state.get("selected_brand_name", "All Brands")

    st.markdown(f"## Voice of Customer AI - {brand_name}")
    st.markdown("AI reads customer reviews and surfaces themes, anomalies, and executive summaries.")

    df = load_data(brand_id)

    if df is None or df.empty:
        st.error(
            "No data found for this brand. Run the extractor first:\n"
            "```bash\npython module1_voice_of_customer/01_extract_reviews.py\n```"
        )
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    sources = sorted(df["source"].dropna().unique()) if "source" in df.columns else []
    if sources:
        sel_sources = st.sidebar.multiselect("Source", options=sources, default=sources)
        df = df[df["source"].isin(sel_sources)] if sel_sources else df

    states = sorted(df["state"].dropna().unique()) if "state" in df.columns else []
    selected_states = st.sidebar.multiselect("States", options=states, default=states)

    valid_dates = df["date"].dropna()
    if len(valid_dates):
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
    else:
        from datetime import date
        min_date = max_date = date.today()

    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )
    star_filter = st.sidebar.multiselect("Star ratings", options=[1,2,3,4,5], default=[1,2,3,4,5])

    mask = pd.Series([True] * len(df), index=df.index)
    if selected_states and "state" in df.columns:
        mask &= df["state"].isin(selected_states)
    if len(date_range) == 2 and "date" in df.columns:
        mask &= (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    if star_filter:
        mask &= df["stars"].isin(star_filter)
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No reviews match the selected filters.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    loc_col = "place_name" if "place_name" in filtered.columns else (
              "business_id" if "business_id" in filtered.columns else None)
    c1.metric("Total Reviews", f"{len(filtered):,}")
    c2.metric("Avg Rating",    f"{filtered['stars'].mean():.2f} ⭐")
    c3.metric("Locations",     filtered[loc_col].nunique() if loc_col else "-")
    c4.metric("States",        filtered["state"].nunique() if "state" in filtered.columns else "-")

    # ── Source breakdown ──────────────────────────────────────────────────────
    if "source" in filtered.columns and filtered["source"].nunique() > 1:
        src_counts = filtered["source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        src_fig = px.pie(src_counts, names="Source", values="Count",
                         color_discrete_sequence=["#3b82f6","#10b981","#f59e0b"])
        src_fig.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor="rgba(0,0,0,0)")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.markdown("**By Source**")
            st.plotly_chart(src_fig, use_container_width=True)
    else:
        col_b = st.container()

    # ── Rating distribution ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Rating Distribution")
    rc = filtered["stars"].value_counts().sort_index().reset_index()
    rc.columns = ["Stars", "Count"]
    fig = px.bar(rc, x="Stars", y="Count", color="Stars",
                 color_continuous_scale=["#E24B4A","#EF9F27","#FAC775","#97C459","#1D9E75"])
    fig.update_layout(showlegend=False, height=260, margin=dict(l=0,r=0,t=10,b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Rating over time ──────────────────────────────────────────────────────
    if "date" in filtered.columns and len(filtered) > 50:
        monthly = filtered.dropna(subset=["date"]).set_index("date")["stars"].resample("ME").mean().reset_index()
        monthly.columns = ["Month", "Avg Stars"]
        fig2 = px.line(monthly, x="Month", y="Avg Stars", line_shape="spline",
                       title="Average Rating Over Time")
        fig2.update_traces(line_color="#185FA5", line_width=2)
        fig2.update_layout(height=240, margin=dict(l=0,r=0,t=40,b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[1,5]))
        st.plotly_chart(fig2, use_container_width=True)

    # ── AI Theme Analysis ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### AI Theme Analysis")
    st.markdown("AI reads a sample of reviews and identifies the top recurring themes.")

    n = min(60, len(filtered))
    sample = filtered["text"].dropna().sample(n, random_state=42).tolist() if len(filtered) >= 1 else []

    if st.button("Run AI Theme Analysis", type="primary"):
        try:
            client = get_groq_client()
        except ValueError as e:
            st.error(str(e))
            return
        with st.spinner(f"Analyzing {n} reviews with AI..."):
            result = cluster_themes(sample, client)
        themes = result.get("themes", [])
        if not themes:
            st.warning("Could not extract themes. Check your Groq API key.")
        else:
            st.session_state["themes"] = themes

    if "themes" in st.session_state:
        colors = {"positive": "#1D9E75", "negative": "#E24B4A", "mixed": "#BA7517"}
        cols = st.columns(2)
        for i, t in enumerate(st.session_state["themes"]):
            c = colors.get(t.get("sentiment","mixed"), "#888")
            with cols[i % 2]:
                st.markdown(
                    f"""<div style="border:1px solid {c}33; border-left:4px solid {c};
                        border-radius:8px; padding:12px 14px; margin-bottom:12px;">
                        <div style="font-weight:600; font-size:15px; margin-bottom:4px;">{t.get('name','')}</div>
                        <div style="font-size:13px; color:#666; margin-bottom:6px;">{t.get('description','')}</div>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <span style="background:{c}22;color:{c};font-size:11px;padding:2px 8px;border-radius:4px;font-weight:500;">{t.get('sentiment','')}</span>
                            <span style="font-size:12px;color:#888;">~{t.get('percent',0)}% of reviews</span>
                        </div>
                        <div style="font-size:12px;color:#888;margin-top:6px;font-style:italic;">"{t.get('example_quote','')}"</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Anomaly detection ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Locations Needing Attention")
    st.markdown("Locations where recent ratings dropped vs their historical average.")

    anomalies = detect_anomalies(filtered)

    if anomalies.empty:
        st.success("No significant rating drops detected in the current filter period.")
    else:
        # Enrich with city/state if available
        loc_col_a = "place_name" if "place_name" in filtered.columns else (
                    "business_id" if "business_id" in filtered.columns else None)
        if loc_col_a and "city" in filtered.columns and "business_id" in anomalies.columns:
            info = (filtered[[loc_col_a, "city", "state"]]
                    .drop_duplicates(loc_col_a)
                    .rename(columns={loc_col_a: "business_id"}))
            anomalies = anomalies.merge(info, on="business_id", how="left")

        display_cols = [c for c in ["business_id","city","state","historical_avg",
                                     "recent_avg","rating_drop","recent_reviews"]
                        if c in anomalies.columns]
        shown = anomalies[display_cols].head(10).copy()
        for col in ["historical_avg","recent_avg","rating_drop"]:
            if col in shown.columns:
                shown[col] = shown[col].round(2)
        st.dataframe(shown, column_config={
            "rating_drop":    st.column_config.ProgressColumn("Rating Drop", min_value=0, max_value=2, format="%.2f ⭐"),
            "historical_avg": st.column_config.NumberColumn("Historical Avg ⭐", format="%.2f"),
            "recent_avg":     st.column_config.NumberColumn("Recent Avg ⭐", format="%.2f"),
        }, use_container_width=True, hide_index=True)

    # ── Executive Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Executive Summary")
    st.markdown("One click generates a polished summary ready for leadership.")

    if st.button("Generate Executive Summary", type="primary"):
        if "themes" not in st.session_state:
            st.warning("Run AI Theme Analysis first.")
        else:
            try:
                client = get_groq_client()
            except ValueError as e:
                st.error(str(e))
                return
            _vd = filtered["date"].dropna()
            d_min = _vd.min().strftime("%b %d, %Y") if len(_vd) else "N/A"
            d_max = _vd.max().strftime("%b %d, %Y") if len(_vd) else "N/A"
            with st.spinner("Writing summary..."):
                summary = write_exec_summary(
                    themes=st.session_state["themes"],
                    anomaly_stores=anomalies if not anomalies.empty else pd.DataFrame(),
                    total_reviews=len(filtered),
                    avg_rating=filtered["stars"].mean(),
                    date_range=f"{d_min} – {d_max}",
                    client=client,
                    brand_name=brand_name,
                )
            st.markdown(
                f"""<div style="background:#F1EFE8;border-radius:12px;padding:20px 24px;border:1px solid #D3D1C7;">
                    <div style="font-size:11px;font-weight:600;letter-spacing:0.08em;color:#888;text-transform:uppercase;margin-bottom:12px;">Executive Summary · {brand_name} · AI Generated</div>
                    <div style="font-size:15px;line-height:1.8;color:#2C2C2A;">{summary.replace(chr(10),'<br><br>')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.code(summary, language=None)

    # ── Raw data ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.checkbox("Show raw reviews"):
        cols = [c for c in ["date","stars","source","city","state","text"] if c in filtered.columns]
        st.dataframe(filtered[cols].sort_values("date", ascending=False).head(200),
                     use_container_width=True, hide_index=True)
