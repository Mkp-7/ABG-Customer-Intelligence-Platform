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
sys.path.insert(0, MOD_DIR)

from config import REVIEWS_CSV
from voc_analyzer import get_groq_client, cluster_themes, detect_anomalies, write_exec_summary


@st.cache_data(show_spinner=False)
def load_data(brand_ids_key: str):
    """Cache key is comma-joined sorted brand_ids so cache busts on brand change."""
    if not os.path.exists(REVIEWS_CSV):
        return None
    df = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    brand_ids = [b for b in brand_ids_key.split(",") if b]
    if brand_ids and "brand_id" in df.columns:
        df = df[df["brand_id"].isin(brand_ids)]
    return df


def show():
    from config import BRANDS as _BRANDS
    brand_ids   = st.session_state.get("selected_brand_ids", [])
    brand_names = st.session_state.get("selected_brand_names", ["All Brands"])
    # Sort brand names by their order in config so label is always Avis, Budget, Zipcar
    _order = {b["name"]: i for i, b in enumerate(_BRANDS)}
    brand_names = sorted(brand_names, key=lambda n: _order.get(n, 99))
    brand_label = ", ".join(brand_names) if brand_names else "All Brands"
    cache_key   = ",".join(sorted(brand_ids))

    st.markdown(f"## Voice of Customer AI - {brand_label}")
    st.markdown("AI reads customer reviews and surfaces themes, anomalies, and executive summaries.")

    df = load_data(cache_key)

    if df is None or df.empty:
        st.error("No data found. Run the extractor first:\n"
                 "```bash\npython module1_voice_of_customer/01_extract_reviews.py\n```")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    sources = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []
    if sources:
        sel_sources = st.sidebar.multiselect("Source", options=sources, default=sources)
        if sel_sources:
            df = df[df["source"].isin(sel_sources)]

    states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    sel_states = st.sidebar.multiselect("States", options=states, default=states)

    valid_dates = df["date"].dropna()
    if len(valid_dates):
        min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    else:
        from datetime import date
        min_d = max_d = date.today()

    date_range  = st.sidebar.date_input("Date range", value=(min_d, max_d),
                                         min_value=min_d, max_value=max_d)
    star_filter = st.sidebar.multiselect("Star ratings", options=[1,2,3,4,5], default=[1,2,3,4,5])

    mask = pd.Series([True] * len(df), index=df.index)
    if sel_states and "state" in df.columns:
        # Keep reviews that match selected states OR have no state (App Store reviews)
        has_state = df["state"].fillna("").str.strip() != ""
        mask &= (~has_state) | df["state"].isin(sel_states)
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
    c1, c2, c3, c4, c5 = st.columns(5)
    loc_col = "place_name" if "place_name" in filtered.columns else (
              "business_id" if "business_id" in filtered.columns else None)
    c1.metric("Total Reviews", f"{len(filtered):,}")
    c2.metric("Avg Rating",    f"{filtered['stars'].mean():.2f} ⭐")
    c3.metric("Locations",     filtered[loc_col].nunique() if loc_col else "-")
    c4.metric("States",        filtered["state"].nunique() if "state" in filtered.columns else "-")
    c5.metric("Brands",        len(brand_ids) if brand_ids else (filtered["brand_id"].nunique() if "brand_id" in filtered.columns else "-"))

    # ── Brand + Source breakdown ───────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    if "brand_name" in filtered.columns and filtered["brand_name"].nunique() > 1:
        brc = filtered["brand_name"].value_counts().reset_index()
        brc.columns = ["Brand", "Count"]
        with col_l:
            st.markdown("**By Brand**")
            fig_b = px.pie(brc, names="Brand", values="Count",
                           color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"])
            fig_b.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_b, use_container_width=True)

    if "source" in filtered.columns and filtered["source"].nunique() > 1:
        src = filtered["source"].value_counts().reset_index()
        src.columns = ["Source", "Count"]
        with col_r:
            st.markdown("**By Source**")
            fig_s = px.pie(src, names="Source", values="Count",
                           color_discrete_sequence=["#6366f1", "#ec4899"])
            fig_s.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Rating distribution ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Rating Distribution")
    rc = filtered["stars"].value_counts().sort_index().reset_index()
    rc.columns = ["Stars", "Count"]
    fig = px.bar(rc, x="Stars", y="Count", color="Stars",
                 color_continuous_scale=["#E24B4A", "#EF9F27", "#FAC775", "#97C459", "#1D9E75"])
    fig.update_layout(showlegend=False, height=260, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Rating by brand side-by-side if multiple brands selected
    if "brand_name" in filtered.columns and filtered["brand_name"].nunique() > 1:
        brand_rc = (filtered.groupby(["brand_name", "stars"])
                    .size().reset_index(name="count"))
        fig_br = px.bar(brand_rc, x="stars", y="count", color="brand_name", barmode="group",
                        color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
                        labels={"stars": "Stars", "count": "Count", "brand_name": "Brand"})
        fig_br.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_br, use_container_width=True)

    # ── Rating over time ──────────────────────────────────────────────────────
    if "date" in filtered.columns and len(filtered) > 20:
        if "brand_name" in filtered.columns and filtered["brand_name"].nunique() > 1:
            monthly = (filtered.dropna(subset=["date"])
                       .groupby([pd.Grouper(key="date", freq="ME"), "brand_name"])["stars"]
                       .mean().reset_index())
            monthly.columns = ["Month", "Brand", "Avg Stars"]
            fig2 = px.line(monthly, x="Month", y="Avg Stars", color="Brand",
                           line_shape="spline", title="Average Rating Over Time by Brand",
                           color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"])
        else:
            monthly = (filtered.dropna(subset=["date"])
                       .set_index("date")["stars"].resample("ME").mean().reset_index())
            monthly.columns = ["Month", "Avg Stars"]
            fig2 = px.line(monthly, x="Month", y="Avg Stars", line_shape="spline",
                           title="Average Rating Over Time")
            fig2.update_traces(line_color="#185FA5", line_width=2)

        fig2.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig2, use_container_width=True)

    # ── AI Theme Analysis ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### AI Theme Analysis")
    st.markdown("AI reads a sample of reviews and identifies the top recurring themes.")
    n = min(60, len(filtered))
    sample = filtered["text"].dropna().sample(n, random_state=42).tolist() if n > 0 else []

    if st.button("Run AI Theme Analysis", type="primary"):
        try:
            client = get_groq_client()
        except ValueError as e:
            st.error(str(e)); return
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
            c = colors.get(t.get("sentiment", "mixed"), "#888")
            with cols[i % 2]:
                st.markdown(
                    f"""<div style="border:1px solid {c}33;border-left:4px solid {c};
                        border-radius:8px;padding:12px 14px;margin-bottom:12px;">
                        <div style="font-weight:600;font-size:15px;margin-bottom:4px;">{t.get('name','')}</div>
                        <div style="font-size:13px;color:#aaa;margin-bottom:6px;">{t.get('description','')}</div>
                        <div style="display:flex;gap:8px;align-items:center;">
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
        loc_col_a = "place_name" if "place_name" in filtered.columns else (
                    "business_id" if "business_id" in filtered.columns else None)
        if loc_col_a and "city" in filtered.columns and "business_id" in anomalies.columns:
            info = (filtered[[loc_col_a, "city", "state"]]
                    .drop_duplicates(loc_col_a)
                    .rename(columns={loc_col_a: "business_id"}))
            anomalies = anomalies.merge(info, on="business_id", how="left")
        display_cols = [c for c in ["business_id", "city", "state", "historical_avg",
                                    "recent_avg", "rating_drop", "recent_reviews"]
                        if c in anomalies.columns]
        shown = anomalies[display_cols].head(10).copy()
        for col in ["historical_avg", "recent_avg", "rating_drop"]:
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
                st.error(str(e)); return
            _vd   = filtered["date"].dropna()
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
                    brand_name=brand_label,
                )
            st.markdown(
                f"""<div style="background:#F1EFE8;border-radius:12px;padding:20px 24px;border:1px solid #D3D1C7;">
                    <div style="font-size:11px;font-weight:600;letter-spacing:0.08em;color:#888;text-transform:uppercase;margin-bottom:12px;">
                        Executive Summary · {brand_label}</div>
                    <div style="font-size:15px;line-height:1.8;color:#2C2C2A;">
                        {summary.replace(chr(10), '<br><br>')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.code(summary, language=None)

    # ── Raw reviews with full text wrap ───────────────────────────────────────
    st.markdown("---")
    if st.checkbox("Show raw reviews"):
        cols = [c for c in ["date", "stars", "brand_name", "source", "city", "state", "text"]
                if c in filtered.columns]
        raw = filtered[cols].sort_values("date", ascending=False).head(200).copy()
        raw["date"] = raw["date"].astype(str).str[:10]

        rows_html = ""
        for _, row in raw.iterrows():
            cells = ""
            for col in cols:
                val = str(row[col]) if pd.notna(row[col]) else ""
                style = ("white-space:pre-wrap;word-break:break-word;"
                         "min-width:320px;max-width:520px;") if col == "text" else "white-space:nowrap;"
                cells += (f'<td style="padding:6px 10px;border-bottom:1px solid #334155;'
                          f'vertical-align:top;font-size:12px;{style}">{val}</td>')
            rows_html += f"<tr>{cells}</tr>"

        headers = "".join(
            f'<th style="padding:8px 10px;text-align:left;background:#1e293b;color:#94a3b8;'
            f'font-size:11px;text-transform:uppercase;letter-spacing:0.06em;white-space:nowrap;">{c}</th>'
            for c in cols
        )
        st.markdown(
            f"""<div style="overflow-x:auto;border-radius:10px;border:1px solid #334155;">
            <table style="width:100%;border-collapse:collapse;background:#0f172a;color:#e2e8f0;">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows_html}</tbody>
            </table></div>""",
            unsafe_allow_html=True,
        )
