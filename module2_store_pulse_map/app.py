"""
Module 2 - Store Pulse Map
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import REVIEWS_CSV, BUSINESSES_CSV, PEER_GROUP_COLUMN, SIGNIFICANT_DELTA_STARS


def add_jitter(series: pd.Series, amount: float = 0.018) -> pd.Series:
    np.random.seed(42)
    return series + np.random.uniform(-amount, amount, size=len(series))


def load_and_process(brand_id: str = None):
    if not os.path.exists(BUSINESSES_CSV):
        return None, None

    biz     = pd.read_csv(BUSINESSES_CSV)
    reviews = None

    if brand_id and "brand_id" in biz.columns:
        biz = biz[biz["brand_id"] == brand_id]

    if os.path.exists(REVIEWS_CSV):
        reviews = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
        reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")
        if brand_id and "brand_id" in reviews.columns:
            reviews = reviews[reviews["brand_id"] == brand_id]

        # Aggregate reviews by place_name (new schema) or business_id (old schema)
        loc_col = "business_id" if "business_id" in reviews.columns else "place_name"
        agg = (reviews.groupby(loc_col)["stars"]
               .agg(avg_rating="mean", review_count="count")
               .reset_index()
               .rename(columns={loc_col: "business_id"}))

        if "review_count" in biz.columns:
            biz = biz.drop(columns=["review_count"])

        biz = biz.merge(agg, on="business_id", how="left")
        biz["avg_rating"]   = biz["avg_rating"].fillna(biz["stars"])
        biz["review_count"] = biz["review_count"].fillna(0).astype(int)
    else:
        biz["avg_rating"]   = biz["stars"]
        biz["review_count"] = 0

    biz = biz.dropna(subset=["latitude", "longitude", "avg_rating"])
    biz["avg_rating"] = pd.to_numeric(biz["avg_rating"], errors="coerce")
    biz = biz.dropna(subset=["avg_rating"])

    peer_col = PEER_GROUP_COLUMN if PEER_GROUP_COLUMN in biz.columns else "state"
    biz["peer_avg"] = biz.groupby(peer_col)["avg_rating"].transform("mean")
    biz["vs_peer"]  = (biz["avg_rating"] - biz["peer_avg"]).round(2)

    delta = SIGNIFICANT_DELTA_STARS

    def status(d):
        if d >= delta:  return "Above Peer"
        if d <= -delta: return "Below Peer"
        return "On Par"

    biz["status"] = biz["vs_peer"].apply(status)

    biz["lat_display"] = add_jitter(biz["latitude"])
    biz["lon_display"] = add_jitter(biz["longitude"])

    # Fix 1: use "name" fallback instead of "business_id"
    biz["label"] = (biz["name"].astype(str)
                    + "<br>" + biz.get("address", pd.Series("", index=biz.index)).astype(str)
                    + "<br>" + biz.get("city", pd.Series("", index=biz.index)).astype(str)
                    + ", " + biz.get("state", pd.Series("", index=biz.index)).astype(str))

    biz["short_label"] = (biz.get("city", pd.Series("", index=biz.index)).astype(str)
                          + ", " + biz.get("state", pd.Series("", index=biz.index)).astype(str)
                          + "  " + biz["avg_rating"].round(1).astype(str) + "⭐")

    return biz, reviews


def show():
    brand_id   = st.session_state.get("selected_brand_id")
    brand_name = st.session_state.get("selected_brand_name", "All Brands")

    st.markdown(f"## 🗺️ Store Pulse Map - {brand_name}")
    st.markdown(
        "Every location benchmarked against its **state peer group**. "
        "🔴 Below peer · 🟡 On par · 🟢 Above peer - "
        "pins are slightly spread so overlapping stores are visible individually."
    )

    biz, reviews = load_and_process(brand_id)

    if biz is None or len(biz) == 0:
        st.error("No location data found. Run:\n"
                 "```\npython module1_voice_of_customer/01_extract_reviews.py\n```")
        return

    st.sidebar.markdown("### 🗺️ Map Filters")
    states = sorted(biz["state"].dropna().unique()) if "state" in biz.columns else []
    sel_states = st.sidebar.multiselect("States", options=states, default=states)

    sel_status = st.sidebar.multiselect(
        "Status",
        options=["Above Peer", "On Par", "Below Peer"],
        default=["Above Peer", "On Par", "Below Peer"],
    )
    min_rev = st.sidebar.slider("Min reviews per location", 1, 30, 1)

    view_mode = st.sidebar.radio(
        "Map view",
        options=["📍 Individual pins (jittered)", "🔵 Cluster mode"],
        index=0,
    )

    mask = biz["status"].isin(sel_status) & (biz["review_count"] >= min_rev)
    if sel_states:
        mask &= biz["state"].isin(sel_states)
    filtered = biz[mask].copy()

    if filtered.empty:
        st.warning("No locations match filters. Try lowering the min reviews slider.")
        return

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Locations",     len(filtered))
    c2.metric("Avg Rating",    f"{filtered['avg_rating'].mean():.2f} ⭐")
    c3.metric("Total Reviews", f"{int(filtered['review_count'].sum()):,}")
    c4.metric("States",        filtered["state"].nunique() if "state" in filtered.columns else "-")
    c5.metric("🔴 Below Peer", int((filtered["status"]=="Below Peer").sum()))
    c6.metric("🟢 Above Peer", int((filtered["status"]=="Above Peer").sum()))

    st.markdown("---")

    color_map = {"Above Peer":"#1D9E75","On Par":"#F59E0B","Below Peer":"#E24B4A"}

    if "Cluster" in view_mode:
        fig = go.Figure()
        for status_val, color in color_map.items():
            sub = filtered[filtered["status"] == status_val]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=sub["lat_display"],
                lon=sub["lon_display"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=14, color=color, opacity=0.85),
                cluster=dict(enabled=True, color=color, size=20, step=3),
                text=sub["label"],
                customdata=np.stack([
                    sub["avg_rating"].round(2),
                    sub["peer_avg"].round(2),
                    sub["vs_peer"],
                    sub["review_count"],
                    sub["city"].fillna(""),        # Fix 2: direct column access
                    sub["state"].fillna(""),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[4]}, %{customdata[5]}</b><br>"
                    "Rating: %{customdata[0]}⭐<br>"
                    "State avg: %{customdata[1]}⭐<br>"
                    "vs Peers: %{customdata[2]:+.2f}⭐<br>"
                    "Reviews: %{customdata[3]}<extra></extra>"
                ),
                name=status_val,
            ))
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3.5,
            mapbox_center={"lat":37.5,"lon":-96},
            height=560,
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(title="vs State Peers", bgcolor="rgba(255,255,255,0.9)"),
            dragmode="zoom",
        )

    else:
        filtered["marker_size"] = np.clip(
            8 + (filtered["review_count"] / filtered["review_count"].max()) * 14,
            8, 22
        )
        fig = go.Figure()
        for status_val, color in color_map.items():
            sub = filtered[filtered["status"] == status_val]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=sub["lat_display"],
                lon=sub["lon_display"],
                mode="markers+text",
                marker=go.scattermapbox.Marker(size=sub["marker_size"], color=color, opacity=0.88),
                text=sub["avg_rating"].round(1).astype(str) + "⭐",
                textposition="top right",
                textfont=dict(size=9, color="#333"),
                customdata=np.stack([
                    sub["avg_rating"].round(2),
                    sub["peer_avg"].round(2),
                    sub["vs_peer"],
                    sub["review_count"],
                    sub["city"].fillna(""),        # Fix 2: direct column access
                    sub["state"].fillna(""),
                    sub["address"].fillna(""),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[4]}, %{customdata[5]}</b><br>"
                    "%{customdata[6]}<br>"
                    "──────────────<br>"
                    "⭐ Rating:      <b>%{customdata[0]}</b><br>"
                    "📊 State avg:   %{customdata[1]}<br>"
                    "📈 vs Peers:    <b>%{customdata[2]:+.2f}</b><br>"
                    "💬 Reviews:     %{customdata[3]}<extra></extra>"
                ),
                name=status_val,
            ))
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3.8,
            mapbox_center={"lat":37.5,"lon":-96},
            height=560,
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(title="vs State Peers", bgcolor="rgba(255,255,255,0.9)"),
            dragmode="zoom",
        )

    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {"format": "png", "filename": "store_pulse_map"},
    })
    st.caption("💡 Tip: Pins are slightly spread so overlapping stores are individually visible. "
               "Hover for full address and rating details. Switch to Cluster mode in the sidebar to see density.")

    st.markdown("---")
    st.markdown("### 🔴 Locations Needing Most Attention")
    st.caption("Stores furthest below their state peer average - priority Field Leader calls.")

    disp_cols = [c for c in ["short_label","state","avg_rating","peer_avg","vs_peer","review_count"] if c in filtered.columns]
    bottom = (filtered[filtered["status"]=="Below Peer"]
              .sort_values("vs_peer")[disp_cols].head(15).copy())

    if bottom.empty:
        st.success("✅ No locations significantly below their state peer group.")
    else:
        for col in ["avg_rating","peer_avg","vs_peer"]:
            if col in bottom.columns:
                bottom[col] = bottom[col].round(2)
        st.dataframe(bottom, column_config={
            "short_label":  st.column_config.TextColumn("Location"),
            "avg_rating":   st.column_config.NumberColumn("Rating ⭐", format="%.2f"),
            "peer_avg":     st.column_config.NumberColumn("State Avg ⭐", format="%.2f"),
            "vs_peer":      st.column_config.ProgressColumn("Gap vs Peers", min_value=-2, max_value=0, format="%.2f ⭐"),
            "review_count": st.column_config.NumberColumn("Reviews"),
        }, use_container_width=True, hide_index=True)

    st.markdown("### 🟢 Top Performing Locations")
    st.caption("Best-practice targets - stores significantly outperforming their state peers.")

    top = (filtered[filtered["status"]=="Above Peer"]
           .sort_values("vs_peer", ascending=False)[disp_cols].head(15).copy())

    if top.empty:
        st.info("No locations significantly above peer group with current filters.")
    else:
        for col in ["avg_rating","peer_avg","vs_peer"]:
            if col in top.columns:
                top[col] = top[col].round(2)
        st.dataframe(top, column_config={
            "short_label":  st.column_config.TextColumn("Location"),
            "avg_rating":   st.column_config.NumberColumn("Rating ⭐", format="%.2f"),
            "peer_avg":     st.column_config.NumberColumn("State Avg ⭐", format="%.2f"),
            "vs_peer":      st.column_config.ProgressColumn("Gap vs Peers", min_value=0, max_value=2, format="+%.2f ⭐"),
            "review_count": st.column_config.NumberColumn("Reviews"),
        }, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Average Rating by State")

    if "state" in filtered.columns:
        sa = (filtered.groupby("state")
              .agg(avg_rating=("avg_rating","mean"), locations=("name","count"))  # Fix 3: use "name"
              .sort_values("avg_rating", ascending=False).reset_index())
        sa["avg_rating"] = sa["avg_rating"].round(2)
        chain_avg = filtered["avg_rating"].mean()

        fig2 = px.bar(
            sa, x="state", y="avg_rating",
            color="avg_rating",
            color_continuous_scale=["#E24B4A","#F59E0B","#1D9E75"],
            range_color=[2.0, 4.5],
            text="avg_rating",
            hover_data={"locations":True},
            labels={"state":"State","avg_rating":"Avg Rating"},
        )
        fig2.add_hline(y=chain_avg, line_dash="dot", line_color="#60a5fa",
                       annotation_text=f"Chain avg: {chain_avg:.2f} ⭐",
                       annotation_position="top right")
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(
            height=360,
            margin=dict(l=0,r=0,t=20,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rating over time ──────────────────────────────────────────────────────
    if reviews is not None:
        st.markdown("### 📈 Rating Trend Over Time")
        # Fix 3: filter reviews by selected states instead of business_id
        rev_f = reviews.copy()
        if sel_states and "state" in rev_f.columns:
            rev_f = rev_f[rev_f["state"].isin(sel_states)]
        if not rev_f.empty and "date" in rev_f.columns:
            monthly = (rev_f.set_index("date")["stars"]
                       .resample("ME").mean().reset_index())
            monthly.columns = ["Month","Avg Rating"]
            fig3 = px.line(monthly, x="Month", y="Avg Rating", line_shape="spline")
            fig3.update_traces(line_color="#60a5fa", line_width=2.5)
            fig3.update_layout(
                height=240,
                margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[1,5]),
            )
            st.plotly_chart(fig3, use_container_width=True)
