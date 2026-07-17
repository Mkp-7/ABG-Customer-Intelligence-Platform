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


@st.cache_data(show_spinner=False)
def load_and_process(brand_ids_key: str):
    """brand_ids_key is comma-joined sorted brand_ids for cache invalidation."""
    brand_ids = [b for b in brand_ids_key.split(",") if b]

    if not os.path.exists(BUSINESSES_CSV):
        return None, None

    biz = pd.read_csv(BUSINESSES_CSV)
    if brand_ids and "brand_id" in biz.columns:
        biz = biz[biz["brand_id"].isin(brand_ids)]

    if biz.empty:
        return None, None

    reviews = None
    if os.path.exists(REVIEWS_CSV):
        reviews = pd.read_csv(REVIEWS_CSV, parse_dates=["date"])
        reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")
        if brand_ids and "brand_id" in reviews.columns:
            reviews = reviews[reviews["brand_id"].isin(brand_ids)]

        # New schema: place_name in reviews matches name in businesses
        # Old schema: business_id direct join
        if "business_id" in reviews.columns:
            agg = (reviews.groupby("business_id")["stars"]
                   .agg(avg_rating="mean", review_count="count")
                   .reset_index())
            if "review_count" in biz.columns:
                biz = biz.drop(columns=["review_count"])
            biz = biz.merge(agg, on="business_id", how="left")
        else:
            agg = (reviews.groupby("place_name")["stars"]
                   .agg(avg_rating="mean", review_count="count")
                   .reset_index()
                   .rename(columns={"place_name": "name"}))
            if "review_count" in biz.columns:
                biz = biz.drop(columns=["review_count"])
            biz = biz.merge(agg, on="name", how="left")

        biz["avg_rating"]   = biz["avg_rating"].fillna(biz["stars"])
        biz["review_count"] = biz["review_count"].fillna(0).astype(int)
    else:
        biz["avg_rating"]   = biz["stars"]
        biz["review_count"] = 0

    biz = biz.dropna(subset=["latitude", "longitude"])
    biz["avg_rating"] = pd.to_numeric(biz["avg_rating"], errors="coerce")
    biz = biz.dropna(subset=["avg_rating"])

    if biz.empty:
        return None, None

    peer_col = PEER_GROUP_COLUMN if PEER_GROUP_COLUMN in biz.columns else "state"
    biz["peer_avg"] = biz.groupby(peer_col)["avg_rating"].transform("mean")
    biz["vs_peer"]  = (biz["avg_rating"] - biz["peer_avg"]).round(2)

    delta = SIGNIFICANT_DELTA_STARS

    def status(d):
        if d >= delta:  return "Above Peer"
        if d <= -delta: return "Below Peer"
        return "On Par"

    biz["status"]      = biz["vs_peer"].apply(status)
    biz["lat_display"] = add_jitter(biz["latitude"])
    biz["lon_display"] = add_jitter(biz["longitude"])

    biz["city"]    = biz.get("city",    pd.Series("", index=biz.index)).fillna("")
    biz["state"]   = biz.get("state",   pd.Series("", index=biz.index)).fillna("")
    biz["address"] = biz.get("address", pd.Series("", index=biz.index)).fillna("")

    biz["label"] = (biz["name"].astype(str)
                    + "<br>" + biz["address"]
                    + "<br>" + biz["city"] + ", " + biz["state"])

    biz["short_label"] = (biz["city"] + ", " + biz["state"]
                          + "  " + biz["avg_rating"].round(1).astype(str) + "⭐")

    return biz, reviews


def show():
    brand_ids   = st.session_state.get("selected_brand_ids", [])
    brand_names = st.session_state.get("selected_brand_names", ["All Brands"])
    brand_label = ", ".join(brand_names) if brand_names else "All Brands"
    cache_key   = ",".join(sorted(brand_ids))

    st.markdown(f"## 🗺️ Store Pulse Map - {brand_label}")
    st.markdown(
        "Every location benchmarked against its **state peer group**. "
        "🔴 Below peer · 🟡 On par · 🟢 Above peer"
    )

    biz, reviews = load_and_process(cache_key)

    if biz is None or len(biz) == 0:
        st.warning("No location data for the selected brand(s). "
                   "Zipcar has app-only reviews with no store locations.")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🗺️ Map Filters")
    states     = sorted(biz["state"].dropna().unique())
    sel_states = st.sidebar.multiselect("States", options=states, default=states)

    sel_status = st.sidebar.multiselect(
        "Status",
        options=["Above Peer", "On Par", "Below Peer"],
        default=["Above Peer", "On Par", "Below Peer"],
    )
    min_rev   = st.sidebar.slider("Min reviews per location", 1, 30, 1)
    view_mode = st.sidebar.radio(
        "Map view",
        options=["📍 Individual pins (jittered)", "🔵 Cluster mode"],
        index=0,
    )

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = biz["status"].isin(sel_status) & (biz["review_count"] >= min_rev)
    if sel_states:
        mask &= biz["state"].isin(sel_states)
    filtered = biz[mask].copy()

    if filtered.empty:
        st.warning("No locations match the current filters.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Locations",     len(filtered))
    c2.metric("Avg Rating",    f"{filtered['avg_rating'].mean():.2f} ⭐")
    c3.metric("Total Reviews", f"{int(filtered['review_count'].sum()):,}")
    c4.metric("States",        filtered["state"].nunique())
    c5.metric("🔴 Below Peer", int((filtered["status"] == "Below Peer").sum()))
    c6.metric("🟢 Above Peer", int((filtered["status"] == "Above Peer").sum()))

    st.markdown("---")

    # Color: if multiple brands, color by brand_id; otherwise by status
    multi_brand = "brand_id" in filtered.columns and filtered["brand_id"].nunique() > 1

    if multi_brand:
        brand_color_map = {
            bid: col for bid, col in zip(
                filtered["brand_id"].unique(),
                ["#3b82f6", "#10b981", "#f59e0b"]
            )
        }

    status_color_map = {"Above Peer": "#1D9E75", "On Par": "#F59E0B", "Below Peer": "#E24B4A"}

    # ── Map ───────────────────────────────────────────────────────────────────
    fig = go.Figure()

    if "Cluster" in view_mode:
        group_col  = "brand_id" if multi_brand else "status"
        color_map  = brand_color_map if multi_brand else status_color_map
        name_col   = "brand_name" if multi_brand else "status"

        for group_val, color in color_map.items():
            sub = filtered[filtered[group_col] == group_val]
            if sub.empty: continue
            fig.add_trace(go.Scattermapbox(
                lat=sub["lat_display"], lon=sub["lon_display"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=14, color=color, opacity=0.85),
                cluster=dict(enabled=True, color=color, size=20, step=3),
                text=sub["label"],
                customdata=np.stack([
                    sub["avg_rating"].round(2), sub["peer_avg"].round(2),
                    sub["vs_peer"], sub["review_count"],
                    sub["city"], sub["state"],
                    sub.get("brand_name", pd.Series(group_val, index=sub.index)).fillna(""),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[4]}, %{customdata[5]}</b><br>"
                    "Brand: %{customdata[6]}<br>"
                    "Rating: %{customdata[0]}⭐  State avg: %{customdata[1]}⭐<br>"
                    "vs Peers: %{customdata[2]:+.2f}⭐  Reviews: %{customdata[3]}<extra></extra>"
                ),
                name=sub[name_col].iloc[0] if name_col in sub.columns else group_val,
            ))
        fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3.5,
                          mapbox_center={"lat": 37.5, "lon": -96}, height=560,
                          margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(title="Brand" if multi_brand else "vs State Peers",
                                      bgcolor="rgba(255,255,255,0.9)"))
    else:
        max_rc = filtered["review_count"].max() or 1
        filtered["marker_size"] = np.clip(8 + (filtered["review_count"] / max_rc) * 14, 8, 22)

        group_col = "brand_id" if multi_brand else "status"
        color_map = brand_color_map if multi_brand else status_color_map
        name_col  = "brand_name" if multi_brand else "status"

        for group_val, color in color_map.items():
            sub = filtered[filtered[group_col] == group_val]
            if sub.empty: continue
            fig.add_trace(go.Scattermapbox(
                lat=sub["lat_display"], lon=sub["lon_display"],
                mode="markers+text",
                marker=go.scattermapbox.Marker(size=sub["marker_size"], color=color, opacity=0.88),
                text=sub["avg_rating"].round(1).astype(str) + "⭐",
                textposition="top right",
                textfont=dict(size=9, color="#333"),
                customdata=np.stack([
                    sub["avg_rating"].round(2), sub["peer_avg"].round(2),
                    sub["vs_peer"], sub["review_count"],
                    sub["city"], sub["state"], sub["address"],
                    sub.get("brand_name", pd.Series(group_val, index=sub.index)).fillna(""),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[4]}, %{customdata[5]}</b><br>"
                    "%{customdata[6]}<br>"
                    "Brand: %{customdata[7]}<br>"
                    "──────────────<br>"
                    "⭐ Rating: <b>%{customdata[0]}</b>  📊 State avg: %{customdata[1]}<br>"
                    "📈 vs Peers: <b>%{customdata[2]:+.2f}</b>  💬 Reviews: %{customdata[3]}"
                    "<extra></extra>"
                ),
                name=sub[name_col].iloc[0] if name_col in sub.columns else group_val,
            ))
        fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3.8,
                          mapbox_center={"lat": 37.5, "lon": -96}, height=560,
                          margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(title="Brand" if multi_brand else "vs State Peers",
                                      bgcolor="rgba(255,255,255,0.9)"))

    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {"format": "png", "filename": "store_pulse_map"},
    })
    st.caption("💡 Hover pins for full details. Pins colored by brand when multiple selected, "
               "by peer status when one brand selected.")

    # ── Attention tables ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔴 Locations Needing Most Attention")
    st.caption("Furthest below state peer average - priority Field Leader calls.")

    disp_cols = [c for c in ["short_label", "brand_name", "state", "avg_rating",
                              "peer_avg", "vs_peer", "review_count"]
                 if c in filtered.columns]
    bottom = (filtered[filtered["status"] == "Below Peer"]
              .sort_values("vs_peer")[disp_cols].head(15).copy())

    if bottom.empty:
        st.success("✅ No locations significantly below their state peer group.")
    else:
        for col in ["avg_rating", "peer_avg", "vs_peer"]:
            if col in bottom.columns: bottom[col] = bottom[col].round(2)
        st.dataframe(bottom, column_config={
            "short_label":  st.column_config.TextColumn("Location"),
            "brand_name":   st.column_config.TextColumn("Brand"),
            "avg_rating":   st.column_config.NumberColumn("Rating ⭐", format="%.2f"),
            "peer_avg":     st.column_config.NumberColumn("State Avg ⭐", format="%.2f"),
            "vs_peer":      st.column_config.ProgressColumn("Gap vs Peers", min_value=-2, max_value=0, format="%.2f ⭐"),
            "review_count": st.column_config.NumberColumn("Reviews"),
        }, use_container_width=True, hide_index=True)

    st.markdown("### 🟢 Top Performing Locations")
    top = (filtered[filtered["status"] == "Above Peer"]
           .sort_values("vs_peer", ascending=False)[disp_cols].head(15).copy())

    if top.empty:
        st.info("No locations significantly above peer group with current filters.")
    else:
        for col in ["avg_rating", "peer_avg", "vs_peer"]:
            if col in top.columns: top[col] = top[col].round(2)
        st.dataframe(top, column_config={
            "short_label":  st.column_config.TextColumn("Location"),
            "brand_name":   st.column_config.TextColumn("Brand"),
            "avg_rating":   st.column_config.NumberColumn("Rating ⭐", format="%.2f"),
            "peer_avg":     st.column_config.NumberColumn("State Avg ⭐", format="%.2f"),
            "vs_peer":      st.column_config.ProgressColumn("Gap vs Peers", min_value=0, max_value=2, format="+%.2f ⭐"),
            "review_count": st.column_config.NumberColumn("Reviews"),
        }, use_container_width=True, hide_index=True)

    # ── State bar chart ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Average Rating by State")

    if multi_brand:
        sa = (filtered.groupby(["state", "brand_name"])
              .agg(avg_rating=("avg_rating", "mean"), locations=("name", "count"))
              .reset_index())
        sa["avg_rating"] = sa["avg_rating"].round(2)
        fig2 = px.bar(sa, x="state", y="avg_rating", color="brand_name", barmode="group",
                      text="avg_rating",
                      color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
                      labels={"state": "State", "avg_rating": "Avg Rating", "brand_name": "Brand"})
    else:
        sa = (filtered.groupby("state")
              .agg(avg_rating=("avg_rating", "mean"), locations=("name", "count"))
              .sort_values("avg_rating", ascending=False).reset_index())
        sa["avg_rating"] = sa["avg_rating"].round(2)
        chain_avg = filtered["avg_rating"].mean()
        fig2 = px.bar(sa, x="state", y="avg_rating", color="avg_rating",
                      color_continuous_scale=["#E24B4A", "#F59E0B", "#1D9E75"],
                      range_color=[2.0, 4.5], text="avg_rating",
                      hover_data={"locations": True},
                      labels={"state": "State", "avg_rating": "Avg Rating"})
        fig2.add_hline(y=chain_avg, line_dash="dot", line_color="#60a5fa",
                       annotation_text=f"Chain avg: {chain_avg:.2f} ⭐",
                       annotation_position="top right")

    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig2.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Rating trend ──────────────────────────────────────────────────────────
    if reviews is not None and not reviews.empty and "date" in reviews.columns:
        st.markdown("### 📈 Rating Trend Over Time")
        rev_f = reviews.copy()
        if sel_states and "state" in rev_f.columns:
            rev_f = rev_f[rev_f["state"].isin(sel_states)]

        if not rev_f.empty:
            if multi_brand and "brand_name" in rev_f.columns:
                monthly = (rev_f.dropna(subset=["date"])
                           .groupby([pd.Grouper(key="date", freq="ME"), "brand_name"])["stars"]
                           .mean().reset_index())
                monthly.columns = ["Month", "Brand", "Avg Rating"]
                fig3 = px.line(monthly, x="Month", y="Avg Rating", color="Brand",
                               line_shape="spline",
                               color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"])
            else:
                monthly = (rev_f.dropna(subset=["date"])
                           .set_index("date")["stars"].resample("ME").mean().reset_index())
                monthly.columns = ["Month", "Avg Rating"]
                fig3 = px.line(monthly, x="Month", y="Avg Rating", line_shape="spline")
                fig3.update_traces(line_color="#60a5fa", line_width=2.5)

            fig3.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(range=[1, 5]))
            st.plotly_chart(fig3, use_container_width=True)
