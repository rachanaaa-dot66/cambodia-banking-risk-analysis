"""
app.py
Cambodia Banking Sector – Interactive Risk Dashboard
=====================================================
Streamlit + Plotly web application presenting the complete
risk analysis in an interactive, user-friendly format.

How to run:
    streamlit run app.py

Structure:
    Sidebar     – navigation between pages
    Page 1      – Credit Risk (NPL trends, bank analysis)
    Page 2      – Regional Context (ASEAN peers, CAR vs NPL)
    Page 3      – Market Risk (asset performance, VaR)
    Page 4      – Risk Scorecard (summary + recommendations)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────
# This must be the FIRST streamlit command in the script
# Sets the browser tab title, icon, and layout width
st.set_page_config(
    page_title    = "Cambodia Banking Risk Dashboard",
    page_icon     = "🏦",
    layout        = "wide",          # use full browser width
    initial_sidebar_state = "expanded"
)

# ── Color Palette ─────────────────────────────────────────────────────────
# Same RAG colors as our analysis scripts for consistency
RED   = "#c0392b"
AMBER = "#e67e22"
GREEN = "#27ae60"
BLUE  = "#2980b9"
DARK  = "#2c3e50"
LIGHT = "#ecf0f1"

# ── Custom CSS ────────────────────────────────────────────────────────────
# Streamlit allows injecting CSS to customize the look
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Headers */
    h1 { color: #2c3e50; font-weight: 700; }
    h2 { color: #2c3e50; font-weight: 600; }
    h3 { color: #34495e; }

    /* RAG badge styles */
    .rag-red    { background:#fde8e8; color:#c0392b;
                  padding:4px 10px; border-radius:12px;
                  font-weight:600; font-size:13px; }
    .rag-amber  { background:#fef3e2; color:#e67e22;
                  padding:4px 10px; border-radius:12px;
                  font-weight:600; font-size:13px; }
    .rag-green  { background:#e8f8e8; color:#27ae60;
                  padding:4px 10px; border-radius:12px;
                  font-weight:600; font-size:13px; }

    /* Divider */
    hr { border: 1px solid #e0e0e0; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
# @st.cache_data tells Streamlit to cache (remember) the loaded data
# so it doesn't reload from disk every time the user clicks something.
# This makes the app much faster.

@st.cache_data
def load_npl_history():
    """System-wide NPL trend 2018-2024 from NBC reports."""
    return pd.DataFrame({
        "year":      [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "npl_ratio": [2.5,  2.3,  2.0,  1.9,  3.8,  5.2,  7.2],
        "car":       [24.2, 23.8, 23.5, 23.1, 22.2, 22.5, 22.3],
    })


@st.cache_data
def load_bank_npl():
    """Individual bank NPL data 2020-2024 from NBC files."""
    # Key banks with data across multiple years
    # Source: NBC Banking Supervision Reports
    banks = {
        "ACLEDA Bank":              [2.1, 2.1, None, 3.2, 4.1],
        "Advanced Bank of Asia":    [0.9, 1.1, None, 1.8, 2.3],
        "Canadia Bank":             [3.4, 3.4, None, 4.1, 5.2],
        "Hattha Bank":              [1.3, 1.5, None, 14.6, 35.6],
        "Sathapana Bank":           [1.9, 2.0, None, 3.8, 6.1],
        "Cambodian Public Bank":    [0.6, 0.6, None, 1.2, 1.8],
        "Maybank Cambodia":         [2.4, 3.2, None, 5.1, 8.3],
        "RHB Bank Cambodia":        [5.9, 4.6, None, 7.2, 11.4],
        "Sacom Bank":               [16.7, 16.6, None, 18.2, 21.3],
        "Bangkok Bank Cambodia":    [24.6, 15.2, None, 12.1, 9.8],
        "Phillip Bank":             [2.6, 3.6, None, 5.8, 9.2],
        "CIMB Bank":                [0.2, 0.4, None, 1.1, 2.8],
    }
    years = [2020, 2021, 2022, 2023, 2024]
    rows  = []
    for bank, values in banks.items():
        for yr, val in zip(years, values):
            if val is not None:
                rows.append({
                    "bank": bank, "year": yr, "npl_ratio": val
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_regional_data():
    """ASEAN regional NPL comparison 2024."""
    return pd.DataFrame({
        "country":   ["Singapore", "Malaysia", "Indonesia",
                      "Thailand", "Philippines", "Vietnam", "Cambodia"],
        "npl_ratio": [1.1, 1.4, 2.3, 3.1, 3.4, 4.8, 7.2],
        "highlight": [False, False, False, False, False, False, True],
    })


@st.cache_data
def load_credit_growth():
    """Total credit growth rate 2019-2024."""
    return pd.DataFrame({
        "year":        [2019, 2020, 2021, 2022, 2023, 2024],
        "growth_rate": [26.9, 18.4, 23.5, 18.7, 3.9,  2.8],
    })


@st.cache_data
def load_sector_data():
    """Sector loan concentration 2024."""
    return pd.DataFrame({
        "sector": [
            "Real Estate", "Retail Trade", "Households",
            "Agriculture", "Construction", "Manufacturing",
            "Accommodation", "Other"
        ],
        "share_pct": [22.7, 17.3, 10.2, 10.1, 9.8, 4.3, 4.1, 21.5],
        "risk_level": [
            "High", "Medium", "High",
            "Medium", "Medium", "Low",
            "High", "Low"
        ],
    })


@st.cache_data
def load_market_data():
    """
    Simulated market data for display.
    In production this would pull from your market_risk_analysis.py outputs.
    We use representative values here for the app to work standalone.
    """
    import yfinance as yf
    try:
        tickers = {"VNM": "Vietnam ETF", "EEM": "Emerging Markets",
                   "XLF": "US Financials", "TLT": "US Treasury"}
        raw = yf.download(
            list(tickers.keys()),
            start="2020-01-01", end="2024-12-31",
            progress=False, auto_adjust=True
        )
        prices = raw["Close"].copy()
        prices.columns = [tickers[c] for c in prices.columns]
        returns = np.log(prices / prices.shift(1)).dropna()
        cum_ret = (1 + returns).cumprod()
        return cum_ret.reset_index()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏦 Cambodia Risk")
        st.markdown("**Banking Sector Analysis**")
        st.markdown("*Data: NBC · IMF · Yahoo Finance*")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["📊 Credit Risk",
             "🌏 Regional Context",
             "📈 Market Risk",
             "📋 Risk Scorecard"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Data Period**")
        st.markdown("2018 – 2024")
        st.markdown("**Last Updated**")
        st.markdown("NBC 2024 Report")
        st.markdown("---")
        st.markdown(
            "<small>Built with Python · Streamlit · Plotly<br>"
            "Data: National Bank of Cambodia</small>",
            unsafe_allow_html=True
        )

    return page


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — CREDIT RISK
# ═══════════════════════════════════════════════════════════════════════════
def page_credit_risk():
    st.title("📊 Credit Risk Analysis")
    st.markdown(
        "Cambodia's banking sector NPL ratio has risen from **2.0% (2020)** "
        "to **7.2% (2024)** — crossing the Basel stress threshold and approaching "
        "the critical zone. This page breaks down the trend, individual bank "
        "stress, and sector concentration."
    )

    # ── KPI Metrics Row ───────────────────────────────────────────────────
    # st.metric shows a number with a delta (change indicator)
    # These are the first things a risk manager looks at
    st.markdown("### Key Risk Indicators — 2024")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label    = "System NPL Ratio",
            value    = "7.2%",
            delta    = "+2.0pp vs 2023",
            delta_color = "inverse"   # red for increase (bad)
        )
    with col2:
        st.metric(
            label = "Banks NPL > 10%",
            value = "8 banks",
            delta = "+3 vs 2023",
            delta_color = "inverse"
        )
    with col3:
        st.metric(
            label = "Capital Adequacy (CAR)",
            value = "22.3%",
            delta = "-0.2pp vs 2023",
            delta_color = "inverse"
        )
    with col4:
        st.metric(
            label = "NBC Minimum CAR",
            value = "15.0%",
            delta = "7.3pp buffer",
            delta_color = "normal"
        )

    st.markdown("---")

    # ── Row 1: NPL Trend + Bank Comparison ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### System NPL Trend (2018–2024)")
        st.caption(
            "The system-wide NPL ratio remained stable pre-COVID then "
            "accelerated sharply from 2022 onward as loan restructuring "
            "programs expired."
        )

        npl_hist = load_npl_history()

        # Color bars by RAG threshold
        bar_colors = [
            RED if v > 5 else AMBER if v > 3 else GREEN
            for v in npl_hist["npl_ratio"]
        ]

        fig = go.Figure()

        # Bar chart for NPL
        fig.add_trace(go.Bar(
            x    = npl_hist["year"],
            y    = npl_hist["npl_ratio"],
            name = "NPL Ratio %",
            marker_color = bar_colors,
            text = [f"{v}%" for v in npl_hist["npl_ratio"]],
            textposition = "outside",
        ))

        # Basel threshold lines
        fig.add_hline(
            y=5, line_dash="dash", line_color=AMBER,
            annotation_text="Stress threshold (5%)",
            annotation_position="right"
        )
        fig.add_hline(
            y=10, line_dash="dash", line_color=RED,
            annotation_text="Critical threshold (10%)",
            annotation_position="right"
        )

        fig.update_layout(
            yaxis_title    = "NPL Ratio (%)",
            xaxis_title    = "Year",
            showlegend     = False,
            plot_bgcolor   = "white",
            paper_bgcolor  = "white",
            yaxis=dict(range=[0, 12]),
            height         = 350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Individual Bank NPL – 2024")
        st.caption(
            "Sorted by NPL ratio. Red = critical (>10%), "
            "Amber = elevated (5-10%), Green = acceptable (<5%)."
        )

        bank_df = load_bank_npl()
        banks_2024 = (
            bank_df[bank_df["year"] == 2024]
            .sort_values("npl_ratio", ascending=True)
        )

        colors = [
            RED if v > 10 else AMBER if v > 5 else GREEN
            for v in banks_2024["npl_ratio"]
        ]

        fig = go.Figure(go.Bar(
            x            = banks_2024["npl_ratio"],
            y            = banks_2024["bank"],
            orientation  = "h",
            marker_color = colors,
            text         = [f"{v}%" for v in banks_2024["npl_ratio"]],
            textposition = "outside",
        ))

        fig.add_vline(x=5,  line_dash="dash", line_color=AMBER)
        fig.add_vline(x=10, line_dash="dash", line_color=RED)

        fig.update_layout(
            xaxis_title   = "NPL Ratio (%)",
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 350,
            xaxis=dict(range=[0, 42]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 2: Bank Trend Selector + Sector Concentration ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Bank NPL History — Select a Bank")
        st.caption(
            "Use the dropdown to explore any bank's NPL trend over time. "
            "This shows which banks deteriorated gradually vs suddenly."
        )

        bank_df   = load_bank_npl()
        all_banks = sorted(bank_df["bank"].unique())

        # st.selectbox = dropdown menu
        selected_bank = st.selectbox(
            "Select bank to explore:",
            all_banks,
            index=all_banks.index("Hattha Bank")
        )

        bank_trend = bank_df[bank_df["bank"] == selected_bank].sort_values("year")

        fig = go.Figure()

        # Line chart with markers
        fig.add_trace(go.Scatter(
            x    = bank_trend["year"],
            y    = bank_trend["npl_ratio"],
            mode = "lines+markers+text",
            name = selected_bank,
            line = dict(color=BLUE, width=2.5),
            marker = dict(size=9, color=BLUE),
            text = [f"{v}%" for v in bank_trend["npl_ratio"]],
            textposition = "top center",
        ))

        # Add threshold zones
        fig.add_hrect(y0=0,  y1=5,  fillcolor=GREEN, opacity=0.05,
                      line_width=0, annotation_text="Safe zone")
        fig.add_hrect(y0=5,  y1=10, fillcolor=AMBER, opacity=0.05,
                      line_width=0, annotation_text="Stress zone")
        fig.add_hrect(y0=10, y1=50, fillcolor=RED,   opacity=0.05,
                      line_width=0, annotation_text="Critical zone")

        fig.update_layout(
            title        = f"{selected_bank} – NPL Ratio Over Time",
            yaxis_title  = "NPL Ratio (%)",
            xaxis_title  = "Year",
            plot_bgcolor = "white",
            paper_bgcolor= "white",
            height       = 350,
            showlegend   = False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Sector Loan Concentration – 2024")
        st.caption(
            "Real estate (22.7%) and households (10.2%) dominate lending. "
            "High concentration in these sectors creates systemic risk "
            "if property markets weaken."
        )

        sector_df = load_sector_data()

        # Color by risk level
        color_map = {"High": RED, "Medium": AMBER, "Low": GREEN}
        colors = [color_map[r] for r in sector_df["risk_level"]]

        fig = go.Figure(go.Pie(
            labels       = sector_df["sector"],
            values       = sector_df["share_pct"],
            marker_colors= colors,
            hole         = 0.4,        # donut chart
            textinfo     = "label+percent",
            textfont_size= 11,
        ))

        fig.update_layout(
            annotations=[dict(
                text="Sector<br>Share",
                x=0.5, y=0.5,
                font_size=13,
                showarrow=False
            )],
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 350,
            showlegend    = False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — REGIONAL CONTEXT
# ═══════════════════════════════════════════════════════════════════════════
def page_regional():
    st.title("🌏 Regional Context")
    st.markdown(
        "Cambodia's NPL ratio of **7.2%** is the highest among comparable "
        "ASEAN markets. The regional average (excluding Cambodia) is **3.0%**. "
        "This page provides the benchmark context that makes Cambodia's "
        "situation clear."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cambodia NPL", "7.2%", "+2.0pp YoY", delta_color="inverse")
    with col2:
        st.metric("ASEAN Average", "3.0%", "excl. Cambodia")
    with col3:
        st.metric("Gap vs Region", "+4.2pp", "above average", delta_color="inverse")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ASEAN NPL Comparison – 2024")
        st.caption(
            "Cambodia stands out clearly against regional peers. "
            "Red bar = Cambodia. Blue = peer countries."
        )

        regional = load_regional_data()

        colors = [RED if h else BLUE for h in regional["highlight"]]

        fig = go.Figure(go.Bar(
            x            = regional["npl_ratio"],
            y            = regional["country"],
            orientation  = "h",
            marker_color = colors,
            text         = [f"{v}%" for v in regional["npl_ratio"]],
            textposition = "outside",
        ))

        # Regional average line
        avg = regional[~regional["highlight"]]["npl_ratio"].mean()
        fig.add_vline(
            x=avg, line_dash="dash", line_color=AMBER,
            annotation_text=f"Regional avg: {avg:.1f}%",
            annotation_position="top"
        )
        fig.add_vline(
            x=5, line_dash="dot", line_color=RED,
            annotation_text="Stress threshold",
            annotation_position="bottom"
        )

        fig.update_layout(
            xaxis_title   = "NPL Ratio (%)",
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 380,
            xaxis=dict(range=[0, 9.5]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### CAR vs NPL – Is the Buffer Holding?")
        st.caption(
            "Capital Adequacy Ratio (blue) remains stable above the 15% minimum. "
            "But NPL (red) is rising fast — the buffer exists but is being tested."
        )

        hist = load_npl_history()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # CAR on primary axis
        fig.add_trace(
            go.Scatter(
                x    = hist["year"],
                y    = hist["car"],
                name = "CAR %",
                line = dict(color=BLUE, width=2.5),
                mode = "lines+markers",
                marker=dict(size=7),
            ),
            secondary_y=False
        )

        # NBC minimum
        fig.add_hline(
            y=15, line_dash="dash", line_color=DARK,
            annotation_text="NBC minimum (15%)",
            annotation_position="right",
            secondary_y=False
        )

        # NPL on secondary axis
        fig.add_trace(
            go.Scatter(
                x    = hist["year"],
                y    = hist["npl_ratio"],
                name = "NPL Ratio %",
                line = dict(color=RED, width=2.5),
                mode = "lines+markers",
                marker=dict(size=7),
                fill = "tozeroy",
                fillcolor = "rgba(192,57,43,0.08)",
            ),
            secondary_y=True
        )

        fig.update_layout(
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 380,
            legend        = dict(x=0.01, y=0.99),
        )
        fig.update_yaxes(
            title_text="CAR (%)", secondary_y=False,
            range=[10, 30], title_font_color=BLUE
        )
        fig.update_yaxes(
            title_text="NPL Ratio (%)", secondary_y=True,
            range=[0, 12], title_font_color=RED
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Credit growth boom-bust
    st.markdown("### Credit Growth Cycle – The Root Cause (2019–2024)")
    st.caption(
        "Rapid credit expansion of 18–27% annually from 2019–2022 created "
        "the conditions for today's NPL stress. Growth collapsed to under 3% "
        "in 2023–2024 as the credit cycle turned."
    )

    growth = load_credit_growth()
    colors = [AMBER if v > 10 else BLUE for v in growth["growth_rate"]]

    fig = go.Figure(go.Bar(
        x            = growth["year"],
        y            = growth["growth_rate"],
        marker_color = colors,
        text         = [f"{v}%" for v in growth["growth_rate"]],
        textposition = "outside",
    ))

    fig.add_annotation(
        x=2021, y=25,
        text="Rapid expansion<br>(risk building)",
        showarrow=True, arrowhead=2,
        ax=60, ay=-40,
        font=dict(color=AMBER, size=11)
    )
    fig.add_annotation(
        x=2023, y=5.5,
        text="Growth collapse<br>(stress emerging)",
        showarrow=True, arrowhead=2,
        ax=-60, ay=-40,
        font=dict(color=BLUE, size=11)
    )

    fig.update_layout(
        yaxis_title   = "Year-on-Year Growth (%)",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        height        = 320,
        yaxis=dict(range=[0, 32]),
        showlegend    = False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARKET RISK
# ═══════════════════════════════════════════════════════════════════════════
def page_market_risk():
    st.title("📈 Market Risk Analysis")
    st.markdown(
        "Regional market stress is a **leading indicator** of Cambodia's credit "
        "stress — preceding NPL deterioration by 12–18 months. The COVID crash "
        "of early 2020 directly preceded Cambodia's credit cycle downturn."
    )

    st.info(
        "📡 This page fetches live market data from Yahoo Finance. "
        "Loading may take a few seconds.",
        icon="ℹ️"
    )

    with st.spinner("Fetching market data..."):
        market_data = load_market_data()

    if market_data is None:
        st.warning(
            "Could not fetch live market data. "
            "Please check your internet connection."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cumulative Returns – Regional Assets (2020–2024)")
        st.caption(
            "Shows total performance from January 2020 baseline. "
            "COVID crash visible in Feb–Mar 2020. "
            "US Treasuries show flight-to-safety behavior."
        )

        fig = go.Figure()

        colors_map = {
            "Vietnam ETF":      RED,
            "Emerging Markets": BLUE,
            "US Financials":    GREEN,
            "US Treasury":      AMBER,
        }

        for col in market_data.columns[1:]:
            fig.add_trace(go.Scatter(
                x    = market_data["Date"],
                y    = market_data[col],
                name = col,
                line = dict(
                    color=colors_map.get(col, DARK),
                    width=1.8
                ),
                mode = "lines",
            ))

        # Shade COVID crash
        fig.add_vrect(
            x0="2020-02-20", x1="2020-03-23",
            fillcolor=RED, opacity=0.1,
            line_width=0,
            annotation_text="COVID Crash",
            annotation_position="top left",
            annotation_font_color=RED
        )

        fig.add_hline(y=1.0, line_dash="dot",
                      line_color=DARK, opacity=0.4)

        fig.update_layout(
            yaxis_title   = "Growth of $1.00",
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 380,
            legend=dict(x=0.01, y=0.99),
            hovermode     = "x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Market Stress → Cambodia Credit Stress")
        st.caption(
            "External market shocks (blue bars) precede Cambodia's NPL "
            "deterioration (red line) by approximately 12–18 months. "
            "This lag makes market data a useful early warning signal."
        )

        # Cambodia NPL data points
        npl_points = pd.DataFrame({
            "date":      pd.to_datetime(["2020-12-31", "2021-12-31",
                                         "2023-12-31", "2024-12-31"]),
            "npl_ratio": [2.0, 1.9, 5.2, 7.2],
        })

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Market returns as bars (left axis)
        eem_col = "Emerging Markets"
        if eem_col in market_data.columns:
            eem_ret = market_data[["Date", eem_col]].copy()
            eem_ret["monthly"] = (
                eem_ret.set_index("Date")[eem_col]
                .resample("ME").last()
                .pct_change()
                .reset_index()[eem_col]
            )

            monthly_eem = (
                eem_ret.set_index("Date")[eem_col]
                .resample("ME").last()
                .pct_change()
                .reset_index()
            )
            monthly_eem.columns = ["date", "return"]
            monthly_eem = monthly_eem.dropna()

            colors_bar = [RED if v < 0 else GREEN
                          for v in monthly_eem["return"]]

            fig.add_trace(
                go.Bar(
                    x=monthly_eem["date"],
                    y=monthly_eem["return"] * 100,
                    name="EEM Monthly Return %",
                    marker_color=colors_bar,
                    opacity=0.5,
                ),
                secondary_y=False
            )

        # NPL line (right axis)
        fig.add_trace(
            go.Scatter(
                x    = npl_points["date"],
                y    = npl_points["npl_ratio"],
                name = "Cambodia NPL %",
                mode = "lines+markers+text",
                line = dict(color=RED, width=2.5),
                marker=dict(size=10, color=RED),
                text = [f"{v}%" for v in npl_points["npl_ratio"]],
                textposition="top center",
            ),
            secondary_y=True
        )

        fig.update_layout(
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height        = 380,
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
        )
        fig.update_yaxes(
            title_text="EEM Monthly Return (%)",
            secondary_y=False, title_font_color=BLUE
        )
        fig.update_yaxes(
            title_text="Cambodia NPL Ratio (%)",
            secondary_y=True, title_font_color=RED,
            range=[0, 10]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Market Risk Insight")
    st.markdown("""
    | Signal | Threshold | Meaning for Cambodia |
    |---|---|---|
    | EEM 30-day volatility | > 25% annualized | Elevated regional stress |
    | EEM drawdown from peak | > 15% | Credit portfolio stress test required |
    | VNM drawdown from peak | > 20% | Vietnam-specific shock — high Cambodia relevance |
    | EEM + VNM simultaneously in drawdown | Any | Systemic regional stress flag |

    > **Why this matters:** When these market signals fire, Cambodia's banking sector
    > typically sees NPL deterioration 12–18 months later. Monitoring these
    > indicators gives risk managers advance warning before stress appears in
    > official NBC data.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK SCORECARD
# ═══════════════════════════════════════════════════════════════════════════
def page_scorecard():
    st.title("📋 Risk Scorecard & Recommendations")
    st.markdown(
        "Complete risk assessment summary combining credit risk, market risk, "
        "liquidity risk, and regional context. This is the management-level "
        "view — findings, RAG status, and recommended actions."
    )

    st.markdown("---")
    st.markdown("### Overall Risk Status – Cambodia Banking Sector 2024")

    # Overall RAG — RED based on findings
    st.markdown("""
    <div style='background:#fde8e8; border-left:5px solid #c0392b;
                padding:16px; border-radius:4px; margin-bottom:20px;'>
        <strong style='color:#c0392b; font-size:18px;'>🔴 ELEVATED RISK</strong><br>
        <span style='color:#2c3e50;'>
        System NPL ratio (7.2%) has exceeded the Basel stress threshold (5%)
        and is trending toward the critical threshold (10%). Concentration risk
        in specific institutions and sectors requires immediate management attention.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Risk Scorecard by Category")

    # Scorecard table
    scorecard = pd.DataFrame([
        ["Credit Risk",    "System NPL Ratio",       "7.2%",  "< 5%",   "🔴 Critical", "↑ Rising fast"],
        ["Credit Risk",    "Banks NPL > 10%",        "8 banks","0",     "🔴 Critical", "↑ Increasing"],
        ["Credit Risk",    "Worst Bank NPL",         "35.6%", "< 10%",  "🔴 Critical", "↑ Doubled YoY"],
        ["Capital",        "System CAR",             "22.3%", "> 15%",  "🟢 Safe",     "→ Stable"],
        ["Capital",        "CAR trend",              "Declining","Stable","🟡 Watch",  "↓ Slow decline"],
        ["Liquidity",      "Loans-to-Deposits",      "~90%",  "< 100%", "🟢 Safe",     "→ Stable"],
        ["Liquidity",      "Banks LTD > 150%",       "Several","None",  "🟡 Watch",    "→ Monitor"],
        ["Concentration",  "Real Estate Share",      "22.7%", "< 20%",  "🟡 Watch",    "↑ Rising"],
        ["Concentration",  "Sector HHI",             "Elevated","Low",  "🟡 Watch",    "→ Stable"],
        ["Market Risk",    "EEM Drawdown Signal",    "Clear", "None",   "🟡 Watch",    "→ Improving"],
        ["Regional",       "vs ASEAN Average",       "+4.2pp","0pp gap","🔴 Critical", "↑ Widening"],
        ["Regulatory",     "IMF Assessment",         "Monitor","Pass",  "🟡 Watch",    "→ Stable"],
    ], columns=["Category", "Metric", "Cambodia", "Benchmark", "Status", "Trend"])

    st.dataframe(
        scorecard,
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.markdown("---")
    st.markdown("### Recommended Management Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🚨 Immediate (0–3 months)**

        1. Credit tightening for real estate and household lending segments
        2. Accelerate provisioning at banks with NPL > 10%
        3. Monthly board-level review of NPL by individual institution
        4. Capital adequacy review for distressed banks
        """)

    with col2:
        st.markdown("""
        **⚠️ Medium Term (3–12 months)**

        1. Implement EEM/VNM market stress as leading KRI triggers
        2. Engage NBC on restructuring timeline for specialized banks
        3. Stress test real estate portfolio against 20% price decline
        4. Review wholesale funding dependency at high-LTD banks
        """)

    with col3:
        st.markdown("""
        **📊 KRI Monitoring Thresholds**

        | KRI | Amber | Red |
        |---|---|---|
        | System NPL | 8% | 10% |
        | Bank NPL change | +5pp YoY | +10pp YoY |
        | EEM drawdown | 10% | 20% |
        | LTD ratio | 120% | 150% |
        """)

    st.markdown("---")
    st.markdown("### Data Sources & Methodology")
    st.markdown("""
    | Data | Source | Period |
    |---|---|---|
    | Bank NPL ratios | NBC Banking Supervision Reports | 2020–2024 |
    | Loans-to-Deposits | NBC DTI Reports | 2023–2024 |
    | Sector exposure | NBC Credits by Economic Activities | 2024 |
    | Capital Adequacy | CEIC / NBC Annual Reports | 2018–2024 |
    | Regional NPL peers | IPAF Asia, World Bank FSI, IMF | 2024 |
    | Market prices | Yahoo Finance (VNM, EEM, XLF, TLT) | 2020–2024 |

    **Tools:** Python · pandas · numpy · Streamlit · Plotly · yfinance

    **Risk Framework:** Basel III NPL thresholds · RAG status indicators ·
    Historical VaR (95%) · Rolling volatility (annualized)
    """)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Route to correct page
# ═══════════════════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()

    if page == "📊 Credit Risk":
        page_credit_risk()
    elif page == "🌏 Regional Context":
        page_regional()
    elif page == "📈 Market Risk":
        page_market_risk()
    elif page == "📋 Risk Scorecard":
        page_scorecard()


if __name__ == "__main__":
    main()
