"""
comparative_analysis.py
Cambodia Banking Sector – Regional Comparison & Capital Adequacy
================================================================
This is the final analytical module. It adds two critical context layers
that transform our analysis from "internal data review" to
"actionable risk assessment with regional benchmarks."

Why benchmarks matter:
    Raw numbers without comparison are hard to act on.
    "Cambodia NPL is 7.2%" — is that bad?
    "Cambodia NPL is 7.2% vs Malaysia 1.4% and regional average 3%" — NOW
    you can see it is serious. Context creates urgency.

What this script adds:
    Chart 1 – Regional NPL Peer Comparison (2024)
    Chart 2 – Cambodia CAR vs NPL Trend (is the buffer shrinking?)
    Chart 3 – Credit Growth Collapse (boom → bust cycle)
    Chart 4 – Complete Risk Summary Scorecard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

RED   = "#c0392b"
AMBER = "#e67e22"
GREEN = "#27ae60"
BLUE  = "#2980b9"
DARK  = "#2c3e50"
GRAY  = "#95a5a6"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA — Regional Benchmarks
# ═══════════════════════════════════════════════════════════════════════════
# Sources:
#   Cambodia NPL    : NBC Banking Supervision Report 2024
#   Regional peers  : IPAF Asia NPL Data 2024, World Bank FSI, IMF FSI
#   Cambodia CAR    : CEIC / NBC data 2018–2024
#   IMF assessment  : IMF Article IV Cambodia 2023/2024
# ─────────────────────────────────────────────────────────────────────────

# Regional NPL ratios 2024 (%)
# These are system-wide NPL ratios from central bank / IMF sources
REGIONAL_NPL = {
    "Singapore":   1.1,   # MAS Financial Stability Review 2024
    "Malaysia":    1.4,   # Bank Negara Malaysia 2024
    "Indonesia":   2.3,   # OJK Indonesia 2024
    "Thailand":    3.1,   # Bank of Thailand 2024
    "Philippines": 3.4,   # BSP Philippines 2024
    "Vietnam":     4.8,   # State Bank of Vietnam 2024
    "Cambodia":    7.2,   # NBC Banking Supervision Report 2024
}

# Cambodia Capital Adequacy Ratio (CAR) history — system-wide %
# Source: CEIC / National Bank of Cambodia
# NBC minimum requirement: 15%
# Basel III international minimum: 8% (Tier 1) / 10.5% (with buffer)
CAR_HISTORY = {
    2018: 24.2,
    2019: 23.8,
    2020: 23.5,
    2021: 23.1,
    2022: 22.2,
    2023: 22.5,
    2024: 22.3,
}

# Cambodia NPL history — system-wide %
# Source: NBC Banking Supervision Reports
NPL_HISTORY = {
    2018: 2.5,
    2019: 2.3,
    2020: 2.0,
    2021: 1.9,
    2022: 3.8,
    2023: 5.2,
    2024: 7.2,
}

# Cambodia total credit growth rate (%) — year on year
# Source: NBC DTI Credits by Sector report (bottom section)
# This is the "boom to bust" credit cycle story
CREDIT_GROWTH = {
    2018: None,       # base year
    2019: 26.9,
    2020: 18.4,
    2021: 23.5,
    2022: 18.7,
    2023: 3.9,
    2024: 2.8,
}


# ═══════════════════════════════════════════════════════════════════════════
# BUILD COMPARATIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def build_comparative_dashboard():
    """
    4-panel dashboard providing regional context and capital adequacy picture.

    Panel 1 — Regional NPL comparison
        Shows Cambodia as a clear outlier vs ASEAN peers.
        This is the single most impactful chart for a management audience
        because it immediately answers "how bad is this really?"

    Panel 2 — CAR vs NPL trend
        Dual-axis chart showing the gap between capital buffer (CAR)
        and rising NPL stress over time.
        The key insight: CAR is stable but NPL is rising fast —
        the buffer exists but is being tested.

    Panel 3 — Credit growth history
        Shows the boom-bust cycle: 20%+ growth → collapse to 3%
        This explains WHY NPLs are rising — loans made during the
        high-growth period are now defaulting.

    Panel 4 — Risk scorecard
        Summary table showing Cambodia's key metrics vs benchmarks.
        This is what goes on the first slide of a management presentation.
    """

    print("[1] Building comparative dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Cambodia Banking Sector – Regional Context & Capital Adequacy\n"
        "Sources: NBC, CEIC, IMF, World Bank, IPAF Asia | 2024",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── PANEL 1: Regional NPL Peer Comparison ─────────────────────────────
    # Question: How does Cambodia compare to its regional peers?
    ax = axes[0, 0]

    countries = list(REGIONAL_NPL.keys())
    npls      = list(REGIONAL_NPL.values())

    # Color: Cambodia = red (it's the subject), others = blue gradient
    # This visual technique draws attention immediately to Cambodia
    colors = [RED if c == "Cambodia" else BLUE for c in countries]
    alphas = [1.0 if c == "Cambodia" else 0.65 for c in countries]

    bars = ax.barh(countries, npls, color=colors,
                   edgecolor="white", height=0.6)

    # Add value labels
    for bar, val, country in zip(bars, npls, countries):
        weight = "bold" if country == "Cambodia" else "normal"
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val}%",
            va="center", fontsize=9, fontweight=weight,
            color=RED if country == "Cambodia" else DARK
        )

    # Regional average line
    avg = np.mean([v for k, v in REGIONAL_NPL.items() if k != "Cambodia"])
    ax.axvline(avg, color=AMBER, linewidth=1.5, linestyle="--",
               label=f"Regional avg excl. Cambodia: {avg:.1f}%")

    # Basel stress threshold
    ax.axvline(5, color=RED, linewidth=1, linestyle=":",
               alpha=0.6, label="Stress threshold (5%)")

    ax.set_title("NPL Ratio – ASEAN Regional Comparison (2024)")
    ax.set_xlabel("NPL Ratio (%)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    # ── PANEL 2: CAR vs NPL Trend ─────────────────────────────────────────
    # Question: Is Cambodia's capital buffer adequate given rising NPLs?
    ax  = axes[0, 1]
    ax2 = ax.twinx()

    years    = list(CAR_HISTORY.keys())
    car_vals = list(CAR_HISTORY.values())
    npl_vals = [NPL_HISTORY[y] for y in years]

    # CAR on left axis (blue — stable, reassuring)
    ax.plot(years, car_vals, color=BLUE, marker="s",
            markersize=7, linewidth=2, label="CAR %", zorder=4)
    ax.fill_between(years, car_vals, 15,
                    where=[c > 15 for c in car_vals],
                    color=BLUE, alpha=0.08,
                    label="Capital buffer above minimum")

    # NBC minimum CAR requirement
    ax.axhline(15, color=GRAY, linewidth=1.2, linestyle="--",
               label="NBC minimum CAR (15%)")

    # NPL on right axis (red — rising, alarming)
    ax2.plot(years, npl_vals, color=RED, marker="o",
             markersize=7, linewidth=2, label="NPL Ratio %", zorder=5)
    ax2.fill_between(years, npl_vals, alpha=0.08, color=RED)

    # Annotate the key divergence point
    ax2.annotate(
        "Divergence:\nNPL accelerating\nCAR stable",
        xy=(2022, 3.8),
        xytext=(2019.5, 6.5),
        fontsize=7.5, color=DARK,
        arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8)
    )

    ax.set_title("Capital Adequacy (CAR) vs NPL Ratio – 2018 to 2024")
    ax.set_ylabel("CAR (%)", color=BLUE)
    ax2.set_ylabel("NPL Ratio (%)", color=RED)
    ax.set_ylim(10, 30)
    ax2.set_ylim(0, 12)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=RED)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7.5, loc="upper left")

    # ── PANEL 3: Credit Growth Boom-Bust Cycle ────────────────────────────
    # Question: Why are NPLs rising now? Because of rapid lending before.
    # This is the causal explanation behind the NPL deterioration.
    ax = axes[1, 0]

    growth_years = [y for y, v in CREDIT_GROWTH.items() if v is not None]
    growth_vals  = [v for v in CREDIT_GROWTH.values() if v is not None]

    # Color bars: high growth = amber (risk building),
    #             low growth  = blue (credit cycle slowing)
    bar_colors = [AMBER if v > 10 else BLUE for v in growth_vals]

    bars = ax.bar(growth_years, growth_vals,
                  color=bar_colors, edgecolor="white", width=0.6)

    # Add value labels
    for bar, val in zip(bars, growth_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    # Annotation explaining the risk logic
    ax.annotate(
        "Rapid lending\n(risk building)",
        xy=(2021, 23.5), xytext=(2019.3, 19),
        fontsize=8, color=AMBER,
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8)
    )
    ax.annotate(
        "Growth collapse\n(stress emerging)",
        xy=(2023, 3.9), xytext=(2022.2, 11),
        fontsize=8, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8)
    )

    ax.set_title(
        "Total Credit Growth Rate – Boom to Bust Cycle (2019–2024)\n"
        "Rapid growth in 2019–2022 explains rising NPLs in 2023–2024"
    )
    ax.set_ylabel("Year-on-Year Growth (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 32)

    # ── PANEL 4: Risk Scorecard ───────────────────────────────────────────
    # Question: What is the overall risk picture at a glance?
    # This is what goes on slide 1 of a board presentation.
    ax = axes[1, 1]
    ax.axis("off")   # no axes for a table panel

    scorecard_data = [
        # Metric, Cambodia Value, Benchmark, RAG Status, Trend
        ["System NPL Ratio",      "7.2%",  "< 5% (Basel stress)",    "🔴 RED",   "↑ Rising"],
        ["Capital Adequacy (CAR)","22.3%", "> 15% (NBC minimum)",    "🟢 GREEN", "→ Stable"],
        ["Loans-to-Deposits",     "~90%",  "< 100% (safe zone)",     "🟢 GREEN", "→ Stable"],
        ["Credit Growth (2024)",  "2.8%",  "> 8% (healthy growth)",  "🟡 AMBER", "↓ Slowing"],
        ["Real Estate Exposure",  "22.7%", "< 20% (concentration)",  "🟡 AMBER", "↑ Rising"],
        ["# Banks NPL > 10%",     "8 banks","0 (ideal)",             "🔴 RED",   "↑ Rising"],
        ["vs Regional Average",   "7.2%",  "3.0% (ASEAN excl. KH)", "🔴 RED",   "↑ Worsening"],
        ["IMF Assessment",        "Monitoring required",
                                           "Well capitalized",       "🟡 AMBER", "→ Watch"],
    ]

    columns = ["Metric", "Cambodia", "Benchmark", "Status", "Trend"]

    table = ax.table(
        cellText  = scorecard_data,
        colLabels = columns,
        loc       = "center",
        cellLoc   = "left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.8)

    # Style header row
    for j in range(len(columns)):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style data rows alternating
    for i in range(1, len(scorecard_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f8f9fa")
            # Color the status column
            if j == 3:
                text = scorecard_data[i-1][3]
                if "RED" in text:
                    table[i, j].set_facecolor("#fde8e8")
                elif "AMBER" in text:
                    table[i, j].set_facecolor("#fef3e2")
                elif "GREEN" in text:
                    table[i, j].set_facecolor("#e8f8e8")

    ax.set_title(
        "Risk Scorecard – Cambodia Banking Sector 2024",
        fontweight="bold", pad=15
    )

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/comparative_analysis.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Comparative dashboard saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL INTEGRATED SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def write_final_summary():
    """
    The complete integrated summary combining all three modules:
    credit risk + market risk + regional context.

    This is what you present in an interview when asked:
    "Tell me about your project."
    You should be able to say this in 2 minutes from memory.
    """

    print("[2] Writing final integrated summary...")

    summary = """
╔══════════════════════════════════════════════════════════════════════════╗
║   CAMBODIA BANKING SECTOR RISK ASSESSMENT – INTEGRATED SUMMARY         ║
║   Credit Risk | Market Risk | Regional Context | Capital Adequacy      ║
║   Data: NBC, Yahoo Finance, IMF, CEIC | Period: 2018–2024              ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE CORE STORY (2 minutes version for interviews)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cambodia's banking sector grew rapidly between 2019 and 2022, with
credit expanding at 18–27% annually. This rapid growth, combined with
the COVID-19 shock in 2020, created the conditions for a credit cycle
downturn that is now clearly visible in official data.

By 2024, the system-wide NPL ratio reached 7.2% — more than double the
regional ASEAN average of 3.0% and above the Basel stress threshold of 5%.
Critically, this headline figure masks severe concentration: 8 individual
banks exceed 10% NPL, with several mid-sized institutions above 25–35%.

The system remains adequately capitalized at the aggregate level (CAR:
22.3% vs 15% minimum), but this buffer is meaningless for individual
banks already in distress. Market risk analysis confirms that regional
external shocks — particularly the COVID crash of 2020 — preceded the
domestic credit deterioration by 12–18 months, validating market stress
indicators as leading signals for Cambodia's banking credit cycle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY FINDINGS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CREDIT RISK
  ► NPL rose from 2.0% (2020) → 7.2% (2024) — 260% increase in 4 years
  ► 8 banks above critical 10% NPL threshold
  ► Hattha Bank: 14.6% (2023) → 35.6% (2024) — largest single-year jump
  ► Specialized banks average NPL: 9.1% vs commercial banks: 1.9%
  ► Real estate (22.7%) and retail trade (17.3%) dominate sector exposure

MARKET RISK
  ► EEM (Emerging Markets) fell 34% during COVID crash (Feb–Mar 2020)
  ► Regional market stress preceded Cambodia NPL rise by 12–18 months
  ► VNM and EEM highly correlated — no regional diversification buffer
  ► Market KRI threshold breach: EEM drawdown > 15% = credit risk alert

LIQUIDITY RISK
  ► Several banks with loans-to-deposits > 150% — reliant on wholesale funding
  ► Combination of high NPL + high LTD ratio = double vulnerability
  ► Most concentrated in mid-sized and specialized institutions

CAPITAL ADEQUACY
  ► System CAR: 22.3% (2024) — above NBC 15% minimum (GREEN)
  ► CAR has been declining slowly: 24.2% (2018) → 22.3% (2024)
  ► System-level adequacy masks individual bank vulnerability
  ► IMF: "well capitalized but asset quality requires close monitoring"

REGIONAL CONTEXT
  ► Cambodia NPL (7.2%) vs Malaysia (1.4%), Indonesia (2.3%),
    Thailand (3.1%), Philippines (3.4%), Vietnam (4.8%)
  ► Cambodia is the highest NPL ratio in comparable ASEAN markets
  ► Credit growth collapsed: 23.5% (2021) → 2.8% (2024)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED MANAGEMENT ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMMEDIATE (0–3 months)
  1. Credit tightening for real estate and household segments
  2. Accelerate provisioning at banks with NPL > 10%
  3. Monthly board review of NPL by institution (not just system average)

MEDIUM TERM (3–12 months)
  4. Conduct individual capital adequacy reviews for distressed banks
  5. Implement EEM/VNM market stress as leading KRI triggers
  6. Engage NBC on restructuring timeline for specialized banks

MONITORING (Ongoing)
  7. KRI thresholds:
     → System NPL > 8% = AMBER escalation
     → System NPL > 10% = RED — board action required
     → Any bank NPL change > 10pp YoY = immediate review
     → EEM drawdown > 15% = credit portfolio stress test required

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data Sources   : NBC Banking Supervision Reports (2020–2024)
                 Yahoo Finance (VNM, EEM, XLF, TLT, AAXJ)
                 IMF Financial Soundness Indicators
                 CEIC / NBC Capital Adequacy Data
Tools Used     : Python (pandas, numpy, matplotlib, seaborn, yfinance)
Risk Framework : Basel III NPL thresholds, RAG status indicators,
                 Historical VaR (95%), Rolling volatility (annualized)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    path = f"{OUTPUT_DIR}/final_integrated_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"    Final summary saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Cambodia Banking Sector – Comparative Analysis")
    print("  Regional Benchmarks & Capital Adequacy")
    print("=" * 60)

    build_comparative_dashboard()
    write_final_summary()

    print("=" * 60)
    print("  Comparative analysis complete.")
    print("=" * 60)