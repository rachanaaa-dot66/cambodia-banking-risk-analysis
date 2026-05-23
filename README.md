# Cambodia Banking Sector – Credit & Market Risk Analysis

A risk management portfolio project analyzing Cambodia's banking sector
using real data from the **National Bank of Cambodia (NBC)** and regional
market data from Yahoo Finance.

Built as part of a financial engineering internship preparation project,
covering credit risk, market risk, and the transmission mechanism between
external market shocks and domestic credit stress.

---

## Project Motivation

Cambodia's banking sector headline NPL ratio of ~7% (2024) appears
manageable — but granular NBC data reveals a more complex picture:
rising credit stress concentrated in specific institutions, dangerous
liquidity mismatches, and a clear link between regional market shocks
and domestic loan deterioration.

This project investigates that full picture using real regulatory data.

---

## Key Findings

- System-wide NPL ratio rose from **2.0% (2020) → 7.2% (2024)** — a 260% increase
- Several mid-sized banks exceed **25–35% NPL** — masked by large stable banks
- **8 banks** are in the critical zone (NPL > 10%) as of 2024
- Regional market stress (COVID crash 2020) preceded Cambodia's credit
  deterioration by approximately **12–18 months** — confirming market risk
  as a leading indicator of credit stress
- High correlation between Vietnam (VNM) and Emerging Markets (EEM)
  means regional shocks hit Cambodia with no diversification buffer

---

## Project Structure

```
cambodia-banking-risk-analysis/
├── credit_risk_analysis.py     # Part 1: NBC credit data analysis
├── market_risk_analysis.py     # Part 2: Regional market risk + connection
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Excludes data and output files
```

> **Note:** Raw data files (NBC Excel/PDF) and generated chart outputs
> are not included in this repository. Run the scripts to generate outputs.
> NBC data is publicly available at [nbcambodia.org](https://www.nbcambodia.org)

---

## How To Run

**1. Clone the repository**
```bash
git clone git@github.com:yourusername/cambodia-banking-risk-analysis.git
cd cambodia-banking-risk-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add NBC data files to `/data` folder**

Download from [nbcambodia.org](https://www.nbcambodia.org):
- Banking Supervision Report (Table 14) — Loans and NPLs by bank
- DTI Loans and Non-Performing Loans (2023–2024)
- DTI Loans to Deposits (2023–2024)
- DTI Credits by Economic Activities (2024)

**4. Run the analysis**
```bash
# Credit risk analysis
python credit_risk_analysis.py

# Market risk analysis (fetches live data from Yahoo Finance)
python market_risk_analysis.py
```

Outputs are saved to the `/outputs` folder.

---

## Risk Concepts Covered

| Concept | Description |
|---|---|
| NPL Ratio | Non-Performing Loans / Total Loans — core credit health metric |
| Loans-to-Deposits | Liquidity structure — reliance on non-deposit funding |
| Sector Concentration | Which industries dominate lending — concentration risk |
| RAG Framework | Red/Amber/Green thresholds — standard bank risk dashboard |
| Log Returns | Daily asset return calculation — standard in market risk |
| Rolling Volatility | 30-day annualized volatility — market stress indicator |
| Historical VaR | 95% confidence 1-day loss threshold — market risk metric |
| Drawdown | Distance from peak — measures depth of market stress |
| Lagged Correlation | Market stress leading credit stress by 12–18 months |

---

## Data Sources

| Data | Source |
|---|---|
| Bank NPL ratios 2020–2024 | National Bank of Cambodia (NBC) Annual Reports |
| Loans-to-Deposits ratios | NBC Banking Supervision Reports |
| Sector loan exposure | NBC Credits by Economic Activities |
| Market prices 2020–2024 | Yahoo Finance (VNM, EEM, XLF, TLT, AAXJ) |

---

## Tools & Libraries

- **Python** — pandas, numpy, matplotlib, seaborn, scipy
- **yfinance** — real market data retrieval
- **openpyxl** — NBC Excel file processing
- **Power BI** *(next phase)* — interactive KRI dashboard

---

## Author

Financial Engineering Student | Risk Management Specialization
Cambodia | 2025