# CAISO Day-Ahead Portfolio Optimization Dashboard

A Python/Streamlit tool that solves a 24-hour linear programming economic dispatch for a 430 MW mixed-resource portfolio (hydro, solar PV, gas peaker) using real CAISO Day-Ahead LMP prices. Includes Monte Carlo price risk analysis and a NERC/FERC compliance tracker.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs offline by default using a California duck-curve LMP profile. Toggle **Use CAISO OASIS API** in the sidebar to fetch live Day-Ahead prices.

## What It Does

- **Economic Dispatch** — LP optimizer (PuLP/CBC) minimizes production cost across hydro, solar, and gas peaker over 24 hours subject to load balance, ramp limits, FERC water budgets, and NERC spinning reserve requirements
- **Risk Analysis** — Monte Carlo price simulation (500 scenarios, log-normal shocks) computes VaR-95 and CVaR-95 revenue distributions across configurable volatility levels
- **Compliance Tracking** — hourly checks against NERC BAL-001-2, CPS1 proxy, FERC Order 764 ramp adequacy, and FERC Part 12 hydro water release budget
- **Live Data** — CAISO OASIS API integration (PRC_LMP, DAM) with XML/ZIP parsing and duck-curve fallback

## Portfolio

| Resource   | Capacity | Node           | Marginal Cost |
|------------|----------|----------------|---------------|
| Hydro      | 150 MW   | NP15_GEN-APND  | $15/MWh       |
| Solar PV   | 80 MW    | SP15_GEN-APND  | $0/MWh        |
| Gas Peaker | 200 MW   | SP15_GEN-APND  | $89.25/MWh    |

## Project Structure

```
├── app.py                      # Streamlit dashboard (4 tabs)
├── src/
│   ├── optimizer/
│   │   ├── resources.py        # Resource dataclasses with embedded constraints
│   │   └── dispatch.py         # 24-hour LP economic dispatch model
│   ├── data/
│   │   ├── caiso_client.py     # CAISO OASIS API client
│   │   └── sample_data.py      # Duck-curve LMP + load profile
│   ├── risk/
│   │   └── sensitivity.py      # Monte Carlo CVaR/VaR engine
│   └── compliance/
│       └── tracker.py          # NERC/FERC compliance checks
└── requirements.txt
```

## Tech Stack

PuLP · pandas · numpy · Plotly · Streamlit · requests · CAISO OASIS API
