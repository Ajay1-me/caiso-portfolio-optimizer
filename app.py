"""
CAISO Day-Ahead Portfolio Optimization Dashboard
================================================
Economic Dispatch Simulator | FERC/NERC Compliant | Real CAISO LMP Data

Run: streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.compliance.tracker import build_compliance_report
from src.data.caiso_client import fetch_lmp
from src.data.sample_data import get_sample_lmp, get_sample_load, get_solar_availability
from src.optimizer.dispatch import run_dispatch
from src.optimizer.resources import build_portfolio
from src.risk.sensitivity import sensitivity_analysis

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CAISO Portfolio Optimizer",
    page_icon="⚡",
    layout="wide",
)

RESOURCE_COLORS = {
    "hydro": "#1E88E5",   # blue
    "solar": "#FDD835",   # amber
    "gas":   "#E53935",   # red
}
RESOURCE_LABELS = {
    "hydro": "Hydro (150 MW)",
    "solar": "Solar PV (80 MW)",
    "gas":   "Gas Peaker (200 MW)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Optimization Parameters")

    op_date = st.date_input(
        "Operating Date",
        value=date.today() - timedelta(days=3),
        help="Day-Ahead market clears the night before this date.",
    )

    node = st.selectbox(
        "CAISO Pricing Node",
        ["SP15_GEN-APND", "NP15_GEN-APND", "ZP26_GEN-APND"],
        help="SP15 = Southern CA (SCE/SDG&E), NP15 = Northern CA (PG&E)",
    )

    reserve_pct = st.slider(
        "NERC BAL-001 Reserve Margin (%)",
        min_value=3, max_value=15, value=5,
        help="Spinning reserve floor as % of hourly load (NERC BAL-001-2).",
    )

    volatility_pct = st.slider(
        "Price Volatility for Risk Analysis (%)",
        min_value=5, max_value=40, value=20,
        help="Log-normal price shock σ applied in Monte Carlo scenarios.",
    )

    n_scenarios = st.select_slider(
        "Monte Carlo Scenarios",
        options=[100, 250, 500, 1000],
        value=500,
    )

    use_api = st.toggle(
        "Use CAISO OASIS API (requires network)",
        value=False,
        help="Fetches real Day-Ahead LMP data. Disable to use sample duck-curve.",
    )

    st.divider()
    if st.button("🔄 Rerun Optimization", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Source: CAISO OASIS public API — PRC_LMP, DAM")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3_600, show_spinner="Fetching LMP prices from CAISO OASIS…")
def _load_lmp(date_str: str, node_id: str, use_api_flag: bool) -> tuple[np.ndarray, str]:
    if use_api_flag:
        prices = fetch_lmp(date_str, node_id)
        if prices is not None:
            return prices, "CAISO OASIS API (live)"
    return get_sample_lmp(date_str), "Sample Data (duck curve)"


lmp_prices, data_source = _load_lmp(str(op_date), node, use_api)
load_profile = get_sample_load()

portfolio = build_portfolio()
portfolio["solar"].set_availability(get_solar_availability())

reserve_margin = reserve_pct / 100.0

# ── Run optimization ──────────────────────────────────────────────────────────

with st.spinner("Running economic dispatch (PuLP/CBC)…"):
    result = run_dispatch(portfolio, lmp_prices, load_profile, reserve_margin)

schedule: pd.DataFrame = result["schedule"]

with st.spinner(f"Running {n_scenarios:,} Monte Carlo price scenarios…"):
    risk = sensitivity_analysis(
        schedule, lmp_prices, n_scenarios, volatility_pct / 100.0
    )

compliance = build_compliance_report(result, portfolio, reserve_margin)

# ── Header + KPIs ─────────────────────────────────────────────────────────────

st.title("⚡ CAISO Day-Ahead Portfolio Optimization")
st.caption(
    f"Operating date: **{op_date}** · Node: **{node}** · "
    f"Solver: **{result['status']}** · Data: **{data_source}**"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Production Cost", f"${result['cost_usd']:,.0f}")
col2.metric("Portfolio Revenue (LMP)", f"${result['revenue_usd']:,.0f}")
col3.metric(
    "Contribution Margin",
    f"${result['margin_usd']:,.0f}",
    delta=f"${result['margin_usd'] - risk['var_95']:,.0f} VaR-95 buffer",
)
col4.metric(
    "NERC BAL-001 Compliance",
    f"{compliance['summary']['bal001_compliance_pct']}%",
    delta="Violations: " + str(compliance["summary"]["bal001_violations"]),
    delta_color="inverse",
)
col5.metric(
    "Hydro Water Used",
    f"{compliance['summary']['ferc12_water_used_mwh']:,.0f} MWh",
    delta=f"of {compliance['summary']['ferc12_water_budget_mwh']:,.0f} MWh budget",
    delta_color="off",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dispatch Schedule", "⚠️ Risk Analysis", "✅ Compliance", "ℹ️ Resource Details"]
)

# ── Tab 1: Dispatch Schedule ──────────────────────────────────────────────────

with tab1:
    hours = list(range(24))
    he_labels = [f"HE{h+1:02d}" for h in hours]

    fig = go.Figure()

    for r_key in ["hydro", "solar", "gas"]:
        fig.add_trace(go.Bar(
            name=RESOURCE_LABELS[r_key],
            x=he_labels,
            y=schedule[r_key].tolist(),
            marker_color=RESOURCE_COLORS[r_key],
            opacity=0.85,
        ))

    # Overlay LMP price on secondary axis
    fig.add_trace(go.Scatter(
        name="LMP ($/MWh)",
        x=he_labels,
        y=lmp_prices.tolist(),
        mode="lines+markers",
        line=dict(color="#AB47BC", width=2.5, dash="dot"),
        marker=dict(size=5),
        yaxis="y2",
    ))

    # Overlay load profile
    fig.add_trace(go.Scatter(
        name="Load (MW)",
        x=he_labels,
        y=load_profile.tolist(),
        mode="lines",
        line=dict(color="#263238", width=2, dash="dash"),
    ))

    fig.update_layout(
        barmode="stack",
        title="Hourly Economic Dispatch — Optimized Resource Mix",
        xaxis_title="Hour Ending (HE)",
        yaxis_title="Dispatch (MW)",
        yaxis2=dict(title="LMP ($/MWh)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Donut — energy mix
    col_a, col_b = st.columns([1, 2])
    with col_a:
        totals_mwh = {RESOURCE_LABELS[r]: schedule[r].sum() for r in ["hydro", "solar", "gas"]}
        donut = go.Figure(go.Pie(
            labels=list(totals_mwh.keys()),
            values=list(totals_mwh.values()),
            hole=0.55,
            marker_colors=[RESOURCE_COLORS[r] for r in ["hydro", "solar", "gas"]],
        ))
        donut.update_layout(
            title="Daily Energy Mix (MWh)",
            height=320,
            showlegend=True,
            legend=dict(orientation="v"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(donut, use_container_width=True)

    with col_b:
        # Resource status table
        res_rows = []
        for r_key, res in portfolio.items():
            dispatch_mwh = schedule[r_key].sum()
            avg_mw = schedule[r_key].mean()
            cf = avg_mw / res.capacity_mw * 100 if res.capacity_mw > 0 else 0
            at_min = int((schedule[r_key] <= res.min_output_mw + 0.1).sum())
            at_max_count = int((schedule[r_key] >= res.capacity_mw - 0.5).sum())

            ferc_flag = ""
            if r_key == "hydro":
                if compliance["summary"]["ferc12_water_used_mwh"] >= 0.95 * compliance["summary"]["ferc12_water_budget_mwh"]:
                    ferc_flag = "⚠️ Near budget"
                else:
                    ferc_flag = "✅ Within budget"
            else:
                ferc_flag = "N/A"

            ferc764_ok = compliance["summary"]["ferc764_capable"].get(r_key, True)

            res_rows.append({
                "Resource": res.name,
                "Node": res.node,
                "Avg Dispatch (MW)": round(avg_mw, 1),
                "Cap Factor (%)": round(cf, 1),
                "Total Energy (MWh)": round(dispatch_mwh, 1),
                "Hours at Min": at_min,
                "Hours at Max": at_max_count,
                "FERC-12 Status": ferc_flag,
                "FERC-764 Ramp": "✅" if ferc764_ok else "❌",
            })

        st.dataframe(
            pd.DataFrame(res_rows).set_index("Resource"),
            use_container_width=True,
        )

# ── Tab 2: Risk Analysis ──────────────────────────────────────────────────────

with tab2:
    st.subheader("Revenue Distribution Across LMP Scenarios")
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        rev_arr = risk["scenario_revenues"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rev_arr,
            nbinsx=50,
            name="Revenue scenarios",
            marker_color="#42A5F5",
            opacity=0.75,
        ))
        fig_hist.add_vline(
            x=risk["expected_revenue"], line_dash="dash", line_color="#1B5E20",
            annotation_text=f"E[Rev] ${risk['expected_revenue']:,.0f}", annotation_position="top right",
        )
        fig_hist.add_vline(
            x=risk["var_95"], line_dash="dot", line_color="#B71C1C",
            annotation_text=f"VaR-95 ${risk['var_95']:,.0f}", annotation_position="top left",
        )
        fig_hist.add_vline(
            x=risk["cvar_95"], line_dash="solid", line_color="#7B1FA2",
            annotation_text=f"CVaR-95 ${risk['cvar_95']:,.0f}", annotation_position="bottom left",
        )
        fig_hist.update_layout(
            title=f"Daily Revenue Distribution — {n_scenarios:,} Scenarios, σ={volatility_pct}%",
            xaxis_title="Daily Revenue ($)",
            yaxis_title="Count",
            height=380,
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r2:
        st.markdown("**Risk Metrics (Base Scenario)**")
        metrics_df = pd.DataFrame({
            "Metric": ["Expected Revenue", "Std Dev (1σ)", "P25 Revenue", "P75 Revenue", "VaR-95", "CVaR-95 (ES)"],
            "Value ($)": [
                f"${risk['expected_revenue']:>12,.0f}",
                f"${risk['revenue_std']:>12,.0f}",
                f"${risk['revenue_p25']:>12,.0f}",
                f"${risk['revenue_p75']:>12,.0f}",
                f"${risk['var_95']:>12,.0f}",
                f"${risk['cvar_95']:>12,.0f}",
            ],
        }).set_index("Metric")
        st.dataframe(metrics_df, use_container_width=True)

        st.caption(
            "CVaR-95 (Conditional Value at Risk) = average revenue in the "
            "worst 5% of price scenarios. Preferred over VaR for coherent "
            "risk measurement (Artzner et al., 1999)."
        )

    st.subheader("Sensitivity Table — Revenue vs. Price Volatility")
    st.dataframe(
        risk["volatility_table"].style.format("${:,.0f}", subset=["Expected Revenue ($)", "Std Dev ($)", "VaR-95 ($)", "CVaR-95 ($)"]),
        use_container_width=True,
    )

# ── Tab 3: Compliance ─────────────────────────────────────────────────────────

with tab3:
    c_summary = compliance["summary"]
    overall_color = "🟢" if c_summary["overall_status"] == "COMPLIANT" else "🔴"

    st.markdown(
        f"### {overall_color} Overall Status: **{c_summary['overall_status']}**  "
        f"| BAL-001: {c_summary['bal001_compliance_pct']}% compliant "
        f"| CPS1 flags: {c_summary['cps1_flags']} hours "
        f"| FERC-12 utilisation: {c_summary['ferc12_utilisation_pct']}%"
    )

    st.subheader("Hourly Compliance Tracker")
    hourly_df = compliance["hourly"].reset_index()
    st.dataframe(hourly_df, use_container_width=True, height=400)

    st.subheader("FERC Order 764 — 15-min Ramp Adequacy")
    ferc764_rows = [
        {
            "Resource": portfolio[r].name,
            "Ramp Rate (MW/min)": portfolio[r].ramp_rate_mw_min,
            "15-min Capability (MW)": portfolio[r].ramp_rate_mw_min * 15,
            "FERC-764 Adequate": "✅" if v else "❌",
        }
        for r, v in c_summary["ferc764_capable"].items()
    ]
    st.dataframe(pd.DataFrame(ferc764_rows).set_index("Resource"), use_container_width=True)

    st.info(
        "**FERC Order 764** (July 2012) requires balancing authorities and "
        "transmission providers to incorporate 15-minute scheduling to reduce "
        "integration costs for variable renewable resources. Resources that "
        "cannot ramp ±15% of portfolio load within 15 minutes are flagged."
    )

    st.subheader("FERC Part 12 — Hydro Water Release Tracking")
    if "hydro" in portfolio:
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("Water Used Today", f"{c_summary['ferc12_water_used_mwh']:,.1f} MWh")
        col_w2.metric("Daily Budget (FERC license)", f"{c_summary['ferc12_water_budget_mwh']:,.0f} MWh")
        col_w3.metric("Budget Utilisation", f"{c_summary['ferc12_utilisation_pct']:.1f}%")

        # Cumulative hydro generation bar
        cumulative = schedule["hydro"].cumsum().tolist()
        budget_line = [c_summary["ferc12_water_budget_mwh"]] * 24
        fig_water = go.Figure()
        fig_water.add_trace(go.Bar(
            x=[f"HE{h+1:02d}" for h in range(24)],
            y=schedule["hydro"].tolist(),
            name="Hydro Dispatch (MW)",
            marker_color="#1E88E5",
        ))
        fig_water.add_trace(go.Scatter(
            x=[f"HE{h+1:02d}" for h in range(24)],
            y=cumulative,
            name="Cumulative (MWh)",
            line=dict(color="#0D47A1", width=2),
            yaxis="y2",
        ))
        fig_water.add_trace(go.Scatter(
            x=[f"HE{h+1:02d}" for h in range(24)],
            y=budget_line,
            name="FERC Daily Budget",
            line=dict(color="#B71C1C", dash="dash", width=1.5),
            yaxis="y2",
        ))
        fig_water.update_layout(
            title="Hydro Generation vs. FERC Water Release Budget",
            xaxis_title="Hour Ending",
            yaxis_title="Dispatch (MW)",
            yaxis2=dict(title="Cumulative (MWh)", overlaying="y", side="right"),
            height=360,
            plot_bgcolor="#fafafa",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_water, use_container_width=True)

# ── Tab 4: Resource Details ───────────────────────────────────────────────────

with tab4:
    st.subheader("Portfolio Resource Specifications")

    for r_key, res in portfolio.items():
        with st.expander(f"🔧 {res.name} — {res.node}", expanded=False):
            spec_cols = st.columns(3)
            spec_cols[0].metric("Nameplate Capacity", f"{res.capacity_mw:.0f} MW")
            spec_cols[1].metric("Min Output", f"{res.min_output_mw:.0f} MW")
            spec_cols[2].metric("Marginal Cost", f"${res.marginal_cost_per_mwh:.2f}/MWh")
            spec_cols[0].metric("Ramp Rate", f"{res.ramp_rate_mw_min:.1f} MW/min")

            if r_key == "hydro":
                spec_cols[1].metric("FERC Daily Budget", f"{res.daily_energy_budget_mwh:.0f} MWh")  # type: ignore[attr-defined]
                spec_cols[2].metric("FERC Min Flow", f"{res.min_flow_mw:.0f} MW")  # type: ignore[attr-defined]
                st.caption(
                    "FERC licensing constraints: daily water release quota (budget) and "
                    "minimum environmental bypass flow are modelled as hard LP constraints. "
                    "See FERC Part 12 and FERC Order 764."
                )
            elif r_key == "solar":
                st.caption(
                    "Solar PV output is bounded each hour by the irradiance availability profile. "
                    "Zero fuel cost means the optimizer always dispatches solar at max availability "
                    "before committing higher-cost resources (merit-order dispatch)."
                )
            elif r_key == "gas":
                spec_cols[0].metric("Heat Rate", f"{res.heat_rate_mmbtu_per_mwh:.1f} MMBtu/MWh")  # type: ignore[attr-defined]
                spec_cols[1].metric("Fuel Cost", f"${res.fuel_cost_per_mmbtu:.2f}/MMBtu")  # type: ignore[attr-defined]
                spec_cols[2].metric("Startup Cost", f"${res.startup_cost_usd:,.0f}")  # type: ignore[attr-defined]
                st.caption(
                    "Gas peaker dispatches only when LMP is above its marginal cost. "
                    "CAISO Default Energy Bid (DEB) = fuel cost × heat rate + VOM + startup "
                    "amortisation — this is the floor bid price per CAISO tariff Section 39."
                )

    st.divider()
    st.subheader("Hourly Dispatch Detail")
    detail_df = schedule.copy()
    detail_df.index = [f"HE{h+1:02d}" for h in range(24)]
    detail_df["Load (MW)"] = load_profile
    detail_df["LMP ($/MWh)"] = lmp_prices
    detail_df["Revenue ($)"] = schedule.sum(axis=1).values * lmp_prices
    detail_df.columns = [RESOURCE_LABELS.get(c, c) for c in detail_df.columns]
    st.dataframe(detail_df.style.format("{:.1f}"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "CAISO Day-Ahead Portfolio Optimizer · "
    "Regulatory references: FERC Order 764 · FERC Part 12 · "
    "NERC BAL-001-2 · NERC CPS1 · CAISO Tariff Section 39 (DEB) · "
    "WECC RC Standards"
)
