import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from core.ui import default_stations, clean_station_df
from core.physics import (
    availability_from_mtbf_mttr,
    compute_policy_results,
    cycle_time_unit_convert,
    generate_insights,
)

st.set_page_config(page_title="FactoryPulse", page_icon="🏭", layout="wide")

# ---- Minimal styling (simple + elegant) ----
st.markdown(
    """
    <style>
      .kpi { padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.08); background: rgba(255,255,255,0.65); }
      .muted { color: rgba(0,0,0,0.55); font-size: 0.92rem; }
      .title { font-size: 1.6rem; font-weight: 750; margin-bottom: 0.2rem; }
      .delta_pos { color: #0a7a2f; font-weight: 650; }
      .delta_neg { color: #a60000; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🏭 FactoryPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Manufacturing line digital twin: bottlenecks, utilization, cycle time, and CONWIP scenario experiments.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("Controls")

    downtime_model = st.radio(
        "Downtime model",
        options=["A) Availability %", "B) MTBF/MTTR"],
        index=0,
    )

    demand_uph = st.number_input(
        "Demand rate (units/hour)",
        min_value=0.0,
        value=40.0,
        step=1.0,
    )

    ct_unit = st.selectbox(
        "Cycle Time unit",
        options=["minutes", "hours", "days"],
        index=0,
    )

    st.divider()
    st.subheader("Policy (Baseline)")
    baseline_policy = st.radio(
        "Baseline policy",
        options=["PUSH", "CONWIP"],
        index=0,
        horizontal=True,
    )
    baseline_wip = st.number_input(
        "Baseline WIP cap (units) — if CONWIP",
        min_value=1.0,
        value=120.0,
        step=5.0,
    )

    st.divider()
    st.subheader("Scenario (Experiment)")
    scenario_policy = st.radio(
        "Scenario policy",
        options=["PUSH", "CONWIP"],
        index=1,
        horizontal=True,
    )
    scenario_wip = st.number_input(
        "Scenario WIP cap (units) — if CONWIP",
        min_value=1.0,
        value=80.0,
        step=5.0,
    )

# ---------------- Line Builder ----------------
st.subheader("Line Builder (5 stations)")
if "stations_df" not in st.session_state:
    st.session_state.stations_df = default_stations()

edited = st.data_editor(
    st.session_state.stations_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
)

stations_df = clean_station_df(edited)

# Compute availability fraction based on downtime model
if downtime_model.startswith("A"):
    stations_df["Availability"] = (stations_df["Availability (%)"] / 100.0).clip(lower=0.01, upper=1.0)
else:
    stations_df["Availability"] = [
        availability_from_mtbf_mttr(mtbf, mttr)
        for mtbf, mttr in zip(stations_df["MTBF (min)"], stations_df["MTTR (min)"])
    ]

engine_df = stations_df[["Station", "Process Time (min/unit)", "CV", "Availability"]].copy()

# ---------------- Compute Baseline + Scenario ----------------
baseline_res = compute_policy_results(
    engine_df,
    demand_uph=demand_uph,
    policy=baseline_policy,
    wip_cap_units=(baseline_wip if baseline_policy == "CONWIP" else None),
    iters=3,
)

scenario_res = compute_policy_results(
    engine_df,
    demand_uph=demand_uph,
    policy=scenario_policy,
    wip_cap_units=(scenario_wip if scenario_policy == "CONWIP" else None),
    iters=3,
)

def fmt_ct(ct_min: float) -> str:
    val = cycle_time_unit_convert(ct_min, ct_unit)
    return "—" if not np.isfinite(val) else f"{val:.2f}"

def delta_fmt(new: float, old: float, is_good_when_lower: bool):
    if not (np.isfinite(new) and np.isfinite(old)):
        return "—"
    d = new - old
    cls = "delta_pos" if (d < 0 and is_good_when_lower) or (d > 0 and not is_good_when_lower) else "delta_neg"
    sign = "+" if d > 0 else ""
    return f"<span class='{cls}'>{sign}{d:.2f}</span>"

# ---------------- KPI Row: Baseline vs Scenario ----------------
st.write("")
st.subheader("Scenario Compare (Baseline vs Experiment)")

b_TH = float(baseline_res["throughput_uph"])
s_TH = float(scenario_res["throughput_uph"])
b_CT_min = float(baseline_res["CT_total_min"]) if baseline_res["CT_total_min"] is not None else np.nan
s_CT_min = float(scenario_res["CT_total_min"]) if scenario_res["CT_total_min"] is not None else np.nan

b_maxu = float(baseline_res["metrics"]["max_util"])
s_maxu = float(scenario_res["metrics"]["max_util"])

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.caption("Throughput (uph)")
    st.subheader(f"{b_TH:.1f} → {s_TH:.1f}")
    st.markdown(f"<div class='muted'>Δ {delta_fmt(s_TH, b_TH, is_good_when_lower=False)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.caption(f"Cycle Time ({ct_unit})")
    b_ct_disp = cycle_time_unit_convert(b_CT_min, ct_unit)
    s_ct_disp = cycle_time_unit_convert(s_CT_min, ct_unit)
    st.subheader(f"{fmt_ct(b_CT_min)} → {fmt_ct(s_CT_min)}")
    st.markdown(f"<div class='muted'>Δ {delta_fmt(s_ct_disp, b_ct_disp, is_good_when_lower=True)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.caption("Max Utilization (%)")
    st.subheader(f"{b_maxu*100:.0f}% → {s_maxu*100:.0f}%")
    st.markdown(f"<div class='muted'>Δ {delta_fmt(s_maxu*100, b_maxu*100, is_good_when_lower=True)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Insights ----------------
st.write("")
left, right = st.columns(2)

with left:
    st.subheader("Baseline Insights")
    for s in generate_insights(baseline_res):
        st.markdown(f"- {s}")
    st.caption(f"Policy: {baseline_res['policy']}" + (f", WIP cap: {baseline_res['wip_cap_units']:.0f}" if baseline_res["policy"] == "CONWIP" else ""))

with right:
    st.subheader("Scenario Insights")
    for s in generate_insights(scenario_res):
        st.markdown(f"- {s}")
    st.caption(f"Policy: {scenario_res['policy']}" + (f", WIP cap: {scenario_res['wip_cap_units']:.0f}" if scenario_res["policy"] == "CONWIP" else ""))

# ---------------- Compact Charts ----------------
st.write("")
st.subheader("Quick Charts (compact)")

ch1, ch2 = st.columns(2)

# Utilization bars baseline vs scenario
with ch1:
    b_tab = baseline_res["metrics"]["table"].copy()
    s_tab = scenario_res["metrics"]["table"].copy()
    dfu = pd.DataFrame({
        "Station": b_tab["Station"],
        "Baseline Util (%)": (b_tab["Utilization"] * 100.0).round(1),
        "Scenario Util (%)": (s_tab["Utilization"] * 100.0).round(1),
    })
    dfu_m = dfu.melt(id_vars=["Station"], var_name="Case", value_name="Utilization (%)")
    fig_u = px.bar(dfu_m, x="Station", y="Utilization (%)", color="Case", barmode="group", title="Utilization by Station (%)")
    fig_u.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_u, use_container_width=True)

# Cycle time bar baseline vs scenario
with ch2:
    dfct = pd.DataFrame({
        "Case": ["Baseline", "Scenario"],
        f"CT ({ct_unit})": [
            cycle_time_unit_convert(b_CT_min, ct_unit),
            cycle_time_unit_convert(s_CT_min, ct_unit),
        ],
    })
    fig_ct = px.bar(dfct, x="Case", y=f"CT ({ct_unit})", title=f"Cycle Time ({ct_unit}) — Compare")
    fig_ct.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig_ct, use_container_width=True)

# ---------------- Detail Tables ----------------
st.write("")
st.subheader("Station Results (Baseline)")
b_out = baseline_res["metrics"]["table"].copy()
b_out["Availability (%)"] = (b_out["Availability"] * 100.0).round(1)
b_out["Utilization (%)"] = (b_out["Utilization"] * 100.0).round(1)
b_out["CT_station (min)"] = baseline_res["ct_pack"]["CT_by_station_min"]["CT_station_min"].round(3)

st.dataframe(
    b_out[["Station", "Process Time (min/unit)", "Availability (%)", "PT_eff (min/unit)", "Capacity (uph)", "Utilization (%)", "CV", "CT_station_min"]],
    use_container_width=True
)

st.subheader("Station Results (Scenario)")
s_out = scenario_res["metrics"]["table"].copy()
s_out["Availability (%)"] = (s_out["Availability"] * 100.0).round(1)
s_out["Utilization (%)"] = (s_out["Utilization"] * 100.0).round(1)
s_out["CT_station (min)"] = scenario_res["ct_pack"]["CT_by_station_min"]["CT_station_min"].round(3)

st.dataframe(
    s_out[["Station", "Process Time (min/unit)", "Availability (%)", "PT_eff (min/unit)", "Capacity (uph)", "Utilization (%)", "CV", "CT_station_min"]],
    use_container_width=True
)
