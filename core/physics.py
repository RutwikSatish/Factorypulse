import numpy as np
import pandas as pd

# -------------------- Availability / Capacity --------------------
def availability_from_mtbf_mttr(mtbf_min: float, mttr_min: float) -> float:
    mtbf = max(float(mtbf_min), 1e-6)
    mttr = max(float(mttr_min), 0.0)
    return float(mtbf / (mtbf + mttr))

def effective_process_time_min(pt_min: float, availability: float) -> float:
    A = float(np.clip(float(availability), 1e-6, 1.0))
    return float(max(float(pt_min), 1e-6) / A)

def station_capacity_units_per_hour(pt_eff_min: float) -> float:
    return float(60.0 / max(float(pt_eff_min), 1e-6))

def compute_line_metrics(df: pd.DataFrame, throughput_uph: float) -> dict:
    """
    Compute station-level effective PT, capacity, and utilization given a throughput.
    df must have: Station, Process Time (min/unit), CV, Availability (fraction)
    """
    out = df.copy()

    out["PT_eff (min/unit)"] = [
        effective_process_time_min(pt, A)
        for pt, A in zip(out["Process Time (min/unit)"], out["Availability"])
    ]
    out["Capacity (uph)"] = out["PT_eff (min/unit)"].apply(station_capacity_units_per_hour)

    TH = float(max(float(throughput_uph), 0.0))
    out["Utilization"] = out["Capacity (uph)"].apply(lambda cap: (TH / cap) if cap > 0 else np.nan)

    bottleneck_idx = int(out["Capacity (uph)"].idxmin())
    bottleneck_station = str(out.loc[bottleneck_idx, "Station"])
    bottleneck_cap = float(out.loc[bottleneck_idx, "Capacity (uph)"])
    max_util = float(np.nanmax(out["Utilization"].values)) if len(out) else 0.0

    return {
        "table": out,
        "bottleneck_station": bottleneck_station,
        "bottleneck_capacity": bottleneck_cap,
        "throughput_uph": TH,
        "max_util": max_util,
    }

# -------------------- Queueing-aware Cycle Time (simple + explainable) --------------------
def _variability_factor(cv: float) -> float:
    cv = float(np.clip(cv, 0.0, 3.0))
    return float(cv**2)  # simple proxy

def station_waiting_time_multiplier(u: float, cv: float) -> float:
    """
    Multiplier ≈ 1 + V*(u/(1-u)), V ~ cv^2
    Captures delay explosion as u -> 1 and sensitivity to variability.
    """
    if not np.isfinite(u):
        return 1.0
    u = float(u)
    if u >= 1.0:
        return 999.0
    u = float(np.clip(u, 0.0, 0.999))
    V = _variability_factor(cv)
    return float(1.0 + V * (u / (1.0 - u)))

def estimate_cycle_time_min(metrics: dict) -> dict:
    """
    Estimate line total cycle time (min) by summing station times:
      station_CT ≈ PT_eff * multiplier(utilization, variability)
    """
    table = metrics["table"].copy()
    TH = float(metrics["throughput_uph"])
    if TH <= 0:
        table["CT_station_min"] = np.nan
        return {"CT_total_min": np.nan, "CT_by_station_min": table}

    ct_station = []
    for _, r in table.iterrows():
        pt_eff = float(r["PT_eff (min/unit)"])
        u = float(r["Utilization"])
        cv = float(r.get("CV", 1.0))
        mult = station_waiting_time_multiplier(u, cv)
        ct_station.append(pt_eff * mult)

    table["CT_station_min"] = ct_station
    return {"CT_total_min": float(np.nansum(table["CT_station_min"].values)), "CT_by_station_min": table}

# -------------------- CONWIP / Policy Model --------------------
def compute_policy_results(
    df_engine: pd.DataFrame,
    demand_uph: float,
    policy: str = "PUSH",
    wip_cap_units: float | None = None,
    iters: int = 3,
) -> dict:
    """
    Simple policy model:
    - PUSH: TH = min(demand, bottleneck capacity)
    - CONWIP: TH is additionally limited by WIP cap via Little's Law:
        TH <= WIP_cap / CT(TH)
      We solve approximately with a few fixed-point iterations.

    Returns dict with:
      demand_uph, policy, wip_cap_units, throughput_uph, CT_total_min, metrics(table...)
    """
    demand = float(max(float(demand_uph), 0.0))

    # Start by computing capacities (need any TH just to build table/caps)
    temp_metrics = compute_line_metrics(df_engine, throughput_uph=0.0)
    bn_cap = float(temp_metrics["bottleneck_capacity"])

    # Base capacity-limited TH (push-like)
    TH0 = min(demand, bn_cap) if demand > 0 else 0.0

    if policy.upper() == "PUSH" or wip_cap_units is None or wip_cap_units <= 0:
        metrics = compute_line_metrics(df_engine, throughput_uph=TH0)
        ct_pack = estimate_cycle_time_min(metrics)
        return {
            "policy": "PUSH",
            "wip_cap_units": None,
            "demand_uph": demand,
            "throughput_uph": TH0,
            "CT_total_min": ct_pack["CT_total_min"],
            "metrics": metrics,
            "ct_pack": ct_pack,
        }

    # CONWIP approximate fixed-point:
    W = float(wip_cap_units)
    TH = TH0

    for _ in range(max(1, iters)):
        metrics = compute_line_metrics(df_engine, throughput_uph=TH)
        ct_pack = estimate_cycle_time_min(metrics)
        CT = float(ct_pack["CT_total_min"])

        if not np.isfinite(CT) or CT <= 0:
            break

        # Little's Law bound: WIP = TH * CT  -> TH <= W / CT
        TH_cap = W / CT  # units/min because CT is min/unit? Actually CT is minutes per unit, so W/CT gives units/min.
        TH_cap_uph = TH_cap * 60.0  # convert to units/hour

        TH = min(TH0, TH_cap_uph)

    # Final compute at converged TH
    metrics = compute_line_metrics(df_engine, throughput_uph=TH)
    ct_pack = estimate_cycle_time_min(metrics)

    return {
        "policy": "CONWIP",
        "wip_cap_units": W,
        "demand_uph": demand,
        "throughput_uph": TH,
        "CT_total_min": ct_pack["CT_total_min"],
        "metrics": metrics,
        "ct_pack": ct_pack,
    }

# -------------------- Units + Insights --------------------
def cycle_time_unit_convert(ct_min: float, unit: str) -> float:
    unit = unit.lower().strip()
    if not np.isfinite(ct_min):
        return np.nan
    if unit == "minutes":
        return float(ct_min)
    if unit == "hours":
        return float(ct_min / 60.0)
    if unit == "days":
        return float(ct_min / (60.0 * 24.0))
    return float(ct_min)

def generate_insights(result: dict) -> list[str]:
    policy = result["policy"]
    demand = float(result["demand_uph"])
    TH = float(result["throughput_uph"])
    CT = float(result["CT_total_min"]) if result["CT_total_min"] is not None else np.nan
    metrics = result["metrics"]
    bn = metrics["bottleneck_station"]
    bn_cap = float(metrics["bottleneck_capacity"])
    max_u = float(metrics["max_util"])

    insights = []

    if demand <= 0:
        insights.append("Set a positive demand rate to estimate throughput and cycle time.")
        return insights

    if np.isclose(TH, bn_cap, rtol=1e-3, atol=1e-3) and demand >= bn_cap:
        insights.append(f"Throughput is **capacity-limited** by the bottleneck at **{bn}** (~{bn_cap:.1f} uph).")
    elif demand < bn_cap:
        insights.append(f"Throughput is **demand-limited** (~{TH:.1f} uph).")
    else:
        insights.append(f"Throughput is limited by **policy constraints** (~{TH:.1f} uph).")

    if policy == "CONWIP":
        insights.append("CONWIP stabilizes flow by capping WIP; it can reduce lead time but may cap throughput.")

    if max_u >= 0.92 and max_u < 1.0:
        insights.append("Max utilization is **very high (>92%)** — expect queue/lead-time instability under variability.")
    elif max_u >= 1.0:
        insights.append("Some stations are **overloaded (≥100%)** — not feasible without changes.")
    elif max_u <= 0.70:
        insights.append("Max utilization is moderate — you likely have headroom; verify variability/downtime sensitivity.")

    if np.isfinite(CT):
        if max_u >= 0.92:
            insights.append("Cycle time is elevated primarily due to **delay explosion at high utilization**.")
        else:
            insights.append("Cycle time is driven mostly by processing time + moderate queueing delay.")

    return insights
