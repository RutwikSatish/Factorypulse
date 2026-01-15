import numpy as np
import pandas as pd

def availability_from_mtbf_mttr(mtbf_min: float, mttr_min: float) -> float:
    mtbf = max(float(mtbf_min), 1e-6)
    mttr = max(float(mttr_min), 0.0)
    return float(mtbf / (mtbf + mttr))

def effective_process_time_min(pt_min: float, availability: float) -> float:
    A = float(np.clip(float(availability), 1e-6, 1.0))
    return float(max(float(pt_min), 1e-6) / A)

def station_capacity_units_per_hour(pt_eff_min: float) -> float:
    return float(60.0 / max(float(pt_eff_min), 1e-6))

def compute_line_metrics(df: pd.DataFrame, demand_rate_uph: float) -> dict:
    out = df.copy()

    out["PT_eff (min/unit)"] = [
        effective_process_time_min(pt, A)
        for pt, A in zip(out["Process Time (min/unit)"], out["Availability"])
    ]
    out["Capacity (uph)"] = out["PT_eff (min/unit)"].apply(station_capacity_units_per_hour)

    bottleneck_idx = int(out["Capacity (uph)"].idxmin())
    bottleneck_station = str(out.loc[bottleneck_idx, "Station"])
    bottleneck_cap = float(out.loc[bottleneck_idx, "Capacity (uph)"])

    demand = float(max(float(demand_rate_uph), 0.0))
    TH = min(demand, bottleneck_cap) if demand > 0 else 0.0

    out["Utilization"] = out["Capacity (uph)"].apply(lambda cap: (TH / cap) if cap > 0 else np.nan)
    out["Utilization"] = out["Utilization"].astype(float)

    max_util = float(np.nanmax(out["Utilization"].values)) if len(out) else 0.0

    return {
        "table": out,
        "bottleneck_station": bottleneck_station,
        "bottleneck_capacity": bottleneck_cap,
        "throughput_uph": TH,
        "demand_uph": demand,
        "max_util": max_util,
    }

# ---------------- Queueing-aware Cycle Time (simple + explainable) ----------------
def _variability_factor(cv: float) -> float:
    """
    Variability factor ~ (Ca^2 + Cs^2)/2.
    We treat CV as proxy for both arrival and service variability.
    cv in [0,3] -> squared in [0,9].
    """
    cv = float(np.clip(cv, 0.0, 3.0))
    return 0.5 * (cv**2 + cv**2)  # = cv^2

def station_waiting_time_multiplier(u: float, cv: float) -> float:
    """
    A compact, explainable waiting time multiplier:
    WT multiplier ≈ 1 + V * (u/(1-u)), where V ~ cv^2.
    - As u -> 1, explosion occurs.
    - Higher variability increases waiting.
    """
    u = float(u)
    if not np.isfinite(u):
        return 1.0
    # Allow overload flagging
    if u >= 1.0:
        return 999.0
    u = float(np.clip(u, 0.0, 0.999))
    V = _variability_factor(cv)
    return float(1.0 + V * (u / (1.0 - u)))

def estimate_cycle_time_min(metrics: dict) -> dict:
    """
    Estimate total cycle time (min) for the line by summing station effective
    processing times + queueing delays (via multiplier).
    """
    table = metrics["table"].copy()
    TH = float(metrics["throughput_uph"])

    # If no throughput, CT undefined
    if TH <= 0:
        return {"CT_total_min": np.nan, "CT_by_station_min": table.assign(CT_station_min=np.nan)}

    ct_station = []
    for _, row in table.iterrows():
        pt_eff = float(row["PT_eff (min/unit)"])
        u = float(row["Utilization"])
        cv = float(row.get("CV", 1.0))
        mult = station_waiting_time_multiplier(u, cv)
        # Total station time = processing * multiplier (processing + implied waiting)
        ct_station.append(pt_eff * mult)

    table["CT_station_min"] = ct_station
    CT_total = float(np.nansum(table["CT_station_min"].values))

    return {"CT_total_min": CT_total, "CT_by_station_min": table}

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

def generate_insights(metrics: dict, ct_total_min: float) -> list[str]:
    TH = float(metrics["throughput_uph"])
    demand = float(metrics["demand_uph"])
    bn = metrics["bottleneck_station"]
    bn_cap = float(metrics["bottleneck_capacity"])
    max_u = float(metrics["max_util"])

    insights = []

    if demand <= 0:
        insights.append("Set a positive demand rate to estimate throughput and cycle time.")
        return insights

    if np.isclose(TH, bn_cap, rtol=1e-3, atol=1e-3) and demand >= bn_cap:
        insights.append(f"Throughput is **capacity-limited** by the bottleneck at **{bn}** (~{bn_cap:.1f} units/hr).")
        insights.append("Improving non-bottleneck stations won’t increase throughput until the bottleneck is elevated.")
    else:
        insights.append(f"Throughput is **demand-limited** (~{TH:.1f} units/hr). Current capacity can support demand.")

    if max_u >= 0.92 and max_u < 1.0:
        insights.append("Max utilization is **very high (>92%)**. With variability, expect **lead-time instability**.")
    elif max_u >= 1.0:
        insights.append("Some stations are **overloaded (utilization ≥ 100%)**. This plan is **not feasible** without changes.")
    elif max_u <= 0.70:
        insights.append("Max utilization is moderate. You likely have headroom; verify variability and downtime sensitivity.")

    if np.isfinite(ct_total_min):
        if max_u >= 0.92:
            insights.append("Cycle time is elevated primarily due to **queueing delay at high utilization** (delay explosion).")
        else:
            insights.append("Cycle time is driven mostly by processing time + moderate queueing delay.")

    insights.append("Next upgrades: CONWIP (WIP caps), scenario compare, and automation ROI experiments.")
    return insights
