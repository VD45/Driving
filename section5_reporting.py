#!/usr/bin/env python3
"""
Utilities to build Section 5 tables based on the latest unified analysis output.

The functions here derive all metrics directly from the JSON results and the
exported custom cycles so that the resulting CSV tables and textual summaries
always reflect the current dataset without relying on hard-coded exemplar values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from unified_wavelet_module import UnifiedWaveletAnalyzer

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ML" / "outputs"
EXPORT_DIR = OUTPUT_DIR / "exported_cycles"
REPORTS_DIR = OUTPUT_DIR / "reports"
CYCLES_DIR = PROJECT_ROOT / "Data" / "standardized_cycles"

STOP_SPEED_THRESHOLD_MS = 1.0
DEFAULT_TOP_ROUTES = 5

# Column descriptors for the generated tables (kept in a single place so that
# downstream scripts can import them without duplicating strings).
TABLE2_COLUMNS = [
    "Route",
    "Trips",
    "Duration (s)",
    "Distance (km)",
    "Mean Speed (km/h)",
    "V95 (km/h)",
    "Max Speed (km/h)",
    "Idle (%)",
    "PKE (m/s²)",
    "Chaos (%)",
    "Max Accel (m/s²)",
    "Max Decel (m/s²)",
    "Stops",
    "Stops/km",
    "Distribution",
    "Wavelet Entropy",
    "Short Band (%)",
    "Medium Band (%)",
    "Long Band (%)",
    "Transient Events min-1",
    "Predictability",
    "Compliance",
    "RPA (m/s²)",
    "Jerk RMS (m/s³)",
]

TABLE2_SUMMARY_SPEC: List[Tuple[str, Callable[[pd.Series], str]]] = [
    ("Route", lambda row: str(row["Route"])),
    ("Trips", lambda row: str(int(row["Trips"])) if pd.notna(row["Trips"]) else "n/a"),
    (
        "Distance (km)",
        lambda row: f"{row['Distance (km)']:.2f}"
        if pd.notna(row["Distance (km)"])
        else "n/a",
    ),
    (
        "Mean Speed (km/h)",
        lambda row: f"{row['Mean Speed (km/h)']:.1f}"
        if pd.notna(row["Mean Speed (km/h)"])
        else "n/a",
    ),
    (
        "Idle (%)",
        lambda row: f"{row['Idle (%)']:.1f}"
        if pd.notna(row["Idle (%)"])
        else "n/a",
    ),
    (
        "PKE (m/s²)",
        lambda row: f"{row['PKE (m/s²)']:.2f}"
        if pd.notna(row["PKE (m/s²)"])
        else "n/a",
    ),
    (
        "Chaos (%)",
        lambda row: f"{row['Chaos (%)']:.1f}"
        if pd.notna(row["Chaos (%)"])
        else "n/a",
    ),
    (
        "Wavelet Entropy",
        lambda row: f"{row['Wavelet Entropy']:.4f}"
        if pd.notna(row["Wavelet Entropy"])
        else "n/a",
    ),
]

TABLE3_COLUMNS = [
    "Metric",
    "Standard Mean",
    "Real Urban Mean",
    "Ratio (Real/Standard)",
]


@dataclass
class TableContext:
    """Aggregated information returned alongside the Section 5 tables."""

    route_ids: List[str]
    route_labels: List[str]
    trip_counts: Dict[str, int]
    real_metrics: Dict[str, float]
    standard_metrics: Dict[str, float]
    ratios: Dict[str, float]
    closest_standard: Optional[str]
    closest_standard_display: Optional[str]
    wavelet_events_mean: float
    total_distance_km: float


def _standardize_route_label(route_id: str) -> str:
    """Convert route identifiers to human-readable labels."""
    base = route_id
    variant_suffix = ""
    if "_v" in route_id:
        base, variant_suffix = route_id.split("_v", maxsplit=1)

    try:
        route_num = int(base.split("_")[-1]) + 1
        label = f"{route_num}"
    except ValueError:
        label = route_id

    if variant_suffix:
        label += f" (v{variant_suffix})"
    return label


def _load_time_normalized_cycle(
    results: Dict, route_id: str
) -> Optional[pd.DataFrame]:
    """Return the time-normalised representative cycle for a route."""
    route_data = results.get("route_variability", {}).get(route_id)
    if not route_data:
        return None

    cycle_dict = (
        route_data.get("custom_cycles", {}).get("time_normalized") or {}
    )
    if cycle_dict:
        df = pd.DataFrame(cycle_dict)
        if not df.empty and "time_s" in df.columns:
            return df.sort_values("time_s").reset_index(drop=True)

    csv_path = EXPORT_DIR / f"{route_id}_time_normalized_cycle.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty and "time_s" in df.columns:
            return df.sort_values("time_s").reset_index(drop=True)

    return None


def _compute_sampling_interval(time_axis: np.ndarray) -> float:
    """Estimate a representative sampling interval for a time axis."""
    if time_axis.size < 2:
        return 1.0
    diffs = np.diff(time_axis)
    valid = diffs[(diffs > 0) & np.isfinite(diffs)]
    if valid.size == 0:
        return 1.0
    return float(np.median(valid))


def _prepare_dt_arrays(time_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return per-interval and per-sample step sizes plus the total duration."""
    if time_axis.size == 0:
        return np.array([1.0]), np.array([1.0]), 1.0

    if time_axis.size == 1:
        duration = float(max(time_axis[0], 1.0))
        return np.array([duration]), np.array([duration]), duration

    interval_dt = np.diff(time_axis)
    valid = (interval_dt > 0) & np.isfinite(interval_dt)
    if not np.any(valid):
        fallback = 1.0
        interval_dt = np.full(time_axis.size - 1, fallback, dtype=float)
    else:
        fallback = float(np.median(interval_dt[valid]))
        interval_dt = np.where(valid, interval_dt, fallback)

    per_sample_dt = np.concatenate([interval_dt, interval_dt[-1:]])
    duration = float(max(time_axis[-1] - time_axis[0], fallback))
    return interval_dt, per_sample_dt, duration


def _compute_route_metrics(
    route_id: str,
    trip_count: int,
    results: Dict,
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """Compute the metrics for a single route used in Table 2/3."""
    route_data = results.get("route_variability", {}).get(route_id)
    cycle_df = _load_time_normalized_cycle(results, route_id)
    if route_data is None or cycle_df is None or cycle_df.empty:
        return None

    required_cols = {"time_s", "speed_ms"}
    if not required_cols.issubset(set(cycle_df.columns)):
        return None

    time_axis = cycle_df["time_s"].to_numpy(dtype=float)
    speed_ms = cycle_df["speed_ms"].to_numpy(dtype=float)
    speed_kmh = (
        cycle_df["speed_kmh"].to_numpy(dtype=float)
        if "speed_kmh" in cycle_df.columns
        else speed_ms * 3.6
    )

    interval_dt, per_sample_dt, duration = _prepare_dt_arrays(time_axis)
    if duration <= 0:
        duration = float(len(time_axis)) * 0.1

    # Distance is the integral of speed over time. Use left Riemann sum for stability.
    if speed_ms.size > 1:
        distance_m = float(np.sum(speed_ms[:-1] * interval_dt))
    else:
        distance_m = float(speed_ms[0] * duration)
    distance_km = distance_m / 1000.0

    mean_speed_kmh = (
        distance_m / duration * 3.6 if duration > 0 else float(np.nan)
    )
    v95_kmh = float(np.percentile(speed_kmh, 95)) if speed_kmh.size else 0.0
    max_speed_kmh = float(np.max(speed_kmh)) if speed_kmh.size else 0.0

    stop_summary = route_data.get("stop_summary", {})
    stop_idle_pct = stop_summary.get("mean_idle_ratio_pct")
    if stop_idle_pct is not None:
        idle_pct = float(stop_idle_pct)
    else:
        idle_mask = speed_ms <= STOP_SPEED_THRESHOLD_MS
        idle_time = float(
            np.sum(per_sample_dt[idle_mask]) if per_sample_dt.size else 0.0
        )
        idle_pct = idle_time / duration * 100.0 if duration > 0 else 0.0

    if interval_dt.size and speed_ms.size > 1:
        acceleration = np.diff(speed_ms) / interval_dt
    else:
        acceleration = np.array([0.0])

    max_accel = float(np.max(acceleration)) if acceleration.size else 0.0
    max_decel = float(
        np.abs(np.min(acceleration)) if acceleration.size else 0.0
    )
    positive_acc = acceleration[acceleration > 0]
    pke = float(np.sum(np.square(positive_acc))) if positive_acc.size else 0.0

    overall = route_data.get("overall_metrics", {})
    chaos_pct = float(overall.get("route_chaos_score", 0.0)) * 100.0
    predictability = float(overall.get("route_predictability", 0.0))
    compliance = float(overall.get("route_speed_limit_compliance", 0.0))
    rpa = float(
        overall.get(
            "route_rpa", route_data.get("advanced_metrics", {}).get("route_rpa", 0.0)
        )
    )
    jerk_rms = float(
        overall.get(
            "route_jerk_rms",
            route_data.get("advanced_metrics", {}).get("route_jerk_rms", 0.0),
        )
    )

    stops = float(
        stop_summary.get("mean_stops_per_trip")
        or stop_summary.get("median_stops_per_trip")
        or 0.0
    )
    stops_per_km = stops / distance_km if distance_km > 0 else 0.0

    distribution = (
        route_data.get("chaos_classification", "unknown")
        .replace("_", " ")
        .title()
    )

    sampling_interval = _compute_sampling_interval(time_axis)
    sampling_rate = 1.0 / sampling_interval if sampling_interval > 0 else 1.0
    wavelet_analyzer = UnifiedWaveletAnalyzer(sampling_rate_hz=sampling_rate)
    wavelet_results = wavelet_analyzer.analyze_cycle(
        speed_kmh,
        cycle_name=route_id,
        sampling_rate=sampling_rate,
    )
    wave_entropy = float(wavelet_results.get("wavelet_entropy", 0.0))
    band_percentages = wavelet_results.get("cwt", {}).get("band_percentages", {})
    short_pct = float(band_percentages.get("short", 0.0))
    medium_pct = float(band_percentages.get("medium", 0.0))
    long_pct = float(band_percentages.get("long", 0.0))
    transient_events = float(
        wavelet_results.get("events", {}).get("events_per_minute", 0.0)
    )

    route_label = _standardize_route_label(route_id)
    row = {
        "Route": route_label,
        "Trips": int(trip_count),
        "Duration (s)": round(duration, 1),
        "Distance (km)": round(distance_km, 3),
        "Mean Speed (km/h)": round(mean_speed_kmh, 1),
        "V95 (km/h)": round(v95_kmh, 1),
        "Max Speed (km/h)": round(max_speed_kmh, 1),
        "Idle (%)": round(idle_pct, 1),
        "PKE (m/s²)": round(pke, 2),
        "Chaos (%)": round(chaos_pct, 1),
        "Max Accel (m/s²)": round(max_accel, 2),
        "Max Decel (m/s²)": round(max_decel, 2),
        "Stops": round(stops, 1),
        "Stops/km": round(stops_per_km, 2),
        "Distribution": distribution,
        "Wavelet Entropy": round(wave_entropy, 4),
        "Short Band (%)": round(short_pct, 2),
        "Medium Band (%)": round(medium_pct, 2),
        "Long Band (%)": round(long_pct, 2),
        "Transient Events min-1": round(transient_events, 2),
        "Predictability": round(predictability, 3),
        "Compliance": round(compliance, 3),
        "RPA (m/s²)": round(rpa, 3),
        "Jerk RMS (m/s³)": round(jerk_rms, 2),
    }

    aggregates = {
        "chaos_index": chaos_pct / 100.0,
        "idle_pct": idle_pct,
        "pke": pke,
        "wavelet_entropy": wave_entropy,
        "events_per_minute": transient_events,
        "distance_km": distance_km,
        "stops": stops,
    }

    return row, aggregates


def _format_standard_cycle_name(cycle_name: str) -> str:
    """Create a human-friendly label for a standard cycle identifier."""
    tokens = cycle_name.split("_")
    if len(tokens) <= 1:
        return cycle_name
    family = tokens[0]
    descriptor = " ".join(tokens[1:]).replace("-", " ")
    return f"{descriptor} ({family})"


def _load_standard_cycle_dataframe(cycle_name: str) -> Optional[pd.DataFrame]:
    """Locate and load a standard cycle file by its canonical key."""
    if not cycle_name:
        return None

    cycle_lower = cycle_name.lower()
    tokens = {cycle_lower}
    if "_" in cycle_lower:
        tokens.add(cycle_lower.split("_", 1)[-1])

    def load_candidate(path: Path) -> Optional[pd.DataFrame]:
        if path.suffix.lower() not in {".csv", ".scv", ".parquet"}:
            return None
        if path.suffix.lower() in {".csv", ".scv"}:
            df = pd.read_csv(path, comment="#", encoding="utf-8-sig")
        else:
            df = pd.read_parquet(path)

        for col in df.columns:
            col_lower = col.lower()
            if "speed" in col_lower or "vehicle" in col_lower:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if series.empty:
                    continue
                if series.max() > 60:
                    return pd.DataFrame(
                        {
                            "time_s": np.arange(len(series), dtype=float),
                            "speed_kmh": series.to_numpy(),
                            "speed_ms": series.to_numpy() / 3.6,
                        }
                    )
                return pd.DataFrame(
                    {
                        "time_s": np.arange(len(series), dtype=float),
                        "speed_kmh": series.to_numpy() * 3.6,
                        "speed_ms": series.to_numpy(),
                    }
                )
        return None

    for candidate in CYCLES_DIR.rglob("*"):
        if not candidate.is_file():
            continue
        key = f"{candidate.parent.name.lower()}_{candidate.stem.lower()}"
        if any(token and token in key for token in tokens):
            loaded = load_candidate(candidate)
            if loaded is not None:
                return loaded
    return None


def _compute_standard_metrics(
    results: Dict, cycle_name: Optional[str]
) -> Dict[str, float]:
    """Extract comparable metrics for the closest standard cycle."""
    metrics: Dict[str, float] = {
        "chaos_index": float("nan"),
        "idle_pct": float("nan"),
        "pke": float("nan"),
        "wavelet_entropy": float("nan"),
    }

    if not cycle_name:
        return metrics

    std_metrics = (
        results.get("standard_cycles_comprehensive", {})
        .get("metrics", {})
        .get(cycle_name, {})
    )
    df = _load_standard_cycle_dataframe(cycle_name)
    if std_metrics:
        metrics["chaos_index"] = float(std_metrics.get("chaos_index", np.nan))
    if df is not None and not df.empty:
        time_axis = df["time_s"].to_numpy(dtype=float)
        speed_ms = df["speed_ms"].to_numpy(dtype=float)
        interval_dt, per_sample_dt, duration = _prepare_dt_arrays(time_axis)
        if duration > 0:
            idle_mask = speed_ms <= STOP_SPEED_THRESHOLD_MS
            idle_time = float(
                np.sum(per_sample_dt[idle_mask]) if per_sample_dt.size else 0.0
            )
            metrics["idle_pct"] = (idle_time / duration) * 100.0
        else:
            metrics["idle_pct"] = np.nan

        if interval_dt.size and speed_ms.size > 1:
            acceleration = np.diff(speed_ms) / interval_dt
            positive = acceleration[acceleration > 0]
            metrics["pke"] = (
                float(np.sum(np.square(positive))) if positive.size else 0.0
            )
        else:
            metrics["pke"] = 0.0

    if df is not None and not df.empty:
        sampling_rate = 10.0 if len(df) > 5000 else 1.0
        analyzer = UnifiedWaveletAnalyzer(sampling_rate_hz=sampling_rate)
        wavelet = analyzer.analyze_cycle(
            df["speed_kmh"].to_numpy(dtype=float),
            cycle_name=cycle_name,
            sampling_rate=sampling_rate,
        )
        metrics["wavelet_entropy"] = float(wavelet.get("wavelet_entropy", np.nan))
    return metrics


def build_section5_tables(
    results: Dict,
    top_n: int = DEFAULT_TOP_ROUTES,
) -> Tuple[pd.DataFrame, pd.DataFrame, TableContext]:
    """
    Construct the Section 5 tables directly from analysis results.

    Returns
    -------
    table2 : pd.DataFrame
        Route-level summary (Table 2).
    table3 : pd.DataFrame
        Aggregate comparison between the closest standard cycle and real routes.
    context : TableContext
        Additional metadata useful for narrative summaries.
    """
    supported = results.get("routes", {}).get("supported_routes", {})
    if not supported:
        raise ValueError("No supported routes available to build Section 5 tables.")

    sorted_routes = sorted(
        supported.items(), key=lambda item: item[1], reverse=True
    )
    if top_n > 0:
        sorted_routes = sorted_routes[:top_n]

    rows: List[Dict[str, float]] = []
    aggregates: List[Dict[str, float]] = []

    for route_id, trip_count in sorted_routes:
        metrics = _compute_route_metrics(route_id, trip_count, results)
        if metrics is None:
            continue
        row, agg = metrics
        # Ensure consistent column ordering, fill missing columns as needed.
        for col in TABLE2_COLUMNS:
            row.setdefault(col, np.nan)
        rows.append(row)
        aggregates.append(agg)

    if not rows:
        raise ValueError("Unable to compute Section 5 tables – no route metrics available.")

    table2 = pd.DataFrame(rows, columns=TABLE2_COLUMNS)

    real_metrics = {
        key: float(np.nanmean([agg.get(key, np.nan) for agg in aggregates]))
        for key in ["chaos_index", "idle_pct", "pke", "wavelet_entropy"]
    }
    mean_events = float(
        np.nanmean([agg.get("events_per_minute", np.nan) for agg in aggregates])
    )
    total_distance_km = float(
        np.nansum([agg.get("distance_km", 0.0) for agg in aggregates])
    )

    closest_standard = None
    closest_display = None
    best_matches = results.get("cycle_comparison", {}).get("best_matches", [])
    if best_matches:
        closest_standard = best_matches[0][0]
        closest_display = _format_standard_cycle_name(closest_standard)

    standard_metrics = _compute_standard_metrics(results, closest_standard)
    comparison = results.get("wavelet_analysis", {}).get("comparison", {})
    if comparison:
        if comparison.get("real_avg_entropy") is not None:
            real_metrics["wavelet_entropy"] = float(comparison["real_avg_entropy"])
        if comparison.get("standard_avg_entropy") is not None:
            standard_metrics["wavelet_entropy"] = float(
                comparison["standard_avg_entropy"]
            )
        if comparison.get("real_events_per_min") is not None:
            mean_events = float(comparison["real_events_per_min"])

    ratios = {}
    for key, real_val in real_metrics.items():
        std_val = standard_metrics.get(key, np.nan)
        if std_val and not np.isnan(std_val):
            ratios[key] = real_val / std_val
        else:
            ratios[key] = np.nan

    table3_rows = []
    metric_labels = {
        "chaos_index": "Chaos Index",
        "idle_pct": "Idle (%)",
        "pke": "PKE (m/s²)",
        "wavelet_entropy": "Wavelet Entropy",
    }
    for key, label in metric_labels.items():
        table3_rows.append(
            {
                "Metric": label,
                "Standard Mean": round(standard_metrics.get(key, np.nan), 5),
                "Real Urban Mean": round(real_metrics.get(key, np.nan), 5),
                "Ratio (Real/Standard)": round(ratios.get(key, np.nan), 3)
                if not np.isnan(ratios.get(key, np.nan))
                else np.nan,
            }
        )

    table3 = pd.DataFrame(table3_rows, columns=TABLE3_COLUMNS)

    context = TableContext(
        route_ids=[rid for rid, _ in sorted_routes[: len(table2)]],
        route_labels=table2["Route"].tolist(),
        trip_counts={rid: int(cnt) for rid, cnt in sorted_routes},
        real_metrics=real_metrics,
        standard_metrics=standard_metrics,
        ratios=ratios,
        closest_standard=closest_standard,
        closest_standard_display=closest_display,
        wavelet_events_mean=mean_events,
        total_distance_km=total_distance_km,
    )

    return table2, table3, context


def write_section5_tables(
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    timestamp: str,
) -> Dict[str, Path]:
    """Persist the computed tables for archival and publication purposes."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    table2_path = REPORTS_DIR / f"table2_urban_routes_{timestamp}.csv"
    table3_path = REPORTS_DIR / f"table3_standard_vs_real_{timestamp}.csv"

    table2.to_csv(table2_path, index=False)
    table3.to_csv(table3_path, index=False)

    return {"table2": table2_path, "table3": table3_path}


def iter_table2_summary(table2: pd.DataFrame) -> List[str]:
    """Yield markdown table rows for the summary subsection derived from Table 2."""
    if table2 is None or table2.empty:
        return []

    headers = [label for label, _ in TABLE2_SUMMARY_SPEC]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("-" * max(len(label), 3) for label in headers) + " |"

    rows: List[str] = []
    for _, series in table2.iterrows():
        formatted_values = [
            formatter(series) if callable(formatter) else "n/a"
            for _, formatter in TABLE2_SUMMARY_SPEC
        ]
        rows.append("| " + " | ".join(formatted_values) + " |")

    return [header_line, separator_line, *rows]
