#!/usr/bin/env python3
"""
Enhanced route variability analysis with additional metrics and proper JSON serialization.
Fixed: Custom cycles now properly serialize to JSON as dictionaries with full data.
"""

import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import entropy, wasserstein_distance, ks_2samp
import warnings
import zlib

warnings.filterwarnings('ignore')

STOP_SPEED_THRESHOLD_MS = 1.0
STOP_MIN_DWELL_S = 1.0
STOP_SMOOTH_WINDOW_S = 1.2
STOP_GAP_CLOSE_S = 3.0
STOP_SEGMENT_MIN_S = 8.0
SEGMENT_RESAMPLE_POINTS = 100
STOP_ALIGNMENT_MIN_TRIPS = 4
STOP_ALIGNMENT_MIN_STOPS = 2
STOP_ALIGNMENT_TIME_STEP_S = 0.5
STOP_RESAMPLE_POINTS = 32
EDGE_STOP_PROB_THRESHOLD = 0.6
EDGE_SPEED_THRESHOLD_MS = 1.0
EDGE_BUFFER_S = 4.0
EDGE_IDLE_MAX_FRAC = 0.08
EDGE_IDLE_MAX_S = 25.0
STOP_PATTERN_WINDOW = 0.12


def _rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    return (
        pd.Series(series, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def compute_stop_probability_from_profiles(
    aligned_profiles: np.ndarray,
    target_duration_s: float,
    speed_threshold_ms: float = STOP_SPEED_THRESHOLD_MS,
    smooth_window_s: float = STOP_SMOOTH_WINDOW_S,
    dwell_s: float = STOP_MIN_DWELL_S,
    gap_close_s: float = STOP_GAP_CLOSE_S,
) -> np.ndarray:
    """
    Compute stop probability over aligned time samples with enforced dwell requirements.
    """
    if aligned_profiles is None:
        return np.array([])
    arr = np.asarray(aligned_profiles, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.array([])

    n_samples = arr.shape[1]
    if n_samples == 0:
        return np.array([])

    dt_est = float(target_duration_s) / float(max(n_samples - 1, 1))
    dt_est = max(dt_est, 1e-3)
    smooth_window = max(1, int(round(smooth_window_s / dt_est)))
    dwell_window = max(1, int(round(dwell_s / dt_est)))
    gap_window = max(1, int(round(gap_close_s / dt_est)))
    stop_masks = []

    for profile in arr:
        smoothed = _rolling_mean(profile, smooth_window)
        mask = smoothed <= speed_threshold_ms
        if dwell_window > 1:
            dwell_scores = _rolling_mean(mask.astype(float), dwell_window)
            mask = dwell_scores >= 0.8
        if gap_window > 1:
            mask = binary_closing(mask, structure=np.ones(gap_window, dtype=bool))
        stop_masks.append(mask.astype(float))

    stop_prob = np.clip(np.mean(stop_masks, axis=0), 0.0, 1.0)
    return stop_prob


def build_stop_mask(
    speeds_ms: np.ndarray,
    dt_series: np.ndarray,
    speed_threshold_ms: float = STOP_SPEED_THRESHOLD_MS,
    smooth_window_s: float = STOP_SMOOTH_WINDOW_S,
    dwell_s: float = STOP_MIN_DWELL_S,
    gap_close_s: float = STOP_GAP_CLOSE_S,
) -> np.ndarray:
    """
    Build a boolean stop mask for a single trip speed trace with explicit dwell and smoothing.
    """
    if speeds_ms is None or len(speeds_ms) == 0:
        return np.array([])
    if dt_series is None or len(dt_series) == 0:
        dt_series = np.full_like(speeds_ms, 0.5, dtype=float)
    dt_mean = float(np.clip(np.nanmedian(dt_series), 1e-3, None))
    smooth_window = max(1, int(round(smooth_window_s / dt_mean)))
    dwell_window = max(1, int(round(dwell_s / dt_mean)))
    gap_window = max(1, int(round(gap_close_s / dt_mean)))

    smoothed = _rolling_mean(speeds_ms, smooth_window)
    stop_mask = smoothed <= speed_threshold_ms
    if dwell_window > 1:
        dwell_scores = _rolling_mean(stop_mask.astype(float), dwell_window)
        stop_mask = dwell_scores >= 0.8
    if gap_window > 1:
        stop_mask = binary_closing(stop_mask, structure=np.ones(gap_window, dtype=bool))
    return stop_mask.astype(bool)


def mask_to_index_spans(mask: np.ndarray, min_samples: int = 1) -> List[Tuple[int, int]]:
    """
    Convert a boolean mask into contiguous index spans that satisfy a minimum length.
    """
    if mask is None or len(mask) == 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask.astype(bool)):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_samples:
                spans.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_samples:
        spans.append((start, len(mask)))
    return spans


def _ensure_length(arr: np.ndarray, target_len: int, pad_value: float) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_width = target_len - len(arr)
    return np.pad(arr, (0, pad_width), constant_values=pad_value)


def _segment_time_axis(length: int, total_duration: float) -> np.ndarray:
    if length <= 1 or total_duration <= 0:
        return np.linspace(0.0, max(total_duration, 0.0), max(length, 2))
    return np.linspace(0.0, total_duration, length)


def _resample_segment_profile(time_axis: np.ndarray,
                              speeds: np.ndarray,
                              target_points: int) -> np.ndarray:
    if target_points <= 1:
        return np.array([float(np.median(speeds))])
    if time_axis.size != speeds.size:
        time_axis = np.linspace(0.0, float(time_axis[-1] if time_axis.size else 0.0), speeds.size)
    total_duration = float(time_axis[-1]) if time_axis.size else 0.0
    new_axis = np.linspace(0.0, total_duration, target_points)
    if speeds.size == 0:
        return np.zeros(target_points, dtype=float)
    return np.interp(new_axis, time_axis, speeds, left=speeds[0], right=speeds[-1])


def _trim_cycle_idle_edges(cycle_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Limit excessive idle durations at the beginning and end of a representative cycle.
    Keeps a short buffer around the first/last active sample to preserve stop cues.
    """
    if cycle_df is None or cycle_df.empty or 'time_s' not in cycle_df.columns:
        return cycle_df

    df = cycle_df.sort_values('time_s').reset_index(drop=True).copy()
    times = df['time_s'].to_numpy()
    speeds = df.get('speed_ms', df.get('speed_kmh', pd.Series(np.zeros(len(df)), dtype=float))).to_numpy()
    if 'speed_kmh' in df.columns and 'speed_ms' not in df.columns:
        speeds = df['speed_kmh'].to_numpy() / 3.6
    stop_prob = df.get('stop_probability', pd.Series(np.zeros(len(df)), dtype=float)).to_numpy()

    active_mask = (speeds > EDGE_SPEED_THRESHOLD_MS) | (stop_prob < EDGE_STOP_PROB_THRESHOLD)
    if not np.any(active_mask):
        df['time_s'] = df['time_s'] - df['time_s'].iloc[0]
        return df

    first_active_idx = int(np.argmax(active_mask))
    last_active_idx = int(len(df) - 1 - np.argmax(active_mask[::-1]))

    total_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    max_idle = min(EDGE_IDLE_MAX_S, total_duration * EDGE_IDLE_MAX_FRAC)

    start_time = max(0.0, times[first_active_idx] - EDGE_BUFFER_S)
    end_time = min(times[-1], times[last_active_idx] + EDGE_BUFFER_S)

    if start_time > max_idle:
        start_time = max_idle
    if times[-1] - end_time > max_idle:
        end_time = times[-1] - max_idle
        end_time = max(end_time, times[last_active_idx])

    trimmed = df[(df['time_s'] >= start_time) & (df['time_s'] <= end_time)].copy()
    if trimmed.empty:
        trimmed = df.copy()

    trimmed['time_s'] = trimmed['time_s'] - trimmed['time_s'].iloc[0]
    return trimmed


def _extract_stop_aligned_trip(speeds: np.ndarray,
                               dt_series: np.ndarray,
                               min_stop_duration_s: float = STOP_MIN_DWELL_S) -> Optional[Dict[str, Any]]:
    if speeds is None or len(speeds) < 10:
        return None
    if dt_series is None or len(dt_series) == 0:
        dt_series = np.full_like(speeds, 0.5, dtype=float)
    dt_series = _ensure_length(dt_series, len(speeds), float(dt_series[-1]) if len(dt_series) else 0.5)
    dt_series = np.clip(dt_series, 1e-3, None)
    dt_median = float(np.median(dt_series))
    dwell_samples = max(1, int(round(min_stop_duration_s / dt_median)))
    stop_mask = build_stop_mask(
        speeds,
        dt_series,
        speed_threshold_ms=STOP_SPEED_THRESHOLD_MS,
        smooth_window_s=STOP_SMOOTH_WINDOW_S,
        dwell_s=min_stop_duration_s,
        gap_close_s=STOP_GAP_CLOSE_S,
    )
    stop_spans = mask_to_index_spans(stop_mask, min_samples=dwell_samples)
    if not stop_spans:
        return None

    # Ensure timeline start/end anchored by stop markers
    if stop_spans[0][0] > 0:
        stop_spans.insert(0, (0, stop_spans[0][0]))
    if stop_spans[-1][1] < len(speeds):
        stop_spans.append((stop_spans[-1][1], len(speeds)))

    # Merge overlapping or touching stop spans to avoid zero-length drive segments
    merged_spans: List[Tuple[int, int]] = []
    for span in stop_spans:
        if not merged_spans:
            merged_spans.append(span)
            continue
        prev_start, prev_end = merged_spans[-1]
        if span[0] <= prev_end:
            merged_spans[-1] = (prev_start, max(prev_end, span[1]))
        elif span[0] == prev_end:
            merged_spans[-1] = (prev_start, span[1])
        else:
            merged_spans.append(span)
    stop_spans = merged_spans

    stop_durations = []
    for start_idx, end_idx in stop_spans:
        if end_idx <= start_idx:
            stop_durations.append(0.0)
            continue
        stop_durations.append(float(np.sum(dt_series[start_idx:end_idx])))

    drive_segments = []
    drive_durations = []
    for idx in range(len(stop_spans) - 1):
        seg_start = stop_spans[idx][1]
        seg_end = stop_spans[idx + 1][0]
        if seg_end <= seg_start:
            drive_segments.append(None)
            drive_durations.append(0.0)
            continue
        segment_speeds = speeds[seg_start:seg_end]
        segment_dt = dt_series[seg_start:seg_end]
        duration = float(np.sum(segment_dt))
        drive_segments.append({
            'speeds': segment_speeds,
            'dt': segment_dt,
            'duration': duration,
        })
        drive_durations.append(duration)

    return {
        'stop_spans': stop_spans,
        'stop_durations': np.array(stop_durations, dtype=float),
        'drive_segments': drive_segments,
        'drive_durations': np.array(drive_durations, dtype=float),
        'dt_median': dt_median,
        'speeds_raw': speeds,
        'dt_raw': dt_series,
        'stop_mask': stop_mask,
        'total_duration': float(np.sum(stop_durations) + np.sum(drive_durations)),
    }


def _compute_stop_signature(struct: Dict[str, Any]) -> Dict[str, Any]:
    stop_durations = np.asarray(struct.get('stop_durations', []), dtype=float)
    drive_durations = np.asarray(struct.get('drive_durations', []), dtype=float)
    total = float(np.sum(stop_durations) + np.sum(drive_durations))
    if not np.isfinite(total) or total <= 0:
        total = float(len(stop_durations) + len(drive_durations)) or 1.0
    centers = []
    time_cursor = 0.0
    for idx, stop_duration in enumerate(stop_durations):
        center = (time_cursor + stop_duration / 2.0) / total
        centers.append(float(center))
        time_cursor += stop_duration
        if idx < len(drive_durations):
            time_cursor += drive_durations[idx]
    normalized_stop_durations = (stop_durations / total).tolist() if total > 0 else stop_durations.tolist()
    normalized_drive = (drive_durations / total).tolist() if total > 0 else drive_durations.tolist()
    return {
        'stop_count': int(len(stop_durations)),
        'centers': centers,
        'stop_norm': normalized_stop_durations,
        'drive_norm': normalized_drive,
        'total_duration': total,
    }


def _signature_distance(sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> float:
    if sig_a['stop_count'] != sig_b['stop_count']:
        return float('inf')
    centers_a = np.asarray(sig_a['centers'], dtype=float)
    centers_b = np.asarray(sig_b['centers'], dtype=float)
    if centers_a.size != centers_b.size:
        return float('inf')
    center_dist = float(np.max(np.abs(centers_a - centers_b))) if centers_a.size else 0.0
    stop_norm_a = np.asarray(sig_a['stop_norm'], dtype=float)
    stop_norm_b = np.asarray(sig_b['stop_norm'], dtype=float)
    stop_dist = float(np.max(np.abs(stop_norm_a - stop_norm_b))) if stop_norm_a.size else 0.0
    drive_norm_a = np.asarray(sig_a['drive_norm'], dtype=float)
    drive_norm_b = np.asarray(sig_b['drive_norm'], dtype=float)
    drive_dist = float(np.max(np.abs(drive_norm_a - drive_norm_b))) if drive_norm_a.size else 0.0
    return max(center_dist, stop_dist, drive_dist)

@dataclass
class SegmentVariability:
    segment_id: str
    n_observations: int
    speed_mean: float
    speed_std: float
    speed_cv: float
    speed_percentiles: Dict[int, float]
    speed_profiles: List[List[float]] = field(default_factory=list)
    entropy_score: float = 0.0
    predictability_score: float = 0.0
    stop_probability: float = 0.0
    segment_type: str = "map_based"
    # Original metrics
    chaos_index: float = 0.0
    normalized_profiles: List[List[float]] = field(default_factory=list)
    distribution_type: str = "unknown"
    speed_limit_compliance: float = 0.0
    turn_dynamics: Dict = field(default_factory=dict)
    # NEW METRICS
    jerk_rms: float = 0.0
    jerk_percentiles: Dict[int, float] = field(default_factory=dict)
    rpa: float = 0.0
    pke_segment: float = 0.0
    compression_ratio: float = 0.0
    acceleration_aggressiveness: float = 0.0
    deceleration_aggressiveness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class RouteVariabilityAnalyzer:
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, List]:
        """Convert DataFrame to dictionary format that preserves all data for JSON"""
        if df is None or df.empty:
            return {}
        
        # Convert to dictionary with lists (JSON-serializable)
        result = {}
        for col in df.columns:
            values = df[col].values
            # Convert numpy types to Python types
            if values.dtype in [np.float64, np.float32]:
                result[col] = [float(x) if not pd.isna(x) else None for x in values]
            elif values.dtype in [np.int64, np.int32]:
                result[col] = [int(x) if not pd.isna(x) else None for x in values]
            else:
                result[col] = [x if not pd.isna(x) else None for x in values.tolist()]
        
        return result
    
    def _dataframe_from_dict(self, data: Dict[str, List]) -> pd.DataFrame:
        """Rebuild DataFrame from dictionary-of-lists format"""
        if not data:
            return pd.DataFrame()
        rebuild = {}
        for col, values in data.items():
            rebuild[col] = pd.Series(values)
        df = pd.DataFrame(rebuild)
        numeric_cols = [c for c in df.columns if df[c].dtype == object]
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                continue
        return df
    
    def _build_trip_structure(self, trip_df: pd.DataFrame, trip_id: str) -> Optional[Dict[str, Any]]:
        speeds = self._extract_speeds(trip_df)
        if speeds.size < 20:
            return None
        dt_series = self._extract_dt_sequence(trip_df)
        dt_series = _ensure_length(dt_series, len(speeds), dt_series[-1] if len(dt_series) else 0.5)
        struct = _extract_stop_aligned_trip(speeds, dt_series)
        if struct is None:
            return None
        struct['trip_id'] = trip_id
        return struct
    
    def _update_signature(self, existing_sig: Dict[str, Any], new_sig: Dict[str, Any], weight_existing: int, weight_new: int) -> Dict[str, Any]:
        total = weight_existing + weight_new
        if total <= 0:
            return existing_sig
        updated = {
            'stop_count': existing_sig['stop_count'],
            'centers': ((np.array(existing_sig['centers']) * weight_existing) + (np.array(new_sig['centers']) * weight_new)) / total,
            'stop_norm': ((np.array(existing_sig['stop_norm']) * weight_existing) + (np.array(new_sig['stop_norm']) * weight_new)) / total,
            'drive_norm': ((np.array(existing_sig['drive_norm']) * weight_existing) + (np.array(new_sig['drive_norm']) * weight_new)) / total,
            'total_duration': (existing_sig['total_duration'] * weight_existing + new_sig['total_duration'] * weight_new) / total,
        }
        updated['centers'] = updated['centers'].tolist()
        updated['stop_norm'] = updated['stop_norm'].tolist()
        updated['drive_norm'] = updated['drive_norm'].tolist()
        return updated
    
    def _group_trip_infos_by_pattern(self, trip_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for info in trip_infos:
            struct = info.get('struct')
            if struct is None:
                continue
            signature = _compute_stop_signature(struct)
            assigned = False
            for group in groups:
                dist = _signature_distance(signature, group['signature'])
                if dist <= STOP_PATTERN_WINDOW:
                    group['members'].append(info)
                    group['signature'] = self._update_signature(group['signature'], signature, len(group['members']) - 1, 1)
                    assigned = True
                    break
            if not assigned:
                groups.append({
                    'signature': signature,
                    'members': [info]
                })
        groups.sort(key=lambda g: len(g['members']), reverse=True)
        return groups
    
    def _create_stop_aligned_variants(self,
                                      route_id: str,
                                      route_trips: List[pd.DataFrame],
                                      trip_infos: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        improved_cycle = self._create_stop_aligned_profile_improved(route_id, route_trips)
        if improved_cycle is not None and not improved_cycle.empty:
            improved_cycle = _trim_cycle_idle_edges(improved_cycle)

        if not trip_infos:
            if improved_cycle is not None and not improved_cycle.empty:
                return improved_cycle, []
            legacy = self._legacy_time_normalization(route_trips)
            return legacy, []

        groups = self._group_trip_infos_by_pattern(trip_infos)
        variant_entries: List[Dict[str, Any]] = []
        for idx, group in enumerate(groups):
            members = group['members']
            if len(members) < STOP_ALIGNMENT_MIN_TRIPS:
                continue
            member_trips = [info['trip_df'] for info in members]
            cycle_df, extras = self._build_stop_aligned_cycle(member_trips, include_profiles=False)
            if cycle_df is None or cycle_df.empty:
                continue
            cycle_df = _trim_cycle_idle_edges(cycle_df)
            variant_id = f"{route_id}_v{idx+1}"
            variant_entries.append({
                'variant_id': variant_id,
                'cycle': cycle_df,
                'stop_count': group['signature']['stop_count'],
                'n_trips': len(members),
                'total_duration_s': float(cycle_df['time_s'].iloc[-1]) if not cycle_df.empty else 0.0,
                'trip_ids': [info['trip_id'] for info in members]
            })

        if variant_entries:
            variant_entries.sort(key=lambda entry: entry['n_trips'], reverse=True)
            time_cycle = improved_cycle if improved_cycle is not None and not improved_cycle.empty else variant_entries[0]['cycle']
            return time_cycle, variant_entries

        if improved_cycle is not None and not improved_cycle.empty:
            return improved_cycle, []

        legacy_cycle = self._legacy_time_normalization(route_trips)
        return legacy_cycle, []
    
    def _legacy_time_normalization(self, trips: List[pd.DataFrame]) -> pd.DataFrame:
        speed_profiles: List[np.ndarray] = []
        durations: List[float] = []
        for trip in trips:
            speeds = self._extract_speeds(trip)
            if len(speeds) <= 10:
                continue
            duration = self._estimate_trip_duration(trip)
            if duration <= 0:
                continue
            speed_profiles.append(speeds)
            durations.append(duration)
        if len(speed_profiles) < 3:
            return pd.DataFrame()
        target_len = int(np.median([len(profile) for profile in speed_profiles]))
        target_len = max(target_len, 10)
        x_new = np.linspace(0, 1, target_len)
        aligned = []
        for speeds in speed_profiles:
            x_old = np.linspace(0, 1, len(speeds))
            aligned.append(np.interp(x_new, x_old, speeds))
        speeds_array = np.array(aligned, dtype=float)
        target_duration = float(np.median(durations)) if durations else 0.0
        if not np.isfinite(target_duration) or target_duration <= 0:
            sample_ratios = [
                durations[i] / max(len(speed_profiles[i]) - 1, 1)
                for i in range(len(speed_profiles))
                if durations[i] > 0
            ]
            median_step = float(np.median(sample_ratios)) if sample_ratios else 0.5
            if not np.isfinite(median_step) or median_step <= 0:
                median_step = 0.5
            target_duration = median_step * (target_len - 1)
        time_axis = np.linspace(0, target_duration, target_len)
        stop_probability = compute_stop_probability_from_profiles(speeds_array, target_duration)
        speed_p10 = np.percentile(speeds_array, 10, axis=0)
        median_speed = np.median(speeds_array, axis=0)
        if stop_probability.size == median_speed.size:
            stop_mask_mid = stop_probability >= 0.5
            stop_mask_high = stop_probability >= 0.7
            median_speed = median_speed.copy()
            median_speed[stop_mask_mid] = np.minimum(median_speed[stop_mask_mid], speed_p10[stop_mask_mid])
            median_speed[stop_mask_high] = np.minimum(median_speed[stop_mask_high], STOP_SPEED_THRESHOLD_MS * 0.2)
        median_speed = np.clip(median_speed, 0.0, None)
        legacy_df = pd.DataFrame({
            'time_s': time_axis,
            'speed_ms': median_speed,
            'speed_kmh': median_speed * 3.6,
            'speed_std': np.std(speeds_array, axis=0),
            'speed_p10': speed_p10,
            'speed_p25': np.percentile(speeds_array, 25, axis=0),
            'speed_p75': np.percentile(speeds_array, 75, axis=0),
            'stop_probability': stop_probability,
            'n_trips': len(speed_profiles)
        })
        return _trim_cycle_idle_edges(legacy_df)
    
    def analyze_main_route(self, route_id: str = "route_00000", 
                          route_data: pd.DataFrame = None,
                          file_list: List[str] = None) -> Dict:
        """Analyze using map-matched segments with enhanced metrics"""
        
        route_trips = []
        trip_infos: List[Dict[str, Any]] = []
        
        if route_data is not None and file_list is not None:
            for file_name in file_list:
                trip_df = route_data[route_data['source_file'] == file_name].copy()
                if len(trip_df) > 10:
                    trip_df = self._ensure_numeric_speeds(trip_df)
                    struct = self._build_trip_structure(trip_df, file_name)
                    if struct is not None:
                        route_trips.append(trip_df)
                        trip_infos.append({'trip_df': trip_df, 'trip_id': file_name, 'struct': struct})
        
        if not route_trips:
            return {}
        
        print(f"  Processing {len(route_trips)} trips with enhanced metrics")
        
        # Use map-based segmentation
        segments = self._segment_by_map_features(route_trips)
        
        # Analyze intersections and turns
        turn_analysis = self._analyze_turns_and_intersections(route_trips)
        
        results = {
            'route_id': route_id,
            'n_trips': len(route_trips),
            'segments': {},
            'traffic_lights': self._analyze_traffic_infrastructure(route_trips),
            'turn_dynamics': turn_analysis,
            'overall_metrics': {},
            'normalization_methods': {},
            # NEW: Advanced metrics summary
            'advanced_metrics': {
                'route_jerk_rms': 0.0,
                'route_rpa': 0.0,
                'route_compression': 0.0,
                'route_aggressiveness': 0.0
            },
            'custom_cycles': {}  # Will store as dictionaries
        }
        
        # Analyze each segment with enhanced metrics
        all_jerks = []
        all_rpas = []
        all_compressions = []
        
        for seg_id, seg_data in segments.items():
            var = self._analyze_segment_variability_enhanced(seg_data)
            results['segments'][seg_id] = var.to_dict()  # Convert to dict
            
            # Collect for route-level metrics
            if var.jerk_rms > 0:
                all_jerks.append(var.jerk_rms)
            if var.rpa > 0:
                all_rpas.append(var.rpa)
            if var.compression_ratio > 0:
                all_compressions.append(var.compression_ratio)
        
        # Calculate route-level advanced metrics
        if all_jerks:
            results['advanced_metrics']['route_jerk_rms'] = float(np.mean(all_jerks))
        if all_rpas:
            results['advanced_metrics']['route_rpa'] = float(np.mean(all_rpas))
        if all_compressions:
            results['advanced_metrics']['route_compression'] = float(np.mean(all_compressions))
        
        if results['segments']:
            results['overall_metrics'] = self._calculate_route_metrics(results)
            results['chaos_classification'] = self._classify_route_chaos(results)
        
        # Create custom cycles with different normalizations - store as dictionaries
        time_cycle, variant_entries = self._create_stop_aligned_variants(route_id, route_trips, trip_infos)
        dist_cycle = self._create_custom_cycle(route_trips, method='distance')
        stop_cycle = self._create_custom_cycle(route_trips, method='stop_to_stop')
        
        # Convert DataFrames to dictionaries for JSON serialization
        variant_dict = {}
        variant_meta = []
        if variant_entries:
            for entry in variant_entries:
                key = entry['variant_id']
                variant_dict[key] = self._dataframe_to_dict(entry['cycle'])
                variant_meta.append({
                    'variant_id': key,
                    'stop_count': entry['stop_count'],
                    'n_trips': entry['n_trips'],
                    'total_duration_s': entry['total_duration_s'],
                    'trip_ids': entry.get('trip_ids', [])
                })

        results['custom_cycles'] = {
            'time_normalized': self._dataframe_to_dict(time_cycle),
            'distance_normalized': self._dataframe_to_dict(dist_cycle),
            'stop_to_stop': self._dataframe_to_dict(stop_cycle),
            'time_variants': variant_dict
        }
        results['stop_pattern_variants'] = variant_meta
        if variant_dict:
            results['variant_cycles'] = variant_dict
        results['stop_summary'] = self._summarize_stop_patterns(route_trips)
        stop_segment_analysis = self.analyze_stop_to_stop_segments(route_id, route_trips)
        if stop_segment_analysis:
            results['stop_segment_analysis'] = stop_segment_analysis
        
        return results
    
    def _ensure_numeric_speeds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure speed columns are numeric and handle type issues"""
        if 'speed_ms' in df.columns:
            df['speed_ms'] = pd.to_numeric(df['speed_ms'], errors='coerce').fillna(0)
        
        if 'Vehicle speed' in df.columns:
            df['Vehicle speed'] = pd.to_numeric(df['Vehicle speed'], errors='coerce').fillna(0)
            if 'speed_ms' not in df.columns:
                df['speed_ms'] = df['Vehicle speed'] / 3.6
        
        return df
    
    def _calculate_jerk(self, speeds: np.ndarray, dt: float = 0.1) -> Dict:
        """Calculate jerk metrics (rate of acceleration change)"""
        if len(speeds) < 3:
            return {'rms': 0.0, 'percentiles': {}}
        
        # Calculate acceleration
        accel = np.diff(speeds) / dt
        
        # Calculate jerk (m/s³)
        jerk = np.diff(accel) / dt
        
        # RMS jerk (smoothness indicator)
        rms_jerk = float(np.sqrt(np.mean(jerk**2)))
        
        # Percentiles for distribution analysis
        percentiles = {}
        for p in [5, 25, 50, 75, 95]:
            percentiles[p] = float(np.percentile(np.abs(jerk), p))
        
        return {
            'rms': rms_jerk,
            'percentiles': percentiles,
            'max_abs': float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0
        }
    
    def _calculate_rpa(self, speeds: np.ndarray, distances: np.ndarray = None) -> float:
        """
        Calculate Relative Positive Acceleration (RPA)
        RPA = sum(v*a*dt) / total_distance for a > 0
        Units: m/s² or m²/s³ depending on normalization
        """
        if len(speeds) < 2:
            return 0.0
        
        dt = 0.1  # Assuming 10Hz
        
        # Calculate acceleration
        accel = np.diff(speeds) / dt
        
        # Get positive accelerations only
        positive_accel_mask = accel > 0.1  # Threshold to avoid noise
        
        if not any(positive_accel_mask):
            return 0.0
        
        # Calculate v*a for positive accelerations
        v_avg = (speeds[:-1] + speeds[1:]) / 2  # Average speed between points
        va_positive = v_avg[positive_accel_mask] * accel[positive_accel_mask]
        
        # Sum of v*a*dt
        rpa_sum = np.sum(va_positive * dt)
        
        # Calculate total distance
        if distances is not None and len(distances) > 0:
            total_distance = distances[-1] - distances[0]
        else:
            total_distance = np.sum(speeds * dt)
        
        if total_distance > 0:
            rpa = rpa_sum / total_distance
        else:
            rpa = 0.0
        
        return float(rpa)
    
    def _calculate_compression_ratio(self, speeds: np.ndarray) -> float:
        """
        Calculate Lempel-Ziv compression ratio as complexity measure
        Higher ratio = more complex/less predictable pattern
        """
        if len(speeds) < 10:
            return 0.0
        
        # Discretize speeds into symbols (e.g., 10 bins)
        bins = np.linspace(0, max(speeds.max(), 1), 11)
        digitized = np.digitize(speeds, bins)
        
        # Convert to string for compression
        pattern = ''.join(str(d) for d in digitized)
        
        # Calculate compression ratio
        original_size = len(pattern.encode('utf-8'))
        compressed_size = len(zlib.compress(pattern.encode('utf-8')))
        
        if compressed_size > 0:
            compression_ratio = original_size / compressed_size
        else:
            compression_ratio = 1.0
        
        return float(compression_ratio)
    
    def _analyze_segment_variability_enhanced(self, segment_data: Dict) -> SegmentVariability:
        """Enhanced variability metrics including jerk, RPA, etc."""
        
        speeds_matrix = segment_data.get('speeds', [])
        
        if not speeds_matrix:
            return SegmentVariability(
                segment_id="empty",
                n_observations=0,
                speed_mean=0, speed_std=0, speed_cv=0,
                speed_percentiles={}
            )
        
        all_speeds = np.concatenate(speeds_matrix)
        
        # Original metrics
        mean_speed = np.mean(all_speeds)
        std_speed = np.std(all_speeds)
        cv = std_speed / mean_speed if mean_speed > 1e-3 else float('inf')
        
        # Entropy calculation
        speed_hist, bins = np.histogram(all_speeds, bins=20)
        speed_entropy = entropy(speed_hist + 1e-10)
        
        # Predictability and chaos metrics
        predictability = 1.0 / (1.0 + cv)
        chaos_index = self._calculate_chaos_index(all_speeds, speed_entropy, cv)
        
        # Distribution classification
        distribution_type = self._classify_distribution(all_speeds)
        
        # Speed limit compliance
        speed_limit_compliance = 0.0
        if segment_data.get('speed_limits'):
            avg_limit = np.mean(segment_data['speed_limits'])
            speed_limit_compliance = np.mean(all_speeds <= avg_limit * 1.1)
        
        # Stop probability
        stop_probs = [np.sum(s < 0.5) / len(s) for s in speeds_matrix if len(s) > 0]
        
        # NEW: Enhanced metrics
        jerk_metrics = {'rms': 0, 'percentiles': {}}
        rpa_values = []
        pke_values = []
        compression_ratios = []
        accel_aggressive = []
        decel_aggressive = []
        
        for speed_profile in speeds_matrix:
            if len(speed_profile) > 2:
                # Jerk analysis
                jerk = self._calculate_jerk(speed_profile)
                if jerk['rms'] > 0:
                    if jerk_metrics['rms'] == 0:
                        jerk_metrics = jerk
                    else:
                        jerk_metrics['rms'] = (jerk_metrics['rms'] + jerk['rms']) / 2
                
                # RPA calculation
                rpa = self._calculate_rpa(speed_profile)
                if rpa > 0:
                    rpa_values.append(rpa)
                
                # PKE for segment
                accels = np.diff(speed_profile) / 0.1
                positive_accels = accels[accels > 0]
                if len(positive_accels) > 0:
                    pke = np.sum(positive_accels ** 2)
                    pke_values.append(pke)
                
                # Compression ratio
                comp = self._calculate_compression_ratio(speed_profile)
                if comp > 0:
                    compression_ratios.append(comp)
                
                # Aggressiveness metrics
                if len(accels) > 0:
                    # Aggressive acceleration (95th percentile of positive)
                    pos_accels = accels[accels > 0.1]
                    if len(pos_accels) > 0:
                        accel_aggressive.append(np.percentile(pos_accels, 95))
                    
                    # Aggressive deceleration (95th percentile of negative)
                    neg_accels = np.abs(accels[accels < -0.1])
                    if len(neg_accels) > 0:
                        decel_aggressive.append(np.percentile(neg_accels, 95))
        
        # Normalized profiles for comparison
        normalized_profiles = self._normalize_speed_profiles(speeds_matrix)
        
        return SegmentVariability(
            segment_id=f"segment_{len(speeds_matrix)}trips",
            n_observations=len(speeds_matrix),
            speed_mean=float(mean_speed),
            speed_std=float(std_speed),
            speed_cv=float(cv),
            speed_percentiles={
                p: float(np.percentile(all_speeds, p))
                for p in [5, 25, 50, 75, 95]
            },
            speed_profiles=speeds_matrix[:10],
            entropy_score=float(speed_entropy),
            predictability_score=float(predictability),
            stop_probability=float(np.mean(stop_probs)) if stop_probs else 0.0,
            chaos_index=float(chaos_index),
            normalized_profiles=normalized_profiles[:10],
            distribution_type=distribution_type,
            speed_limit_compliance=float(speed_limit_compliance),
            # NEW metrics
            jerk_rms=float(jerk_metrics['rms']),
            jerk_percentiles=jerk_metrics['percentiles'],
            rpa=float(np.mean(rpa_values)) if rpa_values else 0.0,
            pke_segment=float(np.mean(pke_values)) if pke_values else 0.0,
            compression_ratio=float(np.mean(compression_ratios)) if compression_ratios else 0.0,
            acceleration_aggressiveness=float(np.mean(accel_aggressive)) if accel_aggressive else 0.0,
            deceleration_aggressiveness=float(np.mean(decel_aggressive)) if decel_aggressive else 0.0
        )
    
    def _segment_by_map_features(self, trips: List[pd.DataFrame]) -> Dict:
        """Enhanced segmentation using map features"""
        segments = {}
        
        for trip_idx, trip in enumerate(trips):
            trip = trip.reset_index(drop=True)
            
            # Find segmentation points
            segment_points = [0]
            
            # Add intersections
            if 'map_near_intersection' in trip.columns:
                intersections = trip[trip['map_near_intersection'] == True].index.tolist()
                segment_points.extend([int(idx) for idx in intersections[::3]])
            
            # Add traffic lights
            if 'map_near_traffic_light' in trip.columns:
                lights = trip[trip['map_near_traffic_light'] == True].index.tolist()
                segment_points.extend([int(idx) for idx in lights[::2]])
            
            # Add road class changes
            if 'map_road_class' in trip.columns:
                trip['map_road_class'] = trip['map_road_class'].fillna('unknown')
                road_changes = trip[trip['map_road_class'].ne(trip['map_road_class'].shift())].index.tolist()
                segment_points.extend([int(idx) for idx in road_changes])
            
            # Sort and deduplicate
            segment_points = sorted(set(segment_points))
            segment_points.append(len(trip) - 1)
            
            # Create segments
            for i in range(len(segment_points) - 1):
                start = int(segment_points[i])
                end = int(segment_points[i + 1])
                
                if end - start < 5:
                    continue
                
                seg_df = trip.iloc[start:end].copy()
                
                # Create segment ID
                segment_attrs = []
                
                if 'map_road_class' in seg_df.columns:
                    road_class_series = seg_df['map_road_class'].fillna('unknown')
                    if len(road_class_series) > 0:
                        road = road_class_series.mode().iloc[0] if len(road_class_series.mode()) > 0 else 'unknown'
                        segment_attrs.append(f"road_{road}")
                
                if 'map_maxspeed_kph' in seg_df.columns:
                    speed_limit_series = pd.to_numeric(seg_df['map_maxspeed_kph'], errors='coerce')
                    if speed_limit_series.notna().any():
                        limit = int(speed_limit_series.mode().iloc[0]) if len(speed_limit_series.mode()) > 0 else 0
                        segment_attrs.append(f"limit_{limit}")
                
                seg_id = "_".join(segment_attrs) if segment_attrs else f"seg_{i}"
                
                if seg_id not in segments:
                    segments[seg_id] = {
                        'speeds': [], 
                        'conditions': [],
                        'speed_limits': [],
                        'accelerations': []
                    }
                
                # Extract speeds
                speeds = self._extract_speeds(seg_df)
                
                if len(speeds) > 0:
                    segments[seg_id]['speeds'].append(speeds)
                    
                    # Calculate accelerations
                    if 'dt_s' in seg_df.columns:
                        dt = seg_df['dt_s'].values[1:]
                        if len(dt) > 0 and len(speeds) > 1:
                            # Ensure dt has no zeros
                            dt = np.where(dt > 0, dt, 0.1)
                            accel = np.diff(speeds) / dt
                            segments[seg_id]['accelerations'].append(accel)
                    
                    # Store speed limit
                    if 'map_maxspeed_kph' in seg_df.columns:
                        limit_ms = pd.to_numeric(seg_df['map_maxspeed_kph'], errors='coerce').mean() / 3.6
                        if not np.isnan(limit_ms):
                            segments[seg_id]['speed_limits'].append(limit_ms)
        
        # Filter segments with enough data
        segments = {k: v for k, v in segments.items() 
                   if len(v['speeds']) >= 3}
        
        return segments
    
    def _extract_speeds(self, df: pd.DataFrame) -> np.ndarray:
        """Safely extract speed values from dataframe"""
        if 'speed_ms' in df.columns:
            speeds = pd.to_numeric(df['speed_ms'], errors='coerce').values
        elif 'Vehicle speed' in df.columns:
            speeds = pd.to_numeric(df['Vehicle speed'], errors='coerce').values / 3.6
        else:
            return np.array([])
        
        return speeds[~np.isnan(speeds)]

    def _extract_dt_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Extract per-sample delta-time values with reasonable fallbacks"""
        if 'dt_s' in df.columns:
            dt = pd.to_numeric(df['dt_s'], errors='coerce').values
        elif 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            dt = timestamps.diff().dt.total_seconds().fillna(0.1).values
        elif 'Time (sec)' in df.columns:
            rel_time = pd.to_numeric(df['Time (sec)'], errors='coerce').values
            dt = np.diff(rel_time, prepend=rel_time[0])
        else:
            return np.full(len(df), 0.5)
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)
        valid = dt[~np.isnan(dt)]
        fallback = np.median(valid) if valid.size > 0 else 0.5
        if fallback <= 0 or not np.isfinite(fallback):
            fallback = 0.5
        dt = np.where(np.isnan(dt) | (dt <= 0), fallback, dt)
        return dt

    def _build_stop_aligned_cycle(self,
                                  trips: List[pd.DataFrame],
                                  include_profiles: bool = False,
                                  max_trips: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Construct a stop-aligned representative cycle by segmenting trips between sequential stops.
        Returns the aggregated cycle DataFrame and optional diagnostic bundles.
        """
        if not trips:
            return None, {}

        trip_structs: List[Dict[str, Any]] = []
        for idx, trip in enumerate(trips):
            if max_trips is not None and len(trip_structs) >= max_trips:
                break
            speeds = self._extract_speeds(trip)
            if speeds.size < 20:
                continue
            dt_series = self._extract_dt_sequence(trip)
            dt_series = _ensure_length(dt_series, len(speeds), float(dt_series[-1]) if dt_series.size else 0.5)
            struct = _extract_stop_aligned_trip(speeds, dt_series)
            if not struct:
                continue
            trip_id = None
            if 'source_file' in trip.columns:
                trip_id = trip['source_file'].iloc[0]
            struct['trip_id'] = trip_id if trip_id is not None else f"trip_{idx:02d}"
            trip_structs.append(struct)

        if len(trip_structs) < STOP_ALIGNMENT_MIN_TRIPS:
            return None, {}

        stop_counts = [len(ts['stop_durations']) for ts in trip_structs]
        count_counter = Counter(stop_counts)
        target_stop_count, _ = count_counter.most_common(1)[0]
        if target_stop_count < STOP_ALIGNMENT_MIN_STOPS:
            return None, {}

        aligned_structs = [
            ts for ts in trip_structs
            if len(ts['stop_durations']) == target_stop_count
            and len(ts['drive_segments']) == target_stop_count - 1
            and all(seg is not None and seg.get('duration', 0.0) >= 0.25
                    for seg in ts['drive_segments'])
        ]

        if len(aligned_structs) < STOP_ALIGNMENT_MIN_TRIPS:
            return None, {}

        stop_duration_matrix = np.vstack([ts['stop_durations'] for ts in aligned_structs])
        median_stop_durations = np.median(stop_duration_matrix, axis=0)

        segment_duration_matrix = np.vstack([ts['drive_durations'] for ts in aligned_structs])
        median_segment_durations = np.median(segment_duration_matrix, axis=0)

        segment_profiles: List[np.ndarray] = []
        segment_p10: List[np.ndarray] = []
        segment_p25: List[np.ndarray] = []
        segment_p75: List[np.ndarray] = []
        segment_std: List[np.ndarray] = []
        segment_stop_prob: List[np.ndarray] = []
        segment_time_axes: List[np.ndarray] = []
        segment_distance_medians: List[float] = []
        per_trip_segment_profiles: List[List[np.ndarray]] = [[] for _ in aligned_structs]

        for seg_idx in range(target_stop_count - 1):
            seg_duration_samples = []
            resampled_segments = []
            seg_distance_samples = []
            for tr_idx, struct in enumerate(aligned_structs):
                segment = struct['drive_segments'][seg_idx]
                speeds = np.asarray(segment['speeds'], dtype=float)
                duration = float(segment.get('duration', np.sum(segment.get('dt', [])) if segment else 0.0))
                if duration <= 0 or speeds.size == 0:
                    continue
                time_axis = _segment_time_axis(len(speeds), duration)
                median_duration = median_segment_durations[seg_idx]
                target_points = max(
                    8,
                    min(
                        SEGMENT_RESAMPLE_POINTS,
                        int(math.ceil(max(median_duration, duration) / STOP_ALIGNMENT_TIME_STEP_S)) + 1
                    )
                )
                resampled = _resample_segment_profile(time_axis, speeds, target_points)
                resampled_segments.append(resampled)
                per_trip_segment_profiles[tr_idx].append(resampled)
                seg_duration_samples.append(duration)
                dt_segment = segment.get('dt')
                if dt_segment is not None and len(dt_segment) == len(speeds):
                    seg_distance_samples.append(float(np.sum(speeds * dt_segment)))
                else:
                    seg_distance_samples.append(float(np.mean(speeds)) * duration)

            if not resampled_segments:
                return None, {}

            try:
                segment_matrix = np.vstack(resampled_segments)
            except ValueError:
                continue
            segment_profiles.append(np.median(segment_matrix, axis=0))
            segment_std.append(np.std(segment_matrix, axis=0))
            segment_p10.append(np.percentile(segment_matrix, 10, axis=0))
            segment_p25.append(np.percentile(segment_matrix, 25, axis=0))
            segment_p75.append(np.percentile(segment_matrix, 75, axis=0))
            segment_stop_prob.append((segment_matrix <= STOP_SPEED_THRESHOLD_MS).mean(axis=0))

            representative_duration = float(np.median(seg_duration_samples)) if seg_duration_samples else median_segment_durations[seg_idx]
            segment_time_axes.append(np.linspace(0.0, representative_duration, segment_profiles[-1].size))
            if seg_distance_samples:
                segment_distance_medians.append(float(np.median(seg_distance_samples)))
            else:
                segment_distance_medians.append(float(np.mean(segment_profiles[-1]) * representative_duration))

        records: List[Dict[str, float]] = []
        current_time = 0.0
        current_distance = 0.0
        stop_steps = []
        for stop_idx, stop_duration in enumerate(median_stop_durations):
            stop_duration = float(max(stop_duration, 0.0))
            steps = max(2, int(math.ceil(stop_duration / STOP_ALIGNMENT_TIME_STEP_S))) if stop_duration > 0 else 1
            stop_steps.append(steps)
            if stop_duration > 0:
                stop_time_axis = np.linspace(0.0, stop_duration, steps, endpoint=False)
                for dt_offset in stop_time_axis:
                    records.append({
                        'time_s': current_time + dt_offset,
                        'speed_ms': 0.0,
                        'speed_std': 0.0,
                        'speed_p10': 0.0,
                        'speed_p25': 0.0,
                        'speed_p75': 0.0,
                        'stop_probability': 1.0,
                        'segment_index': -1,
                        'stop_index': stop_idx,
                        'distance_m': current_distance,
                    })
                current_time += stop_duration
            if stop_idx >= len(segment_profiles):
                continue
            seg_time = segment_time_axes[stop_idx]
            seg_speed = segment_profiles[stop_idx]
            seg_std = segment_std[stop_idx]
            seg_p10 = segment_p10[stop_idx]
            seg_p25 = segment_p25[stop_idx]
            seg_p75 = segment_p75[stop_idx]
            seg_prob = np.clip(segment_stop_prob[stop_idx], 0.0, 1.0)
            median_distance = segment_distance_medians[stop_idx] if stop_idx < len(segment_distance_medians) else 0.0
            samples = max(seg_time.size, 1)
            for idx_time, local_time in enumerate(seg_time):
                progress = float(idx_time / max(samples - 1, 1)) if samples > 1 else 0.0
                records.append({
                    'time_s': current_time + local_time,
                    'speed_ms': seg_speed[idx_time],
                    'speed_std': seg_std[idx_time],
                    'speed_p10': seg_p10[idx_time],
                    'speed_p25': seg_p25[idx_time],
                    'speed_p75': seg_p75[idx_time],
                    'stop_probability': seg_prob[idx_time],
                    'segment_index': stop_idx,
                    'stop_index': stop_idx,
                    'distance_m': current_distance + progress * median_distance,
                })
            current_time += float(seg_time[-1]) if seg_time.size else 0.0
            current_distance += median_distance

        if not records:
            return None, {}

        cycle_df = pd.DataFrame(records).sort_values('time_s').reset_index(drop=True)
        cycle_df['speed_ms'] = np.clip(cycle_df['speed_ms'], 0.0, None)
        cycle_df['speed_kmh'] = cycle_df['speed_ms'] * 3.6
        cycle_df['stop_probability'] = np.clip(cycle_df['stop_probability'], 0.0, 1.0)
        cycle_df['n_trips'] = len(aligned_structs)
        if 'distance_m' not in cycle_df.columns:
            cycle_df['distance_m'] = np.linspace(0.0, current_distance, len(cycle_df))
        total_distance = float(cycle_df['distance_m'].max())
        if total_distance > 0:
            cycle_df['normalized_distance'] = np.clip(cycle_df['distance_m'] / total_distance, 0.0, 1.0)
        else:
            cycle_df['normalized_distance'] = 0.0

        cycle_df = _trim_cycle_idle_edges(cycle_df)

        extras: Dict[str, Any] = {
            'median_stop_durations': median_stop_durations,
            'median_segment_durations': median_segment_durations,
            'aligned_trip_ids': [ts['trip_id'] for ts in aligned_structs],
            'total_distance_m': float(total_distance),
        }

        if include_profiles:
            base_time = cycle_df['time_s'].to_numpy()
            aligned_profiles = []
            for struct in aligned_structs:
                raw_speeds = np.asarray(struct.get('speeds_raw', []), dtype=float)
                dt_seq = np.asarray(struct.get('dt_raw', []), dtype=float)
                if raw_speeds.size == 0 or dt_seq.size == 0:
                    continue
                if dt_seq.size != raw_speeds.size:
                    dt_seq = _ensure_length(dt_seq, raw_speeds.size, dt_seq[-1] if dt_seq.size else 0.5)
                raw_time = np.cumsum(np.concatenate(([0.0], dt_seq[:-1])))
                interp_speed = np.interp(base_time, raw_time, raw_speeds, left=raw_speeds[0], right=raw_speeds[-1])
                aligned_profiles.append((struct.get('trip_id'), base_time.copy(), interp_speed.copy()))
            if aligned_profiles:
                extras['aligned_profiles'] = aligned_profiles

        return cycle_df, extras

    def build_stop_aligned_bundle(self,
                                  trips: List[pd.DataFrame],
                                  include_profiles: bool = False,
                                  max_trips: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Public helper to construct stop-aligned cycles with optional profile bundles.
        """
        return self._build_stop_aligned_cycle(trips, include_profiles=include_profiles, max_trips=max_trips)

    def _summarize_stop_patterns(self, trips: List[pd.DataFrame],
                                 stop_speed_ms: float = STOP_SPEED_THRESHOLD_MS,
                                 min_stop_duration_s: float = STOP_MIN_DWELL_S) -> Dict:
        """Aggregate stop durations and typical dwell behaviours across trips"""
        stop_events = []
        stop_counts = []
        stop_totals = []
        coords = []
        trip_durations = []
        for trip_idx, trip in enumerate(trips):
            if trip is None or trip.empty:
                continue
            speeds = self._extract_speeds(trip)
            if len(speeds) < 5:
                continue
            dt_seq = self._extract_dt_sequence(trip)
            if len(dt_seq) != len(speeds):
                min_len = min(len(dt_seq), len(speeds))
                speeds = speeds[:min_len]
                dt_seq = dt_seq[:min_len]
            stop_mask = build_stop_mask(speeds, dt_seq, speed_threshold_ms=stop_speed_ms)
            dwell_samples = max(1, int(round(min_stop_duration_s / max(np.median(dt_seq), 1e-3))))
            spans = mask_to_index_spans(stop_mask, min_samples=dwell_samples)
            trip_stop_dur = 0.0
            stop_counter = 0
            trip_duration = float(np.sum(dt_seq))
            trip_durations.append(trip_duration)
            for start_idx, end_idx in spans:
                dur = float(np.sum(dt_seq[start_idx:end_idx]))
                if dur < min_stop_duration_s:
                    continue
                stop_counter += 1
                trip_stop_dur += dur
                lat = None
                lon = None
                if 'Latitude' in trip.columns and start_idx < len(trip):
                    lat_val = trip['Latitude'].iloc[start_idx]
                    if pd.notna(lat_val):
                        lat = float(lat_val)
                        if abs(lat) < 1e-6:
                            lat = None
                if 'Longitude' in trip.columns and start_idx < len(trip):
                    lon_val = trip['Longitude'].iloc[start_idx]
                    if pd.notna(lon_val):
                        lon = float(lon_val)
                        if abs(lon) < 1e-6:
                            lon = None
                event = {
                    'trip_index': trip_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration_s': dur,
                    'latitude': lat,
                    'longitude': lon
                }
                stop_events.append(event)
                if lat is not None and lon is not None:
                    coords.append([lat, lon])
            stop_counts.append(stop_counter)
            stop_totals.append(trip_stop_dur)
        if not stop_events:
            return {
                'n_stop_events': 0,
                'median_duration_s': 0.0,
                'mean_duration_s': 0.0,
                'p90_duration_s': 0.0,
                'median_stops_per_trip': float(np.median(stop_counts)) if stop_counts else 0.0,
                'mean_stops_per_trip': float(np.mean(stop_counts)) if stop_counts else 0.0,
                'mean_idle_ratio_pct': 0.0,
                'clusters': [],
                'events': []
            }
        durations = [ev['duration_s'] for ev in stop_events]
        idle_ratio = 0.0
        if trip_durations:
            idle_ratio = float(np.sum(stop_totals) / np.sum(trip_durations) * 100.0) if np.sum(trip_durations) > 0 else 0.0
        summary = {
            'n_stop_events': len(stop_events),
            'median_duration_s': float(np.median(durations)),
            'mean_duration_s': float(np.mean(durations)),
            'p90_duration_s': float(np.percentile(durations, 90)),
            'median_stops_per_trip': float(np.median(stop_counts)) if stop_counts else 0.0,
            'mean_stops_per_trip': float(np.mean(stop_counts)) if stop_counts else 0.0,
            'mean_idle_ratio_pct': idle_ratio,
        }
        clusters = []
        if len(coords) >= 3:
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.00035, min_samples=2).fit(coords)
                labels = clustering.labels_
                for cluster_id in sorted(set(labels)):
                    if cluster_id < 0:
                        continue
                    cluster_events = [ev for ev, label in zip(stop_events, labels) if label == cluster_id]
                    if not cluster_events:
                        continue
                    durations_cluster = [ev['duration_s'] for ev in cluster_events]
                    latitudes = [ev['latitude'] for ev in cluster_events if ev['latitude'] is not None]
                    longitudes = [ev['longitude'] for ev in cluster_events if ev['longitude'] is not None]
                    lat_median = float(np.median(latitudes)) if latitudes else None
                    lon_median = float(np.median(longitudes)) if longitudes else None
                    if lat_median is not None and abs(lat_median) < 1e-6:
                        lat_median = None
                    if lon_median is not None and abs(lon_median) < 1e-6:
                        lon_median = None
                    clusters.append({
                        'cluster_id': int(cluster_id),
                        'n_events': len(cluster_events),
                        'median_duration_s': float(np.median(durations_cluster)),
                        'mean_duration_s': float(np.mean(durations_cluster)),
                        'p90_duration_s': float(np.percentile(durations_cluster, 90)),
                        'latitude': lat_median,
                        'longitude': lon_median
                    })
            except Exception:
                pass
        summary['clusters'] = clusters
        summary['events'] = stop_events[:100]
        return summary

    def _estimate_trip_duration(self, df: pd.DataFrame) -> float:
        """Estimate trip duration in seconds using dt, timestamps, or fallback heuristics"""
        if df is None or df.empty:
            return 0.0

        if 'dt_s' in df.columns:
            dt = pd.to_numeric(df['dt_s'], errors='coerce')
            dt = dt.replace([np.inf, -np.inf], np.nan)
            if dt.notna().any():
                positive = dt[dt > 0]
                fallback = float(positive.median()) if not positive.empty else 0.1
                if not np.isfinite(fallback) or fallback <= 0:
                    fallback = 0.1
                dt = dt.fillna(fallback)
                dt = dt.clip(lower=0)
                duration = float(dt.sum())
                if duration > 0:
                    return duration

        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
            if len(timestamps) > 1:
                delta = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
                if np.isfinite(delta) and delta > 0:
                    return float(delta)

        if 'Time (sec)' in df.columns:
            rel_time = pd.to_numeric(df['Time (sec)'], errors='coerce').dropna()
            if len(rel_time) > 1:
                delta = rel_time.iloc[-1] - rel_time.iloc[0]
                if np.isfinite(delta) and delta > 0:
                    return float(delta)

        # Fallback assumes moderately slow sample rate if no reliable timing data
        approx = float(len(df)) * 0.5
        return approx if approx > 0 else 0.0
    
    def _calculate_chaos_index(self, speeds: np.ndarray, entropy_val: float, cv: float) -> float:
        """Calculate chaos index from multiple factors"""
        if len(speeds) < 10:
            return 0.0
        
        # Factor 1: Normalized entropy
        max_entropy = np.log(20)
        norm_entropy = entropy_val / max_entropy
        
        # Factor 2: Coefficient of variation
        norm_cv = min(cv, 1.0)
        
        # Factor 3: Acceleration variability
        accels = np.diff(speeds)
        accel_cv = np.std(accels) / (np.mean(np.abs(accels)) + 1e-6)
        norm_accel_cv = min(accel_cv / 2, 1.0)
        
        # Factor 4: Number of peaks
        peaks, _ = find_peaks(speeds, distance=5)
        peak_density = len(peaks) / len(speeds)
        norm_peaks = min(peak_density * 10, 1.0)
        
        # Weighted combination
        chaos_index = (
            0.3 * norm_entropy +
            0.3 * norm_cv +
            0.2 * norm_accel_cv +
            0.2 * norm_peaks
        )
        
        return chaos_index
    
    def _classify_distribution(self, speeds: np.ndarray) -> str:
        """Classify speed distribution pattern"""
        if len(speeds) < 20:
            return "insufficient_data"
        
        # Check for bimodality
        hist, bins = np.histogram(speeds, bins=15)
        peaks, _ = find_peaks(hist, distance=3, prominence=len(speeds)*0.1)
        
        if len(peaks) >= 2:
            return "bimodal"
        
        # Check for uniformity
        _, p_value = ks_2samp(speeds, np.random.uniform(speeds.min(), speeds.max(), len(speeds)))
        if p_value > 0.05:
            return "uniform"
        
        # Check CV for chaos
        cv = np.std(speeds) / (np.mean(speeds) + 1e-6)
        if cv > 0.5:
            return "chaotic"
        
        return "normal"
    
    def _normalize_speed_profiles(self, speed_profiles: List[np.ndarray], 
                                 n_points: int = 100) -> List[List[float]]:
        """Normalize speed profiles to same length"""
        normalized = []
        
        for profile in speed_profiles[:20]:
            if len(profile) < 5:
                continue
            
            x_old = np.linspace(0, 1, len(profile))
            x_new = np.linspace(0, 1, n_points)
            
            try:
                f = interp1d(x_old, profile, kind='linear', fill_value='extrapolate')
                normalized_profile = f(x_new)
                normalized.append(normalized_profile.tolist())
            except:
                continue
        
        return normalized
    
    def _analyze_turns_and_intersections(self, trips: List[pd.DataFrame]) -> Dict:
        """Analyze turn dynamics at intersections"""
        results = {
            'intersection_speeds': [],
            'turn_angles': [],
            'turn_speeds': [],
            'acceleration_patterns': []
        }
        
        for trip in trips:
            if 'map_near_intersection' not in trip.columns:
                continue
            
            intersections = trip[trip['map_near_intersection'] == True].index.tolist()
            
            for idx in intersections:
                start = max(0, idx - 10)
                end = min(len(trip), idx + 10)
                
                if 'speed_ms' in trip.columns:
                    window_speeds = pd.to_numeric(trip.iloc[start:end]['speed_ms'], errors='coerce')
                    if window_speeds.notna().any():
                        results['intersection_speeds'].append({
                            'approach': float(window_speeds.iloc[:10].mean()),
                            'min': float(window_speeds.min()),
                            'exit': float(window_speeds.iloc[-10:].mean())
                        })
                
                # Calculate turn angle if possible
                if 'Latitude' in trip.columns and 'Longitude' in trip.columns:
                    if idx > 5 and idx < len(trip) - 5:
                        lat1, lon1 = trip.iloc[idx-5][['Latitude', 'Longitude']]
                        lat2, lon2 = trip.iloc[idx][['Latitude', 'Longitude']]
                        lat3, lon3 = trip.iloc[idx+5][['Latitude', 'Longitude']]
                        
                        bearing1 = np.arctan2(lon2-lon1, lat2-lat1)
                        bearing2 = np.arctan2(lon3-lon2, lat3-lat2)
                        turn_angle = np.degrees(bearing2 - bearing1) % 360
                        
                        if turn_angle > 180:
                            turn_angle -= 360
                        
                        results['turn_angles'].append(float(turn_angle))
        
        return results
    
    def _analyze_traffic_infrastructure(self, trips: List[pd.DataFrame]) -> Dict:
        """Enhanced traffic infrastructure analysis"""
        results = {
            'traffic_light_stops': [],
            'n_traffic_lights': 0,
            'avg_wait_time': 0.0,
            'light_compliance': []
        }
        
        wait_times = []
        
        for trip in trips:
            if 'map_near_traffic_light' not in trip.columns:
                continue
            
            near_light = trip['map_near_traffic_light'] == True
            
            if 'speed_ms' in trip.columns:
                speeds = pd.to_numeric(trip['speed_ms'], errors='coerce')
                stopped = speeds < 0.5
                
                light_stops = int((near_light & stopped).sum())
                results['traffic_light_stops'].append(light_stops)
                
                if 'dt_s' in trip.columns:
                    dt = trip['dt_s'].fillna(0.1)
                    wait_mask = near_light & stopped
                    if wait_mask.any():
                        wait_time = dt[wait_mask].sum()
                        wait_times.append(wait_time)
                
                if 'map_maxspeed_kph' in trip.columns:
                    limit_ms = pd.to_numeric(trip['map_maxspeed_kph'], errors='coerce') / 3.6
                    compliance = (speeds <= limit_ms * 1.1).mean()
                    results['light_compliance'].append(float(compliance))
        
        if wait_times:
            results['avg_wait_time'] = float(np.mean(wait_times))
        
        results['n_traffic_lights'] = sum(1 for t in trips 
                                         if 'map_near_traffic_light' in t.columns 
                                         and (t['map_near_traffic_light'] == True).any())
        
        return results
    
    def _calculate_route_metrics(self, results: Dict) -> Dict:
        """Enhanced route metrics including new analysis"""
        segments = results.get('segments', {})
        
        if not segments:
            return {}
        
        # Extract metrics from dictionary format
        predictabilities = []
        chaos_indices = []
        jerk_values = []
        rpa_values = []
        compression_values = []
        compliances = []
        
        for seg_dict in segments.values():
            if isinstance(seg_dict, dict):
                predictabilities.append(seg_dict.get('predictability_score', 0))
                chaos_indices.append(seg_dict.get('chaos_index', 0))
                if seg_dict.get('jerk_rms', 0) > 0:
                    jerk_values.append(seg_dict['jerk_rms'])
                if seg_dict.get('rpa', 0) > 0:
                    rpa_values.append(seg_dict['rpa'])
                if seg_dict.get('compression_ratio', 0) > 0:
                    compression_values.append(seg_dict['compression_ratio'])
                if seg_dict.get('speed_limit_compliance', 0) > 0:
                    compliances.append(seg_dict['speed_limit_compliance'])
        
        return {
            'route_predictability': float(np.mean(predictabilities)) if predictabilities else 0,
            'route_chaos_score': float(np.mean(chaos_indices)) if chaos_indices else 0,
            'chaos_std': float(np.std(chaos_indices)) if chaos_indices else 0,
            'n_segments_analyzed': len(segments),
            'route_speed_limit_compliance': float(np.mean(compliances)) if compliances else 0.0,
            'high_chaos_segments': sum(1 for c in chaos_indices if c > 0.7),
            'low_chaos_segments': sum(1 for c in chaos_indices if c < 0.3),
            # NEW metrics
            'route_jerk_rms': float(np.mean(jerk_values)) if jerk_values else 0.0,
            'route_rpa': float(np.mean(rpa_values)) if rpa_values else 0.0,
            'route_compression': float(np.mean(compression_values)) if compression_values else 0.0,
            'route_smoothness': 1.0 / (1.0 + float(np.mean(jerk_values))) if jerk_values else 0.0
        }
    
    def _classify_route_chaos(self, results: Dict) -> str:
        """Classify overall route behavior"""
        metrics = results.get('overall_metrics', {})
        chaos_score = metrics.get('route_chaos_score', 0)
        chaos_std = metrics.get('chaos_std', 0)
        
        if chaos_score < 0.3:
            return "highly_predictable"
        elif chaos_score < 0.5:
            if chaos_std > 0.2:
                return "mixed_predictable"
            else:
                return "moderately_predictable"
        elif chaos_score < 0.7:
            if chaos_std > 0.2:
                return "mixed_chaotic"
            else:
                return "moderately_chaotic"
        else:
            return "highly_chaotic"
    
    def _create_custom_cycle(self, trips: List[pd.DataFrame], 
                           method: str = 'time') -> pd.DataFrame:
        """Create custom cycle with different normalization methods"""
        
        if method == 'time':
            return self._normalize_by_time(trips)
        elif method == 'distance':
            return self._normalize_by_distance(trips)
        elif method == 'stop_to_stop':
            return self._normalize_stop_to_stop(trips)
        else:
            return pd.DataFrame()
    
    def _normalize_by_time(self, trips: List[pd.DataFrame]) -> pd.DataFrame:
        """Time-based normalization that preserves realistic durations"""
        stop_cycle, _ = self._build_stop_aligned_cycle(trips, include_profiles=False)
        if stop_cycle is not None and not stop_cycle.empty:
            return stop_cycle

        speed_profiles: List[np.ndarray] = []
        durations: List[float] = []
        
        for trip in trips:
            speeds = self._extract_speeds(trip)
            if len(speeds) <= 10:
                continue
            duration = self._estimate_trip_duration(trip)
            if duration <= 0:
                continue
            speed_profiles.append(speeds)
            durations.append(duration)
        
        if not speed_profiles:
            return pd.DataFrame()

        target_len = int(np.median([len(profile) for profile in speed_profiles]))
        target_len = max(target_len, 10)
        x_new = np.linspace(0, 1, target_len)
        aligned_speeds = []
        
        for speeds in speed_profiles:
            x_old = np.linspace(0, 1, len(speeds))
            f = interp1d(x_old, speeds, kind='linear', fill_value='extrapolate')
            aligned_speeds.append(f(x_new))
        
        speeds_array = np.array(aligned_speeds, dtype=float)
        target_duration = float(np.median(durations))
        if not np.isfinite(target_duration) or target_duration <= 0:
            # Approximate duration using median step derived from profiles
            sample_ratios = [
                durations[i] / max(len(speed_profiles[i]) - 1, 1)
                for i in range(len(speed_profiles))
                if durations[i] > 0
            ]
            median_step = float(np.median(sample_ratios)) if sample_ratios else 0.5
            if not np.isfinite(median_step) or median_step <= 0:
                median_step = 0.5
            target_duration = median_step * (target_len - 1)

        time_axis = np.linspace(0, target_duration, target_len)
        stop_probability = compute_stop_probability_from_profiles(
            speeds_array,
            target_duration_s=target_duration,
        )
        speed_p10 = np.percentile(speeds_array, 10, axis=0)
        median_speed = np.median(speeds_array, axis=0)
        if stop_probability.size == median_speed.size:
            stop_mask_mid = stop_probability >= 0.5
            stop_mask_high = stop_probability >= 0.7
            median_speed = median_speed.copy()
            median_speed[stop_mask_mid] = np.minimum(
                median_speed[stop_mask_mid],
                speed_p10[stop_mask_mid],
            )
            median_speed[stop_mask_high] = np.minimum(
                median_speed[stop_mask_high],
                STOP_SPEED_THRESHOLD_MS * 0.2,
            )
        median_speed = np.clip(median_speed, 0.0, None)

        result_df = pd.DataFrame({
            'time_s': time_axis,
            'speed_ms': median_speed,
            'speed_kmh': median_speed * 3.6,
            'speed_std': np.std(speeds_array, axis=0),
            'speed_p10': speed_p10,
            'speed_p25': np.percentile(speeds_array, 25, axis=0),
            'speed_p75': np.percentile(speeds_array, 75, axis=0),
            'stop_probability': stop_probability,
            'n_trips': len(speed_profiles)
        })
        return _trim_cycle_idle_edges(result_df)

    def _create_stop_aligned_profile_improved(self,
                                              route_id: str,
                                              route_trips: List[pd.DataFrame],
                                              min_stop_duration_s: float = 5.0) -> pd.DataFrame:
        """
        Build a representative cycle by elastically aligning trips between successive stops.
        Returns legacy-normalised profile if insufficient stop structure is available.
        """
        if not route_trips:
            return pd.DataFrame()

        trip_patterns: List[Dict[str, Any]] = []
        for trip_idx, trip in enumerate(route_trips):
            speeds = self._extract_speeds(trip)
            if speeds.size < 50:
                continue
            dt_series = self._extract_dt_sequence(trip)
            dt_series = _ensure_length(np.asarray(dt_series, dtype=float), speeds.size,
                                       float(dt_series[-1]) if dt_series.size else 0.5)

            stop_mask = self._detect_stops_robust(speeds, dt_series, min_stop_duration_s=min_stop_duration_s)
            stop_segments = self._extract_stop_segments(speeds, dt_series, stop_mask)
            if len(stop_segments.get('stops', [])) < 2:
                continue
            trip_patterns.append({
                'trip_idx': trip_idx,
                'speeds': speeds,
                'dt': dt_series,
                'stop_segments': stop_segments,
                'trip_df': trip
            })

        if len(trip_patterns) < 3:
            return self._legacy_time_normalization(route_trips)

        pattern_groups = self._cluster_by_stop_pattern(trip_patterns)
        if not pattern_groups:
            return self._legacy_time_normalization(route_trips)

        largest_group = max(pattern_groups, key=len)
        aligned_profile = self._build_elastic_aligned_profile(largest_group)
        if aligned_profile.empty:
            print(f"  Stop-aligned profile fallback to legacy for {route_id}: unable to build elastic alignment")
            return self._legacy_time_normalization(route_trips)

        aligned_profile = aligned_profile.sort_values('time_s').reset_index(drop=True)
        return _trim_cycle_idle_edges(aligned_profile)

    def _detect_stops_robust(self,
                             speeds: np.ndarray,
                             dt: np.ndarray,
                             min_stop_duration_s: float = STOP_MIN_DWELL_S,
                             speed_threshold_ms: float = STOP_SPEED_THRESHOLD_MS) -> np.ndarray:
        """Detect stops using smoothed speeds and minimum dwell enforcement."""
        if speeds.size == 0:
            return np.zeros_like(speeds, dtype=bool)

        smoothed = gaussian_filter1d(speeds, sigma=2, mode='nearest')
        stop_mask = smoothed < speed_threshold_ms
        dt = np.where(dt > 0, dt, np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 0.5)
        median_dt = float(np.nanmedian(dt)) if np.any(np.isfinite(dt)) else 0.5
        median_dt = max(median_dt, 1e-3)
        min_samples = max(1, int(round(min_stop_duration_s / median_dt)))

        dt_cumsum = np.cumsum(dt)
        stop_starts: List[int] = []
        stop_ends: List[int] = []
        in_stop = False
        stop_start = 0

        for idx, flag in enumerate(stop_mask.astype(bool)):
            if flag and not in_stop:
                stop_start = idx
                in_stop = True
            elif not flag and in_stop:
                if idx - stop_start >= min_samples:
                    duration = dt_cumsum[idx - 1] - (dt_cumsum[stop_start - 1] if stop_start > 0 else 0.0)
                    if duration >= min_stop_duration_s:
                        stop_starts.append(stop_start)
                        stop_ends.append(idx)
                in_stop = False

        if in_stop and len(stop_mask) - stop_start >= min_samples:
            stop_starts.append(stop_start)
            stop_ends.append(len(stop_mask))

        final_mask = np.zeros_like(stop_mask, dtype=bool)
        for start, end in zip(stop_starts, stop_ends):
            final_mask[start:end] = True
        return final_mask

    def _extract_stop_segments(self,
                               speeds: np.ndarray,
                               dt: np.ndarray,
                               stop_mask: np.ndarray) -> Dict[str, Any]:
        """Extract stop segments and the drive segments between them."""
        segments = {
            'stops': [],
            'drives': [],
            'stop_positions': [],
            'drive_characteristics': []
        }
        if speeds.size == 0 or dt.size == 0:
            return segments

        mask = stop_mask.astype(bool)
        diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        stop_starts = np.where(diff_mask == 1)[0]
        stop_ends = np.where(diff_mask == -1)[0]

        total_time = float(np.sum(dt))
        time_cumsum = np.cumsum(dt)

        for idx, (start, end) in enumerate(zip(stop_starts, stop_ends)):
            stop_duration = float(np.sum(dt[start:end]))
            if total_time > 0:
                stop_position = float(time_cumsum[start] / total_time)
            else:
                stop_position = 0.0
            segments['stops'].append({
                'start_idx': int(start),
                'end_idx': int(end),
                'duration': stop_duration,
                'normalized_position': stop_position,
                'stop_number': idx
            })
            segments['stop_positions'].append(stop_position)

            if idx > 0:
                prev_end = stop_ends[idx - 1]
                drive_speeds = speeds[prev_end:start]
                drive_dt = dt[prev_end:start]
                if drive_speeds.size > 0:
                    duration = float(np.sum(drive_dt))
                    segments['drives'].append({
                        'speeds': drive_speeds.copy(),
                        'duration': duration,
                        'mean_speed': float(np.mean(drive_speeds)),
                        'max_speed': float(np.max(drive_speeds)),
                        'segment_idx': idx - 1
                    })
                    if np.mean(drive_speeds) > 0:
                        segments['drive_characteristics'].append({
                            'mean_speed': float(np.mean(drive_speeds)),
                            'cv': float(np.std(drive_speeds) / (np.mean(drive_speeds) + 1e-6)),
                            'acceleration_events': int(np.sum(np.diff(drive_speeds) > 0.5))
                        })

        return segments

    def _cluster_by_stop_pattern(self,
                                 trip_patterns: List[Dict[str, Any]],
                                 max_position_diff: float = 0.15) -> List[List[Dict[str, Any]]]:
        """Cluster trips by similar stop positions."""
        groups: List[List[Dict[str, Any]]] = []
        for pattern in trip_patterns:
            stop_positions = pattern['stop_segments'].get('stop_positions', [])
            if not stop_positions:
                continue
            matched = False
            for group in groups:
                ref_positions = group[0]['stop_segments']['stop_positions']
                if len(ref_positions) != len(stop_positions):
                    continue
                if all(abs(p1 - p2) < max_position_diff for p1, p2 in zip(stop_positions, ref_positions)):
                    group.append(pattern)
                    matched = True
                    break
            if not matched:
                groups.append([pattern])

        groups = [grp for grp in groups if len(grp) >= 3]
        groups.sort(key=len, reverse=True)
        return groups

    def _build_elastic_aligned_profile(self, pattern_group: List[Dict[str, Any]]) -> pd.DataFrame:
        """Construct a stop-aligned representative profile from clustered trips."""
        if not pattern_group:
            return pd.DataFrame()

        n_trips = len(pattern_group)
        all_positions = [pattern['stop_segments']['stop_positions'] for pattern in pattern_group]
        if not all_positions or any(len(pos) != len(all_positions[0]) for pos in all_positions):
            return pd.DataFrame()

        n_stops = len(all_positions[0])
        profile_records: List[Dict[str, Any]] = []
        current_time = 0.0

        for seg_idx in range(n_stops + 1):
            segment_speeds: List[np.ndarray] = []
            segment_durations: List[float] = []

            for pattern in pattern_group:
                stops = pattern['stop_segments']['stops']
                speeds = pattern['speeds']
                dt = pattern['dt']

                if seg_idx == 0:
                    end_idx = stops[0]['start_idx'] if stops else len(speeds)
                    seg_speeds = speeds[:end_idx]
                    seg_dt = dt[:end_idx]
                elif seg_idx == n_stops:
                    start_idx = stops[-1]['end_idx'] if stops else 0
                    seg_speeds = speeds[start_idx:]
                    seg_dt = dt[start_idx:]
                else:
                    start_idx = stops[seg_idx - 1]['end_idx']
                    end_idx = stops[seg_idx]['start_idx']
                    seg_speeds = speeds[start_idx:end_idx]
                    seg_dt = dt[start_idx:end_idx]

                if seg_speeds.size > 2 and seg_dt.size == seg_speeds.size:
                    segment_speeds.append(seg_speeds)
                    segment_durations.append(float(np.sum(seg_dt)))

            if segment_speeds and segment_durations:
                median_duration = float(np.median(segment_durations))
                if not np.isfinite(median_duration) or median_duration <= 0:
                    valid_durations = [d for d in segment_durations if np.isfinite(d) and d > 0]
                    median_duration = float(np.median(valid_durations)) if valid_durations else 0.5
                median_duration = max(median_duration, 0.5)
                target_samples = max(10, int(round(median_duration * 10)))

                resampled_segments = []
                for seg_speeds in segment_speeds:
                    x_old = np.linspace(0, 1, seg_speeds.size)
                    x_new = np.linspace(0, 1, target_samples)
                    resampled_segments.append(np.interp(x_new, x_old, seg_speeds))

                seg_array = np.vstack(resampled_segments)
                median_speeds = np.median(seg_array, axis=0)
                std_speeds = np.std(seg_array, axis=0)
                time_points = np.linspace(current_time, current_time + median_duration, target_samples, endpoint=False)

                for t, speed, std in zip(time_points, median_speeds, std_speeds):
                    profile_records.append({
                        'time_s': float(t),
                        'speed_ms': float(speed),
                        'speed_std': float(std),
                        'segment_type': 'drive' if seg_idx > 0 else 'start',
                        'segment_idx': int(seg_idx)
                    })

                current_time += median_duration

            if seg_idx < n_stops:
                stop_durations = [
                    pattern['stop_segments']['stops'][seg_idx]['duration']
                    for pattern in pattern_group
                    if seg_idx < len(pattern['stop_segments']['stops'])
                ]
                if stop_durations:
                    stop_duration = float(np.median(stop_durations))
                    if not np.isfinite(stop_duration) or stop_duration <= 0:
                        valid_stops = [d for d in stop_durations if np.isfinite(d) and d > 0]
                        stop_duration = float(np.median(valid_stops)) if valid_stops else 0.5
                    stop_duration = max(stop_duration, 0.5)
                    stop_samples = max(5, int(round(stop_duration * 10)))
                    stop_times = np.linspace(current_time, current_time + stop_duration, stop_samples, endpoint=False)
                    for t in stop_times:
                        profile_records.append({
                            'time_s': float(t),
                            'speed_ms': 0.0,
                            'speed_std': 0.0,
                            'segment_type': 'stop',
                            'segment_idx': int(seg_idx)
                        })
                    current_time += stop_duration

        if not profile_records:
            return pd.DataFrame()

        df = pd.DataFrame(profile_records)
        df = df.sort_values('time_s').drop_duplicates(subset='time_s').reset_index(drop=True)
        df['speed_kmh'] = df['speed_ms'] * 3.6
        df['stop_probability'] = (df['segment_type'] == 'stop').astype(float)
        df['n_trips'] = n_trips
        df['speed_p25'] = np.clip(df['speed_ms'] - df['speed_std'] * 0.67, 0.0, None)
        df['speed_p75'] = df['speed_ms'] + df['speed_std'] * 0.67
        return df

    def visualize_stop_aligned_comparison(self,
                                          route_id: str,
                                          original_df: pd.DataFrame,
                                          aligned_df: pd.DataFrame,
                                          output_path: Optional[Path] = None):
        """Optional helper to compare legacy and improved stop-aligned profiles."""
        if original_df.empty or aligned_df.empty:
            print(f"  Skip stop alignment comparison for {route_id}: missing data")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        if 'time_s' in original_df.columns and 'speed_kmh' in original_df.columns:
            ax1.plot(original_df['time_s'], original_df['speed_kmh'],
                     color='tab:blue', linewidth=1.5, label='Time-normalized')
            if 'stop_probability' in original_df.columns:
                stop_mask = original_df['stop_probability'] > 0.5
                for start, end in self._get_continuous_regions(stop_mask):
                    ax1.axvspan(original_df['time_s'].iloc[start],
                                original_df['time_s'].iloc[end - 1],
                                alpha=0.25, color='tab:red')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title(f'{route_id}: Original Time Normalization')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(aligned_df['time_s'], aligned_df['speed_kmh'],
                 color='tab:green', linewidth=1.5, label='Stop-aligned')
        stop_mask_aligned = (aligned_df.get('segment_type') == 'stop') if 'segment_type' in aligned_df.columns else (aligned_df['speed_ms'] < 1.0)
        for start, end in self._get_continuous_regions(stop_mask_aligned):
            ax2.axvspan(aligned_df['time_s'].iloc[start],
                        aligned_df['time_s'].iloc[end - 1],
                        alpha=0.25, color='tab:red')
        if {'speed_p25', 'speed_p75'}.issubset(aligned_df.columns):
            ax2.fill_between(aligned_df['time_s'],
                             aligned_df['speed_p25'] * 3.6,
                             aligned_df['speed_p75'] * 3.6,
                             alpha=0.2, color='tab:green')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title(f'{route_id}: Stop-Aligned Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        if 'stop_probability' in original_df.columns:
            ax3.plot(original_df['time_s'], original_df['stop_probability'],
                     color='tab:blue', alpha=0.7, label='Original')
        if 'stop_probability' in aligned_df.columns:
            ax3.plot(aligned_df['time_s'], aligned_df['stop_probability'],
                     color='tab:green', alpha=0.7, label='Stop-aligned')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Stop Probability')
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        if output_path is None:
            output_path = self.output_dir / f'{route_id}_stop_alignment_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Stop alignment comparison saved: {output_path}")

    def _get_continuous_regions(self, mask: Any) -> List[Tuple[int, int]]:
        """Find contiguous True regions in a boolean mask (array or Series)."""
        regions: List[Tuple[int, int]] = []
        start = None
        mask_iterable = mask.values if hasattr(mask, 'values') else mask
        for idx, flag in enumerate(mask_iterable):
            is_true = bool(flag)
            if is_true and start is None:
                start = idx
            elif not is_true and start is not None:
                regions.append((start, idx))
                start = None
        if start is not None:
            regions.append((start, len(mask_iterable)))
        return regions
    
    def _normalize_by_distance(self, trips: List[pd.DataFrame]) -> pd.DataFrame:
        """Distance-based normalization"""
        distance_profiles = []
        
        for trip in trips:
            speeds = self._extract_speeds(trip)
            if len(speeds) < 10:
                continue
            
            if 'dt_s' in trip.columns:
                dt = trip['dt_s'].fillna(0.1).values[:len(speeds)]
                distances = np.cumsum(speeds * dt)
            else:
                distances = np.cumsum(speeds * 0.1)
            
            if distances[-1] > 100:
                norm_dist = distances / distances[-1]
                distance_profiles.append((norm_dist, speeds))
        
        if distance_profiles:
            common_dist = np.linspace(0, 1, 100)
            aligned = []
            
            for dist, speed in distance_profiles:
                f = interp1d(dist, speed, kind='linear', fill_value='extrapolate')
                aligned.append(f(common_dist))
            
            speeds_array = np.array(aligned)
            return pd.DataFrame({
                'normalized_distance': common_dist,
                'speed_ms': np.median(speeds_array, axis=0),
                'speed_kmh': np.median(speeds_array, axis=0) * 3.6,
                'speed_std': np.std(speeds_array, axis=0),
                'n_trips': len(aligned)
            })
        
        return pd.DataFrame()
    
    def _normalize_stop_to_stop(self, trips: List[pd.DataFrame]) -> pd.DataFrame:
        """Normalize segments between successive high-confidence stops"""
        stop_segments: List[np.ndarray] = []
        segment_lengths: List[int] = []
        
        for trip in trips:
            speeds = self._extract_speeds(trip)
            if len(speeds) < 20:
                continue
            dt_series = self._extract_dt_sequence(trip)
            if len(dt_series) >= len(speeds):
                dt_series = dt_series[:len(speeds)]
            else:
                pad_val = dt_series[-1] if len(dt_series) else 0.5
                pad_len = len(speeds) - len(dt_series)
                dt_series = np.pad(dt_series, (0, pad_len), constant_values=pad_val)
            dt_median = float(np.clip(np.median(dt_series), 1e-3, None))
            stop_mask = build_stop_mask(speeds, dt_series)
            dwell_samples = max(1, int(round(STOP_MIN_DWELL_S / dt_median)))
            stop_spans = mask_to_index_spans(stop_mask, min_samples=dwell_samples)
            if len(stop_spans) < 2:
                continue
            min_seg_samples = max(5, int(round(STOP_SEGMENT_MIN_S / dt_median)))
            for idx_span in range(len(stop_spans) - 1):
                start_idx = stop_spans[idx_span][0]
                next_start = stop_spans[idx_span + 1][0]
                if next_start - start_idx < min_seg_samples:
                    continue
                segment = speeds[start_idx:next_start]
                if len(segment) >= min_seg_samples:
                    stop_segments.append(segment)
                    segment_lengths.append(len(segment))
        
        if not stop_segments:
            return pd.DataFrame()

        n_points = SEGMENT_RESAMPLE_POINTS
        aligned = []
        for seg in stop_segments:
            x_old = np.linspace(0, 1, len(seg))
            x_new = np.linspace(0, 1, n_points)
            f = interp1d(x_old, seg, kind='linear', fill_value='extrapolate')
            aligned.append(f(x_new))

        speeds_array = np.array(aligned, dtype=float)
        return pd.DataFrame({
            'normalized_position': np.linspace(0, 1, n_points),
            'speed_ms': np.median(speeds_array, axis=0),
            'speed_kmh': np.median(speeds_array, axis=0) * 3.6,
            'speed_std': np.std(speeds_array, axis=0),
            'n_segments': len(aligned),
            'segment_length_mean': float(np.mean(segment_lengths)) if segment_lengths else 0.0
        })
    
    def analyze_stop_to_stop_segments(self, route_id: str, trips: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze route dynamics between consecutive stops to preserve stop structure.
        Returns JSON-serializable statistics and representative segment clusters.
        """
        if not trips:
            return {}

        all_segments: List[Dict[str, Any]] = []
        segment_metadata: List[Dict[str, float]] = []
        stop_durations: List[float] = []
        segments_per_trip: Dict[int, int] = {}

        for trip_idx, trip in enumerate(trips):
            speeds = self._extract_speeds(trip)
            if len(speeds) < 20:
                continue

            dt_series = self._extract_dt_sequence(trip)
            dt_series = _ensure_length(
                np.asarray(dt_series, dtype=float),
                len(speeds),
                float(dt_series[-1]) if len(dt_series) else 0.5
            )

            stop_mask = build_stop_mask(speeds, dt_series)
            dwell_samples = max(1, int(round(STOP_MIN_DWELL_S / max(np.median(dt_series), 1e-3))))
            stop_spans = mask_to_index_spans(stop_mask, min_samples=dwell_samples)

            if len(stop_spans) < 2:
                continue

            # Collect stop duration statistics
            for span_start, span_end in stop_spans:
                duration = float(np.sum(dt_series[span_start:span_end]))
                if duration > 0:
                    stop_durations.append(duration)

            segments_per_trip[trip_idx] = 0
            for span_idx in range(len(stop_spans) - 1):
                seg_start = stop_spans[span_idx][1]
                seg_end = stop_spans[span_idx + 1][0]
                if seg_end <= seg_start:
                    continue

                dt_segment = dt_series[seg_start:seg_end]
                if dt_segment.size == 0:
                    continue

                duration = float(np.sum(dt_segment))
                if duration < STOP_SEGMENT_MIN_S:
                    continue

                segment_speeds = speeds[seg_start:seg_end]
                if segment_speeds.size < 5:
                    continue

                time_axis = np.cumsum(dt_segment)
                distance = float(np.sum(segment_speeds * dt_segment))
                if dt_segment.size > 1:
                    accel_dt = np.clip(dt_segment[1:], 1e-3, None)
                    accel = np.diff(segment_speeds) / accel_dt
                else:
                    accel = np.array([])
                accel_events = int(np.sum(accel > 1.0)) if accel.size else 0
                decel_events = int(np.sum(accel < -1.0)) if accel.size else 0

                segment_entry = {
                    'speeds': segment_speeds.copy(),
                    'time': time_axis.copy(),
                    'trip_idx': trip_idx,
                    'segment_idx': span_idx,
                    'duration': duration,
                    'distance': distance
                }
                all_segments.append(segment_entry)
                segment_metadata.append({
                    'mean_speed': float(np.mean(segment_speeds)),
                    'max_speed': float(np.max(segment_speeds)),
                    'acceleration_events': accel_events,
                    'deceleration_events': decel_events
                })
                segments_per_trip[trip_idx] += 1

        if not all_segments:
            return {
                'route_id': route_id,
                'n_segments': 0,
                'n_trips': len(trips),
                'segment_clusters': [],
                'segment_variability': {},
                'stop_statistics': {}
            }

        segment_clusters = self._cluster_stop_segments(all_segments, segment_metadata)
        variability = self._summarize_stop_segment_variability(all_segments, segment_metadata)
        stop_stats = self._summarize_stop_durations(stop_durations, segments_per_trip)

        return {
            'route_id': route_id,
            'n_segments': len(all_segments),
            'n_trips': len(trips),
            'segment_clusters': segment_clusters,
            'segment_variability': variability,
            'stop_statistics': stop_stats
        }

    def _cluster_stop_segments(self,
                               segments: List[Dict[str, Any]],
                               metadata: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Cluster stop-to-stop segments using k-means on duration/speed features."""
        if len(segments) < 3:
            profiles = self._create_representative_profile(segments)
            return [{
                'cluster_id': 0,
                'n_segments': len(segments),
                'summary': self._summarize_cluster_segments(segments, metadata),
                'representative_profile': profiles,
                'sample_segments': [
                    self._serialize_segment(seg) for seg in segments[:min(len(segments), 5)]
                ]
            }]

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("Warning: scikit-learn not available, skipping segment clustering")
            profiles = self._create_representative_profile(segments)
            return [{
                'cluster_id': 0,
                'n_segments': len(segments),
                'summary': self._summarize_cluster_segments(segments, metadata),
                'representative_profile': profiles,
                'sample_segments': [
                    self._serialize_segment(seg) for seg in segments[:min(len(segments), 5)]
                ]
            }]

        feature_rows: List[List[float]] = []
        for seg, meta in zip(segments, metadata):
            feature_rows.append([
                meta['mean_speed'],
                meta['max_speed'],
                seg['duration'],
                seg['distance'],
                meta['acceleration_events'],
                meta['deceleration_events']
            ])

        features = np.array(feature_rows, dtype=float)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        n_clusters = max(2, min(5, len(segments) // 3))
        if n_clusters <= 1:
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_scaled)

        clusters: List[Dict[str, Any]] = []
        for cluster_id in range(n_clusters):
            idx_mask = labels == cluster_id
            if not np.any(idx_mask):
                continue
            cluster_segments = [segments[i] for i, flag in enumerate(idx_mask) if flag]
            cluster_meta = [metadata[i] for i, flag in enumerate(idx_mask) if flag]
            summary = self._summarize_cluster_segments(cluster_segments, cluster_meta)
            representative = self._create_representative_profile(cluster_segments)
            sample_segments = [
                self._serialize_segment(cluster_segments[i])
                for i in range(min(len(cluster_segments), 5))
            ]
            clusters.append({
                'cluster_id': int(cluster_id),
                'n_segments': len(cluster_segments),
                'summary': summary,
                'representative_profile': representative,
                'sample_segments': sample_segments
            })

        clusters.sort(key=lambda c: c['n_segments'], reverse=True)
        return clusters

    def _create_representative_profile(self, segments: List[Dict[str, Any]]) -> List[float]:
        """Create a median representative profile for segments normalized to 100 samples."""
        if not segments:
            return []

        target_len = 100
        normalized_profiles: List[np.ndarray] = []

        for seg in segments:
            speeds = np.asarray(seg['speeds'], dtype=float)
            if speeds.size < 5:
                continue
            x_old = np.linspace(0, 1, speeds.size)
            x_new = np.linspace(0, 1, target_len)
            normalized = np.interp(x_new, x_old, speeds)
            normalized_profiles.append(normalized)

        if not normalized_profiles:
            return []

        median_profile = np.median(np.vstack(normalized_profiles), axis=0)
        return median_profile.tolist()

    def _serialize_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy-heavy segment structure into JSON serializable form."""
        return {
            'trip_idx': int(segment['trip_idx']),
            'segment_idx': int(segment['segment_idx']),
            'duration': float(segment['duration']),
            'distance': float(segment['distance']),
            'speeds_ms': np.asarray(segment['speeds'], dtype=float).tolist(),
            'time_s': np.asarray(segment['time'], dtype=float).tolist()
        }

    def _summarize_cluster_segments(self,
                                    segments: List[Dict[str, Any]],
                                    metadata: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute aggregate statistics for a cluster of stop-to-stop segments."""
        if not segments:
            return {}

        durations = [float(seg['duration']) for seg in segments]
        distances = [float(seg['distance']) for seg in segments]
        mean_speeds = [meta['mean_speed'] for meta in metadata]
        max_speeds = [meta['max_speed'] for meta in metadata]
        accel_events = [meta['acceleration_events'] for meta in metadata]
        decel_events = [meta['deceleration_events'] for meta in metadata]

        return {
            'duration_mean_s': float(np.mean(durations)),
            'duration_std_s': float(np.std(durations)),
            'duration_p95_s': float(np.percentile(durations, 95)),
            'distance_mean_m': float(np.mean(distances)),
            'distance_std_m': float(np.std(distances)),
            'mean_speed_ms': float(np.mean(mean_speeds)),
            'max_speed_ms': float(np.mean(max_speeds)),
            'accel_events_mean': float(np.mean(accel_events)),
            'decel_events_mean': float(np.mean(decel_events))
        }

    def _summarize_stop_segment_variability(self,
                                            segments: List[Dict[str, Any]],
                                            metadata: List[Dict[str, float]]) -> Dict[str, float]:
        """Summarize variability across all stop-to-stop segments."""
        if not segments:
            return {}

        durations = np.array([seg['duration'] for seg in segments], dtype=float)
        distances = np.array([seg['distance'] for seg in segments], dtype=float)
        mean_speeds = np.array([meta['mean_speed'] for meta in metadata], dtype=float)
        max_speeds = np.array([meta['max_speed'] for meta in metadata], dtype=float)

        return {
            'duration_mean_s': float(np.mean(durations)),
            'duration_std_s': float(np.std(durations)),
            'duration_cv': float(np.std(durations) / max(np.mean(durations), 1e-6)),
            'distance_mean_m': float(np.mean(distances)),
            'distance_std_m': float(np.std(distances)),
            'distance_cv': float(np.std(distances) / max(np.mean(distances), 1e-6)),
            'mean_speed_ms': float(np.mean(mean_speeds)),
            'mean_speed_std_ms': float(np.std(mean_speeds)),
            'max_speed_ms': float(np.mean(max_speeds)),
            'max_speed_std_ms': float(np.std(max_speeds)),
            'n_unique_trips': int(len({seg['trip_idx'] for seg in segments}))
        }

    def _summarize_stop_durations(self,
                                  durations: List[float],
                                  segments_per_trip: Dict[int, int]) -> Dict[str, Any]:
        """Summarize stop duration distribution and segment counts."""
        if not durations:
            return {}

        durations_arr = np.array(durations, dtype=float)
        return {
            'total_stops': int(len(durations)),
            'median_stop_duration_s': float(np.median(durations_arr)),
            'p90_stop_duration_s': float(np.percentile(durations_arr, 90)),
            'mean_stop_duration_s': float(np.mean(durations_arr)),
            'std_stop_duration_s': float(np.std(durations_arr)),
            'segments_per_trip': {
                str(trip_idx): int(count) for trip_idx, count in segments_per_trip.items()
            }
        }
    
    def visualize_route_variability(self, results: Dict):
        """Enhanced visualization with new metrics"""
        if not results or 'segments' not in results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract metrics from dictionary format segments
        segments_data = results['segments']
        chaos_indices = []
        jerk_values = []
        rpa_values = []
        chaos_for_rpa = []
        compression_values = []
        compliances = []
        
        for seg_dict in segments_data.values():
            if isinstance(seg_dict, dict):
                chaos_indices.append(seg_dict.get('chaos_index', 0))
                if seg_dict.get('jerk_rms', 0) > 0:
                    jerk_values.append(seg_dict['jerk_rms'])
                if seg_dict.get('rpa', 0) > 0:
                    rpa_values.append(seg_dict['rpa'])
                    chaos_for_rpa.append(seg_dict.get('chaos_index', 0))
                if seg_dict.get('compression_ratio', 0) > 0:
                    compression_values.append(seg_dict['compression_ratio'])
                if seg_dict.get('speed_limit_compliance', 0) > 0:
                    compliances.append(seg_dict['speed_limit_compliance'])
        
        # 1. Chaos index distribution (original)
        ax = axes[0, 0]
        if chaos_indices:
            ax.hist(chaos_indices, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(chaos_indices), color='red', linestyle='--', label='Mean')
            ax.set_xlabel('Chaos Index')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Chaos Distribution ({results.get("chaos_classification", "unknown")})')
            ax.legend()
        
        # 2. NEW: Jerk distribution
        ax = axes[0, 1]
        if jerk_values:
            ax.hist(jerk_values, bins=20, edgecolor='black', alpha=0.7, color='green')
            ax.axvline(np.mean(jerk_values), color='red', linestyle='--', label=f'Mean: {np.mean(jerk_values):.2f}')
            ax.set_xlabel('RMS Jerk (m/s³)')
            ax.set_ylabel('Frequency')
            ax.set_title('Driving Smoothness (Lower = Smoother)')
            ax.legend()
        
        # 3. NEW: RPA vs Chaos scatter
        ax = axes[0, 2]
        if rpa_values and chaos_for_rpa:
            ax.scatter(chaos_for_rpa, rpa_values, alpha=0.5)
            ax.set_xlabel('Chaos Index')
            ax.set_ylabel('RPA (m/s²)')
            ax.set_title('Aggressiveness vs Chaos')
            
            # Add correlation
            if len(rpa_values) > 2:
                corr = np.corrcoef(chaos_for_rpa, rpa_values)[0, 1]
                ax.text(0.1, 0.9, f'Correlation: {corr:.3f}', transform=ax.transAxes)
        
        # 4. NEW: Compression ratio
        ax = axes[1, 0]
        if compression_values:
            ax.hist(compression_values, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax.axvline(np.mean(compression_values), color='red', linestyle='--', label=f'Mean: {np.mean(compression_values):.2f}')
            ax.set_xlabel('Compression Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title('Pattern Complexity (Higher = More Complex)')
            ax.legend()
        
        # 5. Speed limit compliance (original)
        ax = axes[1, 1]
        if compliances:
            ax.hist(compliances, bins=20, edgecolor='black', alpha=0.7, color='blue')
            ax.axvline(np.mean(compliances), color='red', linestyle='--', label='Mean')
            ax.set_xlabel('Compliance Rate')
            ax.set_ylabel('Frequency')
            ax.set_title('Speed Limit Compliance')
            ax.legend()
        
        # 6. Custom cycles comparison (original)
        ax = axes[1, 2]
        custom_cycles = results.get('custom_cycles', {})
        for method, cycle_data in custom_cycles.items():
            if cycle_data and 'speed_ms' in cycle_data:
                speeds = cycle_data['speed_ms']
                if speeds:
                    x = range(len(speeds))
                    ax.plot(x, speeds, label=method, alpha=0.8)
        
        ax.set_xlabel('Normalized Position')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Custom Cycles by Normalization')
        ax.legend()
        
        plt.suptitle(f"Route {results['route_id']} Enhanced Variability Analysis")
        plt.tight_layout()
        
        output_file = self.output_dir / f"route_{results['route_id']}_variability.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Enhanced visualization saved to {output_file}")
