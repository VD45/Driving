#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality visualizations for standard drive cycles
Updated: Added PKE, V95, probability distributions, and full cycle wavelet visualization
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Set
from collections import defaultdict
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import binary_closing
from io import StringIO

from cycle_name_normalizer import canonicalize_name, build_display_name
from analysis_shared import fmt_p

# ======================== Config ========================

A4_PORTRAIT = (8.27, 11.69)  # inches
A4_LANDSCAPE = (11.69, 8.27)  # inches
DPI = 300
STOP_SPEED_THRESHOLD_KMH = 3.6  # corresponds to ~1 m/s
STOP_MIN_DURATION_S = 5.0
STOP_SHADE_COLOR = '#b0b0b0'
STOP_LINE_COLOR = '#b2182b'
STOP_PROB_THRESHOLD = 0.2
STOP_SMOOTH_WINDOW_S = 1.2
STOP_GAP_CLOSE_S = 3.0
ROUTE_COLOR_SEQUENCE = [
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
    '#1f78b4', '#33a02c', '#fb9a99', '#b15928'
]
ROUTE_LINESTYLES = ['-', '--', '-.', ':', (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1, 2, 1))]
SHOW_STOP_WINDOWS_FIG3 = False

plt.rcParams.update({
    "figure.figsize": A4_PORTRAIT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.facecolor": "white",      # WHITE background
    "axes.edgecolor": "black",      # Black border
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#222222",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.0,
})

# Family colors
FAMILY_COLORS = {
    'WLTP_Europe': '#0173B2',     # Blue
    'EPA': '#DE8F05',             # Orange
    'Artemis': '#029E73',         # Green
    'Asia': '#CC78BC',            # Light purple
    'Special': '#8B4513',         # Brown
}

FAMILY_HATCHES = {
    'WLTP_Europe': '',
    'EPA': '//',
    'Artemis': '\\',
    'Asia': '..',
    'Special': 'xx',
}

# Line styles for different families (for grayscale readability)
LINE_STYLES = {
    'WLTP_Europe': '-',          # Solid
    'EPA': '--',                 # Dashed
    'Artemis': '-.',            # Dash-dot
    'Asia': ':',                # Dotted
    'Special': (0, (5, 2)),     # Custom dash pattern
}

SET2_CONTRAST_COLORS = [
    '#66c2a5',  # soft teal
    '#fc8d62',  # orange
    '#8da0cb',  # lavender
    '#e78ac3',  # pink
    '#a6d854',  # lime
    '#ffd92f',  # yellow
    '#e5c494',  # beige
    '#b3b3b3',  # grey
]

PREFERRED_DISPLAY_NAMES = {
    "CADC HW150",
    "CADC HW130",
    "CADC Motorway 130 km/h",
    "CADC Motorway 150 km/h",
    "CADC Rural",
    "CADC Urban",
    "CLTC-P",
    "ECE-15 (Cold)",
    "EUDC",
    "FTP-75",
    "FTP-75 (10 Hz)",
    "FTP-75 (Cold)",
    "Grossglockner Downhill",
    "Grossglockner Uphill",
    "HWFET",
    "HWFET (Short)",
    "HWFET (Cold)",
    "IM240",
    "J10 (Cold)",
    "J10-15 (Cold)",
    "J15 (Cold)",
    "JC08",
    "LA92 dynamo",
    "LA92 (Cold)",
    "LA92 (Short)",
    "NEDC",
    "NYCC",
    "SC03",
    "SFTP US06",
    "UDDS",
    "US06",
    "US06 (Cold)",
    "WLTC Class 1",
    "WLTC Class 2",
    "WLTC class 3",
    "WLTC Global",
}

# Duration groups with EXACT max time scales
DURATION_GROUPS = [
    ("Short (<600s)", 0, 600, 400),           # tighter max_time_display = 400
    ("Medium (600-900s)", 600, 900, 900),     # max_time_display = 900
    ("Standard (900-1200s)", 900, 1200, 1200), # max_time_display = 1200
    ("Long (1200-1500s)", 1200, 1500, 1500),  # max_time_display = 1500
    ("Extended (1500-2500s)", 1500, 2500, 1900), # max_time_display = 1900
    ("Marathon (>2500s)", 2500, 10000, 3200),    # max_time_display = 3200
]

class ImprovedPublicationVisualizer:
    """Publication-quality visualizations with duration-based grouping and integrated wavelet analysis"""

    def __init__(self,
                 results_json: Optional[Path] = None,
                 cycles_dir: Optional[Path] = None,
                 export_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 sample_rate_hz: float = 10.0):

        # Default paths
        self.cycles_dir = Path(cycles_dir) if cycles_dir else Path(
            '${PROJECT_ROOT}/Data/standardized_cycles'
        )
        self.export_dir = Path(export_dir) if export_dir else Path(
            '${PROJECT_ROOT}/ML/outputs/exported_cycles'
        )
        self.output_dir = Path(output_dir) if output_dir else Path(
            '${PROJECT_ROOT}/ML/outputs/publication_figures'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        if results_json and Path(results_json).exists():
            with open(results_json, "r") as f:
                self.results = json.load(f)

        self.sr_hz = float(sample_rate_hz)
        self.standard_cycles: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, dict] = {}
        self.wavelet_metrics: Dict[str, dict] = {}  # Store wavelet metrics
        self._route_cycle_cache: Dict[str, pd.DataFrame] = {}
        self._route_metric_cache: Dict[str, dict] = {}
        self._route_best_matches: Dict[str, dict] = {}
        self.route_speed_limits: Dict[str, float] = self.results.get('route_speed_limits', {})
        routes_meta = self.results.get('routes', {})
        self.variant_map: Dict[str, str] = routes_meta.get('variant_map', {})
        self.variant_trip_counts: Dict[str, int] = routes_meta.get('variant_trip_counts', {})
        self.variant_metadata: Dict[str, Dict[str, Any]] = routes_meta.get('variant_metadata', {})

        self._load_standard_cycles()

        self.reference_standard_key, self.reference_standard_metrics = self._determine_reference_standard()
        self.reference_standard_label = self.reference_standard_metrics.get('display_name', 'Standard cycle')
        self.reference_standard_df = self.standard_cycles.get(self.reference_standard_key, pd.DataFrame())

    def _read_csv_compat(self, file_path, comment='#', encoding='utf-8-sig'):
        """Compatible CSV reader that handles encoding errors gracefully"""
        try:
            return pd.read_csv(file_path, comment=comment, encoding=encoding)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(file_path, comment=comment, encoding='latin-1')
            except Exception:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read().decode(encoding, errors='ignore')
                    return pd.read_csv(StringIO(content), comment=comment)
                except Exception as e:
                    raise Exception(f"Could not read CSV file: {e}")

    def _extract_display_name_from_csv(self, csv_path: Path) -> str:
        """Extract display name from comment header if present"""
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if s and s.startswith('#'):
                        name = s[1:].strip()
                        if name:
                            return name
        except Exception:
            pass
        return csv_path.stem

    def _calculate_pke(self, speed_ms: np.ndarray, sampling_rate: float) -> float:
        """Calculate Positive Kinetic Energy (PKE) in m/s²"""
        if len(speed_ms) < 2:
            return 0.0
        
        dt = 1.0 / sampling_rate
        # Calculate accelerations
        accel = np.diff(speed_ms) / dt
        
        # Only positive accelerations contribute to PKE
        positive_accels = accel[accel > 0]
        
        # PKE = sum of (v_i+1 - v_i)² / distance for positive accelerations
        # Using the simplified formula: sum of positive accel squared * dt / total_distance
        total_distance = np.sum(speed_ms) * dt  # meters
        
        if total_distance > 0:
            pke = np.sum(positive_accels ** 2) / (total_distance / 1000.0)  # per km
        else:
            pke = 0.0
        
        return float(pke)
    
    def _fit_probability_distribution(self, speed_kmh: np.ndarray) -> Dict:
        """Fit and identify the best probability distribution for speed data"""
        if len(speed_kmh) < 10:
            return {'distribution': 'insufficient_data', 'parameters': {}}
        
        # Remove zeros for better fitting (idle removed)
        speeds_nonzero = speed_kmh[speed_kmh > 1.0]
        if len(speeds_nonzero) < 10:
            speeds_nonzero = speed_kmh
        
        # Test these distributions
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'weibull': stats.weibull_min,
            'beta': stats.beta,
            'exponential': stats.expon
        }
        
        best_dist = None
        best_ks_stat = float('inf')
        best_params = {}
        
        for name, distribution in distributions.items():
            try:
                # Fit distribution
                params = distribution.fit(speeds_nonzero)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.kstest(speeds_nonzero, lambda x: distribution.cdf(x, *params))
                
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_dist = name
                    best_params = {
                        'params': params,
                        'ks_statistic': round(ks_stat, 4),
                        'ks_pvalue': float(ks_pval),
                        'ks_pvalue_fmt': fmt_p(ks_pval),
                    }
            except:
                continue
        
        # Check for bimodality
        hist, bins = np.histogram(speed_kmh, bins=30)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                peaks.append(i)
        
        if len(peaks) >= 2:
            best_dist = 'bimodal'
            best_params['n_modes'] = len(peaks)
        
        return {
            'distribution': best_dist if best_dist else 'unknown',
            'parameters': best_params
        }

    def _compute_enhanced_metrics(self, speed_kmh: np.ndarray, sampling_rate: float) -> dict:
        """Compute metrics including chaos index, PKE, and V95"""
        if speed_kmh is None or len(speed_kmh) < 3:
            return {}

        dt = 1.0 / max(sampling_rate, 1e-9)
        v_ms = np.asarray(speed_kmh, dtype=float) / 3.6
        a_ms2 = np.diff(v_ms) / dt

        # Distance (km)
        distance_km = float(np.sum(v_ms) * dt / 1000.0)

        # Chaos index (entropy * coefficient of variation) - compute as percentage
        hist, _ = np.histogram(speed_kmh, bins=20, range=(0, 140))
        if hist.sum() > 0:
            p = hist / hist.sum()
            entropy = -np.sum(p * np.log(p + 1e-10)) / np.log(20)
            cv = np.std(speed_kmh) / (np.mean(speed_kmh) + 1e-6)
            chaos_index = float(cv * entropy)
            chaos_pct = int(chaos_index * 100)  # Convert to percentage
        else:
            chaos_pct = 0

        # Idle %
        idle_pct = float(np.mean(speed_kmh < 1.0) * 100.0)
        
        # PKE (Positive Kinetic Energy)
        pke = self._calculate_pke(v_ms, sampling_rate)
        
        # V95 (95th percentile speed)
        v95 = float(np.percentile(speed_kmh, 95)) if len(speed_kmh) > 0 else 0.0

        # Stops count
        stopped = speed_kmh < 1.0
        stops = np.diff(np.concatenate(([0], stopped.astype(int), [0])))
        n_stops = int(np.sum(stops == 1))
        stops_per_km = float(n_stops / max(distance_km, 1e-9))

        return {
            'distance_km': distance_km,
            'chaos_pct': chaos_pct,
            'idle_pct': idle_pct,
            'pke': pke,  # Added PKE
            'v95': v95,  # Added V95
            'n_stops': n_stops,
            'stops_per_km': stops_per_km,
            'accel_max_ms2': float(np.nanmax(a_ms2)) if a_ms2.size else 0.0,
            'decel_min_ms2': float(np.nanmin(a_ms2)) if a_ms2.size else 0.0,
        }

    def _compute_wavelet_metrics(self, speed_kmh: np.ndarray, sampling_rate: float) -> dict:
        """Compute wavelet metrics for a single cycle"""
        try:
            from unified_wavelet_module import UnifiedWaveletAnalyzer
            
            analyzer = UnifiedWaveletAnalyzer(sampling_rate_hz=sampling_rate)
            results = analyzer.analyze_cycle(speed_kmh, cycle_name="temp", sampling_rate=sampling_rate)
            
            return {
                'wavelet_entropy': float(results['wavelet_entropy']),
                'band_micro_pct': round(results['cwt']['band_percentages'].get('micro', 0), 2),
                'band_short_pct': round(results['cwt']['band_percentages'].get('short', 0), 2),
                'band_medium_pct': round(results['cwt']['band_percentages'].get('medium', 0), 2),
                'band_long_pct': round(results['cwt']['band_percentages'].get('long', 0), 2),
                'band_macro_pct': round(results['cwt']['band_percentages'].get('macro', 0), 2),
                'transient_events_per_min': round(results['events']['events_per_minute'], 2)
            }
        except ImportError:
            print("Warning: Wavelet module not available")
            return {}
        except Exception as e:
            print(f"Warning: Wavelet computation failed: {e}")
            return {}

    def _detect_stop_spans(self,
                           time_values: np.ndarray,
                           speed_values: np.ndarray,
                           threshold_kmh: float = STOP_SPEED_THRESHOLD_KMH,
                           min_duration_s: float = STOP_MIN_DURATION_S) -> List[Tuple[float, float]]:
        """Identify continuous low-speed intervals using raw speeds"""
        if time_values.size == 0 or speed_values.size == 0:
            return []
        stop_mask = speed_values <= threshold_kmh
        return self._mask_to_spans(time_values, stop_mask, min_duration_s)

    def _probability_stop_spans(self,
                                time_values: np.ndarray,
                                stop_probability: np.ndarray,
                                threshold: float = STOP_PROB_THRESHOLD,
                                min_duration_s: float = STOP_MIN_DURATION_S) -> List[Tuple[float, float]]:
        if time_values.size == 0 or stop_probability.size != time_values.size:
            return []
        stop_mask = stop_probability >= threshold
        return self._mask_to_spans(time_values, stop_mask, min_duration_s)

    def _mask_to_spans(self,
                       time_values: np.ndarray,
                       mask: np.ndarray,
                       min_duration_s: float) -> List[Tuple[float, float]]:
        spans: List[Tuple[float, float]] = []
        idx = 0
        n = mask.size
        while idx < n:
            if mask[idx]:
                start_idx = idx
                while idx < n and mask[idx]:
                    idx += 1
                end_idx = idx - 1
                start_time = float(time_values[start_idx])
                end_time = float(time_values[min(end_idx, n - 1)])
                duration = end_time - start_time
                if idx < n:
                    end_time = float(time_values[idx - 1])
                    duration = end_time - start_time
                if duration >= min_duration_s:
                    spans.append((start_time, end_time))
            else:
                idx += 1
        return spans

    def _get_top_route_ids(self, limit: int = 5) -> List[str]:
        """Return the top recurring route identifiers by trip count"""
        routes_meta = self.results.get('routes', {})
        display_order = routes_meta.get('route_display_order')
        display_limit = routes_meta.get('route_display_limit', limit)
        if display_limit is None or display_limit <= 0:
            display_limit = limit
        candidate_ids: List[str] = []
        if display_order:
            candidate_ids.extend(display_order)
        else:
            routes = routes_meta.get('top_routes', {})
            candidate_ids.extend(sorted(routes.keys(), key=self._route_sort_key))
        # Append variants not already included
        for variant_id in sorted(self.variant_map.keys(), key=self._route_sort_key):
            if variant_id not in candidate_ids:
                candidate_ids.append(variant_id)
        candidate_ids = sorted(candidate_ids, key=self._route_sort_key)
        return candidate_ids[:display_limit]

    def _get_route_trip_count(self, route_id: str) -> int:
        """Helper to fetch the number of trips recorded for a route"""
        if route_id in self.variant_trip_counts:
            return int(self.variant_trip_counts.get(route_id, 0))
        routes = self.results.get('routes', {}).get('top_routes', {})
        return int(routes.get(route_id, 0))

    def _get_route_cycle(self, route_id: str) -> pd.DataFrame:
        """Fetch cached time-normalised custom cycle for a route"""
        if route_id in self._route_cycle_cache:
            return self._route_cycle_cache[route_id]

        base_route = route_id
        variant_cycles = {}
        if route_id in self.variant_map:
            base_route = self.variant_map[route_id]
        route_data = self.results.get('route_variability', {}).get(base_route, {})
        custom_cycles = route_data.get('custom_cycles', {}) if route_data else {}
        cycle_dict = None
        if route_id in self.variant_map:
            variant_cycles = custom_cycles.get('time_variants', {})
            cycle_dict = variant_cycles.get(route_id)
        else:
            cycle_dict = custom_cycles.get('time_normalized')

        if not cycle_dict:
            export_csv = self.export_dir / f"{route_id}_time_normalized_cycle.csv"
            if export_csv.exists():
                try:
                    df_cycle = pd.read_csv(export_csv)
                    for col in df_cycle.columns:
                        if df_cycle[col].dtype.kind in {'O', 'U'}:
                            df_cycle[col] = pd.to_numeric(df_cycle[col], errors='coerce')
                    if 'time_s' in df_cycle.columns:
                        df_cycle = df_cycle.sort_values('time_s').reset_index(drop=True)
                    self._route_cycle_cache[route_id] = df_cycle
                    return df_cycle
                except Exception as exc:
                    print(f"Warning: Could not read exported cycle for {route_id}: {exc}")
            self._route_cycle_cache[route_id] = pd.DataFrame()
            return self._route_cycle_cache[route_id]

        df_cycle = pd.DataFrame(cycle_dict)
        numeric_cols = [c for c in df_cycle.columns if df_cycle[c].dtype.kind in {'O', 'U'}]
        for col in numeric_cols:
            df_cycle[col] = pd.to_numeric(df_cycle[col], errors='coerce')

        if 'time_s' in df_cycle.columns:
            df_cycle = df_cycle.sort_values('time_s').reset_index(drop=True)

        self._route_cycle_cache[route_id] = df_cycle
        return df_cycle

    def _get_stop_profile_df(self, route_id: str) -> pd.DataFrame:
        """Return the detailed stop-to-stop profile if the augmented extraction is available."""
        stop_profiles = self.results.get('stop_to_stop_profiles', {})
        entry = stop_profiles.get(route_id, {})
        profile_data = entry.get('profile')
        if not profile_data:
            return pd.DataFrame()

        df = pd.DataFrame(profile_data)
        if df.empty:
            return df

        for col in df.columns:
            if col in {'segment_type', 'route_id'}:
                continue
            if df[col].dtype.kind in {'O', 'U'}:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'time_s' in df.columns and df['time_s'].notna().any():
            df = df.sort_values('time_s').reset_index(drop=True)
        elif 'distance_m' in df.columns and df['distance_m'].notna().any():
            df = df.sort_values('distance_m').reset_index(drop=True)
        elif 'normalized_distance' in df.columns and df['normalized_distance'].notna().any():
            df = df.sort_values('normalized_distance').reset_index(drop=True)

        metadata = entry.get('metadata', {}) or {}
        stop_count = len(metadata.get('stop_summaries', []) or [])
        n_trips = self._get_route_trip_count(route_id)
        if stop_count > 0:
            df['stop_count'] = float(stop_count)
        if n_trips:
            df['n_trips'] = float(n_trips)
        return df

    def _get_route_cycle_by_mode(self, route_id: str, mode: str = "time") -> pd.DataFrame:
        """Return representative cycle for a route by normalization mode."""
        stop_profile_df: Optional[pd.DataFrame] = None
        if mode == "stop":
            stop_profile_df = self._get_stop_profile_df(route_id)
            if stop_profile_df is not None and not stop_profile_df.empty:
                return stop_profile_df
        elif mode == "distance":
            stop_profile_df = self._get_stop_profile_df(route_id)

        route_data = self.results.get('route_variability', {}).get(route_id, {})
        custom = route_data.get('custom_cycles', {}) if route_data else {}
        if mode == "stop":
            data = custom.get('stop_to_stop', {})
        elif mode == "distance":
            data = custom.get('distance_normalized', {})
        else:
            data = custom.get('time_normalized', {})

        if data:
            df = pd.DataFrame(data)
            for c in df.columns:
                if df[c].dtype.kind in {'O', 'U'}:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            if 'time_s' in df.columns:
                df = df.sort_values('time_s').reset_index(drop=True)
            if mode in {"stop", "distance"}:
                df = self._ensure_stop_cycle_distance(df, route_id=route_id)
            return df

        suffix_map = {
            'stop': 'stop_to_stop',
            'distance': 'distance_normalized',
        }
        suffix = suffix_map.get(mode, 'time_normalized')
        csv_path = self.export_dir / f"{route_id}_{suffix}_cycle.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for c in df.columns:
                if df[c].dtype.kind in {'O', 'U'}:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            if 'time_s' in df.columns:
                df = df.sort_values('time_s').reset_index(drop=True)
            if mode in {"stop", "distance"}:
                df = self._ensure_stop_cycle_distance(df, route_id=route_id)
            return df

        if stop_profile_df is not None and not stop_profile_df.empty:
            if mode == "distance":
                stop_profile_df = self._ensure_stop_cycle_distance(stop_profile_df.copy(), route_id=route_id)
            return stop_profile_df

        return pd.DataFrame()

    def _ensure_stop_cycle_distance(self, df: pd.DataFrame, route_id: Optional[str] = None) -> pd.DataFrame:
        if df.empty:
            return df

        total_distance = None
        if route_id:
            route_meta = self.results.get('route_variability', {}).get(route_id, {}) or {}
            stop_analysis = route_meta.get('stop_segment_analysis', {}) or {}
            if isinstance(stop_analysis, dict):
                total_distance = stop_analysis.get('total_distance_m')
                if not total_distance:
                    profile_meta = stop_analysis.get('profile', {}) or {}
                    total_distance = profile_meta.get('total_distance_m')
            if (not total_distance) and route_meta:
                custom_cycles = route_meta.get('custom_cycles', {}) or {}
                time_cycle = custom_cycles.get('time_normalized', {}) or {}
                if time_cycle:
                    df_time = pd.DataFrame(time_cycle)
                    if not df_time.empty:
                        time_series = pd.to_numeric(df_time.get('time_s', pd.Series()), errors='coerce')
                        speed_series = None
                        if 'speed_ms' in df_time.columns:
                            speed_series = pd.to_numeric(df_time['speed_ms'], errors='coerce')
                        elif 'speed_kmh' in df_time.columns:
                            speed_series = pd.to_numeric(df_time['speed_kmh'], errors='coerce') / 3.6
                        if speed_series is not None:
                            combo = pd.DataFrame({
                                'time_s': time_series,
                                'speed_ms': speed_series
                            }).dropna()
                            if not combo.empty:
                                combo = combo.sort_values('time_s')
                                time_vals = combo['time_s'].to_numpy()
                                speed_vals = combo['speed_ms'].to_numpy()
                                if time_vals.size == speed_vals.size and time_vals.size > 1:
                                    distance_est = float(np.trapz(speed_vals, time_vals))
                                    if np.isfinite(distance_est) and distance_est > 0:
                                        total_distance = distance_est

        distance_series = None
        if 'distance_m' in df.columns:
            distance_series = pd.to_numeric(df['distance_m'], errors='coerce')

        normalized_series = None
        if 'normalized_distance' in df.columns:
            normalized_series = pd.to_numeric(df['normalized_distance'], errors='coerce')
        elif 'normalized_position' in df.columns:
            normalized_series = pd.to_numeric(df['normalized_position'], errors='coerce')

        normalized_base = None
        if normalized_series is not None and normalized_series.notna().any():
            normalized_base = normalized_series.fillna(method='ffill').fillna(method='bfill')

        if total_distance and normalized_base is not None:
            total_distance = float(total_distance)
            if total_distance > 0:
                df['distance_m'] = normalized_base.to_numpy() * total_distance
                df['normalized_distance'] = normalized_base.to_numpy()
                distance_series = pd.Series(df['distance_m'])
        elif distance_series is None and normalized_base is not None:
            if {'segment_length_mean', 'n_segments'}.issubset(df.columns):
                seg_len = float(pd.to_numeric(df['segment_length_mean'], errors='coerce').fillna(0).iloc[0])
                n_seg = float(pd.to_numeric(df['n_segments'], errors='coerce').fillna(len(df)).iloc[0])
                total_distance_est = seg_len * n_seg
            else:
                total_distance_est = 0.0
            if total_distance_est > 0:
                df['distance_m'] = normalized_base.to_numpy() * total_distance_est
                df['normalized_distance'] = normalized_base.to_numpy()
                distance_series = pd.Series(df['distance_m'])

        if distance_series is not None and total_distance:
            current_max = float(distance_series.max()) if not distance_series.dropna().empty else np.nan
            if not np.isfinite(current_max) or current_max <= 0:
                current_max = float(distance_series.dropna().max()) if not distance_series.dropna().empty else np.nan
            if np.isfinite(current_max) and current_max > 0:
                deviation = abs(current_max - total_distance) / max(total_distance, 1e-6)
                if deviation > 0.2 and normalized_base is not None:
                    df['distance_m'] = normalized_base.to_numpy() * total_distance
                    df['normalized_distance'] = normalized_base.to_numpy()

        if 'distance_m' in df.columns and 'normalized_distance' not in df.columns:
            distance_series = pd.to_numeric(df['distance_m'], errors='coerce')
            max_distance = float(distance_series.max()) if not distance_series.dropna().empty else 0.0
            if np.isfinite(max_distance) and max_distance > 0:
                df['normalized_distance'] = (distance_series / max_distance).to_numpy()

        return df

    def _get_route_stop_probability(self, route_id: str) -> np.ndarray:
        df_cycle = self._get_route_cycle(route_id)
        if df_cycle.empty:
            return np.array([])
        for candidate in ['stop_probability', 'stop_prob', 'stop_pct']:
            if candidate in df_cycle.columns:
                base = pd.to_numeric(df_cycle[candidate], errors='coerce').fillna(0).to_numpy()
                break
        else:
            base = None
        dt_est = self._estimate_cycle_dt(df_cycle)
        if base is None or base.size == 0:
            base = self._stop_probability_from_speed(df_cycle, dt_est)
        if base.size == 0:
            return np.array([])
        base = np.asarray(base, dtype=float)
        cycle_len = len(df_cycle)
        if base.size != cycle_len and cycle_len > 0:
            x_src = np.linspace(0, 1, base.size)
            x_dst = np.linspace(0, 1, cycle_len)
            base = np.interp(x_dst, x_src, base)
        return self._smooth_stop_probability(base, dt_est)

    def _get_route_stop_summary(self, route_id: str) -> Dict:
        return self.results.get('route_variability', {}).get(route_id, {}).get('stop_summary', {})

    def _compute_stop_durations_from_series(self,
                                            time_values: np.ndarray,
                                            speed_values: np.ndarray,
                                            threshold_kmh: float = STOP_SPEED_THRESHOLD_KMH,
                                            min_duration_s: float = STOP_MIN_DURATION_S) -> List[float]:
        if time_values.size == 0 or speed_values.size == 0:
            return []
        mask = speed_values <= threshold_kmh
        spans = self._mask_to_spans(time_values, mask, min_duration_s)
        durations = [end - start for start, end in spans]
        return durations

    def _compute_route_cycle_metrics(self, route_id: str) -> Dict[str, float]:
        """Compute enhanced metrics for a route's representative cycle"""
        if route_id in self._route_metric_cache:
            return self._route_metric_cache[route_id]

        df_cycle = self._get_route_cycle(route_id)
        if df_cycle.empty:
            self._route_metric_cache[route_id] = {}
            return {}

        if 'speed_kmh' in df_cycle.columns:
            speeds_kmh = pd.to_numeric(df_cycle['speed_kmh'], errors='coerce').fillna(0).to_numpy()
        elif 'speed_ms' in df_cycle.columns:
            speeds_kmh = pd.to_numeric(df_cycle['speed_ms'], errors='coerce').fillna(0).to_numpy() * 3.6
        else:
            self._route_metric_cache[route_id] = {}
            return {}

        if speeds_kmh.size < 3:
            self._route_metric_cache[route_id] = {}
            return {}

        if 'time_s' in df_cycle.columns and df_cycle['time_s'].notna().sum() > 1:
            time_vals = pd.to_numeric(df_cycle['time_s'], errors='coerce').fillna(0).to_numpy()
            time_vals = np.asarray(time_vals, dtype=float)
            dt = np.diff(time_vals)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            sampling_rate = 1.0 / np.mean(dt) if dt.size else self.sr_hz
        else:
            sampling_rate = self.sr_hz

        enhanced = self._compute_enhanced_metrics(speeds_kmh, sampling_rate)
        enhanced['mean_speed_kmh'] = float(np.mean(speeds_kmh))
        enhanced['duration_s'] = float(len(speeds_kmh) / max(sampling_rate, 1e-6))
        enhanced['v95'] = float(np.percentile(speeds_kmh, 95)) if speeds_kmh.size else 0.0
        wavelet = self._compute_wavelet_metrics(speeds_kmh, sampling_rate)
        if wavelet:
            enhanced['wavelet_entropy'] = wavelet.get('wavelet_entropy')

        self._route_metric_cache[route_id] = enhanced
        return enhanced

    def _get_route_palette(self, route_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate grayscale-safe colour/line-style pairs per route."""
        palette: Dict[str, Dict[str, Any]] = {}
        if not route_ids:
            return palette
        for idx, route_id in enumerate(route_ids):
            color = ROUTE_COLOR_SEQUENCE[idx % len(ROUTE_COLOR_SEQUENCE)]
            linestyle = ROUTE_LINESTYLES[idx % len(ROUTE_LINESTYLES)]
            palette[route_id] = {
                'color': color,
                'linestyle': linestyle,
            }
        return palette

    def _determine_speed_axis_limit(self, max_speed: float) -> float:
        """Snap the panel y-limit to sensible urban speed bands (30/40/50/60/70...)."""
        thresholds = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0]
        target = max(float(max_speed), 10.0)
        for threshold in thresholds:
            if target <= threshold * 0.98:
                return threshold
        rounded = math.ceil(target / 10.0) * 10.0
        return float(rounded)

    def _draw_speed_limit_line(self,
                               ax: Axes,
                               existing_limits: List[float],
                               limit_value: float) -> None:
        """Draw a labelled speed-limit guide line if it hasn't been rendered already."""
        if not np.isfinite(limit_value):
            return
        for existing in existing_limits:
            if abs(existing - limit_value) < 0.25:
                return
        ax.axhline(limit_value, color=STOP_LINE_COLOR, linestyle='--', linewidth=1.2, alpha=0.9)
        ax.text(0.98,
                limit_value,
                f"{int(round(limit_value))} km/h speed limit",
                color=STOP_LINE_COLOR,
                fontsize=8,
                ha='right',
                va='bottom',
                transform=ax.get_yaxis_transform())
        existing_limits.append(limit_value)

    def _get_route_style(self, route_id: str, palette: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
        style = palette.get(route_id, {})
        color = style.get('color', '#111111')
        linestyle = style.get('linestyle', '-')
        return color, linestyle

    def _route_sort_key(self, route_id: str) -> Tuple[int, int]:
        base_id = route_id
        variant_idx = 0
        if route_id in self.variant_map:
            base_id = self.variant_map[route_id]
            if "_v" in route_id:
                try:
                    variant_idx = int(route_id.split("_v")[-1])
                except ValueError:
                    variant_idx = 0
        try:
            base_num = int(base_id.split('_')[-1])
        except ValueError:
            base_num = 0
        return (base_num, variant_idx)

    def _format_route_label(self, route_id: str, trips: int) -> str:
        base_id = route_id
        variant_suffix = ""
        if route_id in self.variant_map:
            base_id = self.variant_map[route_id]
            if "_v" in route_id:
                variant_suffix = f" variant {route_id.split('_v')[-1]}"
        try:
            base_num = int(base_id.split('_')[-1]) + 1
        except ValueError:
            base_num = base_id
        return f"Route {base_num}{variant_suffix} (n={trips})"

    def _find_best_standard_cycle(self, target_metrics: Dict[str, float], metric_keys: List[str]) -> Tuple[str, Dict]:
        """Identify closest standard cycle using scaled Euclidean distance across metrics"""
        if not self.metrics or not target_metrics:
            return "", {}

        standard_vectors = []
        for key, metrics in self.metrics.items():
            vector = [float(metrics.get(mk, 0.0)) for mk in metric_keys]
            standard_vectors.append((key, vector, metrics))

        if not standard_vectors:
            return "", {}

        scales = []
        for idx, mk in enumerate(metric_keys):
            values = np.array([vec[idx] for _, vec, _ in standard_vectors], dtype=float)
            scale = np.std(values)
            if scale <= 1e-6:
                scale = np.mean(np.abs(values)) or 1.0
            scales.append(scale)

        target_vector = np.array([float(target_metrics.get(mk, 0.0)) for mk in metric_keys], dtype=float)
        best_key = ""
        best_metrics = {}
        best_distance = np.inf

        for cycle_key, vector, metrics in standard_vectors:
            diff = (target_vector - np.array(vector, dtype=float)) / scales
            distance = float(np.sum(diff ** 2))
            if distance < best_distance:
                best_distance = distance
                best_key = cycle_key
                best_metrics = metrics

        return best_key, best_metrics

    def _resample_curve(self, time_values: np.ndarray, speed_values: np.ndarray, target_points: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """Resample a speed trajectory onto a uniform 0-1 grid"""
        if time_values.size == 0 or speed_values.size == 0:
            return np.array([]), np.array([])

        time_values = np.asarray(time_values, dtype=float)
        speed_values = np.asarray(speed_values, dtype=float)

        if not np.isfinite(time_values).all():
            time_values = np.nan_to_num(time_values, nan=0.0)
        if not np.isfinite(speed_values).all():
            speed_values = np.nan_to_num(speed_values, nan=0.0)

        if time_values.max() - time_values.min() < 1e-6:
            normalized_time = np.linspace(0.0, 1.0, speed_values.size)
            resampled_speed = np.interp(np.linspace(0.0, 1.0, target_points), normalized_time, speed_values)
            return np.linspace(0.0, 1.0, target_points), resampled_speed

        normalized_time = (time_values - time_values.min()) / (time_values.max() - time_values.min())
        normalized_time[0] = 0.0
        normalized_time[-1] = 1.0
        grid = np.linspace(0.0, 1.0, target_points)
        resampled = np.interp(grid, normalized_time, speed_values)
        return grid, resampled

    def _determine_reference_standard(self) -> Tuple[str, Dict]:
        """Select the reference standard cycle (prefer WLTC Class 1)"""
        comparison = self.results.get('cycle_comparison', {}).get('best_matches', [])
        for entry in comparison:
            cycle_key = entry[0]
            if cycle_key in self.metrics:
                return cycle_key, self.metrics[cycle_key]

        # Prefer WLTC Class 1 explicitly if present
        for key, metrics in self.metrics.items():
            display = metrics.get('display_name', '')
            if 'WLTC Class 1' in display:
                return key, metrics

        # Fall back to any available cycle
        if self.metrics:
            key, metrics = next(iter(self.metrics.items()))
            return key, metrics

        return '', {}

    def _snap_speed_limit(self, raw_value: float) -> float:
        """Snap derived speed limits to typical posted values"""
        if not np.isfinite(raw_value) or raw_value <= 0:
            return 50.0
        if raw_value < 35:
            return 30.0
        if raw_value < 52:
            return 40.0
        if raw_value < 70:
            return 60.0
        return round(raw_value / 10.0) * 10.0

    def _derive_speed_limit(self, route_id: str, speed_series: np.ndarray) -> float:
        """Estimate a representative speed limit for the route"""
        limit = float(self.route_speed_limits.get(route_id, 0.0))
        v95 = float(np.percentile(speed_series, 95)) if speed_series.size else 0.0
        candidates = [val for val in (limit, v95) if np.isfinite(val) and val > 0]
        if not candidates:
            estimate = 50.0
        else:
            estimate = max(candidates)
        return self._snap_speed_limit(estimate)

    def _estimate_cycle_dt(self, df_cycle: pd.DataFrame) -> float:
        """Infer sampling interval for a representative cycle"""
        if df_cycle is None or df_cycle.empty:
            return 1.0 / max(self.sr_hz, 1e-6)
        if 'time_s' in df_cycle.columns:
            time_vals = pd.to_numeric(df_cycle['time_s'], errors='coerce').to_numpy()
            diffs = np.diff(time_vals[np.isfinite(time_vals)])
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                return float(np.median(diffs))
        return 1.0 / max(self.sr_hz, 1e-6)

    def _rolling_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        return (
            pd.Series(values, dtype=float)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    def _smooth_stop_probability(self, stop_prob: np.ndarray, dt_est: float) -> np.ndarray:
        if stop_prob.size == 0:
            return stop_prob
        dt_est = max(dt_est, 1e-3)
        smooth_window = max(1, int(round(STOP_SMOOTH_WINDOW_S / dt_est)))
        refined = self._rolling_mean(stop_prob, smooth_window)
        gap_window = max(1, int(round(STOP_GAP_CLOSE_S / dt_est)))
        if gap_window > 1:
            mask = refined >= STOP_PROB_THRESHOLD
            mask = binary_closing(mask, structure=np.ones(gap_window, dtype=bool))
            refined = np.where(mask, refined, refined)
        return np.clip(refined, 0.0, 1.0)

    def _stop_probability_from_speed(self, df_cycle: pd.DataFrame, dt_est: float) -> np.ndarray:
        speeds = None
        if 'speed_ms' in df_cycle.columns:
            speeds = pd.to_numeric(df_cycle['speed_ms'], errors='coerce').fillna(0).to_numpy()
        elif 'speed_kmh' in df_cycle.columns:
            speeds = pd.to_numeric(df_cycle['speed_kmh'], errors='coerce').fillna(0).to_numpy() / 3.6
        if speeds is None:
            return np.array([])
        smooth_window = max(1, int(round(STOP_SMOOTH_WINDOW_S / dt_est)))
        smoothed = self._rolling_mean(speeds, smooth_window)
        mask = smoothed <= (STOP_SPEED_THRESHOLD_KMH / 3.6)
        dwell_window = max(1, int(round(STOP_MIN_DURATION_S / dt_est)))
        if dwell_window > 1:
            dwell = self._rolling_mean(mask.astype(float), dwell_window)
            mask = dwell >= 0.8
        gap_window = max(1, int(round(STOP_GAP_CLOSE_S / dt_est)))
        if gap_window > 1:
            mask = binary_closing(mask, structure=np.ones(gap_window, dtype=bool))
        return mask.astype(float)

    def create_urban_routes_speed_comparison(self,
                                             save_path: Optional[Path] = None,
                                             normalization: str = "time",
                                             show_stop_windows: bool = True,
                                             max_routes_per_panel: int = 3) -> Optional[Path]:
        """Figure 3: Overlay representative speed profiles grouped by duration"""
        route_ids = self._get_top_route_ids(limit=15)
        if not route_ids:
            print("No recurring routes available for comparison figure")
            return None
        route_ids = sorted(route_ids, key=self._route_sort_key)
        palette = self._get_route_palette(route_ids)
        plot_meta: List[Dict[str, Any]] = []
        max_speed = 0.0

        for route_id in route_ids:
            mode_lookup = {
                "stop": "stop",
                "distance": "distance",
            }
            df_cycle = self._get_route_cycle_by_mode(route_id, mode_lookup.get(normalization, "time"))
            if df_cycle.empty:
                continue
            time_values = pd.to_numeric(df_cycle.get('time_s', pd.Series()), errors='coerce').fillna(method='ffill').fillna(0).to_numpy()
            if time_values.size == 0:
                time_values = np.arange(len(df_cycle)) / max(self.sr_hz, 1e-6)
            if 'speed_kmh' in df_cycle.columns:
                speed_values = pd.to_numeric(df_cycle['speed_kmh'], errors='coerce').fillna(0).to_numpy()
            elif 'speed_ms' in df_cycle.columns:
                speed_values = pd.to_numeric(df_cycle['speed_ms'], errors='coerce').fillna(0).to_numpy() * 3.6
            else:
                continue
            if speed_values.size < 10 or time_values.size != speed_values.size:
                continue
            stop_prob = None
            for cand in ['stop_probability', 'stop_prob', 'stop_pct']:
                if cand in df_cycle.columns:
                    stop_prob = pd.to_numeric(df_cycle[cand], errors='coerce').fillna(0).to_numpy()
                    break
            if stop_prob is None:
                stop_prob = self._stop_probability_from_speed(df_cycle, self._estimate_cycle_dt(df_cycle))
            stop_prob = np.asarray(stop_prob, dtype=float)
            finite_stop_prob = stop_prob[np.isfinite(stop_prob)]
            stop_prob_max = float(finite_stop_prob.max()) if finite_stop_prob.size else float('nan')
            duration = float(time_values[-1]) if time_values.size else 0.0
            max_speed = max(max_speed, float(np.nanmax(speed_values)))
            limit_value = self._derive_speed_limit(route_id, speed_values)
            trips = self._get_route_trip_count(route_id)
            stop_meta = self.results.get('route_variability', {}).get(route_id, {}).get('stop_segment_analysis', {})
            total_distance_m = None
            if isinstance(stop_meta, dict):
                total_distance_m = stop_meta.get('total_distance_m')
                if not total_distance_m:
                    profile_meta = stop_meta.get('profile', {}) or {}
                    total_distance_m = profile_meta.get('total_distance_m')
            total_distance_val = None
            if total_distance_m is not None:
                try:
                    total_distance_val = float(total_distance_m)
                    if not np.isfinite(total_distance_val) or total_distance_val <= 0:
                        total_distance_val = None
                except (TypeError, ValueError):
                    total_distance_val = None

            x_values = time_values
            x_label = 'Time (s)'
            x_is_time = True
            label_is_absolute = False
            if normalization in {"stop", "distance"}:
                distance_values = None
                if 'distance_m' in df_cycle.columns:
                    dist_m = (
                        pd.to_numeric(df_cycle['distance_m'], errors='coerce')
                        .fillna(method='ffill')
                        .fillna(method='bfill')
                        .fillna(0.0)
                        .to_numpy()
                    )
                    if total_distance_val is not None:
                        dist_m = np.clip(dist_m, 0.0, total_distance_val)
                    distance_values = dist_m / 1000.0
                    label_is_absolute = True
                else:
                    norm_column = None
                    if 'normalized_distance' in df_cycle.columns:
                        norm_column = 'normalized_distance'
                    elif 'normalized_position' in df_cycle.columns:
                        norm_column = 'normalized_position'
                    if norm_column:
                        norm_vals = (
                            pd.to_numeric(df_cycle[norm_column], errors='coerce')
                            .fillna(method='ffill')
                            .fillna(method='bfill')
                            .fillna(0.0)
                            .to_numpy()
                        )
                        if total_distance_val is not None:
                            distance_values = norm_vals * (total_distance_val / 1000.0)
                            label_is_absolute = True
                        else:
                            distance_values = norm_vals

                if distance_values is not None and distance_values.size == speed_values.size:
                    x_values = distance_values
                    x_label = 'Distance (km)' if label_is_absolute else 'Normalized distance'
                    x_is_time = False
            if x_values.size != speed_values.size:
                continue
            axis_extent = float(x_values[-1]) if x_values.size else 0.0
            group_metric = duration if x_is_time else axis_extent
            stop_positions: List[float] = []
            if {'segment_type', 'stop_index'}.issubset(df_cycle.columns):
                stop_slice = df_cycle[df_cycle['segment_type'] == 'stop']
                if not stop_slice.empty:
                    axis_column: Optional[str] = None
                    scale_factor = 1.0
                    if x_is_time and 'time_s' in stop_slice.columns:
                        axis_column = 'time_s'
                    elif not x_is_time:
                        if label_is_absolute and 'distance_m' in stop_slice.columns:
                            axis_column = 'distance_m'
                            scale_factor = 1.0 / 1000.0
                        elif 'normalized_distance' in stop_slice.columns:
                            axis_column = 'normalized_distance'
                        elif 'distance_m' in stop_slice.columns:
                            axis_column = 'distance_m'
                            if total_distance_val and total_distance_val > 0:
                                scale_factor = 1.0 / total_distance_val
                            else:
                                scale_factor = 1.0 / 1000.0
                    if axis_column and axis_column in stop_slice.columns:
                        for _, group in stop_slice.groupby('stop_index'):
                            axis_vals = pd.to_numeric(group[axis_column], errors='coerce').to_numpy()
                            if axis_vals.size == 0:
                                continue
                            axis_val = float(np.nanmedian(axis_vals))
                            if not np.isfinite(axis_val):
                                continue
                            stop_positions.append(axis_val * scale_factor)
            plot_meta.append({
                'route_id': route_id,
                'x': x_values,
                'x_label': x_label,
                'x_is_time': x_is_time,
                'axis_extent': axis_extent,
                'speed': speed_values,
                'stop_prob': stop_prob,
                'duration': duration,
                'group_metric': group_metric,
                'stop_meta': stop_meta,
                'limit_value': limit_value,
                'label': self._format_route_label(route_id, trips),
                'stop_prob_max': stop_prob_max,
                'total_distance_km': (total_distance_val / 1000.0) if total_distance_val is not None else None,
                'stop_positions': stop_positions,
                'stop_marker_count': len(stop_positions),
            })

        if not plot_meta:
            print("No usable custom cycles for urban route comparison")
            return None

        plot_meta.sort(key=lambda m: m['group_metric'])

        duration_bins: List[Tuple[str, float, float]] = [
            ("≤ 300 s", 0.0, 300.0),
            ("300–450 s", 300.0, 450.0),
            ("450–600 s", 450.0, 600.0),
            ("600–750 s", 600.0, 750.0),
            ("> 750 s", 750.0, float('inf')),
        ]

        assigned_routes = set()
        bin_buckets: List[Tuple[str, List[Dict[str, Any]]]] = []
        for base_label, lower, upper in duration_bins:
            if upper == float('inf'):
                metas = [m for m in plot_meta
                         if m['route_id'] not in assigned_routes and m['duration'] > lower]
            elif lower == 0.0:
                metas = [m for m in plot_meta
                         if m['route_id'] not in assigned_routes and m['duration'] <= upper]
            else:
                metas = [m for m in plot_meta
                         if m['route_id'] not in assigned_routes and lower < m['duration'] <= upper]
            assigned_routes.update(m['route_id'] for m in metas)
            bin_buckets.append((base_label, metas))

        for idx in range(len(bin_buckets) - 1):
            label, metas = bin_buckets[idx]
            next_label, next_metas = bin_buckets[idx + 1]
            while len(metas) > max_routes_per_panel and next_metas is not None:
                overflow = metas[max_routes_per_panel:]
                metas[:] = metas[:max_routes_per_panel]
                next_metas[:0] = overflow
            bin_buckets[idx] = (label, metas)
            bin_buckets[idx + 1] = (next_label, next_metas)

        groups: List[Dict[str, Any]] = []
        label_counter: Dict[str, int] = defaultdict(int)
        for base_label, metas in bin_buckets:
            if not metas:
                continue
            for chunk_idx in range(0, len(metas), max_routes_per_panel):
                chunk = metas[chunk_idx:chunk_idx + max_routes_per_panel]
                label_counter[base_label] += 1
                display_label = base_label
                if label_counter[base_label] > 1:
                    display_label = f"{base_label} (panel {label_counter[base_label]})"
                if normalization == "distance":
                    dist_values = [m.get('total_distance_km') for m in chunk if m.get('total_distance_km') is not None]
                    if dist_values:
                        display_label += f" – distance≈{float(np.nanmedian(dist_values)):.1f} km"
                groups.append({
                    'base': base_label,
                    'label': display_label,
                    'metas': chunk
                })

        def _stop_spans_from_probability(axis_values: np.ndarray,
                                         stop_prob: np.ndarray,
                                         speeds: np.ndarray,
                                         axis_is_time: bool,
                                         threshold: float = 0.6,
                                         min_duration_s: float = STOP_MIN_DURATION_S) -> List[Tuple[float, float]]:
            spans: List[Tuple[float, float]] = []
            if axis_values.size == 0 or stop_prob.size != axis_values.size or speeds.size != axis_values.size:
                return spans
            mask = np.asarray(stop_prob) >= threshold
            idx = 0
            min_width = min_duration_s if axis_is_time else 0.002 * max(axis_values[-1] - axis_values[0], 1.0)
            while idx < len(mask):
                if mask[idx]:
                    start_idx = idx
                    while idx < len(mask) and mask[idx]:
                        idx += 1
                    end_idx = max(start_idx, idx - 1)
                    start_val = float(axis_values[start_idx])
                    end_val = float(axis_values[end_idx])
                    if (end_val - start_val) >= min_width:
                        segment_speeds = np.asarray(speeds[start_idx:end_idx + 1])
                        if segment_speeds.size == 0:
                            idx += 1
                            continue
                        if np.nanmedian(segment_speeds) <= STOP_SPEED_THRESHOLD_KMH:
                            spans.append((start_val, end_val))
                else:
                    idx += 1
            return spans

        def _approx_stop_spans(stop_meta: Dict[str, Any], axis_is_time: bool, total_extent: float) -> List[Tuple[float, float]]:
            spans: List[Tuple[float, float]] = []
            if not stop_meta:
                return spans
            stop_list = stop_meta.get('stop_summaries', []) or []
            if axis_is_time:
                seg_list = stop_meta.get('segment_summaries', []) or []
                cursor = 0.0
                for idx, stop_info in enumerate(stop_list):
                    dur = float(stop_info.get('median_duration_s', 0.0))
                    start = cursor
                    end = cursor + dur
                    if end > start:
                        spans.append((start, end))
                    cursor = end
                    if idx < len(seg_list):
                        cursor += float(seg_list[idx].get('median_duration_s', 0.0))
            else:
                total_dist_m = float(stop_meta.get('total_distance_m', total_extent * 1000.0))
                total_dist = total_dist_m / 1000.0 if total_dist_m > 0 else total_extent
                if total_dist <= 0:
                    return spans
                band_width = max(0.01 * total_dist, 0.01)
                for stop_info in stop_list:
                    dist = stop_info.get('distance_m')
                    if dist is None:
                        continue
                    dist = float(dist) / 1000.0
                    start = max(0.0, dist - band_width / 2.0)
                    end = min(total_dist, dist + band_width / 2.0)
                    if end > start:
                        spans.append((start, end))
            return spans

        fig_height = A4_LANDSCAPE[1] if len(groups) == 1 else A4_LANDSCAPE[1] + 3.0
        fig, axes = plt.subplots(len(groups), 1, figsize=(A4_LANDSCAPE[0], fig_height), facecolor='white', sharex=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        fig.subplots_adjust(right=0.96, top=0.9, hspace=0.52)
        any_stop_shading = False
        any_stop_markers = False

        for idx_group, group_info in enumerate(groups):
            base_label = group_info.get('base') if isinstance(group_info, dict) else group_info[0]
            display_label = group_info.get('label') if isinstance(group_info, dict) else group_info[0]
            metas = group_info.get('metas') if isinstance(group_info, dict) else group_info[1]
            ax = axes[idx_group]
            ax.set_title(f"Routes {display_label}", fontsize=11, fontweight='bold', loc='left')
            unique_limits: List[float] = []
            max_axis_group = 0.0
            panel_handles: List[Any] = []
            panel_labels: List[str] = []
            panel_marker_specs: List[Tuple[np.ndarray, str]] = []
            panel_max_speed = 0.0
            panel_limits: List[float] = []
            forced_limits: Set[float] = set()
            for idx_meta, meta in enumerate(metas):
                color = ROUTE_COLOR_SEQUENCE[idx_meta % len(ROUTE_COLOR_SEQUENCE)]
                linestyle = ROUTE_LINESTYLES[idx_meta % len(ROUTE_LINESTYLES)]
                stop_summary = self._get_route_stop_summary(meta['route_id']) or {}
                stop_meta = meta.get('stop_meta', {}) or {}
                stop_positions = meta.get('stop_positions') or []
                marker_count = meta.get('stop_marker_count') or 0
                median_stops = stop_summary.get('median_stops_per_trip')
                mean_stops = stop_summary.get('mean_stops_per_trip')
                stop_label: Optional[str] = None
                if median_stops is not None and median_stops > 0:
                    stop_label = f"median stops/trip ≈{median_stops:.1f}"
                elif mean_stops is not None and mean_stops > 0:
                    stop_label = f"mean stops/trip ≈{mean_stops:.1f}"
                elif marker_count:
                    stop_label = f"{int(marker_count)} stop markers"
                has_stops = stop_label is not None
                spans: List[Tuple[float, float]] = []
                if show_stop_windows and has_stops:
                    stop_prob_array = meta.get('stop_prob')
                    prob_max = meta.get('stop_prob_max', 0.0)
                    if stop_prob_array is not None and np.isfinite(prob_max) and prob_max >= STOP_PROB_THRESHOLD:
                        spans = _stop_spans_from_probability(meta['x'], stop_prob_array, meta['speed'], meta['x_is_time'])
                    if (not spans) and stop_meta:
                        spans = _approx_stop_spans(stop_meta, meta['x_is_time'], meta['axis_extent'])
                    for t0, t1 in spans:
                        ax.axvspan(t0, t1, facecolor=STOP_SHADE_COLOR, alpha=0.35, linewidth=0)
                        any_stop_shading = True
                label_lines = [meta['label']]
                if stop_label:
                    label_lines.append(stop_label)
                else:
                    label_lines.append("no stop stats")
                label_text = "\n".join(label_lines)
                line, = ax.plot(meta['x'], meta['speed'], color=color, linestyle=linestyle,
                                linewidth=1.6, alpha=0.95, label=label_text)
                panel_handles.append(line)
                panel_labels.append(label_text)
                if stop_positions:
                    panel_marker_specs.append((np.asarray(stop_positions, dtype=float), color))
                    any_stop_markers = True
                if meta['x'].size:
                    max_axis_group = max(max_axis_group, float(np.nanmax(meta['x'])))
                if meta['speed'].size:
                    panel_max_speed = max(panel_max_speed, float(np.nanmax(meta['speed'])))
                limit_value = meta['limit_value']
                if np.isfinite(limit_value) and limit_value > 0:
                    panel_limits.append(float(limit_value))
            if max_axis_group > 0:
                ax.set_xlim(0, max_axis_group * 1.02)
            limit_max = max(panel_limits) if panel_limits else 0.0
            target_max = max(panel_max_speed, limit_max)
            if target_max <= 0:
                target_max = 10.0
            axis_limit = self._determine_speed_axis_limit(target_max)
            if normalization == "stop":
                base_str = str(base_label).strip() if base_label else ""
                if base_str.startswith("≤"):
                    axis_limit = max(axis_limit, 40.0)
                    forced_limits.update({30.0})
                elif base_str.startswith("300–"):
                    axis_limit = max(axis_limit, 50.0)
                    forced_limits.update({30.0, 40.0})
                elif base_str.startswith("450–") or base_str.startswith("600–"):
                    axis_limit = max(axis_limit, 70.0)
                    forced_limits.update({30.0, 40.0, 60.0})
                elif base_str.startswith(">"):
                    axis_limit = max(axis_limit, 70.0)
                    forced_limits.update({30.0, 40.0, 60.0})
            ax.set_ylim(0, axis_limit)
            ax.set_ylabel('Speed (km/h)')
            ax.grid(True, which='major', color='#4f4f4f', linewidth=0.6, alpha=0.45)
            ax.grid(False, which='minor')
            ticks = list(np.arange(0, axis_limit + 1e-6, 10.0))
            if ticks:
                ax.set_yticks(ticks)
            for limit_value in sorted(set(panel_limits)):
                if limit_value <= axis_limit + 0.1:
                    self._draw_speed_limit_line(ax, unique_limits, float(limit_value))
            for forced_value in sorted(forced_limits):
                if forced_value <= axis_limit + 0.1:
                    self._draw_speed_limit_line(ax, unique_limits, float(forced_value))
            if panel_marker_specs:
                marker_level = max(axis_limit * 0.045, 0.8)
                for positions, color in panel_marker_specs:
                    y_vals = np.full(len(positions), marker_level)
                    ax.scatter(positions,
                               y_vals,
                               marker='o',
                               s=28,
                               facecolor='white',
                               edgecolor=color,
                               linewidth=1.0,
                               zorder=4)
            if panel_handles:
                legend = ax.legend(panel_handles, panel_labels,
                                   loc='upper center',
                                   bbox_to_anchor=(0.5, 1.22),
                                   ncol=min(3, len(panel_handles)),
                                   fontsize=8,
                                   frameon=True,
                                   columnspacing=1.2,
                                   handlelength=2.2,
                                   borderaxespad=0.2,
                                   labelspacing=0.6)
                if legend:
                    frame = legend.get_frame()
                    if frame is not None:
                        frame.set_facecolor('white')
                        frame.set_edgecolor('#555555')
                        frame.set_alpha(0.85)
                    legend_box = getattr(legend, "_legend_box", None)
                    if legend_box is not None:
                        legend_box.align = "center"

        if plot_meta:
            if normalization == "time":
                axes[-1].set_xlabel('Time (s)')
            else:
                axes[-1].set_xlabel(plot_meta[-1]['x_label'])
        else:
            axes[-1].set_xlabel('Time (s)')

        if any_stop_shading:
            fig.text(0.02, 0.02, f"\u25A0 Stop probability ≥60%", fontsize=8, ha='left', color=STOP_SHADE_COLOR)
        if any_stop_markers:
            fig.text(0.22, 0.02, "\u25CB Median stop markers", fontsize=8, ha='left', color='#2f2f2f')
        has_variant = any('_v' in meta['route_id'] for meta in plot_meta)
        if has_variant:
            fig.text(0.98, 0.02, 'Variants share the base route but differ in stop patterns.', fontsize=8, ha='right')

        if save_path is None:
            suffix_map = {
                'stop': 'stop_to_stop',
                'distance': 'distance_normalized',
            }
            suffix = suffix_map.get(normalization, 'time_normalized')
            save_path = self.output_dir / f'figure_03_urban_routes_{suffix}.png'

        norm_desc = {
            'stop': 'stop-aligned',
            'distance': 'distance-normalised',
            'time': 'time-normalised'
        }
        descriptor = norm_desc.get(normalization, norm_desc['time'])
        fig.suptitle(f'Figure 3. Urban routes speed comparison ({descriptor}, {len(plot_meta)} series)',
                     fontsize=13, fontweight='bold', x=0.07, ha='left', y=0.97)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ Saved urban routes comparison figure: {save_path}")
        return save_path


    def create_stop_cluster_map(self, route_id: Optional[str] = None,
                                save_path: Optional[Path] = None) -> Optional[Path]:
        route_candidates = self._get_top_route_ids(limit=10)
        if route_id and route_id in route_candidates:
            route_candidates = [route_id] + [rid for rid in route_candidates if rid != route_id]
        if not route_candidates:
            print("Stop cluster map skipped: no route candidates")
            return None

        selected_route = None
        selected_clusters: List[Dict] = []
        route_events: List[Dict] = []
        for rid in route_candidates:
            summary = self._get_route_stop_summary(rid)
            clusters = summary.get('clusters', []) if summary else []
            clusters = [c for c in clusters if c.get('latitude') is not None and c.get('longitude') is not None]
            events = summary.get('events', []) if summary else []
            events = [ev for ev in events if ev.get('latitude') is not None and ev.get('longitude') is not None]
            if clusters:
                selected_route = rid
                selected_clusters = clusters
                route_events = events
                break
            if not selected_route and events:
                selected_route = rid
                route_events = events
        if not selected_route or not selected_clusters:
            if route_events:
                selected_clusters = [{
                    'latitude': ev['latitude'],
                    'longitude': ev['longitude'],
                    'median_duration_s': ev.get('duration_s', 0.0),
                    'n_events': 1
                } for ev in route_events]
            else:
                print("Stop cluster map skipped: no clusters with coordinates")
                return None

        latitudes = [c['latitude'] for c in selected_clusters]
        longitudes = [c['longitude'] for c in selected_clusters]
        durations = [c.get('median_duration_s', 0) for c in selected_clusters]
        weights = [max(c.get('n_events', 1), 1) for c in selected_clusters]

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(longitudes, latitudes, s=np.array(weights) * 30.0,
                             c=durations, cmap='Reds', alpha=0.85, edgecolor='k')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Stop clusters for {selected_route} (median duration size-coded)', fontsize=12, fontweight='bold')
        for cluster in selected_clusters:
            label = f"{cluster['median_duration_s']:.1f}s"
            ax.annotate(label,
                        xy=(cluster['longitude'], cluster['latitude']),
                        xytext=(4, 4), textcoords='offset points', fontsize=8)
        fig.colorbar(scatter, ax=ax, label='Median stop duration (s)')
        ax.grid(True, linestyle='--', alpha=0.4)

        if save_path is None:
            save_path = self.output_dir / f'stop_clusters_{selected_route}.png'
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.savefig(save_path, dpi=DPI, facecolor='white')
        plt.close(fig)
        print(f"✓ Saved stop cluster map: {save_path}")
        return save_path

    def create_wltp_stop_profile(self, route_id: Optional[str] = None,
                                 save_path: Optional[Path] = None) -> Optional[Path]:
        route_ids = self._get_top_route_ids(limit=10)
        if not route_ids:
            print("WLTP stop profile skipped: no routes available")
            return None
        if route_id and route_id in route_ids:
            route_ids = [route_id] + [rid for rid in route_ids if rid != route_id]
        selected_route = route_ids[0]

        df_cycle = self._get_route_cycle(selected_route)
        if df_cycle.empty:
            print("WLTP stop profile skipped: missing cycle data")
            return None

        time_values = pd.to_numeric(df_cycle.get('time_s', pd.Series()), errors='coerce').fillna(0).to_numpy()
        speed_values = np.zeros_like(time_values)
        if 'speed_kmh' in df_cycle.columns:
            speed_values = pd.to_numeric(df_cycle['speed_kmh'], errors='coerce').fillna(0).to_numpy()
        elif 'speed_ms' in df_cycle.columns:
            speed_values = pd.to_numeric(df_cycle['speed_ms'], errors='coerce').fillna(0).to_numpy() * 3.6

        stop_prob = self._get_route_stop_probability(selected_route)
        if stop_prob.size != time_values.size:
            stop_prob = np.zeros_like(time_values)
        stop_summary = self._get_route_stop_summary(selected_route) or {}
        route_stop_durations = [ev.get('duration_s', 0) for ev in stop_summary.get('events', [])] or \
                               self._compute_stop_durations_from_series(time_values, speed_values)

        ref_df = self.reference_standard_df
        if ref_df.empty:
            print("WLTP stop profile skipped: missing reference standard")
            return None
        ref_time = pd.to_numeric(ref_df.get('time_s', pd.Series()), errors='coerce').fillna(0).to_numpy()
        if 'speed_kmh' in ref_df.columns:
            ref_speed = pd.to_numeric(ref_df['speed_kmh'], errors='coerce').fillna(0).to_numpy()
        elif 'speed_ms' in ref_df.columns:
            ref_speed = pd.to_numeric(ref_df['speed_ms'], errors='coerce').fillna(0).to_numpy() * 3.6
        else:
            ref_speed = np.zeros_like(ref_time)
        wltp_stop_durations = self._compute_stop_durations_from_series(ref_time, ref_speed)

        fig, (ax_route, ax_hist, ax_wltp) = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=False)

        ax_route.plot(time_values, speed_values, color='#045a8d', linewidth=1.2, label='Route speed')
        ax_route.set_ylabel('Route speed (km/h)')
        ax_route.set_title(f'{selected_route} stop probability vs speed', fontsize=12, fontweight='bold')
        ax_route.grid(True, linestyle='--', alpha=0.4)
        ax_route2 = ax_route.twinx()
        ax_route2.plot(time_values, stop_prob, color='#d73027', linewidth=1.0, alpha=0.7, label='Stop probability')
        ax_route2.set_ylabel('Stop probability')
        ax_route2.set_ylim(0, 1)
        ax_route.fill_between(time_values, 0, speed_values, where=stop_prob >= STOP_PROB_THRESHOLD,
                              color=STOP_SHADE_COLOR, alpha=0.3, label='High stop probability')
        ax_route.legend(loc='upper right')

        bins = np.linspace(0, max(route_stop_durations + wltp_stop_durations + [5]), 15)
        ax_hist.hist(route_stop_durations, bins=bins, alpha=0.6, label=f'{selected_route} stops', color='#3690c0')
        ax_hist.hist(wltp_stop_durations, bins=bins, alpha=0.6, label=f'{self.reference_standard_label} stops', color='#ef6548')
        ax_hist.set_xlabel('Stop duration (s)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Stop-duration distribution comparison')
        ax_hist.legend()
        ax_hist.grid(True, linestyle='--', alpha=0.4)

        ax_wltp.plot(ref_time, ref_speed, color='#636363', linewidth=1.1)
        ax_wltp.set_xlabel('Time (s)')
        ax_wltp.set_ylabel(f'{self.reference_standard_label} speed (km/h)')
        ax_wltp.set_title(f'{self.reference_standard_label} profile')
        ax_wltp.grid(True, linestyle='--', alpha=0.4)
        wltp_mask = ref_speed <= STOP_SPEED_THRESHOLD_KMH
        wltp_spans = self._mask_to_spans(ref_time, wltp_mask, STOP_MIN_DURATION_S)
        for start_time, end_time in wltp_spans:
            ax_wltp.axvspan(start_time, end_time, color=STOP_SHADE_COLOR, alpha=0.25)

        if save_path is None:
            save_path = self.output_dir / f'route_vs_wltp_stop_profile_{selected_route}.png'
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.savefig(save_path, dpi=DPI, facecolor='white')
        plt.close(fig)
        print(f"✓ Saved WLTP stop profile comparison: {save_path}")
        return save_path

    def create_route_trip_stack(self, route_id: str, max_traces: int = 25,
                                save_path: Optional[Path] = None) -> Optional[Path]:
        profile_path_parquet = self.export_dir / f"{route_id}_time_normalized_profiles.parquet"
        profile_path_csv = self.export_dir / f"{route_id}_time_normalized_profiles.csv"
        if profile_path_parquet.exists():
            df_profiles = pd.read_parquet(profile_path_parquet)
        elif profile_path_csv.exists():
            df_profiles = pd.read_csv(profile_path_csv)
        else:
            print(f"Route trip stack skipped: missing profiles for {route_id}")
            return None
        if df_profiles.empty or 'time_s' not in df_profiles.columns:
            print(f"Route trip stack skipped: malformed profiles for {route_id}")
            return None

        time_values = pd.to_numeric(df_profiles['time_s'], errors='coerce').fillna(0).to_numpy()
        trip_cols = [c for c in df_profiles.columns if c.startswith('trip_')]
        if not trip_cols:
            print(f"Route trip stack skipped: no trip traces for {route_id}")
            return None
        trip_cols = trip_cols[:max_traces]

        trace_df = df_profiles[['time_s'] + trip_cols].copy()
        for col in trip_cols:
            trace_df[col] = pd.to_numeric(trace_df[col], errors='coerce')

        df_cycle = self._get_route_cycle(route_id)
        stop_prob = self._get_route_stop_probability(route_id)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        max_trace_speed = float(np.nanmax(trace_df[trip_cols].to_numpy()))
        if not np.isfinite(max_trace_speed) or max_trace_speed <= 0:
            max_trace_speed = 60.0
        for col in trip_cols:
            ax.plot(time_values, trace_df[col].fillna(0).to_numpy(),
                    linewidth=0.8, alpha=0.35)
        median_series = np.nanmedian(trace_df[trip_cols].to_numpy(), axis=1)
        ax.plot(time_values, np.nan_to_num(median_series, nan=0.0),
                color='#1b9e77', linewidth=1.4, label='Median profile')
        combined_stop_prob = np.mean((trace_df[trip_cols] <= STOP_SPEED_THRESHOLD_KMH), axis=1)
        spans = self._probability_stop_spans(time_values, combined_stop_prob,
                                             threshold=STOP_PROB_THRESHOLD,
                                             min_duration_s=STOP_MIN_DURATION_S)
        for start_time, end_time in spans:
            ax.axvspan(start_time, end_time, color=STOP_SHADE_COLOR, alpha=0.25)
        ax.set_title(f"Route {route_id} time-normalized stacks ({len(trip_cols)} traces)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax2 = ax.twinx()
        ax2.plot(time_values, combined_stop_prob, color=STOP_LINE_COLOR, linewidth=0.9, alpha=0.7, label='Stop probability')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Stop probability')
        ax.set_ylim(0, max(max_trace_speed * 1.15, ax.get_ylim()[1]))
        if save_path is None:
            save_path = self.output_dir / f"route_trip_stack_{route_id}.png"
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        fig.savefig(save_path, dpi=DPI, facecolor='white')
        plt.close(fig)
        print(f"✓ Saved route trip stack figure: {save_path}")
        return save_path

    def create_route_markov_surface(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """Figure 4: 3D heatmap of speed-bin Markov transitions for the dominant routes"""
        route_models = self.results.get('route_markov_models', {})
        if not route_models:
            print("Route-level Markov data not found; skipping Figure 4")
            return None

        route_ids = self._get_top_route_ids(limit=5)
        matrices = {}
        max_probability = 0.0
        for rid in route_ids:
            matrix = np.array(route_models.get(rid, {}).get('bin_transition_matrix', []), dtype=float)
            if matrix.size == 0:
                continue
            matrices[rid] = matrix
            max_probability = max(max_probability, float(matrix.max()))

        if not matrices:
            print("Top routes missing Markov matrices; skipping Figure 4")
            return None

        route_ids = [rid for rid in route_ids if rid in matrices]
        if not route_ids:
            print("Route list filtered out all Markov matrices; skipping Figure 4")
            return None

        aggregate_info: Optional[Dict[str, Any]] = None
        if len(route_ids) >= 2:
            stack = np.stack([matrices[rid] for rid in route_ids], axis=0)
            weights = np.array([route_models[rid].get('n_transitions', 0) or 1 for rid in route_ids], dtype=float)
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                weight_norm = weights / weight_sum
                mean_matrix = np.tensordot(weight_norm, stack, axes=(0, 0))
                diff = stack - mean_matrix
                std_matrix = np.sqrt(np.tensordot(weight_norm, diff**2, axes=(0, 0)))
            else:
                mean_matrix = stack.mean(axis=0)
                std_matrix = stack.std(axis=0)
            aggregate_info = {
                'matrix': mean_matrix,
                'std': std_matrix,
                'weights': weights,
                'weight_sum': weight_sum
            }
            max_probability = max(max_probability, float(mean_matrix.max()))

        palette = self._get_route_palette(route_ids)

        norm = Normalize(vmin=0.0, vmax=max_probability if max_probability > 0 else 1.0)
        cmap = plt.cm.plasma

        if aggregate_info is not None and len(route_ids) >= 2:
            mean_matrix = aggregate_info['matrix']
            std_matrix = aggregate_info['std']
            weight_sum = int(aggregate_info.get('weight_sum') or 0)
            n_bins = mean_matrix.shape[0]

            vmax = max(max_probability, float(np.nanmax(mean_matrix))) if max_probability > 0 else float(np.nanmax(mean_matrix))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 0.2

            fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor='white')
            im = ax.imshow(mean_matrix,
                           origin='lower',
                           cmap=cmap,
                           vmin=0.0,
                           vmax=max(vmax, 0.2))

            speed_edges = route_models[route_ids[0]].get('speed_bins_kmh') or route_models[route_ids[0]].get('speed_bins_ms')
            use_kmh = bool(route_models[route_ids[0]].get('speed_bins_kmh'))
            if speed_edges:
                edges = np.array(speed_edges, dtype=float)
                if len(edges) == n_bins - 1:
                    edges = np.concatenate(([0.0], edges))
                labels_edges = edges if use_kmh else edges * 3.6
                labels_edges = np.round(labels_edges).astype(int)
                bin_labels = []
                for b in range(n_bins):
                    if b == 0 and len(labels_edges) > 1:
                        bin_labels.append(f"< {labels_edges[1]}")
                    elif b == n_bins - 1:
                        bin_labels.append(f">= {labels_edges[-1]}")
                    else:
                        bin_labels.append(f"{labels_edges[b]}–{labels_edges[b+1]}")
            else:
                bin_labels = [f"Bin {i}" for i in range(n_bins)]

            ticks = np.arange(n_bins)
            ax.set_xticks(ticks)
            ax.set_xticklabels(bin_labels, rotation=32, ha='right', fontsize=9)
            ax.set_yticks(ticks)
            ax.set_yticklabels(bin_labels, rotation=32, ha='right', fontsize=9)
            ax.set_xlabel('Current speed bin')
            ax.set_ylabel('Next speed bin')

            diag = np.arange(n_bins)
            ax.plot(diag, diag, color='#222222', linewidth=1.1, linestyle='--', alpha=0.8, label='No-change boundary')

            thresholds = [0.05, 0.10, 0.20]
            threshold_colors = ['#74add1', '#fdae61', '#d73027']
            for thr, color in zip(thresholds, threshold_colors):
                try:
                    contour = ax.contour(mean_matrix,
                                         levels=[thr],
                                         colors=color,
                                         linewidths=1.2,
                                         alpha=0.85)
                except ValueError:
                    continue
                collections = getattr(contour, "collections", None)
                if collections:
                    collections[0].set_label(f"{thr:.2f} prob contour")

            high_prob_mask = mean_matrix >= 0.25
            if np.any(high_prob_mask):
                high_y, high_x = np.where(high_prob_mask)
                ax.scatter(high_x,
                           high_y,
                           s=60,
                           facecolor='none',
                           edgecolor='#000000',
                           linewidth=1.2,
                           label='≥0.25 probability')

            cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label('Transition probability')

            mean_sigma = float(np.nanmean(std_matrix))
            max_sigma = float(np.nanmax(std_matrix))
            ax.text(0.02, 1.02,
                    f"Weighted by transitions (total={weight_sum}). Mean σ={mean_sigma:.3f}, max σ={max_sigma:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='left',
                    va='bottom')
            ax.text(0.02, -0.12,
                    "Contours highlight 0.05 / 0.10 / 0.20 probability bands; circles mark cells ≥0.25",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha='left',
                    va='top',
                    color='#333333')

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.8)

            suffix = " (weighted aggregate)"
            fig.suptitle(f'Figure 4. Markov transition probabilities{suffix}', fontsize=14, fontweight='bold', x=0.5, y=0.98)

            if save_path is None:
                save_path = self.output_dir / 'figure_04_route_markov_surface.png'

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"✓ Saved route Markov aggregate figure: {save_path}")
            return save_path

        # Fallback: single-route view or insufficient aggregate data
        n_routes = len(route_ids)
        fig = plt.figure(figsize=(9.5, 7.0), facecolor='white')
        axes = []
        for idx, rid in enumerate(route_ids):
            matrix = matrices[rid]
            ax = fig.add_subplot(1, n_routes, idx + 1, projection='3d')
            axes.append(ax)

            n_bins = matrix.shape[0]
            x_idx = np.arange(n_bins)
            y_idx = np.arange(n_bins)
            xpos, ypos = np.meshgrid(x_idx, y_idx, indexing='ij')
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos, dtype=float)
            dx = dy = 0.6
            dz = matrix.flatten()

            colors = cmap(norm(dz))
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.85, shade=True, zsort='max', edgecolor='none')
            ax.set_facecolor('white')
            ax.view_init(elev=32, azim=135)
            ax.set_box_aspect((1, 1, 0.6))

            speed_edges = route_models[rid].get('speed_bins_kmh') or route_models[rid].get('speed_bins_ms')
            bin_labels = []
            if speed_edges:
                edges = np.array(speed_edges, dtype=float)
                if len(edges) == n_bins - 1:
                    edges = np.concatenate(([0.0], edges))
                if route_models[rid].get('speed_bins_kmh'):
                    labels_edges = edges
                else:
                    labels_edges = edges * 3.6
                labels_edges = np.round(labels_edges).astype(int)
                for b in range(n_bins):
                    if b == 0:
                        bin_labels.append(f"< {labels_edges[1]}")
                    elif b == n_bins - 1:
                        bin_labels.append(f">= {labels_edges[-1]}")
                    else:
                        bin_labels.append(f"{labels_edges[b]}–{labels_edges[b+1]}")
            else:
                bin_labels = [f"Bin {i}" for i in range(n_bins)]

            tick_positions = np.arange(n_bins) + 0.3
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(bin_labels, rotation=32, ha='right', fontsize=8)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(bin_labels, rotation=32, ha='right', fontsize=8)
            ax.tick_params(axis='x', pad=6, labelsize=8)
            ax.tick_params(axis='y', pad=6, labelsize=8)
            ax.tick_params(axis='z', labelsize=8)
            ax.set_zlim(0, max(0.2, matrix.max() * 1.1))
            ax.set_xlabel('Current speed bin', fontsize=9)
            ax.set_ylabel('Next speed bin', fontsize=9)
            ax.set_zlabel('Probability', fontsize=9)

            route_number = rid.split('_')[-1] if '_' in rid else rid
            try:
                route_number = str(int(route_number) + 1)
            except ValueError:
                route_number = route_number
            transitions = route_models[rid].get('n_transitions', 0)
            ax.set_title(f"Route {route_number} (n={transitions})", fontsize=11, pad=14,
                         color=self._get_route_style(rid, palette)[0])

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if axes:
            cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
            cbar.set_label('Transition probability', fontsize=10)

        fig.suptitle('Figure 4. Route-level Markov transition surfaces', fontsize=14, fontweight='bold')

        if save_path is None:
            save_path = self.output_dir / 'figure_04_route_markov_surface.png'

        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ Saved route Markov surface figure: {save_path}")
        return save_path

    def create_real_vs_standard_comparison(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """Figure 5: Multi-metric comparison between real routes and WLTC Class 1 reference"""
        route_ids = self._get_top_route_ids(limit=5)
        if not route_ids:
            print("No recurring routes available for real vs standard comparison")
            return None

        palette = self._get_route_palette(route_ids)
        metric_config = [
            ('chaos_pct', 'Chaos (%)', True),
            ('pke', 'PKE (m/s²)', True),
            ('idle_pct', 'Idle (%)', False),
            ('stops_per_km', 'Stops per km', True),
            ('mean_speed_kmh', 'Mean speed (km/h)', False),
            ('wavelet_entropy', 'Wavelet entropy', False)
        ]

        # Gather route metrics and identify matched standards
        route_metrics: Dict[str, Dict[str, float]] = {}
        for rid in route_ids:
            metrics = self._compute_route_cycle_metrics(rid)
            if metrics:
                route_metrics[rid] = metrics

        if not route_metrics:
            print("Route metrics unavailable for comparison figure")
            return None

        metric_keys = [key for key, _, _ in metric_config]
        route_best: Dict[str, Dict] = {}
        available_routes: List[str] = []
        for rid in route_ids:
            metrics = route_metrics.get(rid)
            if not metrics:
                continue
            best_key, standard_metrics = self._find_best_standard_cycle(metrics, metric_keys)
            if standard_metrics:
                available_routes.append(rid)
                route_best[rid] = {
                    'cycle_key': best_key,
                    'display_name': standard_metrics.get('display_name', best_key.split('_')[-1] if best_key else 'Standard cycle'),
                    'metrics': standard_metrics
                }
                self._route_best_matches[rid] = route_best[rid]

        if not available_routes:
            print("No valid route-standard matches for comparison figure")
            return None

        palette = self._get_route_palette(available_routes)

        n_metrics = len(metric_config)
        n_cols = min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 7.0), facecolor='white')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        axes_flat = axes.flatten()
        x_positions = np.arange(len(available_routes))
        bar_width = 0.32

        for idx, (metric_key, metric_label, use_log) in enumerate(metric_config):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]

            real_values = []
            reference_values = []
            for rid in available_routes:
                real_values.append(float(route_metrics[rid].get(metric_key, np.nan)))
                reference_values.append(float(route_best[rid]['metrics'].get(metric_key, np.nan)))

            plotted_real_values: List[float] = []
            plotted_std_values: List[float] = []

            for pos, rid in enumerate(available_routes):
                real_val = real_values[pos]
                std_val = reference_values[pos]
                if np.isnan(real_val) or np.isnan(std_val):
                    continue

                plotted_real = real_val
                plotted_std = std_val
                if use_log:
                    plotted_real = max(plotted_real, 1e-3)
                    plotted_std = max(plotted_std, 1e-3)

                bar_color = self._get_route_style(rid, palette)[0]
                ax.bar(x_positions[pos] - bar_width / 2,
                       plotted_real,
                       width=bar_width,
                       color=bar_color,
                       edgecolor='black', linewidth=0.3)
                std_info = route_best.get(rid, {})
                std_family = std_info.get('metrics', {}).get('family', 'Standard')
                std_color = FAMILY_COLORS.get(std_family, '#d0d0d0')
                std_hatch = FAMILY_HATCHES.get(std_family, '')

                ax.bar(x_positions[pos] + bar_width / 2,
                       plotted_std,
                       width=bar_width,
                       color=std_color,
                       edgecolor='black', linewidth=0.3,
                       hatch=std_hatch)

                plotted_real_values.append(plotted_real)
                plotted_std_values.append(plotted_std)

            ax.set_title(metric_label, fontsize=11, fontweight='bold', loc='left', pad=6)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [self._format_route_label(r, self._get_route_trip_count(r)).replace('Route ', 'R') for r in available_routes],
                rotation=90, fontsize=8, ha='center', va='top'
            )
            if idx % n_cols == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            else:
                ax.set_ylabel('')
            if use_log:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"))

            finite_vals = [val for val in plotted_real_values + plotted_std_values if np.isfinite(val) and val > 0]
            if finite_vals:
                ymin = min(finite_vals)
                ymax = max(finite_vals)
                if use_log:
                    ax.set_ylim(ymin * 0.6, ymax * 1.6)
                else:
                    ax.set_ylim(0, ymax * 1.25)

            ax.grid(True, which='major', color='#7a7a7a', linewidth=0.65, alpha=0.55)
            if use_log:
                ax.grid(True, which='minor', color='#b0b0b0', linewidth=0.4, alpha=0.4)

        # Hide any unused axes
        for ax in axes_flat[len(metric_config):]:
            ax.axis('off')

        best_labels = sorted({route_best[r]['display_name'] for r in available_routes})
        fig.suptitle("Figure 5. Real urban routes vs matched standard cycles",
                     fontsize=14, fontweight='bold', x=0.07, ha='left', y=0.97)
        if best_labels:
            wrapped = "; ".join(best_labels)
            fig.text(0.07, 0.915, f"Matched standards: {wrapped}",
                     ha='left', fontsize=9, color='#333333')

        family_handles: Dict[str, Patch] = {}
        for rid in available_routes:
            std_info = route_best.get(rid, {})
            family = std_info.get('metrics', {}).get('family', 'Standard')
            family_color = FAMILY_COLORS.get(family, '#c0c0c0')
            hatch = FAMILY_HATCHES.get(family, '') if 'FAMILY_HATCHES' in globals() else ''
            if family not in family_handles:
                family_handles[family] = Patch(facecolor=family_color, edgecolor='black', hatch=hatch, label=family, linewidth=0.4)

        legend_handles: List[Patch] = []
        if family_handles:
            legend_handles = list(family_handles.values())
        fig.legend(legend_handles,
                   [handle.get_label() for handle in legend_handles],
                   loc='upper right', ncol=1, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.98, 0.98))

        if save_path is None:
            save_path = self.output_dir / 'figure_05_real_vs_standard_ratios.png'

        fig.tight_layout(rect=[0, 0.11, 1, 0.9])
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ Saved real vs standard comparison figure: {save_path}")
        return save_path

    def create_normalized_speed_pattern(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """Figure 6: Aggregate normalized speed envelope vs closest standard cycle"""
        route_ids = self._get_top_route_ids(limit=5)
        if not route_ids:
            print("No routes available for normalized speed pattern figure")
            return None

        resampled_curves = []
        base_stop_cycles = []
        for rid in route_ids:
            df_cycle = self._get_route_cycle_by_mode(rid, mode="stop")
            if df_cycle.empty:
                df_cycle = self._get_route_cycle(rid)
            if df_cycle.empty:
                continue

            if 'time_s' in df_cycle.columns:
                time_values = pd.to_numeric(df_cycle['time_s'], errors='coerce').fillna(method='ffill').fillna(0).to_numpy()
            else:
                time_values = np.arange(len(df_cycle)) / self.sr_hz
            if time_values.size < 2:
                continue

            if 'speed_kmh' in df_cycle.columns:
                speed_values = pd.to_numeric(df_cycle['speed_kmh'], errors='coerce').fillna(0).to_numpy()
            elif 'speed_ms' in df_cycle.columns:
                speed_values = pd.to_numeric(df_cycle['speed_ms'], errors='coerce').fillna(0).to_numpy() * 3.6
            else:
                continue

            grid, resampled = self._resample_curve(time_values, speed_values, target_points=600)
            if resampled.size == 0:
                continue
            resampled_curves.append(resampled)

            if 'stop_probability' in df_cycle.columns:
                stop_profile = pd.to_numeric(df_cycle['stop_probability'], errors='coerce').fillna(0).to_numpy()
            elif 'segment_type' in df_cycle.columns:
                stop_profile = df_cycle['segment_type'].astype(str).str.contains('stop', case=False, na=False).astype(float).to_numpy()
            else:
                stop_profile = (speed_values <= 2.0).astype(float)

            time_norm = time_values - time_values[0]
            if time_norm[-1] <= 0:
                continue
            base_stop_cycles.append({
                'route_id': rid,
                'time': time_norm,
                'speed': speed_values,
                'stop': np.clip(stop_profile, 0.0, 1.0),
                'duration': float(time_norm[-1])
            })

        if not resampled_curves:
            print("No usable cycles for normalized speed pattern")
            return None

        stacked = np.vstack(resampled_curves)
        p25 = np.percentile(stacked, 25, axis=0)
        p75 = np.percentile(stacked, 75, axis=0)

        reference_df = self.reference_standard_df
        reference_curve = None
        reference_duration = None
        if reference_df is not None and not reference_df.empty:
            reference_speeds = pd.to_numeric(reference_df['speed_kmh'], errors='coerce').fillna(0).to_numpy()
            reference_time = pd.to_numeric(reference_df['time_s'], errors='coerce').fillna(method='ffill').fillna(0).to_numpy()
            if reference_time.size > 0:
                reference_duration = float(reference_time[-1])
            _, reference_curve = self._resample_curve(reference_time, reference_speeds, target_points=stacked.shape[1])

        if reference_duration is None or reference_duration <= 0:
            reference_duration = float(np.mean([c['duration'] for c in base_stop_cycles])) if base_stop_cycles else 1.0

        target_time = np.linspace(0, reference_duration, 600)

        # Build a composite stop-aligned urban profile by tiling the longest cycles
        composite_speed = None
        composite_stop = None
        urban_duration = 0.0
        if base_stop_cycles:
            base_stop_cycles.sort(key=lambda item: item['duration'], reverse=True)
            combined_time_segments: List[np.ndarray] = []
            combined_speed_segments: List[np.ndarray] = []
            combined_stop_segments: List[np.ndarray] = []
            acc_time = 0.0
            idx = 0
            max_cycles = max(6, len(base_stop_cycles) * 3)
            while acc_time < reference_duration and idx < max_cycles:
                cycle = base_stop_cycles[idx % len(base_stop_cycles)]
                t_vals = cycle['time']
                s_vals = cycle['speed']
                stop_vals = cycle['stop']
                if t_vals.size < 2:
                    idx += 1
                    continue
                seg_time = t_vals + acc_time
                seg_speed = s_vals
                seg_stop = stop_vals
                if combined_time_segments:
                    seg_time = seg_time[1:]
                    seg_speed = seg_speed[1:]
                    seg_stop = seg_stop[1:]
                combined_time_segments.append(seg_time)
                combined_speed_segments.append(seg_speed)
                combined_stop_segments.append(seg_stop)
                acc_time = seg_time[-1]
                idx += 1

            if combined_time_segments:
                combined_time = np.concatenate(combined_time_segments)
                combined_speed = np.concatenate(combined_speed_segments)
                combined_stop = np.concatenate(combined_stop_segments)
                if combined_time.size >= 2:
                    composite_speed = np.interp(target_time, combined_time, combined_speed,
                                                left=combined_speed[0], right=combined_speed[-1])
                    composite_stop = np.clip(
                        np.interp(target_time, combined_time, combined_stop,
                                  left=combined_stop[0], right=combined_stop[-1]), 0.0, 1.0)
                    urban_duration = min(acc_time, reference_duration)

        scaled_profiles = []
        if base_stop_cycles:
            for cycle in base_stop_cycles:
                if cycle['duration'] <= 0:
                    continue
                scaled_time = (cycle['time'] / cycle['duration']) * reference_duration
                if scaled_time.size < 2:
                    continue
                interp_speed = np.interp(target_time, scaled_time, cycle['speed'],
                                         left=cycle['speed'][0], right=cycle['speed'][-1])
                scaled_profiles.append(interp_speed)

        p25 = None
        p75 = None
        if scaled_profiles:
            scaled_stack = np.vstack(scaled_profiles)
            p25 = np.percentile(scaled_stack, 25, axis=0)
            p75 = np.percentile(scaled_stack, 75, axis=0)
        else:
            p25 = np.percentile(stacked, 25, axis=0)
            p75 = np.percentile(stacked, 75, axis=0)

        if composite_speed is None:
            composite_speed = np.median(np.vstack(scaled_profiles) if scaled_profiles else stacked, axis=0)
            composite_stop = None
            urban_duration = reference_duration

        wltc_time_axis = target_time
        median_profile = composite_speed
        stop_envelope = composite_stop if composite_stop is not None else None

        fig, ax = plt.subplots(figsize=A4_LANDSCAPE, facecolor='white')
        ax.fill_between(wltc_time_axis, p25, p75, color='#9ecae1', alpha=0.55, edgecolor='#6baed6', label='Urban IQR')
        ax.plot(wltc_time_axis, median_profile, color='#08519c', linewidth=2.4, label='Urban stop-aligned median')

        if reference_curve is not None and reference_curve.size:
            ax.plot(wltc_time_axis, reference_curve, color='#d7301f', linestyle='--', linewidth=1.9,
                    label=f'{self.reference_standard_label}')

        # Stop probability belt intentionally omitted for clarity

        max_speed = max(np.max(stacked), np.max(reference_curve) if reference_curve is not None else 0)
        ax.set_xlim(0, wltc_time_axis[-1] if wltc_time_axis.size else reference_duration)
        ax.set_ylim(0, max_speed * 1.2 if max_speed > 0 else None)
        ax.set_xlabel(f'{self.reference_standard_label} timeline (s)', fontsize=11)
        ax.set_ylabel('Speed (km/h)', fontsize=11)
        ax.set_title('Figure 6. Stop-aligned urban speed envelope vs WLTC reference', fontsize=13, fontweight='bold', loc='left')
        ax.minorticks_on()
        ax.grid(True, which='major', color='#5c5c5c', linewidth=0.8, alpha=0.7)
        ax.grid(True, which='minor', color='#8f8f8f', linewidth=0.5, alpha=0.55)
        ax.legend(loc='upper right', framealpha=0.9)

        def wltc_to_urban(sec: float) -> float:
            return (sec / reference_duration) * urban_duration if reference_duration else sec

        def urban_to_wltc(sec: float) -> float:
            return (sec / urban_duration) * reference_duration if urban_duration else sec

        top_axis = ax.secondary_xaxis('top', functions=(wltc_to_urban, urban_to_wltc))
        top_axis.set_xlabel(f'Urban stop-aligned median duration ≈ {urban_duration:.0f} s', fontsize=10)
        top_axis.tick_params(labelsize=9, colors='#08519c')
        top_axis.spines['top'].set_color('#08519c')
        top_axis.spines['top'].set_position(('axes', 1.08))

        if save_path is None:
            save_path = self.output_dir / 'figure_06_urban_speed_pattern.png'

        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ Saved normalized speed pattern figure: {save_path}")
        return save_path

    def _load_standard_cycles(self):
        """Load all standard cycles and compute all metrics including wavelets and distributions"""
        families = ["WLTP_Europe", "EPA", "Artemis", "Asia", "Special"]
        
        print("\nLoading standard cycles with wavelet and distribution analysis...")
        
        for family in families:
            folder = self.cycles_dir / family
            if not folder.exists():
                continue
            
            files = list(folder.glob("*.csv")) + list(folder.glob("*.scv")) + list(folder.glob("*.parquet"))
            
            for f in files:
                try:
                    # Read data
                    if f.suffix == ".parquet":
                        df = pd.read_parquet(f)
                    else:
                        df = self._read_csv_compat(f, comment="#", encoding="utf-8-sig")
                    
                    # Find speed column
                    speed_data = None
                    probable_cols = [
                        "Vehicle speed (km/h)", " Vehicle speed (km/h)",
                        "speed_kph", "speed_kmh", "Speed (km/h)", "Speed", "speed"
                    ]
                    
                    for col in probable_cols:
                        if col in df.columns:
                            speed_data = pd.to_numeric(df[col], errors="coerce").dropna().values
                            break
                    
                    # Last resort: scan all columns
                    if speed_data is None:
                        for col in df.columns:
                            c = col.strip().lower()
                            if any(k in c for k in ["speed", "vehicle", "velocity"]):
                                try:
                                    speed_data = pd.to_numeric(df[col], errors="coerce").dropna().values
                                    if len(speed_data) > 10:
                                        break
                                except Exception:
                                    continue
                    
                    if speed_data is None or len(speed_data) < 10:
                        continue
                    
                    # Determine if data is in km/h or m/s
                    max_speed = np.max(speed_data)
                    if max_speed < 50:  # Likely m/s
                        speed_data = speed_data * 3.6
                    
                    # Determine sampling rate
                    sampling_rate = 10.0 if len(speed_data) > 5000 else 1.0
                    duration_s = len(speed_data) / sampling_rate
                    
                    cycle_key = f"{family}_{f.stem}"
                    raw_name = self._extract_display_name_from_csv(f) if f.suffix != ".parquet" else f.stem
                    stem_lower = f.stem.lower()
                    if stem_lower == "us06col":
                        # Treat the cold-start variant as a distinct short label
                        raw_name = "US06 (Cold)"
                    elif stem_lower == "hwy10hz":
                        # High-frequency highway trace corresponds to the shorter HWFET cycle
                        raw_name = "HWFET (Short)"
                    elif stem_lower == "la92col":
                        # Column-oriented LA92 trace represents the cold-start version
                        raw_name = "LA92 (Cold)"
                    elif stem_lower == "la92dds":
                        # Dynamometer schedule variant should surface as its own short label
                        raw_name = "LA92 dynamo"
                    # Canonicalise the drive-cycle name so plots and exports share identical labels
                    canonical_name = canonicalize_name(raw_name)
                    display_name = build_display_name(canonical_name)

                    # Store cycle
                    self.standard_cycles[cycle_key] = pd.DataFrame({
                        "speed_kmh": speed_data,
                        "time_s": np.arange(len(speed_data)) / sampling_rate
                    })

                    # Calculate basic metrics
                    metrics = {
                        'display_name': display_name,
                        'canonical_name': canonical_name,
                        'family': family,
                        'duration_s': duration_s,
                        'max_speed_kmh': float(np.max(speed_data)),
                        'mean_speed_kmh': float(np.mean(speed_data)),
                        'sampling_rate_hz': sampling_rate,
                        'original_points': len(speed_data),
                    }
                    
                    # Add enhanced metrics (includes PKE and V95)
                    metrics.update(self._compute_enhanced_metrics(speed_data, sampling_rate))
                    
                    # Add probability distribution analysis
                    distribution_info = self._fit_probability_distribution(speed_data)
                    metrics['prob_distribution'] = distribution_info['distribution']
                    params_display = dict(distribution_info['parameters'])
                    if 'ks_pvalue' in params_display:
                        params_display['ks_pvalue'] = fmt_p(params_display['ks_pvalue'])
                    metrics['distribution_params'] = str(params_display)
                    
                    # Calculate wavelet metrics
                    wavelet_metrics = self._compute_wavelet_metrics(speed_data, sampling_rate)
                    if wavelet_metrics:
                        metrics.update(wavelet_metrics)
                        print(f"  ✓ {display_name} ({family}): {len(speed_data)} pts, PKE={metrics.get('pke', 0):.2f}, dist={distribution_info['distribution']}")
                    else:
                        print(f"  ✓ {display_name} ({family}): {len(speed_data)} pts, {duration_s:.0f}s")
                    
                    self.metrics[cycle_key] = metrics
                    
                except Exception as e:
                    print(f"  ✗ Error loading {f.name}: {e}")
        
        print(f"Loaded {len(self.standard_cycles)} cycles total")

    def create_duration_grouped_figure(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """Create figure with cycles grouped by duration, each group with individual scales"""
        
        if not self.standard_cycles:
            print("No cycles loaded!")
            return None
        
        # Group cycles by duration
        grouped_cycles = {}
        for group_name, min_dur, max_dur, _ in DURATION_GROUPS:
            grouped_cycles[group_name] = []

        available_labels = {m.get('display_name', '') for m in self.metrics.values()}
        if PREFERRED_DISPLAY_NAMES:
            missing_pref = sorted(label for label in PREFERRED_DISPLAY_NAMES if label not in available_labels)
            if missing_pref:
                print(f"Warning: Missing preferred cycles ({len(missing_pref)}): {', '.join(missing_pref)}")

        seen_labels = set()

        for cycle_key, metrics in self.metrics.items():
            display_label = metrics.get('display_name', '')
            if PREFERRED_DISPLAY_NAMES and display_label not in PREFERRED_DISPLAY_NAMES:
                continue
            if display_label in seen_labels:
                continue

            duration = metrics['duration_s']
            for group_name, min_dur, max_dur, _ in DURATION_GROUPS:
                if min_dur <= duration < max_dur:
                    grouped_cycles[group_name].append((cycle_key, metrics))
                    seen_labels.add(display_label)
                    break

        # Remove empty groups
        grouped_cycles = {k: v for k, v in grouped_cycles.items() if v}
        n_groups = len(grouped_cycles)
        total_cycles_plotted = len(seen_labels)

        if n_groups == 0:
            print("No cycles to plot!")
            return None

        print(f"\nCreating duration-grouped figure with {total_cycles_plotted} cycles")
        
        # Create figure with white background
        fig = plt.figure(figsize=A4_PORTRAIT, dpi=100, facecolor='white')
        gs = gridspec.GridSpec(n_groups, 1, hspace=0.3, top=0.96, bottom=0.04, left=0.08, right=0.98)
        
        # Get the display time for each group
        group_display_times = {}
        for group_name, min_dur, max_dur, display_time in DURATION_GROUPS:
            group_display_times[group_name] = display_time
        
        for panel_idx, (group_name, cycles) in enumerate(grouped_cycles.items()):
            ax = fig.add_subplot(gs[panel_idx])
            
            # Sort cycles within group by duration
            cycles_sorted = sorted(cycles, key=lambda x: x[1]['duration_s'])
            
            # Get the specified display time for this group
            max_time_display = group_display_times[group_name]
            max_speed = max(m['max_speed_kmh'] for _, m in cycles_sorted)
            
            # Plot each cycle with different line styles - HIGHER ZORDER
            for i, (cycle_key, metrics) in enumerate(cycles_sorted):
                if cycle_key not in self.standard_cycles:
                    continue
                
                df = self.standard_cycles[cycle_key]

                # Cycle colors through a high-contrast palette (similar to Figure 3 styling)
                color = SET2_CONTRAST_COLORS[i % len(SET2_CONTRAST_COLORS)]
                
                # Use different line styles for better grayscale distinction
                line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
                linestyle = line_styles[i % len(line_styles)]
                
                # Create label with metrics - use chaos_pct
                chaos_pct = metrics.get('chaos_pct', 0)
                label = f"{metrics['display_name']} ({metrics['duration_s']:.0f}s, chaos={chaos_pct}%)"
                
                # Plot with HIGHER ZORDER so lines appear above everything
                ax.plot(df['time_s'], df['speed_kmh'], 
                       color=color, linestyle=linestyle, alpha=0.9, linewidth=1.0, 
                       label=label, zorder=10)  # High zorder for lines
            
            # Format axes with EXACT specified limits
            ax.set_xlim(0, max_time_display)
            ax.set_ylim(0, max_speed * 1.15)  # 15% extra for legend
            
            # Grid with BETTER VISIBILITY - LOWER ZORDER
            ax.grid(True, which='major', alpha=0.7, color='gray', linewidth=0.5, zorder=0)
            ax.grid(True, which='minor', alpha=0.3, color='lightgray', linewidth=0.3, zorder=0)
            ax.minorticks_on()
            
            # Set background to white
            ax.set_facecolor('white')
            
            # Labels
            ax.set_title(f"{group_name}: {len(cycles)} cycles", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel('Speed (km/h)', fontsize=11)
            
            if panel_idx == n_groups - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            
            # Smart legend positioning
            if panel_idx == 0:
                legend = ax.legend(loc='upper center', fontsize=6, ncol=3,
                                   framealpha=0.7, facecolor='white', edgecolor='gray',
                                   handlelength=2.0, columnspacing=0.8,
                                   borderpad=0.3, labelspacing=0.2,
                                   bbox_to_anchor=(0.5, 1.0))

            elif group_name.startswith("Medium") or group_name.startswith("Standard"):
                ncol = min(4, len(cycles_sorted))
                legend = ax.legend(loc='upper center', fontsize=6, ncol=ncol,
                                   framealpha=0.7, facecolor='white', edgecolor='gray',
                                   handlelength=2.0, columnspacing=0.8,
                                   borderpad=0.3, labelspacing=0.2,
                                   bbox_to_anchor=(0.5, 1.0))

            elif panel_idx == n_groups - 1:
                legend = ax.legend(loc='center left', fontsize=6, ncol=1,
                                   framealpha=0.7, facecolor='white', edgecolor='gray',
                                   handlelength=2.5, columnspacing=1.0,
                                   borderpad=0.5, labelspacing=0.3,
                                   bbox_to_anchor=(0.0, 0.5))
            else:
                n_cycles = len(cycles_sorted)
                if n_cycles <= 6:
                    ncol = 3
                elif n_cycles <= 8:
                    ncol = 4
                else:
                    ncol = 5
                legend = ax.legend(loc='upper center', fontsize=6, ncol=ncol,
                                   framealpha=0.7, facecolor='white', edgecolor='gray',
                                   handlelength=2.0, columnspacing=0.8,
                                   borderpad=0.3, labelspacing=0.2,
                                   bbox_to_anchor=(0.5, 1.0))
            
            # Set legend zorder to be behind the lines
            legend.set_zorder(5)
        
        # Main title
        fig.suptitle(f'Standard Drive Cycles ({total_cycles_plotted} cycles)', 
                    fontsize=14, fontweight='bold', x=0.60)
        
        # Save
        if save_path is None:
            save_path = self.output_dir / 'drive_cycles_duration_grouped.png'
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        print(f"✓ Saved duration-grouped figure: {save_path}")
        plt.close()
        
        return save_path

    def create_simplified_wavelet_figure(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """Create simplified wavelet analysis figure showing ALL cycles in frequency band distribution"""
        
        if not self.metrics:
            print("No metrics available for wavelet visualization")
            return None
        
        # Prepare data - include ALL cycles, computing wavelet metrics if missing
        fallback_cycles: list[str] = []
        cycles_with_wavelet = []

        for cycle_key, metrics in self.metrics.items():
            entropy = metrics.get('wavelet_entropy')
            if (entropy is None or entropy <= 0) and cycle_key in self.standard_cycles:
                df_cycle = self.standard_cycles[cycle_key]
                computed = self._compute_wavelet_metrics(
                    df_cycle['speed_kmh'].to_numpy(),
                    metrics.get('sampling_rate_hz', self.sr_hz)
                )
                if computed:
                    metrics.update(computed)
                    entropy = metrics.get('wavelet_entropy')

            if entropy is None:
                # Keep the cycle visible even if wavelet metrics are unavailable
                metrics['wavelet_entropy'] = 0.0
                metrics.setdefault('band_short_pct', 0.0)
                metrics.setdefault('band_medium_pct', 0.0)
                metrics.setdefault('band_long_pct', 0.0)
                metrics.setdefault('transient_events_per_min', 0.0)
                fallback_cycles.append(metrics.get('display_name', cycle_key))

            cycles_with_wavelet.append((cycle_key, metrics))

        if not cycles_with_wavelet:
            print("No wavelet data available")
            return None

        # Build tidy structure with consistent fields
        fallback_set = set(fallback_cycles)
        records = []
        for cycle_key, metrics in cycles_with_wavelet:
            display_name = metrics.get('display_name', cycle_key.split('_')[-1])
            family = metrics.get('family', 'Special')
            records.append({
                'cycle_key': cycle_key,
                'display_name': display_name,
                'family': family,
                'entropy': float(metrics.get('wavelet_entropy', 0.0) or 0.0),
                'events_per_min': float(metrics.get('transient_events_per_min', 0.0) or 0.0),
                'band_micro_pct': float(metrics.get('band_micro_pct', 0.0) or 0.0),
                'band_short_pct': float(metrics.get('band_short_pct', 0.0) or 0.0),
                'band_medium_pct': float(metrics.get('band_medium_pct', 0.0) or 0.0),
                'band_long_pct': float(metrics.get('band_long_pct', 0.0) or 0.0),
                'band_macro_pct': float(metrics.get('band_macro_pct', 0.0) or 0.0),
                'estimated': display_name in fallback_set
            })

        wavelet_df = pd.DataFrame.from_records(records).sort_values(
            'entropy', ascending=False
        ).reset_index(drop=True)
        n_cycles = len(wavelet_df)

        if fallback_cycles:
            missing_list = ', '.join(sorted(set(fallback_cycles)))
            print(f"Note: wavelet metrics estimated for {len(fallback_cycles)} cycle(s): {missing_list}")

        # Layout tuned for full set while staying within A4 width
        fig_height = max(11, 8 + n_cycles * 0.12)
        fig = plt.figure(figsize=(15, fig_height), facecolor='white')
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.0, 1.25],
            width_ratios=[1.0, 1.0],
            hspace=0.32, wspace=0.25,
            top=0.93, bottom=0.06, left=0.07, right=0.97
        )

        # 1. Wavelet entropy ranking for all cycles
        ax_entropy = fig.add_subplot(gs[0, :])
        colors = wavelet_df['family'].map(FAMILY_COLORS).fillna('#666666')
        y_positions = np.arange(n_cycles)
        bars = ax_entropy.barh(
            y_positions,
            wavelet_df['entropy'],
            color=colors,
            edgecolor='black',
            linewidth=0.4
        )
        ax_entropy.set_yticks(y_positions)
        ax_entropy.set_yticklabels(wavelet_df['display_name'], fontsize=9)
        ax_entropy.invert_yaxis()
        max_entropy = wavelet_df['entropy'].max()
        ax_entropy.set_xlim(0, max_entropy * 1.08 if max_entropy > 0 else 0.5)
        ax_entropy.set_xlabel('Wavelet Entropy (normalized)', fontsize=12, fontweight='bold')
        ax_entropy.set_title(f'Drive Cycle Complexity Ranking ({n_cycles} cycles)', fontsize=12, fontweight='bold')
        ax_entropy.grid(True, alpha=0.3, axis='x')

        def format_entropy_label(val: float) -> str:
            if val >= 0.1:
                return f"{val:.2f}"
            if val >= 0.01:
                return f"{val:.3f}"
            if val > 0:
                return f"{val:.2e}"
            return "0"

        for idx, bar in enumerate(bars):
            if wavelet_df.at[idx, 'estimated']:
                bar.set_hatch('//')
                bar.set_alpha(0.65)
            ax_entropy.text(
                bar.get_width() + 0.006,
                bar.get_y() + bar.get_height() / 2,
                format_entropy_label(wavelet_df.at[idx, 'entropy']),
                ha='left', va='center', fontsize=8
            )

        from matplotlib.patches import Patch
        family_handles = [
            Patch(facecolor=FAMILY_COLORS[fam], edgecolor='black', label=fam)
            for fam in FAMILY_COLORS
            if fam in wavelet_df['family'].values
        ]
        legend_handles = family_handles
        if wavelet_df['estimated'].any():
            legend_handles = legend_handles + [Patch(
                facecolor='white', edgecolor='black', hatch='//', label='Metrics estimated'
            )]
        if legend_handles:
            ax_entropy.legend(handles=legend_handles, loc='lower right', fontsize=9)

        # 2. Entropy vs transient activity scatter, marker size = short-band share
        ax_scatter = fig.add_subplot(gs[1, 0])
        marker_sizes = 40 + wavelet_df['band_short_pct'] * 2.5
        scatter = ax_scatter.scatter(
            wavelet_df['entropy'],
            wavelet_df['events_per_min'],
            s=marker_sizes,
            c=colors,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.4
        )
        ax_scatter.set_xlabel('Wavelet Entropy (normalized)', fontsize=11, fontweight='bold')
        ax_scatter.set_ylabel('Transient Events per Minute', fontsize=11, fontweight='bold')
        ax_scatter.set_title('Transient Activity vs. Complexity', fontsize=11, fontweight='bold')
        ax_scatter.grid(True, alpha=0.45, color='#B0B0B0', linestyle='--', linewidth=0.6)
        ax_scatter.set_xlim(0, max_entropy * 1.05 if max_entropy > 0 else 0.5)
        max_events = wavelet_df['events_per_min'].max()
        if max_events > 0:
            upper_margin = max(1.5, 0.2 * max_events)
            ax_scatter.set_ylim(-0.2, max_events + upper_margin)
        else:
            ax_scatter.set_ylim(-0.2, 1.0)
        ax_scatter.set_xlim(-0.01, max_entropy * 1.05 if max_entropy > 0 else 0.5)

        size_examples = [5, 15, 30]
        size_handles = [
            plt.scatter([], [], s=40 + pct * 2.5, facecolor='none', edgecolor='black',
                        linewidth=0.4, label=f'{pct}% short-band')
            for pct in size_examples
        ]
        family_handles_scatter = [
            plt.Line2D([], [], marker='o', linestyle='', markersize=6,
                       markerfacecolor=FAMILY_COLORS[fam], markeredgecolor='black',
                       label=fam)
            for fam in FAMILY_COLORS
            if fam in wavelet_df['family'].values
        ]

        legend1 = ax_scatter.legend(
            handles=family_handles_scatter,
            fontsize=8,
            title='Family color',
            loc='upper right',
            borderaxespad=0.6
        )
        ax_scatter.add_artist(legend1)
        legend2 = ax_scatter.legend(
            handles=size_handles,
            fontsize=8,
            title='Marker size (short-band %)',
            loc='upper right',
            bbox_to_anchor=(0.72, 0.75)
        )
        ax_scatter.add_artist(legend2)

        # 3. Heatmap of band energy distribution (short → long)
        ax_heatmap = fig.add_subplot(gs[1, 1])
        band_columns = [
            'band_short_pct', 'band_medium_pct', 'band_long_pct'
        ]
        heatmap_data = wavelet_df[band_columns].to_numpy()
        norm = Normalize(vmin=0, vmax=max(heatmap_data.max(), 1))
        im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='viridis', norm=norm)
        ax_heatmap.set_title('Frequency Band Energy Mix (CWT)', fontsize=11, fontweight='bold')
        ax_heatmap.set_xticks(np.arange(len(band_columns)))
        ax_heatmap.set_xticklabels([
            'Short\n(0.1-1 Hz)', 'Medium\n(0.01-0.1 Hz)', 'Long\n(0.001-0.01 Hz)'
        ], fontsize=9)
        ax_heatmap.set_yticks(y_positions)
        ax_heatmap.set_yticklabels(wavelet_df['display_name'], fontsize=8)
        ax_heatmap.tick_params(axis='both', which='both', length=0)
        ax_heatmap.set_ylabel('Drive Cycle', fontsize=11, fontweight='bold')
        ax_heatmap.set_xlabel('Band', fontsize=11, fontweight='bold')
        for spine in ax_heatmap.spines.values():
            spine.set_visible(False)
        ax_heatmap.set_xticks(np.arange(-0.5, len(band_columns), 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-0.5, n_cycles, 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linewidth=0.4)
        cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
        cbar.set_label('Energy share (%)', fontsize=10)

        fig.suptitle(f'Wavelet Transform Analysis - All {n_cycles} Cycles', fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path is None:
            save_path = self.output_dir / 'wavelet_analysis_simplified.png'

        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved wavelet figure with all {n_cycles} cycles: {save_path}")
        plt.close()
        
        return save_path

    def export_metrics_csv(
        self,
        csv_path: Optional[Path] = None,
        wavelet_data: Optional[Dict] = None,
        log_summary: bool = False,
    ) -> Path:
        """
        Export metrics to CSV with PKE, V95, and probability distributions.

        When log_summary is True, emit a detailed console summary. Callers that
        prefer to manage their own messaging should leave it False to avoid
        duplicate log lines.
        """
        
        if not self.metrics:
            print("No metrics to export!")
            return None
        
        rows = []
        for key, m in self.metrics.items():
            # Extract canonical and display names for consistent downstream usage
            display_name = m.get("display_name", key)
            canonical_name = m.get("canonical_name", canonicalize_name(display_name))
            
            row = {
                "Name": canonical_name,  # Canonical name consumed by the normalizer
                "DisplayName": display_name,
                "Category": m.get("family", ""),  # Temporary, for normalizer
                "Duration (s)": int(round(m.get("duration_s", 0))),
                "Distance (km)": round(m.get("distance_km", 0.0), 3),
                "Mean Speed (km/h)": round(m.get("mean_speed_kmh", 0.0), 2),
                "V95 (km/h)": round(m.get("v95", 0.0), 2),  # Added V95 after mean speed
                "Max Speed (km/h)": round(m.get("max_speed_kmh", 0.0), 2),
                "Idle (%)": round(m.get("idle_pct", 0.0), 2),
                "PKE (m/s²)": round(m.get("pke", 0.0), 3),  # Added PKE after idle
                "Chaos (%)": m.get("chaos_pct", 0),
                "Max Accel (m/s²)": round(m.get("accel_max_ms2", 0.0), 2),
                "Max Decel (m/s²)": round(m.get("decel_min_ms2", 0.0), 2),
                "Stops": int(m.get("n_stops", 0)),
                "Stops/km": round(m.get("stops_per_km", 0.0), 2),
                # Probability distribution
                "Distribution": m.get("prob_distribution", ""),
                "Dist. Parameters": m.get("distribution_params", ""),
                # Wavelet metrics
                "Wavelet Entropy": round(m.get("wavelet_entropy", 0), 2) if m.get("wavelet_entropy") else "",
                "Short Band (%)": m.get("band_short_pct", ""),
                "Medium Band (%)": m.get("band_medium_pct", ""),
                "Long Band (%)": m.get("band_long_pct", ""),
                "Transient Events (/min)": m.get("transient_events_per_min", ""),
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Remove duplicates before sorting
        df = df.drop_duplicates(subset=["Name", "Duration (s)"], keep="first")
        df = df.sort_values("Duration (s)").reset_index(drop=True)
        
        # Save temporary CSV
        temp_csv = self.output_dir / "_temp_metrics.csv"
        df.to_csv(temp_csv, index=False)
        
        # Apply cycle name normalizer if available
        try:
            from cycle_name_normalizer import normalize_cycle_table
            
            if csv_path is None:
                csv_path = self.output_dir / "cycle_metrics_publication.csv"
            
            # Normalize names and categories
            normalize_cycle_table(temp_csv, csv_path)
            
            # Read back and remove unnecessary columns
            df_normalized = pd.read_csv(csv_path)
            if "Family" in df_normalized.columns:
                df_normalized = df_normalized.drop(columns=["Family"])
            if "Category" in df_normalized.columns:
                df_normalized = df_normalized.drop(columns=["Category"])
                
            # Use the generated display label when available so figures and CSV share text
            if "DisplayName" in df_normalized.columns:
                df_normalized.insert(0, "Cycle", df_normalized["DisplayName"])
                df_normalized = df_normalized.drop(columns=["DisplayName"])
                if "Name" in df_normalized.columns:
                    df_normalized = df_normalized.drop(columns=["Name"])
                if "LongName" in df_normalized.columns:
                    df_normalized = df_normalized.drop(columns=["LongName"])
            elif "LongName" in df_normalized.columns:
                df_normalized.insert(0, "Cycle", df_normalized.apply(
                    lambda row: build_display_name(row["Name"]), axis=1
                ))
                df_normalized = df_normalized.drop(columns=["Name", "LongName"])
            else:
                # Fallback to the canonical name as the visible label
                df_normalized.rename(columns={"Name": "Cycle"}, inplace=True)

            # Ensure the Cycle column is the leading column for readability
            cols = ["Cycle"] + [col for col in df_normalized.columns if col != "Cycle"]
            df_normalized = df_normalized[cols]
            
            # Save final version
            df_normalized.to_csv(csv_path, index=False)
            
            # Clean up temp file
            temp_csv.unlink(missing_ok=True)
            
            if log_summary:
                print(f"✓ Exported normalized metrics CSV with PKE, V95, and distributions: {csv_path}")
                print(f"  Total cycles: {len(df_normalized)}")
                
                # Print summary of new metrics
                print("\nNew metrics summary:")
                print(f"  PKE: {df_normalized['PKE (m/s²)'].notna().sum()}/{len(df_normalized)} filled")
                print(f"  V95: {df_normalized['V95 (km/h)'].notna().sum()}/{len(df_normalized)} filled")
                print(f"  Distribution: {df_normalized['Distribution'].notna().sum()}/{len(df_normalized)} identified")
            
        except ImportError:
            # Fallback if normalizer not available
            print("Warning: cycle_name_normalizer not available, using basic names")
            
            # Remove Category column in fallback
            if "Category" in df.columns:
                df = df.drop(columns=["Category"])
            if "DisplayName" in df.columns:
                df.insert(0, "Cycle", df["DisplayName"])
                df = df.drop(columns=["DisplayName"])
                if "Name" in df.columns:
                    df = df.drop(columns=["Name"])
            else:
                df.rename(columns={"Name": "Cycle"}, inplace=True)

            if csv_path is None:
                csv_path = self.output_dir / "cycle_metrics_publication.csv"
            
            df.to_csv(csv_path, index=False)
            temp_csv.unlink(missing_ok=True)
            
            if log_summary:
                print(f"✓ Exported metrics CSV: {csv_path} ({len(df)} cycles)")
        
        return csv_path

    def generate_all_improved_figures(self, wavelet_data: Optional[Dict] = None) -> List[Path]:
        """Generate all outputs with integrated analysis"""
        outputs = []
        
        # Figure 3: Urban routes speed comparison (stop-aligned)
        try:
            stop_fig = self.create_urban_routes_speed_comparison(
                save_path=self.output_dir / 'figure_03_urban_routes_stop_to_stop.png',
                normalization="stop",
                show_stop_windows=True)
            if stop_fig:
                outputs.append(stop_fig)
        except Exception as e:
            print(f"Warning: Stop-aligned urban routes comparison failed: {e}")

        # Figure 3b: Urban routes speed comparison (distance-normalised)
        try:
            distance_fig = self.create_urban_routes_speed_comparison(
                save_path=self.output_dir / 'figure_03_urban_routes_distance_normalized.png',
                normalization="distance",
                show_stop_windows=True)
            if distance_fig:
                outputs.append(distance_fig)
        except Exception as e:
            print(f"Warning: Distance-normalised urban routes comparison failed: {e}")

        # Figure 3c: Urban routes speed comparison (time-normalised)
        try:
            time_fig = self.create_urban_routes_speed_comparison(
                save_path=self.output_dir / 'figure_03_urban_routes_time_normalized.png',
                normalization="time",
                show_stop_windows=True)
            if time_fig:
                outputs.append(time_fig)
        except Exception as e:
            print(f"Warning: Time-normalised urban routes comparison failed: {e}")

        # Supplemental: Stop cluster overview
        try:
            cluster_fig = self.create_stop_cluster_map()
            if cluster_fig:
                outputs.append(cluster_fig)
        except Exception as e:
            print(f"Warning: Stop cluster map failed: {e}")

        # Supplemental: Route vs WLTP stop profile
        try:
            stop_profile_fig = self.create_wltp_stop_profile()
            if stop_profile_fig:
                outputs.append(stop_profile_fig)
        except Exception as e:
            print(f"Warning: WLTP stop profile figure failed: {e}")

        # Supplemental: trip stack overlays for supported routes
        supported_routes = self.results.get('routes', {}).get('route_display_order', [])
        for rid in supported_routes[:8]:
            try:
                stack_fig = self.create_route_trip_stack(rid)
                if stack_fig:
                    outputs.append(stack_fig)
            except Exception as e:
                print(f"Warning: Trip stack figure failed for {rid}: {e}")

        # Figure 4: Route-level Markov surfaces
        try:
            markov_fig = self.create_route_markov_surface()
            if markov_fig:
                outputs.append(markov_fig)
        except Exception as e:
            print(f"Warning: Markov surface figure failed: {e}")

        # Figure 5: Real vs standard ratios (log scale)
        try:
            ratio_fig = self.create_real_vs_standard_comparison()
            if ratio_fig:
                outputs.append(ratio_fig)
        except Exception as e:
            print(f"Warning: Real vs standard figure failed: {e}")

        # Figure 6: Normalised speed pattern comparison
        try:
            pattern_fig = self.create_normalized_speed_pattern()
            if pattern_fig:
                outputs.append(pattern_fig)
        except Exception as e:
            print(f"Warning: Normalized speed pattern figure failed: {e}")

        # Duration-grouped figure (Figure 2)
        try:
            fig_path = self.create_duration_grouped_figure()
            if fig_path:
                outputs.append(fig_path)
        except Exception as e:
            print(f"Warning: Duration-grouped figure failed: {e}")
        
        # Simplified wavelet analysis figure with ALL cycles
        try:
            wavelet_fig_path = self.create_simplified_wavelet_figure()
            if wavelet_fig_path:
                outputs.append(wavelet_fig_path)
        except Exception as e:
            print(f"Warning: Wavelet figure failed: {e}")
        
        # Metrics CSV with PKE, V95, and distributions
        try:
            csv_path = self.export_metrics_csv(log_summary=True)
            if csv_path:
                outputs.append(csv_path)
        except Exception as e:
            print(f"Warning: CSV export failed: {e}")
        
        print(f"\nGenerated {len(outputs)} outputs")
        return outputs

    def generate_all_outputs(self, wavelet_data: Optional[Dict] = None):
        """Alias for backward compatibility"""
        return self.generate_all_improved_figures(wavelet_data=wavelet_data)


if __name__ == "__main__":
    viz = ImprovedPublicationVisualizer()
    outputs = viz.generate_all_improved_figures()
    for o in outputs:
        print("->", o)
