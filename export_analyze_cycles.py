#!/usr/bin/env python3
"""
Export custom drive cycles and analyze standard cycles with segmentation.
FINAL VERSION with robust JSON flattening and REAL-cycle fallback.

This module provides:
- CycleExporter class with:
  - export_custom_cycles(): robustly flattens JSON/mixed structures to sample-wise DataFrames,
    ensures speed/index columns, adds metadata, writes CSV/Parquet.
  - analyze_standard_cycles(): segments standard cycles and computes metrics.
  - compare_cycles(): coarse similarity vs. standard cycles.
  - generate_cycle_report(): runs export, analyze, compare and writes a markdown report.

Compatible with unified_analysis_main_all_in_one.py which imports CycleExporter.
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd


# Optional heavy imports guarded where needed
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # pragma: no cover
    plt = None


class CycleExporter:
    def __init__(self, analysis_json_path: Path, cycles_dir: Path, output_dir: Path):
        """Initialize with paths"""
        with open(analysis_json_path, "r") as f:
            self.results = json.load(f)
        self.cycles_dir = Path(cycles_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # NEW: Robust exporter with JSON flattening + REAL fallback
    # -------------------------------------------------------------------------
    def export_custom_cycles(self) -> Dict[str, pd.DataFrame]:
        """
        Export all custom cycles from route analysis to CSV/Parquet.

        Robust handling:
        - Works with dict/list/mixed JSON structures (including 1-row frames with list-like cells).
        - Recursively flattens list-like columns into a proper row-per-sample DataFrame.
        - Ensures presence of speed_ms/speed_kmh.
        - Adds method-specific index columns (time_s / normalized_distance / normalized_position).
        - If the JSON-derived frame is too small (<10 rows), falls back to REAL cycles produced in Phase 3.5.
        - Also scans Phase 3.5 REAL cycles if nothing is exported directly.
        """
        exported_cycles: Dict[str, pd.DataFrame] = {}

        route_variability = self.results.get("route_variability", {}) or {}

        for route_id, route_data in route_variability.items():
            if not isinstance(route_data, dict):
                continue
            custom_cycles = route_data.get("custom_cycles", {}) or {}
            if not custom_cycles:
                continue

            for method in ("time_normalized", "distance_normalized", "stop_to_stop"):
                if method not in custom_cycles:
                    continue

                cycle_data = custom_cycles.get(method)
                if cycle_data in (None, {}, []):
                    continue

                try:
                    # 1) Build DataFrame from JSON (robust flatten)
                    df = self._flatten_mixed_cycle_data(route_id, method, cycle_data)

                    # 2) If we couldn't build a valid frame, or it's too small, try REAL fallback
                    if df is None or df.empty:
                        print(
                            f"Skipping {route_id} {method}: could not create a valid DataFrame; trying REAL fallback…"
                        )
                        df = self._try_load_real_cycle(route_id, method)
                        if df is None:
                            continue
                    elif len(df) < 10:
                        print(
                            f"JSON for {route_id} {method} yielded only {len(df)} rows; trying REAL fallback…"
                        )
                        real_df = self._try_load_real_cycle(route_id, method)
                        if real_df is not None:
                            df = real_df

                    # 3) Ensure speed columns
                    df = self._ensure_speed_columns(df)
                    if df is None or df.empty or len(df) < 10:
                        print(
                            f"Skipping {route_id} {method}: insufficient usable data ({0 if df is None else len(df)} rows)"
                        )
                        continue

                    # 4) Add index columns based on method
                    df = self._add_index_columns(df, method)

                    # 5) Attach metadata (from route_data if available)
                    if "source" not in df.columns:
                        df["source"] = f"{route_id}_{method}"
                    df["n_trips"] = route_data.get("n_trips", 0)

                    overall_metrics = route_data.get("overall_metrics", {}) or {}
                    df["chaos_score"] = pd.to_numeric(
                        overall_metrics.get("route_chaos_score", 0), errors="coerce"
                    )
                    df["predictability"] = pd.to_numeric(
                        overall_metrics.get("route_predictability", 0), errors="coerce"
                    )

                    # 6) Save outputs
                    csv_path = self.output_dir / f"{route_id}_{method}_cycle.csv"
                    parquet_path = self.output_dir / f"{route_id}_{method}_cycle.parquet"
                    df.to_csv(csv_path, index=False)
                    df.to_parquet(parquet_path, index=False)

                    exported_cycles[f"{route_id}_{method}"] = df
                    print(f"Exported {route_id} {method} cycle: {len(df)} samples")

                except Exception as e:
                    print(f"Error exporting {route_id} {method}: {e}")
                    traceback.print_exc()
                    continue

            time_variants = custom_cycles.get("time_variants", {})
            for variant_key, variant_cycle in time_variants.items():
                try:
                    df_variant = self._flatten_mixed_cycle_data(variant_key, "time_normalized", variant_cycle)
                    if df_variant is None or df_variant.empty:
                        continue
                    df_variant = self._ensure_speed_columns(df_variant)
                    df_variant = self._add_index_columns(df_variant, "time_normalized")
                    if "source" not in df_variant.columns:
                        df_variant["source"] = f"{variant_key}_time_normalized"
                    csv_path = self.output_dir / f"{variant_key}_time_normalized_cycle.csv"
                    parquet_path = self.output_dir / f"{variant_key}_time_normalized_cycle.parquet"
                    df_variant.to_csv(csv_path, index=False)
                    df_variant.to_parquet(parquet_path, index=False)
                    exported_cycles[f"{variant_key}_time_normalized"] = df_variant
                    print(f"Exported {variant_key} time_normalized variant: {len(df_variant)} samples")
                except Exception as exc:
                    print(f"Error exporting variant {variant_key}: {exc}")

        # If nothing exported directly, scan and re-export REAL cycles (Phase 3.5 artifacts)
        if not exported_cycles:
            print("\nTrying to load REAL cycles from Phase 3.5...")
            real_cycle_dir = self.output_dir.parent / "exported_cycles"
            if real_cycle_dir.exists():
                for real_file in real_cycle_dir.glob("*_REAL.csv"):
                    try:
                        df = pd.read_csv(real_file)
                        if df.empty:
                            continue
                        df = self._ensure_speed_columns(df)
                        if df is None:
                            continue
                        stem = real_file.stem.replace("_REAL", "")
                        method_guess = (
                            "time_normalized"
                            if "time_normalized" in stem
                            else "distance_normalized"
                            if "distance_normalized" in stem
                            else "stop_to_stop"
                            if "stop_to_stop" in stem
                            else "time_normalized"
                        )
                        df = self._add_index_columns(df, method_guess)
                        if "source" not in df.columns:
                            df["source"] = stem

                        csv_path = self.output_dir / f"{stem}_cycle.csv"
                        parquet_path = self.output_dir / f"{stem}_cycle.parquet"
                        df.to_csv(csv_path, index=False)
                        df.to_parquet(parquet_path, index=False)

                        exported_cycles[stem] = df
                        print(f"Loaded and re-exported REAL cycle: {stem} ({len(df)} samples)")
                    except Exception as e:
                        print(f"Error loading {real_file}: {e}")

        if not exported_cycles:
            print("No cycles exported - check route variability analysis completed properly")

        return exported_cycles

    # -------------------------------------------------------------------------
    # Helpers for export_custom_cycles
    # -------------------------------------------------------------------------
    def _flatten_mixed_cycle_data(
        self, route_id: str, method: str, cycle_data: Any
    ) -> Optional[pd.DataFrame]:
        """
        Try multiple strategies to turn mixed JSON into a row-per-sample DataFrame.
        """
        # Direct cases
        if isinstance(cycle_data, pd.DataFrame):
            df = cycle_data.copy()
            df = self._maybe_expand_listlike_cells(df, route_id, method)
            return df

        if isinstance(cycle_data, list):
            if len(cycle_data) == 0:
                return None
            if isinstance(cycle_data[0], dict):
                df = pd.DataFrame(cycle_data)
                return self._maybe_expand_listlike_cells(df, route_id, method)
            else:
                # Bare list of speed values
                df = pd.DataFrame({"speed_ms": pd.to_numeric(cycle_data, errors="coerce")})
                return df

        if isinstance(cycle_data, dict):
            if not cycle_data:
                return None

            # Case: DataFrame .to_dict('split')
            if all(k in cycle_data for k in ("index", "columns", "data")):
                try:
                    df = pd.DataFrame(
                        data=cycle_data["data"],
                        columns=cycle_data["columns"],
                        index=cycle_data.get("index"),
                    )
                    return self._maybe_expand_listlike_cells(df, route_id, method)
                except Exception:
                    pass

            # If values are all lists with consistent lengths
            list_values = [v for v in cycle_data.values() if isinstance(v, list)]
            if list_values:
                lengths = {len(v) for v in list_values}
                if len(lengths) == 1 and len(list_values) == len(cycle_data):
                    try:
                        df = pd.DataFrame(cycle_data)
                        return df
                    except Exception:
                        pass

            # If values are dicts (index-like orientation)
            if all(isinstance(v, dict) for v in cycle_data.values()):
                try:
                    df = pd.DataFrame.from_dict(cycle_data, orient="index")
                    return self._maybe_expand_listlike_cells(df, route_id, method)
                except Exception:
                    pass

            # Fallback: try json_normalize
            try:
                df = pd.json_normalize(cycle_data)
                df = self._maybe_expand_listlike_cells(df, route_id, method)
                return df
            except Exception:
                pass

        # Last resort: None
        return None

    def _maybe_expand_listlike_cells(
        self, df: pd.DataFrame, route_id: str, method: str
    ) -> pd.DataFrame:
        """
        If a DataFrame contains list-like cells (often 1-row with lists), expand them
        to a proper sample-wise table.
        """
        if df is None or df.empty:
            return df

        # Detect presence of list-like cells
        def _is_listlike(x: Any) -> bool:
            return isinstance(x, (list, tuple, np.ndarray, pd.Series))

        any_listlike = False
        max_len = 0
        if len(df) == 1:
            for col in df.columns:
                val = df.iloc[0][col]
                if _is_listlike(val):
                    any_listlike = True
                    try:
                        max_len = max(max_len, len(val))
                    except Exception:
                        pass

        if any_listlike and max_len > 0:
            # Expand to max_len rows; broadcast scalars
            out = {}
            for col in df.columns:
                val = df.iloc[0][col]
                if _is_listlike(val):
                    seq = list(val)
                    if len(seq) != max_len:
                        # pad/truncate to align
                        if len(seq) < max_len:
                            seq = seq + [np.nan] * (max_len - len(seq))
                        else:
                            seq = seq[:max_len]
                    out[col] = seq
                else:
                    out[col] = [val] * max_len
            new_df = pd.DataFrame(out)
            print(f"Flattened recursively for {route_id} {method} -> {len(new_df)} rows")
            return new_df

        # Also handle multi-row frames that still have list-like cells: explode per column
        # (Only explode columns where every row is list-like with same lengths)
        explode_cols: List[str] = []
        same_lengths: Optional[int] = None
        # Limit to a small number of columns to avoid combinatorial explosions
        MAX_EXPLODE = 6
        for col in df.columns[:MAX_EXPLODE]:
            col_vals = df[col]
            if col_vals.apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).all():
                lens = col_vals.apply(lambda x: len(x) if x is not None else 0)
                if lens.nunique() == 1 and lens.iloc[0] > 1:
                    length = int(lens.iloc[0])
                    if same_lengths in (None, length):
                        same_lengths = length
                        explode_cols.append(col)

        if explode_cols and same_lengths and len(df) == 1:
            new_df = df.copy()
            for col in explode_cols:
                new_df = new_df.explode(col, ignore_index=True)
            print(
                f"Expanded list-like cells for {route_id} {method} -> {len(new_df)} rows (from {len(df)})"
            )
            return new_df

        return df

    def _ensure_speed_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Guarantee both speed_ms and speed_kmh columns exist and are numeric."""
        if df is None or df.empty:
            return None

        cols = set(df.columns.str.lower())

        # Common alternates
        candidates_ms = [c for c in df.columns if c.lower() in ("speed_ms", "speed(m/s)", "v_ms")]
        candidates_kmh = [
            c for c in df.columns if c.lower() in ("speed_kmh", "speed (km/h)", "vehicle speed", "v_kmh")
        ]

        if "speed_ms" not in df.columns and candidates_ms:
            df["speed_ms"] = pd.to_numeric(df[candidates_ms[0]], errors="coerce")
        if "speed_kmh" not in df.columns and candidates_kmh and "speed_ms" not in df.columns:
            df["speed_ms"] = pd.to_numeric(df[candidates_kmh[0]], errors="coerce") / 3.6

        # If neither detected, try generic heuristic
        if "speed_ms" not in df.columns and "speed_kmh" not in df.columns:
            speed_like = [c for c in df.columns if "speed" in c.lower()]
            if speed_like:
                s = pd.to_numeric(df[speed_like[0]], errors="coerce")
                # Guess units: if max > 70, assume km/h
                if s.max(skipna=True) > 70:
                    df["speed_ms"] = s / 3.6
                else:
                    df["speed_ms"] = s

        if "speed_ms" in df.columns and "speed_kmh" not in df.columns:
            df["speed_kmh"] = pd.to_numeric(df["speed_ms"], errors="coerce") * 3.6
        if "speed_kmh" in df.columns and "speed_ms" not in df.columns:
            df["speed_ms"] = pd.to_numeric(df["speed_kmh"], errors="coerce") / 3.6

        if "speed_ms" not in df.columns:
            return None

        df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
        df["speed_kmh"] = pd.to_numeric(df.get("speed_kmh", df["speed_ms"] * 3.6), errors="coerce")

        # Remove rows with all-NaN speed
        df = df[~df["speed_ms"].isna()].reset_index(drop=True)
        return df

    def _add_index_columns(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Add method-specific index vectors if missing."""
        n = len(df)
        if method == "time_normalized":
            if "time_s" not in df.columns:
                # Assume 10 Hz
                df["time_s"] = np.arange(n, dtype=float) * 0.1
        elif method == "distance_normalized":
            if "normalized_distance" not in df.columns:
                df["normalized_distance"] = np.linspace(0.0, 1.0, n)
        elif method == "stop_to_stop":
            if "normalized_position" not in df.columns:
                df["normalized_position"] = np.linspace(0.0, 1.0, n)
        return df

    def _try_load_real_cycle(self, route_id: str, method: str) -> Optional[pd.DataFrame]:
        """
        Try to load Phase 3.5 REAL cycle artifacts from outputs/exported_cycles.
        Expected patterns include:
          {route_id}_{method}_REAL.csv
          {route_id}_{method}_cycle_REAL.csv  (backup)
        """
        real_dir = self.output_dir.parent / "exported_cycles"
        if not real_dir.exists():
            return None

        patterns = [
            f"{route_id}_{method}_REAL.csv",
            f"{route_id}_{method}_cycle_REAL.csv",
        ]
        for pat in patterns:
            for p in real_dir.glob(pat):
                try:
                    df = pd.read_csv(p)
                    if df.empty:
                        continue
                    df = self._ensure_speed_columns(df)
                    if df is None or df.empty:
                        continue
                    df = self._add_index_columns(df, method)
                    if "source" not in df.columns:
                        df["source"] = f"{route_id}_{method}"
                    print(f"Loaded REAL fallback for {route_id} {method}: {p.name} -> {len(df)} rows")
                    return df
                except Exception as e:
                    print(f"Error reading REAL fallback {p}: {e}")
        return None

    # -------------------------------------------------------------------------
    # Standard cycles analysis and comparison (unchanged from prior behavior)
    # -------------------------------------------------------------------------
    def analyze_standard_cycles(self) -> Dict:
        """Segment and analyze standard cycles (WLTP, EPA, etc.)."""
        cycles = self._load_standard_cycles_enhanced()
        if not cycles:
            print("Found 0 standard cycle files")
            return {}

        print(f"Analyzing {len(cycles)} standard cycles")
        standard_analysis: Dict[str, Dict] = {}

        for cycle_name, df in cycles.items():
            try:
                segments = self._segment_standard_cycle(df, cycle_name)
                cycle_analysis = {
                    "cycle_name": cycle_name,
                    "total_duration_s": float(df['time_s'].iloc[-1] if 'time_s' in df.columns else len(df)),
                    "segments": segments,
                    "overall_metrics": self._calculate_cycle_metrics(segments),
                }
                standard_analysis[cycle_name] = cycle_analysis
            except Exception as exc:
                print(f"Error analyzing {cycle_name}: {exc}")

        return standard_analysis

    def _load_standard_cycles_enhanced(self) -> Dict[str, pd.DataFrame]:
        """Load standard cycles (CSV/SCV/Parquet) with unified naming."""
        cycles: Dict[str, pd.DataFrame] = {}

        master = self.cycles_dir / "cycles_master.parquet"
        if master.exists():
            try:
                df_master = pd.read_parquet(master)
                if '_source_file' in df_master.columns:
                    for source in df_master['_source_file'].unique():
                        name = Path(source).stem
                        df_cycle = df_master[df_master['_source_file'] == source].copy()
                        normalized = self._normalize_cycle_dataframe(df_cycle)
                        if normalized is not None:
                            cycles[name] = normalized
            except Exception as exc:
                print(f"Warning: failed to read master standard cycles: {exc}")

        categories = ['WLTP_Europe', 'EPA', 'Artemis', 'Asia', 'Special']
        for category in categories:
            category_path = self.cycles_dir / category
            if not category_path.exists():
                continue
            files = list(category_path.glob('*.csv'))
            files += list(category_path.glob('*.scv'))
            files += list(category_path.glob('*.parquet'))
            for file_path in files:
                name = f"{category}_{file_path.stem}"
                if name in cycles:
                    continue
                try:
                    if file_path.suffix.lower() == '.parquet':
                        raw_df = pd.read_parquet(file_path)
                    else:
                        raw_df = pd.read_csv(file_path, comment='#', encoding='utf-8-sig')
                    normalized = self._normalize_cycle_dataframe(raw_df)
                    if normalized is not None:
                        cycles[name] = normalized
                except Exception as exc:
                    print(f"  Error loading {file_path.name}: {exc}")

        # Fallback: any remaining files under cycles_dir
        for file_path in self.cycles_dir.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {'.csv', '.scv', '.parquet'}:
                continue
            name = file_path.stem if file_path.parent == self.cycles_dir else f"{file_path.parent.name}_{file_path.stem}"
            if name in cycles:
                continue
            try:
                if file_path.suffix.lower() == '.parquet':
                    raw_df = pd.read_parquet(file_path)
                else:
                    raw_df = pd.read_csv(file_path, comment='#', encoding='utf-8-sig')
                normalized = self._normalize_cycle_dataframe(raw_df)
                if normalized is not None:
                    cycles[name] = normalized
            except Exception as exc:
                print(f"  Error loading {file_path.name}: {exc}")

        return cycles

    def _normalize_cycle_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure a standard cycle DataFrame exposes time_s/speed_ms/speed_kmh."""
        candidate = None
        for col in df.columns:
            lower = col.strip().lower()
            if 'speed' in lower:
                candidate = pd.to_numeric(df[col], errors='coerce')
                if candidate.notna().sum() > 0:
                    break
        if candidate is None or candidate.empty:
            return None

        # Drop NaNs and reset index
        speeds = candidate.dropna().reset_index(drop=True)
        if speeds.empty:
            return None

        if speeds.max() > 60:  # km/h
            speed_kmh = speeds
            speed_ms = speeds / 3.6
        else:
            speed_ms = speeds
            speed_kmh = speeds * 3.6

        df_out = pd.DataFrame({
            'time_s': np.arange(len(speed_ms), dtype=float) * 0.1,
            'speed_ms': speed_ms.values,
            'speed_kmh': speed_kmh.values,
        })
        return df_out

    def _segment_standard_cycle(self, df: pd.DataFrame, cycle_name: str) -> List[Dict]:
        """
        Segment standard cycle based on speed patterns
        """
        segments: List[Dict[str, Any]] = []
        speeds = df["speed_ms"].values

        # Find stops (speed < 0.5 m/s)
        stops = speeds < 0.5

        # Find segment boundaries (stops or major speed changes)
        boundaries: List[int] = [0]

        # Add stops as boundaries
        for i in range(1, len(stops) - 1):
            if not stops[i - 1] and stops[i] and stops[i + 1]:
                boundaries.append(i)

        # Add major speed changes (>5 m/s in 1 sample)
        speed_changes = np.abs(np.diff(speeds))
        large_changes = np.where(speed_changes > 5)[0]
        boundaries.extend(large_changes[::5])  # Sample to avoid over-segmentation

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        boundaries.append(len(df) - 1)

        # Create segments
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end - start < 10:  # Skip tiny segments
                continue

            seg_speeds = speeds[start:end]

            # Calculate segment metrics
            segment = self._analyze_cycle_segment(seg_speeds, f"{cycle_name}_seg{i:03d}")
            segment["start_idx"] = start
            segment["end_idx"] = end
            segments.append(segment)

        return segments

    def _analyze_cycle_segment(self, speeds: np.ndarray, seg_id: str) -> Dict:
        """
        Analyze a segment from a standard cycle
        """
        if len(speeds) == 0:
            return {"segment_id": seg_id, "invalid": True}

        from scipy.stats import entropy, skew, kurtosis

        # Speed statistics
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        cv = std_speed / (mean_speed + 1e-6)

        # Distribution metrics
        hist, _ = np.histogram(speeds, bins=20)
        speed_entropy = entropy(hist + 1e-10)

        # Acceleration
        accels = np.diff(speeds)

        # Classify segment type
        if mean_speed < 2:
            seg_type = "idle"
        elif std_speed < 1:
            seg_type = "cruise"
        elif len(accels) > 0 and np.max(accels) > 2:
            seg_type = "acceleration"
        elif len(accels) > 0 and np.min(accels) < -2:
            seg_type = "deceleration"
        else:
            seg_type = "mixed"

        return {
            "segment_id": seg_id,
            "segment_type": seg_type,
            "duration_s": int(len(speeds)),
            "mean_speed_ms": float(mean_speed),
            "std_speed_ms": float(std_speed),
            "cv": float(cv),
            "max_speed_ms": float(np.max(speeds)),
            "min_speed_ms": float(np.min(speeds)),
            "entropy": float(speed_entropy),
            "skewness": float(skew(speeds)),
            "kurtosis": float(kurtosis(speeds)),
            "percentiles": {
                "p5": float(np.percentile(speeds, 5)),
                "p25": float(np.percentile(speeds, 25)),
                "p50": float(np.percentile(speeds, 50)),
                "p75": float(np.percentile(speeds, 75)),
                "p95": float(np.percentile(speeds, 95)),
            },
            "stop_fraction": float(np.sum(speeds < 0.5) / len(speeds)),
            "chaos_index": float(cv * speed_entropy / math.log(20)),  # Normalized chaos
        }

    def _calculate_cycle_metrics(self, segments: List[Dict]) -> Dict:
        """
        Calculate overall metrics for a cycle
        """
        if not segments:
            return {}

        chaos_indices = [s.get("chaos_index", 0) for s in segments]
        cvs = [s.get("cv", 0) for s in segments]

        # Segment type distribution
        seg_types = [s.get("segment_type", "unknown") for s in segments]
        type_dist: Dict[str, float] = {}
        for t in set(seg_types):
            type_dist[t] = seg_types.count(t) / len(seg_types)

        return {
            "mean_chaos": float(np.mean(chaos_indices)),
            "std_chaos": float(np.std(chaos_indices)),
            "mean_cv": float(np.mean(cvs)),
            "n_segments": len(segments),
            "segment_type_distribution": type_dist,
            "total_idle_fraction": float(np.mean([s.get("stop_fraction", 0) for s in segments])),
        }

    def compare_cycles(
        self, custom_cycles: Dict[str, pd.DataFrame], standard_analysis: Dict
    ) -> pd.DataFrame:
        """
        Compare custom cycles with standard cycles
        """
        comparisons = []

        for custom_name, custom_df in custom_cycles.items():
            # Skip if not a DataFrame
            if not isinstance(custom_df, pd.DataFrame):
                continue

            # Skip if DataFrame is empty or too small
            if custom_df.empty or len(custom_df) < 10:
                print(
                    f"Skipping comparison for {custom_name}: insufficient data ({len(custom_df)} samples)"
                )
                continue

            # Ensure speed_ms column exists
            if "speed_ms" not in custom_df.columns:
                if "speed_kmh" in custom_df.columns:
                    custom_df["speed_ms"] = custom_df["speed_kmh"] / 3.6
                else:
                    print(f"Skipping comparison for {custom_name}: no speed data")
                    continue

            custom_speeds = custom_df["speed_ms"].dropna().values

            if len(custom_speeds) < 10:
                print(f"Skipping comparison for {custom_name}: insufficient valid speed data")
                continue

            for std_name, std_data in standard_analysis.items():
                # Calculate similarity metrics
                comparison = {
                    "custom_cycle": custom_name,
                    "standard_cycle": std_name,
                    "custom_samples": len(custom_speeds),
                }

                # Get chaos scores if available
                if "chaos_score" in custom_df.columns and not custom_df["chaos_score"].isna().all():
                    custom_chaos = float(custom_df["chaos_score"].iloc[0])
                else:
                    custom_chaos = 0.0

                comparison["chaos_difference"] = abs(
                    custom_chaos - std_data["overall_metrics"].get("mean_chaos", 0.0)
                )

                # Speed distribution comparison (coarse, based on percentiles)
                std_speeds = []
                for seg in std_data.get("segments", []):
                    if isinstance(seg, dict) and "percentiles" in seg:
                        percentiles = seg["percentiles"]
                        if all(k in percentiles for k in ("p25", "p50", "p75")):
                            std_speeds.extend([percentiles["p25"], percentiles["p50"], percentiles["p75"]])

                if std_speeds and len(custom_speeds) > 0:
                    from scipy.stats import wasserstein_distance

                    custom_sample = np.percentile(custom_speeds, [25, 50, 75])
                    arr_std = np.array(std_speeds[:3]) if len(std_speeds) >= 3 else np.array(std_speeds)
                    if arr_std.size >= 3:
                        comparison["speed_similarity"] = 1.0 / (
                            1.0 + float(wasserstein_distance(custom_sample, arr_std))
                        )

                comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def generate_cycle_report(self, output_path: Path):
        """
        Generate comprehensive cycle analysis report
        """
        # Export custom cycles
        print("\n=== Exporting Custom Cycles ===")
        custom_cycles = self.export_custom_cycles()

        # Analyze standard cycles
        print("\n=== Analyzing Standard Cycles ===")
        standard_analysis = self.analyze_standard_cycles()

        # Compare cycles
        print("\n=== Comparing Cycles ===")
        try:
            comparison_df = self.compare_cycles(custom_cycles, standard_analysis)
        except Exception as e:
            print(f"Warning: Cycle comparison failed: {e}")
            traceback.print_exc()
            comparison_df = pd.DataFrame()

        # Write report
        with open(output_path, "w") as f:
            f.write("# Drive Cycle Export and Analysis Report\n\n")

            # Custom cycles section
            f.write("## Custom Cycles Exported\n\n")
            for cycle_name, df in custom_cycles.items():
                f.write(f"### {cycle_name}\n")
                f.write(f"- Duration: {len(df)} samples\n")
                if "speed_ms" in df.columns:
                    f.write(f"- Mean speed: {df['speed_ms'].mean()*3.6:.1f} km/h\n")
                    f.write(f"- Max speed: {df['speed_ms'].max()*3.6:.1f} km/h\n")
                if "chaos_score" in df.columns and not df["chaos_score"].isna().all():
                    f.write(f"- Chaos score: {df['chaos_score'].iloc[0]:.3f}\n")
                f.write("\n")

            # Standard cycles analysis
            f.write("## Standard Cycles Segmentation\n\n")
            for cycle_name, analysis in standard_analysis.items():
                f.write(f"### {cycle_name}\n")
                metrics = analysis["overall_metrics"]
                f.write(f"- Segments: {metrics.get('n_segments', 0)}\n")
                f.write(f"- Mean chaos: {metrics.get('mean_chaos', 0):.3f}\n")
                f.write(f"- Segment types:\n")
                for seg_type, pct in metrics.get("segment_type_distribution", {}).items():
                    f.write(f"  - {seg_type}: {pct:.1%}\n")
                f.write("\n")

            # Comparison
            if not comparison_df.empty:
                f.write("## Cycle Comparisons\n\n")
                f.write("Best matches (by similarity):\n\n")
                if "speed_similarity" in comparison_df.columns:
                    best_matches = comparison_df.nlargest(min(10, len(comparison_df)), "speed_similarity")
                    for _, row in best_matches.iterrows():
                        f.write(f"- {row['custom_cycle']} ↔ {row['standard_cycle']}: ")
                        f.write(f"similarity={row.get('speed_similarity', 0):.3f}\n")

        print(f"\nReport saved to {output_path}")


# -----------------------------------------------------------------------------
# CLI usage for direct execution (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to unified_analysis JSON",
    )
    parser.add_argument(
        "--cycles-dir",
        type=Path,
        default=Path(
            "${PROJECT_ROOT}/Data/standardized_cycles"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "${PROJECT_ROOT}/ML/outputs/exported_cycles"
        ),
    )

    args = parser.parse_args()

    exporter = CycleExporter(
        analysis_json_path=args.json, cycles_dir=args.cycles_dir, output_dir=args.output_dir
    )

    exporter.generate_cycle_report(args.output_dir / "cycle_analysis_report.md")
