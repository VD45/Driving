#!/usr/bin/env python3
"""
Core analysis functions combining best features from both implementations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from scipy import stats, signal
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

from unified_analysis_config import UnifiedConfig, RouteSegment, TurnEvent, MarkovModel

warnings.filterwarnings('ignore')

# ========== Data Loading ==========

def load_enriched_parquets(config: UnifiedConfig) -> pd.DataFrame:
    """Load all enriched parquet files"""
    files = sorted(config.logged_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {config.logged_dir}")
    
    dfs = []
    for i, f in enumerate(files):
        if config.max_files_per_route > 0 and i >= config.max_files_per_route:
            break
        df = pd.read_parquet(f)
        df['source_file'] = f.stem
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} files, {len(combined):,} total samples")
    return combined

def load_standard_cycles(config: UnifiedConfig) -> Dict[str, pd.DataFrame]:
    """
    Load standard drive cycles from configured directory.
    Supports CSV, SCV, and Parquet files.
    """
    cycles = {}

    # 1. Try master parquet file first
    master = config.cycles_dir / "cycles_master.parquet"
    if master.exists():
        df = pd.read_parquet(master)
        if '_source_file' in df.columns:
            for source in df['_source_file'].unique():
                name = Path(source).stem
                cycles[name] = df[df['_source_file'] == source].copy()

    # 2. Look for individual CSV/SCV/Parquet files in subfolders
    categories = ['WLTP_Europe', 'EPA', 'Artemis', 'Asia', 'Special']
    for category in categories:
        category_path = config.cycles_dir / category
        if not category_path.exists():
            continue

        files = list(category_path.glob("*.csv")) + \
                list(category_path.glob("*.scv")) + \
                list(category_path.glob("*.parquet"))

        for file_path in files:
            try:
                # Load data depending on extension
                if file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path, comment="#", encoding="utf-8-sig")

                # Try to detect a speed column
                speed_data = None
                for col in df.columns:
                    col_clean = col.strip().lower()
                    if any(pattern in col_clean for pattern in ["speed", "vehicle"]):
                        speed_data = pd.to_numeric(df[col], errors="coerce").dropna().values
                        break

                if speed_data is None or len(speed_data) < 10:
                    continue

                # Convert km/h to m/s if needed
                if speed_data.max() > 50:  # assume km/h
                    speed_ms = speed_data / 3.6
                    speed_kmh = speed_data
                else:  # assume already m/s
                    speed_ms = speed_data
                    speed_kmh = speed_data * 3.6

                # Build standardized DataFrame (assuming 0.1s step if not given)
                cycle_df = pd.DataFrame({
                    "speed_ms": speed_ms,
                    "speed_kmh": speed_kmh,
                    "time_s": np.arange(len(speed_ms)) * 0.1
                })

                # Store with meaningful key
                cycle_key = f"{category}_{file_path.stem}"
                cycles[cycle_key] = cycle_df

            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")

    # 3. Include any additional parquet files not in categories
    for f in config.cycles_dir.rglob("*.parquet"):
        if f.name == "cycles_master.parquet":
            continue
        name = f.stem
        if name not in cycles:
            try:
                cycles[name] = pd.read_parquet(f)
            except Exception as e:
                print(f"  Error loading {f.name}: {e}")

    print(f"Loaded {len(cycles)} standard cycles")
    return cycles


# ========== Route Identification ==========

def identify_routes(df: pd.DataFrame, config: UnifiedConfig) -> pd.DataFrame:
    """Cluster trips into routes using GPS start/end points"""
    
    # Group by source file (each file is one trip)
    trips = []
    for file_id, trip_df in df.groupby('source_file'):
        if len(trip_df) < 10:
            continue
        
        # Get key waypoints
        n = len(trip_df)
        indices = [0, n//4, n//2, 3*n//4, n-1]
        
        waypoints = []
        for idx in indices:
            if 0 <= idx < n:
                lat = trip_df.iloc[idx]['Latitude']
                lon = trip_df.iloc[idx]['Longitude']
                waypoints.extend([lat, lon])
        
        trips.append((file_id, waypoints))
    
    if not trips:
        df['route_id'] = 'route_00000'
        return df
    
    # Build distance matrix
    X = np.array([t[1] for t in trips])
    
    def gps_distance_m(a, b):
        """Haversine distance between waypoint sets"""
        from math import radians, sin, cos, sqrt, asin
        dist_sum = 0
        n_points = len(a) // 2
        
        for i in range(n_points):
            lat1, lon1 = radians(a[i*2]), radians(a[i*2+1])
            lat2, lon2 = radians(b[i*2]), radians(b[i*2+1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            h = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            d = 2 * 6371000 * asin(sqrt(min(1, h)))
            dist_sum += d
        
        return dist_sum / n_points
    
    from sklearn.metrics.pairwise import pairwise_distances
    distances = pairwise_distances(X, metric=gps_distance_m)
    
    # Cluster with DBSCAN
    clustering = DBSCAN(
        eps=config.route_similarity_threshold_m,
        min_samples=config.min_route_occurrences,
        metric='precomputed'
    ).fit(distances)
    
    # Map clusters back to dataframe
    file_to_route = {}
    for (file_id, _), label in zip(trips, clustering.labels_):
        if label >= 0:
            file_to_route[file_id] = f"route_{label:05d}"
        else:
            file_to_route[file_id] = f"unique_{file_id}"
    
    df['route_id'] = df['source_file'].map(file_to_route)
    df['route_id'] = df['route_id'].fillna('route_unknown')
    
    # Summary
    route_counts = df.groupby('route_id')['source_file'].nunique()
    print(f"Identified {len(route_counts)} routes:")
    for rid in route_counts.index[:10]:
        if rid.startswith('route_'):
            print(f"  {rid}: {route_counts[rid]} trips")
    
    return df

# ========== Segmentation ==========

def segment_routes(df: pd.DataFrame, config: UnifiedConfig) -> List[RouteSegment]:
    """Segment routes at intersections, turns, and speed discontinuities"""
    
    segments = []
    
    for (route_id, trip_id), trip_df in df.groupby(['route_id', 'source_file']):
        if len(trip_df) < 10:
            continue
        
        trip_df = trip_df.sort_values('timestamp').reset_index(drop=True)
        
        # Find segment boundaries
        boundaries = [0]
        
        # Speed jumps
        if 'speed_ms' in trip_df.columns:
            speed_diff = trip_df['speed_ms'].diff().abs()
            jumps = trip_df[speed_diff > 3.0].index.tolist()
            boundaries.extend(jumps)
        
        # Intersections
        if config.segment_at_intersections and 'map_near_intersection' in trip_df.columns:
            intersections = trip_df[trip_df['map_near_intersection']].index.tolist()
            boundaries.extend(intersections[::5])  # Sample to avoid over-segmentation
        
        # Traffic lights
        if 'map_near_traffic_light' in trip_df.columns:
            lights = trip_df[trip_df['map_near_traffic_light']].index.tolist()
            boundaries.extend(lights[::5])
        
        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        boundaries.append(len(trip_df) - 1)
        
        # Create segments
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            if end_idx - start_idx < 10:  # Too short
                continue
            
            seg_df = trip_df.iloc[start_idx:end_idx]
            
            # Calculate metrics
            if 'dt_s' in seg_df.columns:
                duration = seg_df['dt_s'].sum()
            else:
                duration = len(seg_df) / config.resample_hz
            
            if duration < config.min_segment_duration_s:
                continue
            
            # Distance calculation
            if 'speed_ms' in seg_df.columns and 'dt_s' in seg_df.columns:
                distance = (seg_df['speed_ms'] * seg_df['dt_s']).sum()
            else:
                # Haversine distance
                lats = seg_df['Latitude'].values
                lons = seg_df['Longitude'].values
                distance = calculate_trajectory_distance(lats, lons)
            
            segment = RouteSegment(
                segment_id=f"{route_id}_{trip_id}_seg{i:04d}",
                route_id=route_id,
                trip_id=trip_id,
                start_idx=start_idx,
                end_idx=end_idx,
                duration_s=duration,
                distance_m=distance,
                mean_speed_ms=seg_df['speed_ms'].mean() if 'speed_ms' in seg_df.columns else 0,
                max_speed_ms=seg_df['speed_ms'].max() if 'speed_ms' in seg_df.columns else 0,
            )
            
            # Add context
            if 'map_road_class' in seg_df.columns:
                segment.road_class = seg_df['map_road_class'].mode().iloc[0] if len(seg_df['map_road_class'].mode()) > 0 else None
            
            if 'day_type' in seg_df.columns:
                segment.day_type = seg_df['day_type'].mode().iloc[0] if len(seg_df['day_type'].mode()) > 0 else 'weekday'
            
            if 'time_category' in seg_df.columns:
                segment.time_category = seg_df['time_category'].mode().iloc[0] if len(seg_df['time_category'].mode()) > 0 else 'midday'
            
            if 'is_rush_hour' in seg_df.columns:
                segment.is_rush_hour = seg_df['is_rush_hour'].any()
            
            segments.append(segment)
    
    print(f"Created {len(segments)} segments")
    return segments

def calculate_trajectory_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    """Calculate total distance along GPS trajectory"""
    from math import radians, sin, cos, sqrt, asin
    
    total_dist = 0
    for i in range(len(lats) - 1):
        lat1, lon1 = radians(lats[i]), radians(lons[i])
        lat2, lon2 = radians(lats[i+1]), radians(lons[i+1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        h = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        d = 2 * 6371000 * asin(sqrt(min(1, h)))
        total_dist += d
    
    return total_dist

# ========== Markov Chains ==========

def build_markov_models(df: pd.DataFrame, config: UnifiedConfig, 
                       context_splits: List[str] = ['day_type']) -> Dict[str, MarkovModel]:
    """Build Markov chain models with optional context splits"""
    
    models = {}
    
    # Check and prepare speed column
    if 'speed_ms' not in df.columns:
        if 'Vehicle speed' in df.columns:
            df['speed_ms'] = pd.to_numeric(df['Vehicle speed'], errors='coerce') / 3.6
        else:
            print("Warning: Missing speed for Markov analysis")
            return models
    
    # Calculate acceleration if missing
    if 'accel_ms2' not in df.columns:
        # Drop trips with fewer than two samples to avoid gradient errors on micro-trips
        df = df.groupby('source_file').filter(lambda group: len(group) > 1)

        df['accel_ms2'] = df.groupby('source_file')['speed_ms'].transform(lambda x: np.gradient(x))
    
    speed_bins = config.markov_speed_bins_ms
    speeds = df['speed_ms'].values
    accels = df['accel_ms2'].values
    
    # Create state labels
    speed_idx = np.digitize(speeds, speed_bins)
    accel_sign = np.where(accels > 0.1, 'A', np.where(accels < -0.1, 'D', 'C'))  # Accel/Decel/Cruise
    
    states = [f"S{si}_{ac}" for si, ac in zip(speed_idx, accel_sign)]
    unique_states = sorted(set(states))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    
    # Global model
    global_model = compute_markov_matrix(states, unique_states, state_to_idx, order=1)
    global_model.context_key = "global"
    global_model.n_samples = len(states)
    models["global"] = global_model
    
    # Context-specific models
    for context_col in context_splits:
        if context_col not in df.columns:
            continue
        
        for context_val in df[context_col].unique():
            mask = df[context_col] == context_val
            if mask.sum() < 1000:  # Need enough data
                continue
            
            context_states = [s for s, m in zip(states, mask) if m]
            
            model = compute_markov_matrix(context_states, unique_states, state_to_idx, order=1)
            model.context_key = f"{context_col}_{context_val}"
            model.n_samples = len(context_states)
            models[model.context_key] = model
            
            # Second-order if requested
            if config.second_order_markov and len(context_states) > 5000:
                model2 = compute_markov_matrix(context_states, unique_states, state_to_idx, order=2)
                model2.context_key = f"{context_col}_{context_val}_order2"
                model2.n_samples = len(context_states)
                models[model2.context_key] = model2
    
    print(f"Built {len(models)} Markov models")
    return models

def compute_markov_matrix(states: List[str], unique_states: List[str], 
                          state_to_idx: Dict[str, int], order: int = 1) -> MarkovModel:
    """Compute transition matrix for Markov chain"""
    
    n_states = len(unique_states)
    
    if order == 1:
        trans_counts = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            curr_idx = state_to_idx.get(states[i], -1)
            next_idx = state_to_idx.get(states[i+1], -1)
            if curr_idx >= 0 and next_idx >= 0:
                trans_counts[curr_idx, next_idx] += 1
        
        # Normalize
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_matrix = trans_counts / row_sums
        
    else:  # order == 2
        # Create compound states for pairs
        pairs = [f"{states[i]}|{states[i+1]}" for i in range(len(states)-1)]
        unique_pairs = sorted(set(pairs))
        pair_to_idx = {p: i for i, p in enumerate(unique_pairs)}
        
        trans_counts = np.zeros((len(unique_pairs), n_states))
        
        for i in range(len(pairs) - 1):
            curr_idx = pair_to_idx.get(pairs[i], -1)
            next_idx = state_to_idx.get(states[i+2], -1)
            if curr_idx >= 0 and next_idx >= 0:
                trans_counts[curr_idx, next_idx] += 1
        
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_matrix = trans_counts / row_sums
    
    model = MarkovModel(
        order=order,
        states=unique_states,
        transition_matrix=trans_matrix
    )
    
    # Calculate stationary distribution for order-1 models
    if order == 1:
        try:
            eigenvals, eigenvects = np.linalg.eig(trans_matrix.T)
            stationary_idx = np.argmax(np.abs(eigenvals))
            stationary = np.abs(eigenvects[:, stationary_idx])
            model.stationary_dist = stationary / stationary.sum()
        except:
            pass
    
    return model
