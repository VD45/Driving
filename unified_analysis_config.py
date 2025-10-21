#!/usr/bin/env python3
"""
Unified configuration and data structures for comprehensive drive cycle analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

@dataclass
class UnifiedConfig:
    """Central configuration for all analysis modules"""
    
    # Data paths
    data_root: Path = Path("${PROJECT_ROOT}/Data")
    logged_dir: Path = None  # Set in __post_init__
    cycles_dir: Path = None
    output_dir: Path = Path("${PROJECT_ROOT}/ML/outputs")
    
    # Route identification (from comprehensive_route_analysis)
    route_similarity_threshold_m: float = 50.0
    min_route_occurrences: int = 3
    
    # Segmentation
    segment_at_intersections: bool = True
    segment_at_turns: bool = True
    turn_angle_threshold_deg: float = 20.0
    min_segment_length_m: float = 100.0
    min_segment_duration_s: float = 60.0
    
    # Markov analysis (from all_in_one_drive_analysis)
    markov_speed_bins_ms: List[float] = field(
        default_factory=lambda: [0, 2.5, 7, 13.9, 22.2, 30.6]  # 0, 9, 25, 50, 80, 110 km/h
    )
    second_order_markov: bool = True
    
    # DTW and similarity
    dtw_band_frac: float = 0.1
    
    # Corridor analysis
    n_corridor_points: int = 100
    
    # Turn dynamics
    turn_speed_bins: int = 5
    turn_angle_bins: List[float] = field(
        default_factory=lambda: [-180, -120, -60, -20, 20, 60, 120, 180]
    )
    
    # Speed analysis percentiles
    speed_percentiles: List[int] = field(
        default_factory=lambda: [5, 10, 25, 50, 75, 90, 95]
    )
    
    # Sampling and performance
    resample_hz: float = 10.0
    smooth_window_sec: float = 1.0
    max_files_per_route: int = -1  # -1 for all
    parallel_jobs: int = 4
    
    # Analysis features to enable
    enable_turn_analysis: bool = True
    enable_corridor_analysis: bool = True
    enable_weather_analysis: bool = True
    enable_cycle_comparison: bool = True
    enable_markov_chains: bool = True
    
    def __post_init__(self):
        if self.logged_dir is None:
            self.logged_dir = self.data_root / "logged"
        if self.cycles_dir is None:
            self.cycles_dir = self.data_root / "standardized_cycles"
        self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class RouteSegment:
    """Represents a segment of a route"""
    segment_id: str
    route_id: str
    trip_id: str
    start_idx: int
    end_idx: int
    
    # Basic metrics
    duration_s: float
    distance_m: float
    mean_speed_ms: float
    max_speed_ms: float
    
    # Context
    road_class: Optional[str] = None
    speed_limit_ms: Optional[float] = None
    has_traffic_light: bool = False
    is_intersection: bool = False
    is_turn: bool = False
    turn_angle_deg: float = 0.0
    
    # Temporal context
    day_type: str = "weekday"  # weekday, weekend, holiday
    time_category: str = "midday"  # night, morning, midday, afternoon, evening
    is_rush_hour: bool = False
    
    # Weather context
    weather_condition: Optional[str] = None
    is_wet: bool = False
    temperature_c: Optional[float] = None
    
    # Statistics
    speed_percentiles: Dict[int, float] = field(default_factory=dict)
    accel_percentiles: Dict[int, float] = field(default_factory=dict)
    idle_fraction: float = 0.0
    stop_count: int = 0

@dataclass
class TurnEvent:
    """Represents a turning maneuver"""
    turn_id: str
    route_id: str
    angle_deg: float
    angle_category: str  # slight, moderate, sharp, very_sharp
    direction: str  # left, right
    
    entry_speed_ms: float
    min_speed_ms: float
    exit_speed_ms: float
    speed_reduction_ms: float
    
    road_class: Optional[str] = None
    weather: Optional[str] = None
    day_type: str = "weekday"
    
    lat: float = 0.0
    lon: float = 0.0

@dataclass
class MarkovModel:
    """Markov chain model results"""
    order: int
    states: List[str]
    transition_matrix: np.ndarray
    stationary_dist: Optional[np.ndarray] = None
    
    # Context for this model
    context_key: str = "global"  # e.g., "route_0_weekday_dry"
    n_samples: int = 0
    
    # Derived metrics
    mean_dwell_times: Dict[str, float] = field(default_factory=dict)
    state_frequencies: Dict[str, float] = field(default_factory=dict)
