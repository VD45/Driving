#!/usr/bin/env python3
"""
Detailed segment-by-segment route analyzer with probability distributions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class DetailedRouteAnalyzer:
    def __init__(self, analysis_json_path: Path, parquet_dir: Path):
        """
        Load previous analysis results for deep dive
        """
        with open(analysis_json_path, 'r') as f:
            self.results = json.load(f)
        self.parquet_dir = parquet_dir
        
    def analyze_route_segments(self, route_id: str) -> Dict:
        """
        Deep analysis of all segments in a route
        """
        if route_id not in self.results.get('route_variability', {}):
            print(f"Route {route_id} not found in analysis")
            return {}
            
        route_data = self.results['route_variability'][route_id]
        segments = route_data.get('segments', {})
        
        detailed_analysis = {
            'route_id': route_id,
            'n_trips': route_data.get('n_trips', 0),
            'segment_details': {},
            'probability_distributions': {},
            'predictable_segments': [],
            'chaotic_segments': [],
            'transition_patterns': {}
        }
        
        # Analyze each segment
        for seg_id, seg_data in segments.items():
            seg_analysis = self._analyze_single_segment(seg_id, seg_data)
            detailed_analysis['segment_details'][seg_id] = seg_analysis
            
            # Classify by chaos
            if seg_analysis['chaos_index'] < 0.3:
                detailed_analysis['predictable_segments'].append(seg_analysis)
            elif seg_analysis['chaos_index'] > 0.7:
                detailed_analysis['chaotic_segments'].append(seg_analysis)
        
        # Analyze transitions between segments
        detailed_analysis['transition_patterns'] = self._analyze_transitions(segments)
        
        return detailed_analysis
    
    def _analyze_single_segment(self, seg_id: str, seg_data: Dict) -> Dict:
        """
        Detailed analysis of a single segment
        """
        # Extract segment characteristics from ID
        road_type = 'unknown'
        speed_limit = 0
        
        if 'road_' in seg_id:
            parts = seg_id.split('_')
            for i, part in enumerate(parts):
                if part == 'road' and i+1 < len(parts):
                    road_type = parts[i+1]
                elif part == 'limit' and i+1 < len(parts):
                    speed_limit = int(parts[i+1])
        
        # Get speed profiles
        speed_profiles = seg_data.get('speed_profiles', [])
        all_speeds = []
        for profile in speed_profiles:
            all_speeds.extend(profile)
        
        if not all_speeds:
            return {
                'segment_id': seg_id,
                'road_type': road_type,
                'speed_limit_kph': speed_limit,
                'insufficient_data': True
            }
        
        speeds = np.array(all_speeds)
        
        # Calculate probability distribution metrics
        hist, bins = np.histogram(speeds, bins=30, density=True)
        
        # Fit different distributions
        distributions = self._fit_distributions(speeds)
        
        # Calculate detailed metrics
        analysis = {
            'segment_id': seg_id,
            'road_type': road_type,
            'speed_limit_kph': speed_limit,
            'n_observations': seg_data.get('n_observations', 0),
            
            # Basic statistics
            'speed_mean_ms': float(np.mean(speeds)),
            'speed_std_ms': float(np.std(speeds)),
            'speed_median_ms': float(np.median(speeds)),
            'speed_mode_ms': float(stats.mode(speeds, keepdims=False)[0]) if len(speeds) > 0 else 0,
            
            # Distribution shape
            'skewness': float(skew(speeds)),
            'kurtosis': float(kurtosis(speeds)),
            'entropy': float(entropy(hist + 1e-10)),
            
            # Chaos metrics
            'chaos_index': seg_data.get('chaos_index', 0),
            'predictability_score': seg_data.get('predictability_score', 0),
            'distribution_type': seg_data.get('distribution_type', 'unknown'),
            
            # Percentiles
            'percentiles': {
                'p5': float(np.percentile(speeds, 5)),
                'p25': float(np.percentile(speeds, 25)),
                'p50': float(np.percentile(speeds, 50)),
                'p75': float(np.percentile(speeds, 75)),
                'p95': float(np.percentile(speeds, 95))
            },
            
            # Traffic behavior
            'stop_probability': seg_data.get('stop_probability', 0),
            'speed_limit_compliance': seg_data.get('speed_limit_compliance', 0),
            
            # Best fit distribution
            'best_fit_distribution': distributions['best_fit'],
            'distribution_parameters': distributions['parameters']
        }
        
        # Identify why this segment is predictable or chaotic
        analysis['predictability_factors'] = self._identify_predictability_factors(analysis)
        
        return analysis
    
    def _fit_distributions(self, data: np.ndarray) -> Dict:
        """
        Fit various probability distributions to the data
        """
        distributions = ['norm', 'gamma', 'expon', 'uniform', 'beta']
        best_fit = None
        best_ks = float('inf')
        best_params = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                ks_stat, _ = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best_fit = dist_name
                    best_params = params
            except:
                continue
        
        return {
            'best_fit': best_fit,
            'ks_statistic': best_ks,
            'parameters': best_params
        }
    
    def _identify_predictability_factors(self, analysis: Dict) -> List[str]:
        """
        Identify factors contributing to predictability or chaos
        """
        factors = []
        
        # Low chaos factors
        if analysis['chaos_index'] < 0.3:
            if analysis['road_type'] in ['motorway', 'trunk', 'primary']:
                factors.append('highway_steady_flow')
            if analysis['speed_std_ms'] < 2.0:
                factors.append('consistent_speed')
            if analysis['stop_probability'] < 0.1:
                factors.append('no_stops')
            if abs(analysis['skewness']) < 0.5:
                factors.append('symmetric_distribution')
        
        # High chaos factors  
        elif analysis['chaos_index'] > 0.7:
            if analysis['stop_probability'] > 0.3:
                factors.append('frequent_stops')
            if analysis['speed_std_ms'] > 5.0:
                factors.append('high_speed_variance')
            if analysis['road_type'] in ['residential', 'unclassified']:
                factors.append('complex_urban_road')
            if abs(analysis['skewness']) > 1.0:
                factors.append('asymmetric_distribution')
            if analysis['distribution_type'] == 'bimodal':
                factors.append('bimodal_traffic_pattern')
        
        return factors
    
    def _analyze_transitions(self, segments: Dict) -> Dict:
        """
        Analyze how chaos changes between segments
        """
        seg_list = list(segments.items())
        if len(seg_list) < 2:
            return {}
        
        transitions = {
            'chaos_increases': [],
            'chaos_decreases': [],
            'chaos_stable': []
        }
        
        for i in range(len(seg_list) - 1):
            curr_seg = seg_list[i][1]
            next_seg = seg_list[i+1][1]
            
            curr_chaos = curr_seg.get('chaos_index', 0)
            next_chaos = next_seg.get('chaos_index', 0)
            
            change = next_chaos - curr_chaos
            
            transition = {
                'from': seg_list[i][0],
                'to': seg_list[i+1][0],
                'chaos_change': float(change)
            }
            
            if change > 0.1:
                transitions['chaos_increases'].append(transition)
            elif change < -0.1:
                transitions['chaos_decreases'].append(transition)
            else:
                transitions['chaos_stable'].append(transition)
        
        return transitions
    
    def generate_segment_report(self, route_id: str, output_path: Path):
        """
        Generate detailed segment report
        """
        analysis = self.analyze_route_segments(route_id)
        
        with open(output_path, 'w') as f:
            f.write(f"# Detailed Segment Analysis - {route_id}\n\n")
            f.write(f"Total trips analyzed: {analysis['n_trips']}\n\n")
            
            # Predictable segments
            f.write("## Most Predictable Segments\n\n")
            if analysis['predictable_segments']:
                for seg in sorted(analysis['predictable_segments'], 
                                key=lambda x: x['chaos_index'])[:5]:
                    f.write(f"### {seg['segment_id']}\n")
                    f.write(f"- Road type: {seg['road_type']}\n")
                    f.write(f"- Chaos index: {seg['chaos_index']:.3f}\n")
                    f.write(f"- Mean speed: {seg['speed_mean_ms']*3.6:.1f} km/h\n")
                    f.write(f"- Speed std: {seg['speed_std_ms']*3.6:.1f} km/h\n")
                    f.write(f"- Distribution: {seg['best_fit_distribution']}\n")
                    f.write(f"- Factors: {', '.join(seg['predictability_factors'])}\n\n")
            
            # Chaotic segments
            f.write("## Most Chaotic Segments\n\n")
            if analysis['chaotic_segments']:
                for seg in sorted(analysis['chaotic_segments'], 
                                key=lambda x: x['chaos_index'], reverse=True)[:5]:
                    f.write(f"### {seg['segment_id']}\n")
                    f.write(f"- Road type: {seg['road_type']}\n")
                    f.write(f"- Chaos index: {seg['chaos_index']:.3f}\n")
                    f.write(f"- Stop probability: {seg['stop_probability']:.1%}\n")
                    f.write(f"- Speed range: {seg['percentiles']['p5']*3.6:.1f} - {seg['percentiles']['p95']*3.6:.1f} km/h\n")
                    f.write(f"- Distribution: {seg['distribution_type']}\n")
                    f.write(f"- Factors: {', '.join(seg['predictability_factors'])}\n\n")
            
            # Transition patterns
            f.write("## Segment Transitions\n\n")
            transitions = analysis['transition_patterns']
            if transitions.get('chaos_increases'):
                f.write("### Chaos Increases\n")
                for t in transitions['chaos_increases'][:3]:
                    f.write(f"- {t['from']} → {t['to']}: +{t['chaos_change']:.3f}\n")
                f.write("\n")
            
            if transitions.get('chaos_decreases'):
                f.write("### Chaos Decreases\n")
                for t in transitions['chaos_decreases'][:3]:
                    f.write(f"- {t['from']} → {t['to']}: {t['chaos_change']:.3f}\n")
                f.write("\n")
        
        print(f"Detailed report saved to {output_path}")


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path, required=True,
                       help='Path to unified_analysis JSON file')
    parser.add_argument('--route', type=str, default='route_00000',
                       help='Route ID to analyze')
    parser.add_argument('--output', type=Path, 
                       default=Path('detailed_segment_report.md'))
    
    args = parser.parse_args()
    
    analyzer = DetailedRouteAnalyzer(
        analysis_json_path=args.json,
        parquet_dir=Path('${PROJECT_ROOT}/Data/logged')
    )
    
    analyzer.generate_segment_report(args.route, args.output)
