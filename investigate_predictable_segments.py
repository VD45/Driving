#!/usr/bin/env python3
"""
Investigate why certain segments have low chaos (high predictability)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from analysis_shared import CHAOS_PREDICTABLE

class PredictabilityInvestigator:
    def __init__(self, analysis_json_path: Path, parquet_dir: Path):
        """Initialize with analysis results"""
        with open(analysis_json_path, 'r') as f:
            self.results = json.load(f)
        self.parquet_dir = parquet_dir
        
    def find_predictable_segments(self, chaos_threshold: float = CHAOS_PREDICTABLE) -> List[Dict]:
        """
        Find all segments with chaos index below threshold
        """
        predictable_segments = []
        
        for route_id, route_data in self.results.get('route_variability', {}).items():
            if not isinstance(route_data, dict) or 'segments' not in route_data:
                continue
                
            for seg_id, seg_data in route_data['segments'].items():
                if not isinstance(seg_data, dict):
                    continue
                    
                chaos = seg_data.get('chaos_index', 1.0)
                if chaos < chaos_threshold:
                    predictable_segments.append({
                        'route_id': route_id,
                        'segment_id': seg_id,
                        'chaos_index': chaos,
                        'predictability_score': seg_data.get('predictability_score', 0),
                        'n_observations': seg_data.get('n_observations', 0),
                        'segment_data': seg_data
                    })
        
        # Sort by chaos index (most predictable first)
        predictable_segments.sort(key=lambda x: x['chaos_index'])
        
        return predictable_segments
    
    def analyze_predictability_patterns(self, predictable_segments: List[Dict]) -> Dict:
        """
        Identify common patterns in predictable segments
        """
        patterns = {
            'road_types': {},
            'speed_ranges': [],
            'time_patterns': {},
            'weather_patterns': {},
            'common_characteristics': [],
            'optimization_opportunities': []
        }
        
        for seg in predictable_segments:
            seg_id = seg['segment_id']
            seg_data = seg['segment_data']
            
            # Extract road type from segment ID
            if 'road_' in seg_id:
                try:
                    road_type = seg_id.split('road_')[1].split('_')[0]
                    patterns['road_types'][road_type] = patterns['road_types'].get(road_type, 0) + 1
                except:
                    pass
            
            # Speed characteristics
            if 'speed_mean' in seg_data:
                patterns['speed_ranges'].append({
                    'mean': seg_data['speed_mean'],
                    'std': seg_data.get('speed_std', 0),
                    'cv': seg_data.get('speed_cv', 0)
                })
        
        # Analyze common characteristics
        if patterns['speed_ranges']:
            speeds = [s['mean'] for s in patterns['speed_ranges']]
            patterns['common_characteristics'] = self._identify_common_traits(
                patterns['road_types'],
                speeds,
                predictable_segments
            )
        
        # Identify optimization opportunities
        patterns['optimization_opportunities'] = self._find_optimization_opportunities(
            predictable_segments
        )
        
        return patterns
    
    def _identify_common_traits(self, road_types: Dict, speeds: List, segments: List[Dict]) -> List[str]:
        """
        Identify traits common to predictable segments
        """
        traits = []
        
        # Road type dominance
        if road_types:
            dominant_road = max(road_types, key=road_types.get)
            if road_types[dominant_road] > len(segments) * 0.3:
                traits.append(f"mostly_{dominant_road}_roads")
        
        # Speed consistency
        if speeds:
            mean_speed = np.mean(speeds)
            if 15 < mean_speed < 25:  # 54-90 km/h
                traits.append("optimal_flow_speed")
            elif mean_speed > 25:  # >90 km/h
                traits.append("highway_speeds")
        
        # Low stop probability
        stop_probs = []
        for s in segments:
            if 'segment_data' in s and isinstance(s['segment_data'], dict):
                stop_probs.append(s['segment_data'].get('stop_probability', 0))
        
        if stop_probs and np.mean(stop_probs) < 0.1:
            traits.append("minimal_stops")
        
        # Speed limit compliance
        compliances = []
        for s in segments:
            if 'segment_data' in s and isinstance(s['segment_data'], dict):
                compliances.append(s['segment_data'].get('speed_limit_compliance', 0))
        
        if compliances and np.mean(compliances) > 0.8:
            traits.append("high_speed_limit_compliance")
        
        return traits
    
    def _find_optimization_opportunities(self, segments: List[Dict]) -> List[Dict]:
        """
        Identify opportunities to replicate predictable conditions
        """
        opportunities = []
        
        # Group by route to find consistent patterns
        route_segments = {}
        for seg in segments:
            route = seg['route_id']
            if route not in route_segments:
                route_segments[route] = []
            route_segments[route].append(seg)
        
        # Find routes with multiple predictable segments
        for route, segs in route_segments.items():
            if len(segs) >= 2:
                opportunities.append({
                    'type': 'consistent_route',
                    'route': route,
                    'n_predictable_segments': len(segs),
                    'avg_chaos': np.mean([s['chaos_index'] for s in segs]),
                    'recommendation': 'Replicate conditions from this route'
                })
        
        # Identify time windows with predictability
        opportunities.append({
            'type': 'optimal_timing',
            'recommendation': 'Analyze time-of-day patterns for these segments'
        })
        
        # Speed optimization zones
        optimal_speeds = []
        stricter_threshold = 0.5 * CHAOS_PREDICTABLE
        for s in segments:
            if s['chaos_index'] < stricter_threshold and 'segment_data' in s:
                if isinstance(s['segment_data'], dict) and 'speed_mean' in s['segment_data']:
                    optimal_speeds.append(s['segment_data']['speed_mean'])
        
        if optimal_speeds:
            opportunities.append({
                'type': 'speed_optimization',
                'optimal_speed_ms': float(np.mean(optimal_speeds)),
                'optimal_speed_kmh': float(np.mean(optimal_speeds) * 3.6),
                'recommendation': f'Target {np.mean(optimal_speeds)*3.6:.1f} km/h for predictability'
            })
        
        return opportunities
    
    def investigate_specific_segment(self, route_id: str, segment_id: str) -> Dict:
        """
        Deep dive into why a specific segment is predictable
        """
        investigation = {
            'route_id': route_id,
            'segment_id': segment_id,
            'factors': [],
            'recommendations': []
        }
        
        # Find the segment
        route_data = self.results.get('route_variability', {}).get(route_id, {})
        if not isinstance(route_data, dict):
            return investigation
            
        seg_data = route_data.get('segments', {}).get(segment_id, {})
        
        if not seg_data or not isinstance(seg_data, dict):
            return investigation
        
        chaos = seg_data.get('chaos_index', 1.0)
        
        # Analyze contributing factors
        if chaos < CHAOS_PREDICTABLE:
            # Speed consistency
            cv = seg_data.get('speed_cv', 1.0)
            if cv < 0.2:
                investigation['factors'].append({
                    'factor': 'speed_consistency',
                    'value': cv,
                    'impact': 'high',
                    'description': 'Very consistent speeds reduce chaos'
                })
            
            # Low stop probability
            stop_prob = seg_data.get('stop_probability', 1.0)
            if stop_prob < 0.1:
                investigation['factors'].append({
                    'factor': 'continuous_flow',
                    'value': stop_prob,
                    'impact': 'high',
                    'description': 'Minimal stops maintain flow'
                })
            
            # Distribution type
            dist_type = seg_data.get('distribution_type', 'unknown')
            if dist_type == 'normal':
                investigation['factors'].append({
                    'factor': 'normal_distribution',
                    'value': dist_type,
                    'impact': 'medium',
                    'description': 'Predictable bell-curve speed pattern'
                })
            
            # Speed limit compliance
            compliance = seg_data.get('speed_limit_compliance', 0)
            if compliance > 0.8:
                investigation['factors'].append({
                    'factor': 'traffic_regulation',
                    'value': compliance,
                    'impact': 'medium',
                    'description': 'Traffic follows speed limits'
                })
        
        # Generate recommendations
        investigation['recommendations'] = self._generate_recommendations(investigation['factors'])
        
        return investigation
    
    def _generate_recommendations(self, factors: List[Dict]) -> List[str]:
        """
        Generate recommendations based on predictability factors
        """
        recommendations = []
        
        for factor in factors:
            if factor['factor'] == 'speed_consistency':
                recommendations.append(
                    "Maintain steady throttle and avoid unnecessary speed changes"
                )
            elif factor['factor'] == 'continuous_flow':
                recommendations.append(
                    "Choose routes with fewer intersections and traffic lights"
                )
            elif factor['factor'] == 'normal_distribution':
                recommendations.append(
                    "Follow traffic flow rather than aggressive driving"
                )
            elif factor['factor'] == 'traffic_regulation':
                recommendations.append(
                    "Adhere to speed limits for predictable conditions"
                )
        
        return recommendations
    
    def generate_predictability_report(self, output_path: Path):
        """
        Generate comprehensive predictability investigation report
        """
        # Find predictable segments
        predictable = self.find_predictable_segments(chaos_threshold=CHAOS_PREDICTABLE)
        
        # Analyze patterns
        patterns = self.analyze_predictability_patterns(predictable)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write("# Predictability Investigation Report\n\n")
            f.write(
                f"Found {len(predictable)} highly predictable segments "
                f"(chaos < {CHAOS_PREDICTABLE:.2f})\n\n"
            )
            
            # Most predictable segments
            f.write("## Top 10 Most Predictable Segments\n\n")
            f.write("| Route | Segment | Chaos | Predictability | Observations |\n")
            f.write("|-------|---------|-------|---------------|-------------|\n")
            
            for seg in predictable[:10]:
                f.write(f"| {seg['route_id']} | {seg['segment_id'][:30]} | ")
                f.write(f"{seg['chaos_index']:.3f} | ")
                f.write(f"{seg['predictability_score']:.3f} | ")
                f.write(f"{seg['n_observations']} |\n")
            
            f.write("\n## Common Patterns in Predictable Segments\n\n")
            
            # Road types
            if patterns['road_types']:
                f.write("### Road Type Distribution\n")
                for road, count in sorted(patterns['road_types'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    f.write(f"- {road}: {count} segments\n")
                f.write("\n")
            
            # Common characteristics
            if patterns['common_characteristics']:
                f.write("### Common Characteristics\n")
                for trait in patterns['common_characteristics']:
                    f.write(f"- {trait.replace('_', ' ').title()}\n")
                f.write("\n")
            
            # Optimization opportunities
            f.write("## Optimization Opportunities\n\n")
            for opp in patterns['optimization_opportunities']:
                f.write(f"### {opp['type'].replace('_', ' ').title()}\n")
                if 'route' in opp:
                    f.write(f"- Route: {opp['route']}\n")
                if 'optimal_speed_kmh' in opp:
                    f.write(f"- Optimal speed: {opp['optimal_speed_kmh']:.1f} km/h\n")
                f.write(f"- Recommendation: {opp['recommendation']}\n\n")
            
            # Detailed investigation of most predictable
            if predictable:
                best = predictable[0]
                investigation = self.investigate_specific_segment(
                    best['route_id'], best['segment_id']
                )
                
                f.write("## Deep Dive: Most Predictable Segment\n\n")
                f.write(f"**{best['segment_id']}** (Chaos: {best['chaos_index']:.3f})\n\n")
                
                f.write("### Contributing Factors\n")
                for factor in investigation['factors']:
                    f.write(f"- **{factor['factor']}** ({factor['impact']} impact): ")
                    f.write(f"{factor['description']}\n")
                
                f.write("\n### Recommendations\n")
                for rec in investigation['recommendations']:
                    f.write(f"- {rec}\n")
            
            # Summary statistics
            f.write("\n## Summary Statistics\n\n")
            if predictable:
                chaos_values = [s['chaos_index'] for s in predictable]
                f.write(f"- Mean chaos (predictable segments): {np.mean(chaos_values):.3f}\n")
                f.write(f"- Std chaos: {np.std(chaos_values):.3f}\n")
                f.write(f"- Min chaos: {np.min(chaos_values):.3f}\n")
                
                # Compare with overall
                all_chaos = []
                for route_data in self.results.get('route_variability', {}).values():
                    if isinstance(route_data, dict) and 'overall_metrics' in route_data:
                        overall = route_data.get('overall_metrics', {})
                        if isinstance(overall, dict):
                            all_chaos.append(overall.get('route_chaos_score', 0))
                
                if all_chaos:
                    f.write(f"\n- Overall mean chaos (all routes): {np.mean(all_chaos):.3f}\n")
                    f.write(f"- Predictability improvement potential: ")
                    f.write(f"{(np.mean(all_chaos) - np.mean(chaos_values)):.3f}\n")
        
        print(f"Predictability report saved to {output_path}")
    
    def visualize_predictability(
        self,
        output_path: Path,
        predictable_segments: Optional[List[Dict]] = None,
        chaos_threshold: float = CHAOS_PREDICTABLE,
    ):
        """Create visualizations of predictability patterns."""
        predictable = predictable_segments
        if predictable is None:
            predictable = self.find_predictable_segments(chaos_threshold=chaos_threshold)

        if not predictable:
            print(f"No predictable segments found for chaos < {chaos_threshold}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Chaos distribution
        ax = axes[0, 0]
        chaos_values = [s['chaos_index'] for s in predictable]
        ax.hist(chaos_values, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Chaos Index')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Predictable Segments')
        ax.axvline(chaos_threshold, color='red', linestyle='--', label=f'Threshold ({chaos_threshold:.2f})')
        ax.legend()
        
        # 2. Speed characteristics
        ax = axes[0, 1]
        speeds = []
        cvs = []
        for seg in predictable[:50]:  # Limit for visibility
            seg_data = seg.get('segment_data', {})
            if isinstance(seg_data, dict) and 'speed_mean' in seg_data:
                speeds.append(seg_data['speed_mean'] * 3.6)
                cvs.append(seg_data.get('speed_cv', 0))
        
        if speeds and cvs:
            scatter = ax.scatter(speeds, cvs, c=chaos_values[:len(speeds)], 
                               cmap='RdYlGn_r', alpha=0.6)
            ax.set_xlabel('Mean Speed (km/h)')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Speed vs Variability')
            plt.colorbar(scatter, ax=ax, label='Chaos Index')
        
        # 3. Road type pie chart
        ax = axes[1, 0]
        patterns = self.analyze_predictability_patterns(predictable)
        if patterns['road_types']:
            ax.pie(patterns['road_types'].values(), 
                  labels=patterns['road_types'].keys(),
                  autopct='%1.1f%%')
            ax.set_title('Road Types in Predictable Segments')
        
        # 4. Predictability by route
        ax = axes[1, 1]
        route_chaos = {}
        for seg in predictable:
            route = seg['route_id']
            if route not in route_chaos:
                route_chaos[route] = []
            route_chaos[route].append(seg['chaos_index'])
        
        if route_chaos:
            routes = list(route_chaos.keys())[:10]  # Top 10 routes
            mean_chaos = [np.mean(route_chaos[r]) for r in routes]
            ax.bar(range(len(routes)), mean_chaos, color='green', alpha=0.7)
            ax.set_xticks(range(len(routes)))
            ax.set_xticklabels([r.replace('route_', 'R') for r in routes], rotation=45)
            ax.set_ylabel('Mean Chaos Index')
            ax.set_title('Predictability by Route')
            ax.axhline(chaos_threshold, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Predictability Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_path}")


# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path, required=True,
                       help='Path to unified_analysis JSON')
    parser.add_argument('--output', type=Path,
                       default=Path('predictability_report.md'))
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    
    args = parser.parse_args()
    
    investigator = PredictabilityInvestigator(
        analysis_json_path=args.json,
        parquet_dir=Path('${PROJECT_ROOT}/Data/logged')
    )
    
    investigator.generate_predictability_report(args.output)
    
    if args.visualize:
        viz_path = args.output.with_suffix('.png')
        investigator.visualize_predictability(viz_path)
