#!/usr/bin/env python3
"""
Unified Wavelet Transform Analysis Module for Drive Cycles
Complete integrated module for wavelet analysis in the drive cycle pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pywt
from scipy import signal, stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class UnifiedWaveletAnalyzer:
    """Complete wavelet analysis system for drive cycles"""
    
    def __init__(self, sampling_rate_hz: float = 10.0):
        self.sr = sampling_rate_hz
        
        # Wavelet parameters
        self.cwt_wavelet = 'morl'  # Morlet for CWT
        self.dwt_wavelet = 'db4'    # Daubechies-4 for DWT
        self.max_decomp_level = 6   # For multi-resolution
        
        # Frequency bands of interest for driving (Hz)
        self.bands = {
            'micro': (1.0, 10.0),      # Rapid speed fluctuations
            'short': (0.1, 1.0),       # Short maneuvers (1-10s)
            'medium': (0.01, 0.1),     # Traffic patterns (10-100s)
            'long': (0.001, 0.01),     # Route segments (100-1000s)
            'macro': (0.0001, 0.001)   # Trip-level patterns
        }
        
        # Storage for results
        self.analyzed_cycles = {}
        self.comparison_matrix = None
        
    def continuous_wavelet_transform(self, speed_ms: np.ndarray) -> Dict:
        """Perform CWT for time-frequency analysis"""
        # Remove DC component
        speed_centered = speed_ms - np.mean(speed_ms)
        
        # Adaptive frequency range based on sampling rate
        if self.sr <= 1.0:
            freqs = np.logspace(-3, -0.5, 50)  # For 1Hz data
        else:
            freqs = np.logspace(-2, 0.7, 100)  # For 10Hz data
        
        # Calculate scales with minimum threshold
        scales = self.sr / (2 * freqs * np.pi)
        scales = np.maximum(scales, 1.0)  # Prevent too-small scales
        
        # Compute CWT
        cwt_matrix, freqs_cwt = pywt.cwt(speed_centered, scales, self.cwt_wavelet, 
                                         sampling_period=1/self.sr)
        
        # Power scalogram
        power = np.abs(cwt_matrix) ** 2
        
        # Find dominant frequency at each time
        dominant_freq_idx = np.argmax(power, axis=0)
        dominant_freqs = freqs_cwt[dominant_freq_idx]
        
        # Energy in each band
        band_energies = {}
        for band_name, (f_low, f_high) in self.bands.items():
            band_mask = (freqs_cwt >= f_low) & (freqs_cwt <= f_high)
            if np.any(band_mask):
                band_energies[band_name] = np.sum(power[band_mask, :])
            else:
                band_energies[band_name] = 0
        
        total_energy = np.sum(power)
        band_percentages = {k: v/total_energy*100 if total_energy > 0 else 0 
                          for k, v in band_energies.items()}
        
        return {
            'cwt_matrix': cwt_matrix,
            'power': power,
            'frequencies': freqs_cwt,
            'scales': scales,
            'dominant_freqs': dominant_freqs,
            'band_energies': band_energies,
            'band_percentages': band_percentages
        }
    
    def discrete_wavelet_decomposition(self, speed_ms: np.ndarray, level: Optional[int] = None) -> Dict:
        """Multi-resolution analysis using DWT"""
        if level is None:
            level = min(self.max_decomp_level, pywt.dwt_max_level(len(speed_ms), self.dwt_wavelet))
        
        # Perform decomposition
        coeffs = pywt.wavedec(speed_ms, self.dwt_wavelet, level=level)
        
        # Calculate energy at each level
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energies)
        energy_dist = [e/total_energy*100 if total_energy > 0 else 0 for e in energies]
        
        # Statistical features per level
        features = {}
        for i, (coeffs_level, energy_pct) in enumerate(zip(coeffs, energy_dist)):
            level_name = 'A' if i == 0 else f'D{i}'
            features[level_name] = {
                'energy_pct': energy_pct,
                'mean': float(np.mean(coeffs_level)),
                'std': float(np.std(coeffs_level)),
                'max': float(np.max(np.abs(coeffs_level))),
                'entropy': float(-np.sum(coeffs_level**2 * np.log(coeffs_level**2 + 1e-10)))
                          if np.any(coeffs_level != 0) else 0
            }
        
        return {
            'coefficients': coeffs,
            'level': level,
            'features': features,
            'energy_distribution': energy_dist
        }
    
    def extract_transient_events(self, speed_ms: np.ndarray, threshold_std: float = 2.0) -> Dict:
        """Detect transient events using wavelets"""
        if len(speed_ms) < 10:
            return {
                'accel_events': np.array([]),
                'decel_events': np.array([]),
                'n_accel': 0,
                'n_decel': 0,
                'events_per_minute': 0
            }
        
        # Use detail coefficients at high frequency
        level = min(3, pywt.dwt_max_level(len(speed_ms), self.dwt_wavelet))
        coeffs = pywt.wavedec(speed_ms, self.dwt_wavelet, level=level)
        d1 = coeffs[-1]  # Highest frequency details
        
        if len(d1) == 0:
            return {
                'accel_events': np.array([]),
                'decel_events': np.array([]),
                'n_accel': 0,
                'n_decel': 0,
                'events_per_minute': 0
            }
        
        # Threshold for significant events
        threshold = threshold_std * np.std(d1)
        
        # Find peaks
        peaks_pos, _ = signal.find_peaks(d1, height=threshold)
        peaks_neg, _ = signal.find_peaks(-d1, height=threshold)
        
        # Map back to original time indices
        scale_factor = len(speed_ms) / len(d1)
        accel_events = (peaks_pos * scale_factor).astype(int)
        decel_events = (peaks_neg * scale_factor).astype(int)
        
        # Ensure indices are within bounds
        accel_events = accel_events[accel_events < len(speed_ms)]
        decel_events = decel_events[decel_events < len(speed_ms)]
        
        duration_min = len(speed_ms) / (self.sr * 60)
        events_per_minute = (len(accel_events) + len(decel_events)) / duration_min if duration_min > 0 else 0
        
        return {
            'accel_events': accel_events,
            'decel_events': decel_events,
            'n_accel': len(accel_events),
            'n_decel': len(decel_events),
            'events_per_minute': float(events_per_minute)
        }
    
    def compute_wavelet_entropy(self, speed_ms: np.ndarray) -> float:
        """Compute wavelet entropy as complexity measure"""
        if len(speed_ms) < 10:
            return 0.0
        
        # DWT decomposition
        level = min(5, pywt.dwt_max_level(len(speed_ms), self.dwt_wavelet))
        coeffs = pywt.wavedec(speed_ms, self.dwt_wavelet, level=level)
        
        # Energy at each level
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            return 0
        
        # Probability distribution
        p = energies / total_energy
        
        # Shannon entropy
        entropy = -np.sum(p * np.log(p + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(len(energies))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def analyze_cycle(self, speed_kmh: np.ndarray, cycle_name: str = "Cycle", 
                     sampling_rate: Optional[float] = None) -> Dict:
        """Complete wavelet analysis of a drive cycle"""
        if sampling_rate is not None:
            self.sr = sampling_rate
        
        speed_ms = speed_kmh / 3.6
        
        # All analyses
        cwt_results = self.continuous_wavelet_transform(speed_ms)
        dwt_results = self.discrete_wavelet_decomposition(speed_ms)
        events = self.extract_transient_events(speed_ms)
        entropy = self.compute_wavelet_entropy(speed_ms)
        
        results = {
            'cycle_name': cycle_name,
            'cwt': cwt_results,
            'dwt': dwt_results,
            'events': events,
            'wavelet_entropy': entropy,
            'duration_s': len(speed_ms) / self.sr,
            'sampling_rate': self.sr
        }
        
        # Store for later use
        self.analyzed_cycles[cycle_name] = results
        
        return results
    
    def analyze_standard_cycles(self, cycles_dir: Path, families: Optional[List[str]] = None) -> Dict:
        """Analyze all standard cycles from directory structure"""
        if families is None:
            families = ['WLTP_Europe', 'EPA', 'Artemis', 'Asia', 'Special']
        
        results = {}
        
        for family in families:
            family_dir = cycles_dir / family
            if not family_dir.exists():
                continue
            
            # Process CSV and SCV files
            for file_path in list(family_dir.glob('*.csv')) + list(family_dir.glob('*.scv')):
                try:
                    # Read CSV
                    df = pd.read_csv(file_path, comment='#', encoding='utf-8-sig')
                    
                    # Find speed column
                    speed_data = None
                    for col in df.columns:
                        col_clean = col.strip()
                        if 'speed' in col_clean.lower() or 'vehicle' in col_clean.lower():
                            speed_data = pd.to_numeric(df[col], errors='coerce').dropna().values
                            break
                    
                    if speed_data is None or len(speed_data) < 50:
                        continue
                    
                    # Convert m/s to km/h if needed
                    if speed_data.max() < 10:
                        speed_data = speed_data * 3.6
                    
                    # Determine sampling rate
                    sampling_rate = 10.0 if len(speed_data) > 5000 else 1.0
                    
                    # Analyze
                    cycle_name = f"{family}_{file_path.stem}"
                    cycle_results = self.analyze_cycle(speed_data, cycle_name, sampling_rate)
                    results[cycle_name] = cycle_results
                    
                except Exception as e:
                    print(f"Error analyzing {file_path.name}: {e}")
        
        return results
    
    def analyze_logged_routes(self, df: pd.DataFrame, route_ids: List[str]) -> Dict:
        """Analyze logged route data with wavelets"""
        results = {}
        
        for route_id in route_ids:
            route_df = df[df['route_id'] == route_id]
            
            if 'speed_ms' not in route_df.columns:
                continue
            
            # Get trips for this route
            trips = route_df.groupby('source_file')
            
            # Use longest trip as representative
            longest_trip = None
            max_len = 0
            
            for trip_name, trip_data in trips:
                trip_sorted = trip_data.sort_values('timestamp')
                speeds = trip_sorted['speed_ms'].values * 3.6  # km/h
                
                if len(speeds) > max_len:
                    max_len = len(speeds)
                    longest_trip = speeds
            
            if longest_trip is not None and len(longest_trip) > 100:
                # Determine sampling rate
                sampling_rate = 10.0  # Typical for logged data
                
                # Analyze
                route_analysis = self.analyze_cycle(
                    longest_trip,
                    cycle_name=f"Route_{route_id}",
                    sampling_rate=sampling_rate
                )
                
                results[route_id] = {
                    'wavelet_entropy': route_analysis['wavelet_entropy'],
                    'events_per_minute': route_analysis['events']['events_per_minute'],
                    'micro_band_%': route_analysis['cwt']['band_percentages'].get('micro', 0),
                    'dominant_freq_mean': float(np.mean(route_analysis['cwt']['dominant_freqs']))
                }
        
        return results
    
    def compare_real_vs_standard(self, real_results: Dict, standard_results: Dict) -> Dict:
        """Compare real driving with standard cycles using wavelet metrics"""
        if not real_results or not standard_results:
            return {}
        
        # Calculate averages
        real_entropy = np.mean([r.get('wavelet_entropy', 0) for r in real_results.values()])
        std_entropy = np.mean([r.get('wavelet_entropy', 0) for r in standard_results.values()])
        
        real_events = np.mean([r.get('events_per_minute', 0) for r in real_results.values()])
        std_events = np.mean([r['events']['events_per_minute'] for r in standard_results.values()])
        
        return {
            'real_avg_entropy': float(real_entropy),
            'standard_avg_entropy': float(std_entropy),
            'entropy_ratio': float(real_entropy / std_entropy) if std_entropy > 0 else 0,
            'real_events_per_min': float(real_events),
            'standard_events_per_min': float(std_events),
            'events_ratio': float(real_events / std_events) if std_events > 0 else 0
        }
    
    def create_publication_wavelet_figure(self, save_path: Optional[Path] = None, 
                                         max_cycles: int = 8) -> Path:
        """Create publication-quality wavelet analysis figure"""
        if not self.analyzed_cycles:
            print("No cycles analyzed yet")
            return None
        
        # Select representative cycles
        selected_cycles = list(self.analyzed_cycles.keys())[:max_cycles]
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Entropy comparison
        ax1 = fig.add_subplot(gs[0, :])
        entropies = [self.analyzed_cycles[c]['wavelet_entropy'] for c in selected_cycles]
        colors = ['#0173B2' if 'WLTP' in c or 'WLTC' in c else 
                 '#DE8F05' if 'EPA' in c or 'FTP' in c else
                 '#029E73' if 'Artemis' in c else
                 '#CC78BC' if 'Asia' in c or 'JC08' in c else '#8B4513'
                 for c in selected_cycles]
        
        bars = ax1.bar(range(len(selected_cycles)), entropies, color=colors)
        ax1.set_xticks(range(len(selected_cycles)))
        ax1.set_xticklabels([c.replace('_', ' ')[:20] for c in selected_cycles], rotation=45, ha='right')
        ax1.set_ylabel('Wavelet Entropy')
        ax1.set_title('Wavelet Entropy - Cycle Complexity Measure', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, entropies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Events per minute
        ax2 = fig.add_subplot(gs[1, 0])
        events = [self.analyzed_cycles[c]['events']['events_per_minute'] for c in selected_cycles]
        ax2.barh(range(len(selected_cycles)), events, color=colors)
        ax2.set_yticks(range(len(selected_cycles)))
        ax2.set_yticklabels([c.split('_')[-1][:15] for c in selected_cycles])
        ax2.set_xlabel('Events per Minute')
        ax2.set_title('Transient Events Detection')
        ax2.grid(True, alpha=0.3)
        
        # 3. Frequency band distribution (average)
        ax3 = fig.add_subplot(gs[1, 1])
        band_avgs = {}
        for band in self.bands.keys():
            band_avgs[band] = np.mean([
                self.analyzed_cycles[c]['cwt']['band_percentages'].get(band, 0)
                for c in selected_cycles
            ])
        
        wedges, texts, autotexts = ax3.pie(band_avgs.values(), labels=band_avgs.keys(),
                                            autopct='%1.1f%%', startangle=90)
        ax3.set_title('Average Energy Distribution')
        
        # 4. DWT energy distribution heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        dwt_matrix = []
        for c in selected_cycles[:6]:  # Limit for visibility
            if 'dwt' in self.analyzed_cycles[c]:
                energy_dist = self.analyzed_cycles[c]['dwt']['energy_distribution']
                dwt_matrix.append(energy_dist[:6])  # First 6 levels
        
        if dwt_matrix:
            im = ax4.imshow(dwt_matrix, aspect='auto', cmap='YlOrRd')
            ax4.set_yticks(range(len(dwt_matrix)))
            ax4.set_yticklabels([c.split('_')[-1][:10] for c in selected_cycles[:len(dwt_matrix)]])
            ax4.set_xticks(range(6))
            ax4.set_xticklabels(['A', 'D1', 'D2', 'D3', 'D4', 'D5'])
            ax4.set_xlabel('Decomposition Level')
            ax4.set_title('DWT Energy Distribution (%)')
            plt.colorbar(im, ax=ax4)
        
        # 5. Example scalogram
        if selected_cycles:
            ax5 = fig.add_subplot(gs[2, :])
            example_cycle = selected_cycles[0]
            cwt_data = self.analyzed_cycles[example_cycle]['cwt']
            
            # Plot scalogram
            power_log = np.log10(cwt_data['power'][:, ::10] + 1e-10)  # Downsample for speed
            
            extent = [0, power_log.shape[1], 
                     cwt_data['frequencies'][-1], cwt_data['frequencies'][0]]
            
            im = ax5.imshow(power_log, extent=extent, aspect='auto', 
                          cmap='jet', interpolation='bilinear')
            ax5.set_ylabel('Frequency (Hz)')
            ax5.set_xlabel('Time (samples)')
            ax5.set_yscale('log')
            ax5.set_title(f'Example CWT Scalogram: {example_cycle}')
            plt.colorbar(im, ax=ax5, label='Log10(Power)')
        
        plt.suptitle('Wavelet Transform Analysis of Standard Drive Cycles', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        plt.show()
        return None
    
    def generate_wavelet_report(self, results: Dict) -> str:
        """Generate text report of wavelet analysis findings"""
        report = []
        report.append("# WAVELET TRANSFORM ANALYSIS REPORT\n")
        
        if 'comparison' in results:
            comp = results['comparison']
            report.append("## Key Findings\n")
            report.append(f"- **Entropy Ratio**: {comp.get('entropy_ratio', 0):.2f}x")
            report.append(f"  (Real: {comp.get('real_avg_entropy', 0):.3f}, ")
            report.append(f"Standard: {comp.get('standard_avg_entropy', 0):.3f})\n")
            report.append(f"- **Events Ratio**: {comp.get('events_ratio', 0):.2f}x")
            report.append(f"  (Real: {comp.get('real_events_per_min', 0):.1f}/min, ")
            report.append(f"Standard: {comp.get('standard_events_per_min', 0):.1f}/min)\n")
        
        if 'standard_cycles' in results:
            report.append("\n## Standard Cycles Summary\n")
            std = results['standard_cycles']
            report.append(f"- Cycles Analyzed: {std.get('n_analyzed', 0)}\n")
            report.append(f"- Mean Entropy: {std.get('mean_entropy', 0):.3f}\n")
            report.append(f"- Mean Events/min: {std.get('mean_events_per_min', 0):.1f}\n")
        
        return ''.join(report)


def integrate_wavelet_analysis(
    results: Dict,
    df: pd.DataFrame,
    config,
    timestamp: str,
    show_banner: bool = True,
) -> Dict:
    """
    Main integration function for the unified analysis pipeline
    Enhanced version that stores per-cycle data for visualization.
    """
    if show_banner:
        print("\n" + "="*50)
        print("PHASE 6.5: WAVELET TRANSFORM ANALYSIS")
        print("="*50)
    
    # Initialize analyzer
    analyzer = UnifiedWaveletAnalyzer(sampling_rate_hz=10.0)
    
    # Create output directory
    wavelet_dir = config.output_dir / 'wavelet_analysis'
    wavelet_dir.mkdir(parents=True, exist_ok=True)
    
    wavelet_results = {}
    
    # 1. Analyze standard cycles
    print("\n6.5.1 Analyzing standard cycles with wavelets...")
    std_results = analyzer.analyze_standard_cycles(config.cycles_dir)
    
    if std_results:
        # Store both summary and per-cycle data
        wavelet_results['standard_cycles'] = {
            'n_analyzed': len(std_results),
            'cycles': list(std_results.keys()),
            'mean_entropy': np.mean([r['wavelet_entropy'] for r in std_results.values()]),
            'mean_events_per_min': np.mean([r['events']['events_per_minute'] for r in std_results.values()])
        }
        
        # Store per-cycle data separately for visualization
        wavelet_results['per_cycle'] = std_results
        
        print(f"  Analyzed {len(std_results)} standard cycles")
    
    # --- NEW: Write minimal per-cycle JSON for CSV exporter to consume ---
    # This allows publication_visualizations.export_metrics_csv to merge
    # WaveletEntropy, band percentages, and events/min into the metrics CSV.
    if std_results:
        per_cycle_min = {}
        for key, val in std_results.items():
            per_cycle_min[key] = {
                "wavelet_entropy": val.get("wavelet_entropy", 0),
                "events": {
                    "events_per_minute": val.get("events", {}).get("events_per_minute", 0)
                },
                "cwt": {
                    "band_percentages": val.get("cwt", {}).get("band_percentages", {})
                }
            }
        import json
        wavelet_json_path = wavelet_dir / "wavelet_data.json"
        with open(wavelet_json_path, "w") as f:
            json.dump(per_cycle_min, f, indent=2)
    
    # 2. Analyze logged routes
    print("\n6.5.2 Analyzing logged routes with wavelets...")
    if 'route_variability' in results:
        route_ids = list(results['route_variability'].keys())[:3]  # Top 3 routes
        route_results = analyzer.analyze_logged_routes(df, route_ids)
        wavelet_results['logged_routes'] = route_results
        print(f"  Analyzed {len(route_results)} routes")
    
    # 3. Compare real vs standard
    print("\n6.5.3 Comparing real driving with standards...")
    if 'logged_routes' in wavelet_results and std_results:
        comparison = analyzer.compare_real_vs_standard(
            wavelet_results['logged_routes'],
            std_results
        )
        wavelet_results['comparison'] = comparison
        
        print(f"  Wavelet Entropy - Real: {comparison.get('real_avg_entropy', 0):.3f}, ")
        print(f"                   Standard: {comparison.get('standard_avg_entropy', 0):.3f}")
        print(f"  Ratio: {comparison.get('entropy_ratio', 0):.2f}x more complex")
    
    # 4. Generate publication figure
    print("\n6.5.4 Generating wavelet visualization...")
    fig_path = wavelet_dir / f"wavelet_analysis_{timestamp}.png"
    analyzer.create_publication_wavelet_figure(fig_path)
    
    # 5. Generate recommendations
    recommendations = []
    if 'comparison' in wavelet_results:
        ratio = wavelet_results['comparison'].get('entropy_ratio', 0)
        if ratio > 1.5:
            recommendations.append(
                f"Real driving shows {ratio:.1f}x higher wavelet entropy than standards. "
                "Consider multi-resolution test cycles that better capture transient behaviors."
            )
    
    wavelet_results['recommendations'] = recommendations
    
    # Add to main results
    results['wavelet_analysis'] = wavelet_results
    
    # Generate report
    report_path = wavelet_dir / f"wavelet_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(analyzer.generate_wavelet_report(wavelet_results))
    
    print(f"\n6.5.5 Wavelet analysis complete. Report: {report_path}")
    
    return results


def plot_band_energy_summary(per_cycle_data: Dict, save_path: Optional[Path] = None) -> Path:
    """
    Create publication-quality plot of wavelet band energy distribution across cycles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    
    if not per_cycle_data:
        print("No per-cycle wavelet data available")
        return None
    
    # Extract band energy data
    cycles = []
    band_data = {'micro': [], 'short': [], 'medium': [], 'long': [], 'macro': []}
    entropy_data = []
    events_data = []
    
    for cycle_name, data in per_cycle_data.items():
        if isinstance(data, dict):
            cycles.append(cycle_name.replace('_', ' ')[:25])  # Clean names
            
            # Get band percentages
            if 'cwt' in data and 'band_percentages' in data['cwt']:
                for band in band_data.keys():
                    band_data[band].append(data['cwt']['band_percentages'].get(band, 0))
            else:
                for band in band_data.keys():
                    band_data[band].append(0)
            
            # Get entropy
            entropy_data.append(data.get('wavelet_entropy', 0))
            
            # Get events
            if 'events' in data:
                events_data.append(data['events'].get('events_per_minute', 0))
            else:
                events_data.append(0)
    
    if not cycles:
        print("No valid cycle data found")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25,
                          top=0.94, bottom=0.06, left=0.08, right=0.95)
    
    # Color scheme for bands
    band_colors = {
        'micro': '#FF6B6B',    # Red - highest frequency
        'short': '#4ECDC4',    # Teal
        'medium': '#45B7D1',   # Blue
        'long': '#96CEB4',     # Green
        'macro': '#FFEAA7'     # Yellow - lowest frequency
    }
    
    # 1. Stacked bar chart of band energy distribution
    ax1 = fig.add_subplot(gs[0, :])
    
    # Sort by total high-frequency content (micro + short)
    high_freq_content = [band_data['micro'][i] + band_data['short'][i] 
                        for i in range(len(cycles))]
    sorted_indices = np.argsort(high_freq_content)[::-1][:15]  # Top 15
    
    sorted_cycles = [cycles[i] for i in sorted_indices]
    bottom = np.zeros(len(sorted_indices))
    
    for band in ['macro', 'long', 'medium', 'short', 'micro']:
        values = [band_data[band][i] for i in sorted_indices]
        ax1.bar(range(len(sorted_cycles)), values, bottom=bottom,
                label=f'{band.capitalize()} ({band_freq_description(band)})',
                color=band_colors[band], alpha=0.85)
        bottom += np.array(values)
    
    ax1.set_xticks(range(len(sorted_cycles)))
    ax1.set_xticklabels(sorted_cycles, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Energy Distribution (%)', fontweight='bold')
    ax1.set_title('Wavelet Band Energy Distribution (Top 15 by High-Frequency Content)', 
                  fontweight='bold', fontsize=11)
    ax1.legend(loc='upper right', ncol=5, fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # 2. Entropy vs High-Frequency Content
    ax2 = fig.add_subplot(gs[1, 0])
    
    high_freq_all = [band_data['micro'][i] + band_data['short'][i] 
                    for i in range(len(cycles))]
    
    # Color by cycle category (simplified)
    colors = []
    for cycle in cycles:
        if 'WLTC' in cycle or 'WLTP' in cycle or 'NEDC' in cycle:
            colors.append('#0173B2')  # Blue for Europe
        elif 'FTP' in cycle or 'US06' in cycle or 'EPA' in cycle:
            colors.append('#DE8F05')  # Orange for EPA/US
        elif 'Artemis' in cycle or 'CADC' in cycle:
            colors.append('#029E73')  # Green for Artemis
        elif 'JC08' in cycle or 'CLTC' in cycle:
            colors.append('#CC78BC')  # Purple for Asia
        else:
            colors.append('#8B4513')  # Brown for others
    
    scatter = ax2.scatter(high_freq_all, entropy_data, c=colors, 
                         alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('High-Frequency Content (micro + short) %', fontweight='bold')
    ax2.set_ylabel('Wavelet Entropy', fontweight='bold')
    ax2.set_title('Complexity vs Transient Content', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(high_freq_all, entropy_data, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(high_freq_all), max(high_freq_all), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=1, 
            label=f'Trend (R²={np.corrcoef(high_freq_all, entropy_data)[0,1]**2:.3f})')
    ax2.legend(loc='best', fontsize=8)
    
    # 3. Events per minute comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Sort by events per minute
    sorted_events_idx = np.argsort(events_data)[::-1][:15]
    sorted_events_cycles = [cycles[i] for i in sorted_events_idx]
    sorted_events_values = [events_data[i] for i in sorted_events_idx]
    sorted_events_colors = [colors[i] for i in sorted_events_idx]
    
    bars = ax3.barh(range(len(sorted_events_cycles)), sorted_events_values,
                    color=sorted_events_colors, alpha=0.8)
    ax3.set_yticks(range(len(sorted_events_cycles)))
    ax3.set_yticklabels(sorted_events_cycles, fontsize=8)
    ax3.set_xlabel('Transient Events per Minute', fontweight='bold')
    ax3.set_title('Dynamic Event Frequency', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, sorted_events_values):
        ax3.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', ha='left', va='center', fontsize=7)
    
    # 4. Band energy heatmap
    ax4 = fig.add_subplot(gs[2, :])
    
    # Create matrix for heatmap
    heatmap_data = []
    heatmap_labels = []
    
    for i in sorted_indices[:20]:  # Top 20 cycles
        row = [band_data[band][i] for band in ['micro', 'short', 'medium', 'long', 'macro']]
        heatmap_data.append(row)
        heatmap_labels.append(cycles[i])
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=50)
    
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(['Micro\n(1-10 Hz)', 'Short\n(0.1-1 Hz)', 
                         'Medium\n(0.01-0.1 Hz)', 'Long\n(0.001-0.01 Hz)', 
                         'Macro\n(<0.001 Hz)'], fontsize=9)
    ax4.set_yticks(range(len(heatmap_labels)))
    ax4.set_yticklabels(heatmap_labels, fontsize=7)
    ax4.set_title('Frequency Band Energy Distribution Heatmap (%)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, aspect=40)
    cbar.set_label('Energy Percentage', fontweight='bold')
    
    # Add text annotations
    for i in range(len(heatmap_data)):
        for j in range(5):
            text = ax4.text(j, i, f'{heatmap_data[i][j]:.1f}',
                          ha="center", va="center", color="white" if heatmap_data[i][j] > 25 else "black",
                          fontsize=6)
    
    # Overall title
    fig.suptitle('Wavelet Transform Analysis - Frequency Band Distribution',
                fontsize=14, fontweight='bold')
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def band_freq_description(band: str) -> str:
    """Return human-readable frequency description for a band."""
    descriptions = {
        'micro': '1-10 Hz',
        'short': '0.1-1 Hz',
        'medium': '0.01-0.1 Hz',
        'long': '0.001-0.01 Hz',
        'macro': '<0.001 Hz'
    }
    return descriptions.get(band, '')


def update_metrics_with_wavelet(metrics: Dict, wavelet_results: Dict) -> Dict:
    """
    Update cycle metrics dictionary with wavelet analysis results.
    """
    for cycle_name, wave_data in wavelet_results.items():
        # Find matching metric entry
        matching_key = None
        for key in metrics.keys():
            if cycle_name in key or key in cycle_name:
                matching_key = key
                break
        
        if matching_key and isinstance(wave_data, dict):
            # Add wavelet entropy
            if 'wavelet_entropy' in wave_data:
                metrics[matching_key]['wavelet_entropy'] = wave_data['wavelet_entropy']
            
            # Add band percentages
            if 'cwt' in wave_data and 'band_percentages' in wave_data['cwt']:
                metrics[matching_key]['band_percentages'] = wave_data['cwt']['band_percentages']
            
            # Add events per minute
            if 'events' in wave_data and 'events_per_minute' in wave_data['events']:
                metrics[matching_key]['events_per_minute'] = wave_data['events']['events_per_minute']
    
    return metrics


def add_wavelet_to_publication_viz(visualizer, analyzer: UnifiedWaveletAnalyzer):
    """
    Add wavelet visualizations to publication figures
    Call this from publication_visualizations.py
    """
    # Analyze standard cycles if not done
    if not analyzer.analyzed_cycles:
        cycles_dir = Path('${PROJECT_ROOT}/Data/standardized_cycles')
        analyzer.analyze_standard_cycles(cycles_dir)
    
    # Create wavelet figure
    output_path = visualizer.output_dir / 'wavelet_standard_cycles.png'
    analyzer.create_publication_wavelet_figure(output_path)
    
    return output_path
