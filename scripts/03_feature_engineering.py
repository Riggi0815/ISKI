"""
Feature Engineering for Driver Identification
Extracts statistical and behavioral features from telemetry data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal, stats
from scipy.fft import fft
import sys

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_dataframe,
    save_dataframe,
    save_json,
    get_results_path,
    create_summary_report,
    print_dataframe_info
)


class FeatureEngineer:
    """Extract features from telemetry data for ML"""
    
    def __init__(self, telemetry_df: pd.DataFrame, segment_size=500):
        """
        Args:
            telemetry_df: Raw telemetry DataFrame
            segment_size: Number of samples per segment (default 500 = 10 seconds at 50Hz)
        """
        self.df = telemetry_df
        self.features_df = None
        self.segment_size = segment_size
        
        # Define telemetry channels to analyze
        self.channels = [
            'percent_throttle', 'p_brakeF', 'a_steering',
            'g_long', 'g_lat', 'v_car', 'n_engine',
            'p_tyreFL', 'p_tyreFR', 'p_tyreRL', 'p_tyreRR',
            't_tyreFL', 't_tyreFR', 't_tyreRL', 't_tyreRR'
        ]
        
    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract all feature types and combine into feature DataFrame
        Segments data into time windows for multiple samples per driver
        
        Returns:
            DataFrame with features per segment per driver
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTION (Segmented)")
        print(f"Segment size: {self.segment_size} samples (~{self.segment_size/50:.1f} seconds at 50Hz)")
        print("="*60)
        
        feature_groups = []
        
        # Group by driver
        grouped = self.df.groupby('driver_id')
        
        for driver_id, driver_data in grouped:
            print(f"\nProcessing driver: {driver_id} ({len(driver_data)} samples)")
            
            # Split into segments
            n_segments = len(driver_data) // self.segment_size
            print(f"  Creating {n_segments} segments...")
            
            segment_count = 0
            for segment_idx in range(n_segments):
                start_idx = segment_idx * self.segment_size
                end_idx = start_idx + self.segment_size
                segment_data = driver_data.iloc[start_idx:end_idx]
                
                if len(segment_data) < self.segment_size * 0.8:  # Skip incomplete segments
                    continue
                
                features = {
                    'driver_id': driver_id,
                    'segment_idx': segment_idx
                }
                
                # Extract features from segment
                features.update(self._extract_basic_stats(segment_data))
                features.update(self._extract_driving_style(segment_data))
                features.update(self._extract_braking_patterns(segment_data))
                features.update(self._extract_cornering(segment_data))
                features.update(self._extract_tire_features(segment_data))
                features.update(self._extract_fft_features(segment_data))
                
                feature_groups.append(features)
                segment_count += 1
            
            print(f"  ✓ Created {segment_count} feature sets")
        
        self.features_df = pd.DataFrame(feature_groups)
        
        # Replace NaN values with 0 (occurs in skew/kurtosis when data has little variation)
        self.features_df = self.features_df.fillna(0)
        
        print(f"\n{'='*60}")
        print(f"FEATURE EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total feature sets: {len(self.features_df)}")
        print(f"Features per set: {len(self.features_df.columns) - 2}")  # Exclude driver_id and segment_idx
        print(f"Samples per driver:")
        for driver_id, count in self.features_df['driver_id'].value_counts().items():
            print(f"  {driver_id}: {count} segments")
        
        return self.features_df
    
    def _extract_basic_stats(self, driver_data: pd.DataFrame) -> dict:
        """Extract basic statistical features"""
        features = {}
        
        for channel in self.channels:
            if channel not in driver_data.columns:
                continue
            
            data = driver_data[channel].dropna()
            
            if len(data) == 0:
                continue
            
            prefix = f"{channel}_"
            features[f"{prefix}mean"] = data.mean()
            features[f"{prefix}std"] = data.std()
            features[f"{prefix}min"] = data.min()
            features[f"{prefix}max"] = data.max()
            features[f"{prefix}median"] = data.median()
            features[f"{prefix}q25"] = data.quantile(0.25)
            features[f"{prefix}q75"] = data.quantile(0.75)
            features[f"{prefix}skew"] = stats.skew(data)
            features[f"{prefix}kurtosis"] = stats.kurtosis(data)
        
        return features
    
    def _extract_driving_style(self, driver_data: pd.DataFrame) -> dict:
        """Extract driving style characteristics"""
        features = {}
        
        # Throttle usage
        throttle = driver_data['percent_throttle'].dropna()
        features['throttle_full_pct'] = (throttle > 90).sum() / len(throttle) * 100
        features['throttle_partial_pct'] = ((throttle > 10) & (throttle < 90)).sum() / len(throttle) * 100
        
        # Brake usage
        brake = driver_data['p_brakeF'].dropna()
        features['brake_heavy_pct'] = (brake > 50).sum() / len(brake) * 100
        features['brake_light_pct'] = ((brake > 0) & (brake < 50)).sum() / len(brake) * 100
        
        # Steering smoothness (changes per second)
        steering = driver_data['a_steering'].dropna()
        steering_changes = np.abs(np.diff(steering))
        features['steering_smoothness'] = steering_changes.mean()
        features['steering_aggression'] = np.percentile(steering_changes, 95)
        
        # Speed consistency
        speed = driver_data['v_car'].dropna()
        features['speed_variance'] = speed.var()
        features['speed_cv'] = speed.std() / (speed.mean() + 1e-6)  # Coefficient of variation
        
        # G-Force usage (driving aggression)
        if 'g_long' in driver_data.columns and 'g_lat' in driver_data.columns:
            g_long = driver_data['g_long'].dropna()
            g_lat = driver_data['g_lat'].dropna()
            
            features['g_long_extreme_pct'] = (np.abs(g_long) > 1.0).sum() / len(g_long) * 100
            features['g_lat_extreme_pct'] = (np.abs(g_lat) > 1.0).sum() / len(g_lat) * 100
        
        return features
    
    def _extract_braking_patterns(self, driver_data: pd.DataFrame) -> dict:
        """Analyze braking behavior"""
        features = {}
        
        brake = driver_data['p_brakeF'].dropna().values
        
        # Find braking zones (brake pressure > 10%)
        braking = brake > 10
        
        # Detect braking events
        braking_events = []
        in_braking = False
        brake_start = 0
        
        for i, is_braking in enumerate(braking):
            if is_braking and not in_braking:
                brake_start = i
                in_braking = True
            elif not is_braking and in_braking:
                braking_events.append((brake_start, i))
                in_braking = False
        
        if len(braking_events) > 0:
            # Braking event statistics
            brake_durations = [end - start for start, end in braking_events]
            features['brake_event_count'] = len(braking_events)
            features['brake_avg_duration'] = np.mean(brake_durations)
            features['brake_max_duration'] = np.max(brake_durations)
            
            # Braking intensity
            brake_pressures = [brake[start:end].mean() for start, end in braking_events]
            features['brake_avg_pressure'] = np.mean(brake_pressures)
            features['brake_max_pressure'] = np.max(brake_pressures)
            
            # Trail braking (braking while steering)
            if 'a_steering' in driver_data.columns:
                steering = driver_data['a_steering'].dropna().values
                trail_brake_count = 0
                for start, end in braking_events:
                    if end < len(steering):
                        steering_in_brake = np.abs(steering[start:end])
                        if steering_in_brake.mean() > 5:  # Steering while braking
                            trail_brake_count += 1
                
                features['trail_brake_pct'] = trail_brake_count / len(braking_events) * 100
        else:
            features['brake_event_count'] = 0
            features['brake_avg_duration'] = 0
            features['brake_max_duration'] = 0
            features['brake_avg_pressure'] = 0
            features['brake_max_pressure'] = 0
            features['trail_brake_pct'] = 0
        
        return features
    
    def _extract_cornering(self, driver_data: pd.DataFrame) -> dict:
        """Analyze cornering behavior"""
        features = {}
        
        if 'a_steering' not in driver_data.columns or 'v_car' not in driver_data.columns:
            return features
        
        steering = driver_data['a_steering'].dropna().values
        speed = driver_data['v_car'].dropna().values[:len(steering)]
        
        # Detect corners (significant steering)
        cornering = np.abs(steering) > 10
        
        corner_events = []
        in_corner = False
        corner_start = 0
        
        for i, is_cornering in enumerate(cornering):
            if is_cornering and not in_corner:
                corner_start = i
                in_corner = True
            elif not is_cornering and in_corner:
                corner_events.append((corner_start, i))
                in_corner = False
        
        if len(corner_events) > 0:
            # Corner statistics
            corner_speeds = [speed[start:end].mean() for start, end in corner_events if end < len(speed)]
            corner_angles = [np.abs(steering[start:end]).mean() for start, end in corner_events]
            
            features['corner_count'] = len(corner_events)
            features['corner_avg_speed'] = np.mean(corner_speeds) if corner_speeds else 0
            features['corner_avg_angle'] = np.mean(corner_angles)
            features['corner_max_angle'] = np.max(corner_angles)
            
            # Understeer/Oversteer indicator (simplified)
            if 'g_lat' in driver_data.columns:
                g_lat = driver_data['g_lat'].dropna().values
                lateral_g_in_corners = [np.abs(g_lat[start:end]).mean() 
                                       for start, end in corner_events if end < len(g_lat)]
                features['corner_avg_g_lat'] = np.mean(lateral_g_in_corners) if lateral_g_in_corners else 0
        else:
            features['corner_count'] = 0
            features['corner_avg_speed'] = 0
            features['corner_avg_angle'] = 0
            features['corner_max_angle'] = 0
            features['corner_avg_g_lat'] = 0
        
        return features
    
    def _extract_tire_features(self, driver_data: pd.DataFrame) -> dict:
        """Analyze tire usage and temperatures"""
        features = {}
        
        # Tire pressure differences (indicates weight transfer and driving style)
        tire_pressures = ['p_tyreFL', 'p_tyreFR', 'p_tyreRL', 'p_tyreRR']
        
        if all(col in driver_data.columns for col in tire_pressures):
            fl = driver_data['p_tyreFL'].dropna()
            fr = driver_data['p_tyreFR'].dropna()
            rl = driver_data['p_tyreRL'].dropna()
            rr = driver_data['p_tyreRR'].dropna()
            
            # Front-rear balance
            features['tire_pressure_front_avg'] = (fl.mean() + fr.mean()) / 2
            features['tire_pressure_rear_avg'] = (rl.mean() + rr.mean()) / 2
            features['tire_pressure_f_r_diff'] = features['tire_pressure_front_avg'] - features['tire_pressure_rear_avg']
            
            # Left-right balance
            features['tire_pressure_left_avg'] = (fl.mean() + rl.mean()) / 2
            features['tire_pressure_right_avg'] = (fr.mean() + rr.mean()) / 2
            features['tire_pressure_l_r_diff'] = features['tire_pressure_left_avg'] - features['tire_pressure_right_avg']
        
        # Tire temperature management
        tire_temps = ['t_tyreFL', 't_tyreFR', 't_tyreRL', 't_tyreRR']
        
        if all(col in driver_data.columns for col in tire_temps):
            temps = [driver_data[col].dropna().mean() for col in tire_temps]
            features['tire_temp_avg'] = np.mean(temps)
            features['tire_temp_std'] = np.std(temps)
            features['tire_temp_max'] = np.max(temps)
        
        return features
    
    def _extract_fft_features(self, driver_data: pd.DataFrame) -> dict:
        """Extract frequency domain features using FFT"""
        features = {}
        
        # Analyze steering frequency (how often driver adjusts steering)
        if 'a_steering' in driver_data.columns:
            steering = driver_data['a_steering'].dropna().values
            
            if len(steering) > 100:
                # Compute FFT
                fft_vals = np.abs(fft(steering))
                freqs = np.fft.fftfreq(len(steering), d=1/50)  # 50Hz sampling
                
                # Positive frequencies only
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_vals[:len(fft_vals)//2]
                
                # Dominant frequency
                if len(positive_fft) > 0:
                    dominant_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
                    features['steering_dominant_freq'] = positive_freqs[dominant_idx]
                    features['steering_freq_power'] = positive_fft[dominant_idx]
                    
                    # Low frequency power (smooth steering)
                    low_freq_mask = (positive_freqs < 1.0) & (positive_freqs > 0)
                    features['steering_low_freq_power'] = np.sum(positive_fft[low_freq_mask])
                    
                    # High frequency power (jerky steering)
                    high_freq_mask = positive_freqs > 2.0
                    features['steering_high_freq_power'] = np.sum(positive_fft[high_freq_mask])
        
        # Throttle frequency patterns
        if 'percent_throttle' in driver_data.columns:
            throttle = driver_data['percent_throttle'].dropna().values
            
            if len(throttle) > 100:
                fft_vals = np.abs(fft(throttle))
                freqs = np.fft.fftfreq(len(throttle), d=1/50)
                
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_vals[:len(fft_vals)//2]
                
                if len(positive_fft) > 0:
                    # Throttle modulation frequency
                    dominant_idx = np.argmax(positive_fft[1:]) + 1
                    features['throttle_dominant_freq'] = positive_freqs[dominant_idx]
                    features['throttle_freq_power'] = positive_fft[dominant_idx]
        
        return features


def main():
    """Main execution"""
    print("="*70)
    print("FEATURE ENGINEERING - Driver Identification")
    print("="*70)
    
    # Load parsed telemetry
    print("\nLoading telemetry data...")
    telemetry_df = load_dataframe("telemetry_all", directory="processed_data")
    
    if telemetry_df.empty:
        print("ERROR: No telemetry data found. Run 01_parse_htf.py first.")
        return
    
    print(f"✓ Loaded {len(telemetry_df):,} samples from {telemetry_df['driver_id'].nunique()} drivers")
    
    # Extract features (segmented)
    engineer = FeatureEngineer(telemetry_df, segment_size=500)  # 500 samples = 10 seconds
    features_df = engineer.extract_all_features()
    
    # Save features
    print(f"\n{'='*60}")
    print("SAVING FEATURES")
    print(f"{'='*60}")
    
    save_dataframe(features_df, "driver_features", directory="features")
    
    # Print info
    print_dataframe_info(features_df, "Driver Features")
    
    # Create summary
    summary = {
        "Total Drivers": features_df['driver_id'].nunique(),
        "Total Feature Sets": len(features_df),
        "Total Features": len(features_df.columns) - 2,  # Exclude driver_id and segment_idx
        "Segment Size": "500 samples (10 seconds at 50Hz)",
        "Samples per Driver": features_df['driver_id'].value_counts().to_dict(),
        "Feature Categories": {
            "Basic Statistics": sum(1 for col in features_df.columns if any(stat in col for stat in ['_mean', '_std', '_min', '_max', '_median'])),
            "Driving Style": sum(1 for col in features_df.columns if any(word in col for word in ['throttle_', 'brake_', 'steering_', 'speed_'])),
            "Braking Patterns": sum(1 for col in features_df.columns if 'brake_' in col and 'event' in col or 'duration' in col or 'pressure' in col),
            "Cornering": sum(1 for col in features_df.columns if 'corner' in col),
            "Tire Management": sum(1 for col in features_df.columns if 'tire' in col),
            "Frequency Analysis": sum(1 for col in features_df.columns if 'freq' in col)
        },
        "Drivers": features_df['driver_id'].tolist(),
        "All Features": features_df.columns.tolist()
    }
    
    results_path = get_results_path()
    report_path = results_path / "03_feature_engineering_summary.txt"
    
    create_summary_report(summary, save_path=report_path)
    
    print(f"\n{'='*60}")
    print("✓ FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Features saved: features/driver_features.csv")
    print(f"Summary report: {report_path}")
    print()


if __name__ == "__main__":
    main()
