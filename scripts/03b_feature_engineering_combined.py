"""
Step 3b: Feature Engineering on Combined HTF+LD Data
Extract features from unified telemetry dataset
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft

sys.stdout.reconfigure(encoding='utf-8')

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_processed_data_path, get_features_path,
    get_results_path, load_dataframe, save_dataframe, print_dataframe_info
)


class FeatureEngineer:
    """Extract behavioral features from telemetry data"""

    def __init__(self, telemetry_df, segment_size=500, stride=None):
        """
        Initialize FeatureEngineer

        Args:
            telemetry_df: DataFrame with telemetry data
            segment_size: Number of samples per segment (default: 500 = 10 seconds @ 50Hz)
            stride: Step between windows (default: segment_size // 2 = 50% overlap)
        """
        self.telemetry_df = telemetry_df
        self.segment_size = segment_size
        self.stride = stride if stride is not None else segment_size // 2
        self.min_segment_size = int(segment_size * 0.8)  # Allow 80% minimum

        # Load corner zones for zone feature (optional — skipped if file missing)
        self.corner_zones = None
        zones_file = get_features_path() / 'corner_zones.json'
        if zones_file.exists():
            with open(zones_file) as f:
                self.corner_zones = json.load(f)
            print(f"  Loaded {len(self.corner_zones)} corner zones for zone feature")

        # Telemetry channels to extract features from
        self.feature_channels = [
            'v_car', 'percent_throttle', 'percent_brake', 'steering_angle',
            'g_lat', 'g_long', 'n_engine',
            't_tyreFR', 't_tyreFL', 't_tyreRR', 't_tyreRL',
            'p_tyreFR', 'p_tyreFL', 'p_tyreRR', 'p_tyreRL',
            'v_x', 'v_z'
        ]
        
    def extract_all_features(self):
        """Extract features for all drivers using sliding windows with zone bool features."""
        print(f"\nExtracting features (window={self.segment_size} samples, stride={self.stride}, 50% overlap)...")

        all_features = []

        for driver_id in self.telemetry_df['driver_id'].unique():
            # reset_index so segment.index == positional index == zone_labels positions
            driver_data = self.telemetry_df[
                self.telemetry_df['driver_id'] == driver_id
            ].copy().reset_index(drop=True)

            zone_labels = None
            if self.corner_zones is not None and 'pos_norm' in driver_data.columns:
                from corner_zones import label_samples
                zone_labels, _ = label_samples(driver_data['pos_norm'].values, self.corner_zones)

            segments = self._create_segments(driver_data)
            print(f"  {driver_id}: {len(segments)} segments")

            for seg_idx, segment in enumerate(segments):
                features = self._extract_segment_features(segment, driver_id, seg_idx, zone_labels)
                all_features.append(features)

        features_df = pd.DataFrame(all_features)
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"\n✓ Extracted {len(features_df)} feature sets with {len(features_df.columns) - 2} features each")
        return features_df

    def _create_segments(self, driver_data):
        """Sliding window with 50% overlap."""
        segments = []
        total_samples = len(driver_data)
        for start_idx in range(0, total_samples, self.stride):
            segment = driver_data.iloc[start_idx: start_idx + self.segment_size]
            if len(segment) >= self.min_segment_size:
                segments.append(segment)
        return segments

    def _extract_segment_features(self, segment, driver_id, segment_idx, zone_labels=None):
        """Extract all features from a single segment."""
        # Dominant zone in this window → one-hot bool features
        if zone_labels is not None:
            seg_zones = zone_labels[segment.index[0]: segment.index[-1] + 1]
            dominant_zone = int(np.bincount(seg_zones).argmax())
        else:
            dominant_zone = 0

        features = {
            'driver_id':   driver_id,
            'segment_id':  f"{driver_id}_{segment_idx}",
            'is_straight': int(dominant_zone == 0),
            'is_eingang':  int(dominant_zone == 1),
            'is_mitte':    int(dominant_zone == 2),
            'is_apex':     int(dominant_zone == 3),
        }

        # 1. Statistical features for each channel
        for channel in self.feature_channels:
            if channel in segment.columns:
                data = segment[channel].values
                
                with np.errstate(invalid='ignore', divide='ignore'):
                    features[f'{channel}_mean'] = np.mean(data)
                    features[f'{channel}_std'] = np.std(data)
                    features[f'{channel}_min'] = np.min(data)
                    features[f'{channel}_max'] = np.max(data)
                    features[f'{channel}_skew'] = stats.skew(data, nan_policy='omit')
                    features[f'{channel}_kurtosis'] = stats.kurtosis(data, nan_policy='omit')
        
        # 2. Behavioral features
        features.update(self._extract_behavioral_features(segment))
        
        # 3. Frequency features
        features.update(self._extract_frequency_features(segment))
        
        # 4. Relative features
        features.update(self._extract_relative_features(segment))
        
        return features
    
    def _extract_behavioral_features(self, segment):
        """Extract driving behavior features"""
        features = {}
        
        # Acceleration/jerk
        if 'g_long' in segment.columns:
            g_long = segment['g_long'].values
            jerk = np.diff(g_long)
            features['jerk_mean'] = np.mean(np.abs(jerk))
            features['jerk_max'] = np.max(np.abs(jerk))
        
        # Steering behavior
        if 'steering_angle' in segment.columns:
            steering = segment['steering_angle'].values
            steering_rate = np.diff(steering)
            features['steering_rate_mean'] = np.mean(np.abs(steering_rate))
            features['steering_rate_max'] = np.max(np.abs(steering_rate))
            features['steering_smoothness'] = np.std(steering_rate)
        
        # Throttle behavior
        if 'percent_throttle' in segment.columns:
            throttle = segment['percent_throttle'].values
            throttle_changes = np.diff(throttle)
            features['throttle_changes'] = np.sum(np.abs(throttle_changes) > 0.1)
            features['throttle_smoothness'] = np.std(throttle_changes)
            features['throttle_aggressive'] = np.sum(throttle_changes > 0.3)
        
        # Brake behavior
        if 'percent_brake' in segment.columns:
            brake = segment['percent_brake'].values
            brake_events = np.sum(brake > 0.1)
            features['brake_events'] = brake_events
            features['brake_pct'] = brake_events / len(brake)
            features['brake_max_pressure'] = np.max(brake)
        
        # Combined throttle/brake (trail braking)
        if 'percent_throttle' in segment.columns and 'percent_brake' in segment.columns:
            throttle = segment['percent_throttle'].values
            brake = segment['percent_brake'].values
            trail_brake = np.sum((throttle > 0.1) & (brake > 0.1))
            features['trail_brake_pct'] = trail_brake / len(brake)
        
        # Cornering
        if 'g_lat' in segment.columns:
            g_lat = segment['g_lat'].values
            features['corner_count'] = np.sum(np.abs(g_lat) > 0.5)
            features['g_lat_extreme_pct'] = np.sum(np.abs(g_lat) > 1.0) / len(g_lat)
        
        # Speed variation
        if 'v_car' in segment.columns:
            v_car = segment['v_car'].values
            features['speed_cv'] = np.std(v_car) / (np.mean(v_car) + 1e-6)  # Coefficient of variation
        
        return features
    
    def _extract_frequency_features(self, segment):
        """Extract frequency domain features using FFT"""
        features = {}
        
        for channel in ['steering_angle', 'percent_throttle', 'g_lat']:
            if channel in segment.columns:
                data = segment[channel].values
                
                # Compute FFT
                fft_vals = np.abs(fft(data))
                freqs = np.fft.fftfreq(len(data))
                
                # Get dominant frequency
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_vals[:len(fft_vals)//2]
                
                if len(positive_fft) > 0:
                    dominant_freq_idx = np.argmax(positive_fft)
                    features[f'{channel}_dominant_freq'] = positive_freqs[dominant_freq_idx]
        
        return features
    
    def _extract_relative_features(self, segment):
        """Extract relative/ratio features"""
        features = {}
        
        # Speed per gear ratio (if available)
        if 'v_car' in segment.columns and 'n_engine' in segment.columns:
            v_car = segment['v_car'].values
            n_engine = segment['n_engine'].values
            with np.errstate(divide='ignore', invalid='ignore'):
                gear_ratio = v_car / (n_engine + 1e-6)
                features['gear_ratio_mean'] = np.mean(gear_ratio)
                features['gear_ratio_std'] = np.std(gear_ratio)
        
        # Tire temperature differences (front vs rear)
        if all(col in segment.columns for col in ['t_tyreFR', 't_tyreFL', 't_tyreRR', 't_tyreRL']):
            front_temp = (segment['t_tyreFR'] + segment['t_tyreFL']) / 2
            rear_temp = (segment['t_tyreRR'] + segment['t_tyreRL']) / 2
            features['tire_temp_diff_fr'] = np.mean(front_temp - rear_temp)
        
        # Tire pressure differences
        if all(col in segment.columns for col in ['p_tyreFR', 'p_tyreFL', 'p_tyreRR', 'p_tyreRL']):
            left_pressure = (segment['p_tyreFR'] + segment['p_tyreRL']) / 2
            right_pressure = (segment['p_tyreFL'] + segment['p_tyreRR']) / 2
            features['tire_pressure_diff_lr'] = np.mean(left_pressure - right_pressure)
        
        return features


def main():
    """Main execution function"""
    project_root = get_project_root()
    processed_path = get_processed_data_path()
    features_path = get_features_path()
    results_path = get_results_path()
    
    print(f"{'='*60}")
    print(f"FEATURE ENGINEERING - Combined HTF + LD Data")
    print(f"{'='*60}\n")
    
    # Load combined telemetry data (fall back to LD-only if no combined exists)
    print("Loading combined telemetry data...")
    telemetry_file = processed_path / "telemetry_combined"
    if not (telemetry_file.with_suffix('.pkl')).exists():
        telemetry_file = processed_path / "telemetry_ld"
    telemetry_df = load_dataframe(telemetry_file)
    
    if telemetry_df is None or len(telemetry_df) == 0:
        print("\n⚠ No combined telemetry data found!")
        print("Please run: py -3 scripts\\03a_combine_data.py first")
        return
    
    print_dataframe_info(telemetry_df)
    
    print(f"\nDriver distribution:")
    for driver_id, count in telemetry_df['driver_id'].value_counts().items():
        print(f"  {driver_id}: {count:,} samples")
    
    # Extract features using FeatureEngineer
    print(f"\n{'='*60}")
    print("EXTRACTING FEATURES")
    print(f"{'='*60}")
    
    # 500-sample windows (10s @ 50Hz), 50% overlap (stride=250)
    feature_engineer = FeatureEngineer(telemetry_df, segment_size=500, stride=250)
    features_df = feature_engineer.extract_all_features()
    
    # Save features
    print(f"\n{'='*60}")
    print("SAVING FEATURES")
    print(f"{'='*60}")
    
    output_file = features_path / "driver_features_combined"
    save_dataframe(features_df, output_file)
    
    # Create summary report
    summary_file = results_path / "03b_feature_engineering_combined_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("FEATURE ENGINEERING SUMMARY - Combined HTF+LD Data\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Input samples: {len(telemetry_df):,}\n")
        f.write(f"Output feature sets: {len(features_df):,}\n")
        f.write(f"Features per set: {len(features_df.columns) - 2}\n")
        f.write(f"Segmentation: corner-phase (eingang / mitte / apex)\n\n")
        
        f.write("Feature sets per driver:\n")
        for driver_id, count in features_df['driver_id'].value_counts().items():
            f.write(f"  {driver_id}: {count} segments\n")
    
    print(f"✓ Saved summary: {summary_file.name}")
    
    print(f"\n{'='*60}")
    print("✓ FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Features saved to: {output_file}.pkl / .csv")
    print(f"\nNext step: Train models with combined data")
    print("  py -3 scripts\\04_train_models_combined.py")


if __name__ == "__main__":
    main()
