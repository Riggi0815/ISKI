"""
Step 3b: Combine HTF and LD Data for Unified Training
Merges HTF and LD telemetry data into single dataset with unified column names
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_processed_data_path, get_features_path,
    get_results_path, load_dataframe, save_dataframe
)


class TelemetryCombiner:
    """Combine HTF and LD telemetry data into unified dataset"""
    
    def __init__(self):
        self.project_root = get_project_root()
        self.processed_path = get_processed_data_path()
        
        # Column name mapping: LD -> HTF standard names
        self.ld_to_htf_mapping = {
            # Basic controls
            'throttle': 'percent_throttle',
            'brake': 'p_brakeF',
            'steering': 'a_steering',
            
            # Vehicle state
            'gear': 'gear',
            'speed': 'v_car',  # Changed: HTF uses v_car, not v_vehicle
            'rpm': 'n_engine',
            
            # G-forces (already matching)
            'g_long': 'g_long',
            'g_lat': 'g_lat',
            'g_vert': 'g_vert',
            
            # Tire pressures
            'tire_FL_pressure': 'p_tyreFL',
            'tire_FR_pressure': 'p_tyreFR',
            'tire_RL_pressure': 'p_tyreRL',
            'tire_RR_pressure': 'p_tyreRR',
            
            # Tire temperatures
            'tire_FL_temp': 't_tyreFL',
            'tire_FR_temp': 't_tyreFR',
            'tire_RL_temp': 't_tyreRL',
            'tire_RR_temp': 't_tyreRR',
            
            # Tire velocities
            'tire_FL_vel': 'v_tyreFL',
            'tire_FR_vel': 'v_tyreFR',
            'tire_RL_vel': 'v_tyreRL',
            'tire_RR_vel': 'v_tyreRR',
        }
        
        print(f"{'='*70}")
        print(f"TELEMETRY DATA COMBINER")
        print(f"Combining HTF and LD data sources")
        print(f"{'='*70}\n")
    
    def load_htf_data(self) -> pd.DataFrame:
        """Load HTF telemetry data"""
        print("Loading HTF data...")
        htf_df = load_dataframe(self.processed_path / "telemetry_all")
        
        if htf_df is None or len(htf_df) == 0:
            print("  ⚠ No HTF data found!")
            return pd.DataFrame()
        
        print(f"  ✓ Loaded {len(htf_df):,} HTF samples")
        print(f"  ✓ Drivers: {htf_df['driver_id'].nunique()} unique")
        print(f"    {', '.join(htf_df['driver_id'].unique())}")
        
        return htf_df
    
    def load_ld_data(self) -> pd.DataFrame:
        """Load LD telemetry data"""
        print("\nLoading LD data...")
        ld_df = load_dataframe(self.processed_path / "telemetry_ld")
        
        if ld_df is None or len(ld_df) == 0:
            print("  ⚠ No LD data found!")
            return pd.DataFrame()
        
        print(f"  ✓ Loaded {len(ld_df):,} LD samples")
        print(f"  ✓ Drivers: {ld_df['driver_id'].nunique()} unique")
        print(f"    {', '.join(ld_df['driver_id'].unique())}")
        
        return ld_df
    
    def standardize_ld_columns(self, ld_df: pd.DataFrame) -> pd.DataFrame:
        """Rename LD columns to match HTF standard"""
        print("\nStandardizing LD column names...")
        
        # Rename columns according to mapping
        ld_renamed = ld_df.copy()
        
        # Apply mapping
        for ld_col, htf_col in self.ld_to_htf_mapping.items():
            if ld_col in ld_renamed.columns:
                ld_renamed.rename(columns={ld_col: htf_col}, inplace=True)
        
        print(f"  ✓ Mapped {len(self.ld_to_htf_mapping)} column names")
        
        return ld_renamed
    
    def combine_datasets(self, htf_df: pd.DataFrame, ld_df: pd.DataFrame) -> pd.DataFrame:
        """Combine HTF and LD datasets with unified columns"""
        print("\nCombining datasets...")
        
        if htf_df.empty and ld_df.empty:
            print("  ⚠ No data to combine!")
            return pd.DataFrame()
        
        if htf_df.empty:
            print("  ⚠ No HTF data, using only LD")
            return ld_df
        
        if ld_df.empty:
            print("  ⚠ No LD data, using only HTF")
            return htf_df
        
        # Get common columns (metadata + telemetry channels that exist in both)
        htf_cols = set(htf_df.columns)
        ld_cols = set(ld_df.columns)
        
        # Must have metadata columns
        metadata_cols = ['driver_id', 'circuit', 'vehicle', 'recording_date', 'sample_index']
        
        # Find common telemetry channels
        common_channels = (htf_cols & ld_cols) - set(metadata_cols)
        
        print(f"  Common telemetry channels: {len(common_channels)}")
        
        # Select only common columns for combination
        common_cols = metadata_cols + sorted(list(common_channels))
        
        # Filter to common columns
        htf_filtered = htf_df[common_cols].copy()
        ld_filtered = ld_df[[col for col in common_cols if col in ld_df.columns]].copy()
        
        # Add source column to track origin
        htf_filtered['data_source'] = 'HTF'
        ld_filtered['data_source'] = 'LD'
        
        # Combine
        combined_df = pd.concat([htf_filtered, ld_filtered], ignore_index=True)
        
        # Reset sample_index globally
        combined_df['sample_index'] = range(len(combined_df))
        
        print(f"  ✓ Combined dataset: {len(combined_df):,} samples")
        print(f"    HTF samples: {len(htf_filtered):,}")
        print(f"    LD samples:  {len(ld_filtered):,}")
        print(f"  ✓ Total columns: {len(combined_df.columns)}")
        
        return combined_df
    
    def create_summary_report(self, combined_df: pd.DataFrame):
        """Create summary report of combined data"""
        results_path = get_results_path()
        summary_file = results_path / "03b_data_combination_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("HTF + LD DATA COMBINATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total samples: {len(combined_df):,}\n")
            f.write(f"Total drivers: {combined_df['driver_id'].nunique()}\n")
            f.write(f"Total columns: {len(combined_df.columns)}\n\n")
            
            f.write("Samples by data source:\n")
            for source, count in combined_df['data_source'].value_counts().items():
                f.write(f"  {source}: {count:,} samples\n")
            
            f.write("\nSamples per driver:\n")
            for driver, count in combined_df['driver_id'].value_counts().items():
                source_dist = combined_df[combined_df['driver_id'] == driver]['data_source'].value_counts()
                f.write(f"  {driver}: {count:,} samples")
                if len(source_dist) > 0:
                    sources = ", ".join([f"{src}:{cnt}" for src, cnt in source_dist.items()])
                    f.write(f" ({sources})")
                f.write("\n")
            
            f.write(f"\nCircuits: {', '.join(combined_df['circuit'].unique())}\n")
            f.write(f"Vehicles: {', '.join(combined_df['vehicle'].unique())}\n")
            
            f.write("\nTelemetry channels:\n")
            telemetry_cols = [col for col in combined_df.columns 
                            if col not in ['driver_id', 'circuit', 'vehicle', 
                                          'recording_date', 'sample_index', 
                                          'simulation_setup', 'data_source']]
            f.write(f"  Total: {len(telemetry_cols)} channels\n")
            for col in sorted(telemetry_cols):
                f.write(f"    - {col}\n")
        
        print(f"✓ Saved summary: {summary_file.name}")
    
    def run(self) -> pd.DataFrame:
        """Run complete combination pipeline"""
        
        # Load data
        htf_df = self.load_htf_data()
        ld_df = self.load_ld_data()
        
        # Standardize LD columns
        if not ld_df.empty:
            ld_df = self.standardize_ld_columns(ld_df)
        
        # Combine datasets
        combined_df = self.combine_datasets(htf_df, ld_df)
        
        if combined_df.empty:
            print("\n⚠ No data to save!")
            return combined_df
        
        # Save combined data
        print(f"\n{'='*70}")
        print("SAVING COMBINED DATA")
        print(f"{'='*70}")
        
        output_file = self.processed_path / "telemetry_combined"
        save_dataframe(combined_df, output_file)
        
        # Create summary report
        self.create_summary_report(combined_df)
        
        print(f"\n{'='*70}")
        print("✓ DATA COMBINATION COMPLETE")
        print(f"{'='*70}")
        print(f"Combined data saved to: {output_file}.pkl / .csv")
        print(f"\nSummary:")
        print(f"  Total samples: {len(combined_df):,}")
        print(f"  Unique drivers: {combined_df['driver_id'].nunique()}")
        print(f"  Driver IDs: {', '.join(sorted(combined_df['driver_id'].unique()))}")
        
        return combined_df


def main():
    """Main execution function"""
    combiner = TelemetryCombiner()
    combined_df = combiner.run()
    
    if not combined_df.empty:
        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print(f"{'='*70}")
        print("1. Run feature engineering on combined data:")
        print("   py -3 scripts\\03_feature_engineering_combined.py")
        print("\n2. Train models with all drivers (HTF + LD):")
        print("   py -3 scripts\\04_train_models.py")


if __name__ == "__main__":
    main()
