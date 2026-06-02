"""
Step 3a: Prepare HTF Data for Training
Loads and prepares HTF telemetry data (no LD data anymore)
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


class TelemetryPreparer:
    """Prepare HTF telemetry data for training (no LD data anymore)"""
    
    def __init__(self):
        self.project_root = get_project_root()
        self.processed_path = get_processed_data_path()
        
        print(f"{'='*70}")
        print(f"TELEMETRY DATA PREPARATION")
        print(f"Processing HTF data only (no LD data)")
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
    
    def prepare_dataset(self, htf_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare HTF dataset for training"""
        print("\nPreparing HTF dataset...")
        
        if htf_df.empty:
            print("  ⚠ No data to prepare!")
            return pd.DataFrame()
        
        # Add source column to track origin
        htf_df['data_source'] = 'HTF'
        
        # Reset sample_index globally
        htf_df['sample_index'] = range(len(htf_df))
        
        print(f"  ✓ Prepared dataset: {len(htf_df):,} samples")
        print(f"  ✓ Total columns: {len(htf_df.columns)}")
        
        return htf_df
    
    def create_summary_report(self, prepared_df: pd.DataFrame):
        """Create summary report of prepared data"""
        results_path = get_results_path()
        summary_file = results_path / "03a_data_preparation_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("HTF DATA PREPARATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total samples: {len(prepared_df):,}\n")
            f.write(f"Total drivers: {prepared_df['driver_id'].nunique()}\n")
            f.write(f"Total columns: {len(prepared_df.columns)}\n\n")
            
            f.write("Samples per driver:\n")
            for driver, count in prepared_df['driver_id'].value_counts().items():
                f.write(f"  {driver}: {count:,} samples\n")
            
            f.write(f"\nCircuits: {', '.join(prepared_df['circuit'].unique())}\n")
            f.write(f"Vehicles: {', '.join(prepared_df['vehicle'].unique())}\n")
            
            f.write("\nTelemetry channels:\n")
            telemetry_cols = [col for col in prepared_df.columns 
                            if col not in ['driver_id', 'circuit', 'vehicle', 
                                          'recording_date', 'sample_index', 
                                          'simulation_setup', 'data_source']]
            f.write(f"  Total: {len(telemetry_cols)} channels\n")
            for col in sorted(telemetry_cols):
                f.write(f"    - {col}\n")
        
        print(f"✓ Saved summary: {summary_file.name}")
    
    def run(self) -> pd.DataFrame:
        """Run complete data preparation pipeline"""
        
        # Load HTF data
        htf_df = self.load_htf_data()
        
        # Prepare dataset
        prepared_df = self.prepare_dataset(htf_df)
        
        if prepared_df.empty:
            print("\n⚠ No data to save!")
            return prepared_df
        
        # Save prepared data
        print(f"\n{'='*70}")
        print("SAVING PREPARED DATA")
        print(f"{'='*70}")
        
        output_file = self.processed_path / "telemetry_combined"
        save_dataframe(prepared_df, output_file)
        
        # Create summary report
        self.create_summary_report(prepared_df)
        
        print(f"\n{'='*70}")
        print("✓ DATA PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"Prepared data saved to: {output_file}.pkl / .csv")
        print(f"\nSummary:")
        print(f"  Total samples: {len(prepared_df):,}")
        print(f"  Unique drivers: {prepared_df['driver_id'].nunique()}")
        print(f"  Driver IDs: {', '.join(sorted(prepared_df['driver_id'].unique()))}")
        
        return prepared_df


def main():
    """Main execution function"""
    preparer = TelemetryPreparer()
    prepared_df = preparer.run()
    
    if not prepared_df.empty:
        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print(f"{'='*70}")
        print("1. Run feature engineering:")
        print("   python scripts/03b_feature_engineering_combined.py")
        print("\n2. Train models with HTF data:")
        print("   python scripts/04b_train_models_combined.py")


if __name__ == "__main__":
    main()
