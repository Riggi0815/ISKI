"""
Step 3b: Feature Engineering on Combined HTF+LD Data
Extract features from unified telemetry dataset
"""

import sys
from pathlib import Path
import importlib.util

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_processed_data_path, get_features_path,
    get_results_path, load_dataframe, save_dataframe, print_dataframe_info
)

# Import FeatureEngineer from 03_feature_engineering.py
def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_dir = Path(__file__).parent
feature_engineering = import_module_from_path("feature_engineering", script_dir / "03_feature_engineering.py")
FeatureEngineer = feature_engineering.FeatureEngineer


def main():
    """Main execution function"""
    project_root = get_project_root()
    processed_path = get_processed_data_path()
    features_path = get_features_path()
    results_path = get_results_path()
    
    print(f"{'='*60}")
    print(f"FEATURE ENGINEERING - Combined HTF + LD Data")
    print(f"{'='*60}\n")
    
    # Load combined telemetry data
    print("Loading combined telemetry data...")
    telemetry_file = processed_path / "telemetry_combined"
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
    
    # Use 500-sample segments (10 seconds at 50Hz)
    feature_engineer = FeatureEngineer(telemetry_df, segment_size=500)
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
        f.write(f"Segment size: 500 samples (10 seconds at 50Hz)\n\n")
        
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
