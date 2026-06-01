"""
Step 4b: Train Models on Combined HTF+LD Data
Train ML models using unified dataset with all drivers
"""

import sys
from pathlib import Path
import importlib.util

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_features_path, get_models_path, 
    get_results_path, load_dataframe
)

# Import ModelTrainer from 04_train_models.py
def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_dir = Path(__file__).parent
train_models = import_module_from_path("train_models", script_dir / "04_train_models.py")
ModelTrainer = train_models.ModelTrainer


def main():
    """Main execution function"""
    project_root = get_project_root()
    features_path = get_features_path()
    models_path = get_models_path()
    results_path = get_results_path()
    
    print(f"{'='*70}")
    print(f"ML MODEL TRAINING - Combined HTF+LD Data (All Drivers)")
    print(f"{'='*70}\n")
    
    # Load combined features
    print("Loading combined features...")
    features_file = features_path / "driver_features_combined"
    features_df = load_dataframe(features_file)
    
    if features_df is None or len(features_df) == 0:
        print("\n⚠ No combined features found!")
        print("Please run: py -3 scripts\\03b_feature_engineering_combined.py first")
        return
    
    print(f"✓ Loaded features for {len(features_df)} segments")
    print(f"  Features per segment: {len(features_df.columns) - 2}")
    print(f"  Unique drivers: {features_df['driver_id'].nunique()}")
    
    # Initialize trainer with combined features
    trainer = ModelTrainer(features_df)
    
    # Prepare data (70% train / 30% test, stratified)
    trainer.prepare_data(test_size=0.3, random_state=42)
    
    # Train all models
    print(f"\n{'='*60}")
    print("TRAINING MODELS")
    print(f"{'='*60}\n")
    
    trainer.train_random_forest(n_estimators=100, max_depth=10)
    trainer.train_svm(C=10.0, kernel='rbf')
    trainer.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
    
    # Save models with "_combined" suffix to distinguish from single-source models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")
    
    # Save models to models_combined/ subdirectory
    combined_models_path = models_path / "combined"
    combined_models_path.mkdir(exist_ok=True)
    
    # Save with custom paths
    import joblib
    joblib.dump(trainer.models['random_forest'], combined_models_path / "random_forest_model.pkl")
    print(f"✓ Saved: {combined_models_path / 'random_forest_model.pkl'}")
    
    joblib.dump(trainer.models['svm'], combined_models_path / "svm_model.pkl")
    print(f"✓ Saved: {combined_models_path / 'svm_model.pkl'}")
    
    joblib.dump(trainer.models['xgboost'], combined_models_path / "xgboost_model.pkl")
    print(f"✓ Saved: {combined_models_path / 'xgboost_model.pkl'}")
    
    joblib.dump(trainer.scaler, combined_models_path / "scaler.pkl")
    joblib.dump(trainer.label_encoder, combined_models_path / "label_encoder.pkl")
    print(f"✓ Saved: scaler.pkl and label_encoder.pkl")
    
    # Save metadata
    metadata = {
        'feature_names': trainer.X.columns.tolist() if hasattr(trainer.X, 'columns') else [],
        'driver_names': trainer.label_encoder.classes_.tolist(),
        'n_features': trainer.X.shape[1],
        'n_drivers': len(trainer.label_encoder.classes_),
        'data_source': 'HTF + LD combined'
    }
    
    import json
    with open(combined_models_path / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: model_metadata.json")
    
    # Save results
    results = trainer.results
    
    with open(results_path / "training_results_combined.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON: {results_path / 'training_results_combined.json'}")
    
    # Create comparison report
    trainer._create_comparison_report()
    
    # Rename to combined version
    import shutil
    if (results_path / "04_model_comparison.txt").exists():
        shutil.move(
            str(results_path / "04_model_comparison.txt"),
            str(results_path / "04_model_comparison_combined.txt")
        )
        print(f"✓ Saved: {results_path / '04_model_comparison_combined.txt'}")
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved in: {combined_models_path}")
    print(f"Results saved in: {results_path}")
    
    print(f"\nAll drivers trained:")
    for driver in sorted(trainer.label_encoder.classes_):
        print(f"  - {driver}")
    
    print(f"\nBest model: ", end="")
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"{best_model[0].upper()} ({best_model[1]['test_accuracy']:.2%})")


if __name__ == "__main__":
    main()
