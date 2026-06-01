"""
Step 7: Evaluate Models on Test Data
Tests trained models on held-out test data that was never seen during training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
import importlib.util

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_models_path, get_results_path, get_test_data_path,
    list_test_htf_files, list_test_ld_files
)

# Import modules with numeric prefixes using importlib
def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get script directory
script_dir = Path(__file__).parent

# Import parsers and feature engineer
parse_htf = import_module_from_path("parse_htf", script_dir / "01_parse_htf.py")
parse_ld = import_module_from_path("parse_ld", script_dir / "02_parse_ld.py")
feature_engineering = import_module_from_path("feature_engineering", script_dir / "03b_feature_engineering_combined.py")

HTFParser = parse_htf.HTFParser
LDParser = parse_ld.LDParser
FeatureEngineer = feature_engineering.FeatureEngineer


class TestEvaluator:
    """Evaluate models on held-out test data"""
    
    def __init__(self, model_name: str = 'xgboost'):
        """
        Initialize evaluator with trained model
        
        Args:
            model_name: Name of model to use ('random_forest', 'svm', 'xgboost')
        """
        self.model_name = model_name
        self.project_root = get_project_root()
        self.models_dir = get_models_path() / "combined"
        self.test_data_path = get_test_data_path()
        
        print(f"{'='*60}")
        print(f"TEST DATA EVALUATION - {model_name.upper()}")
        print(f"{'='*60}\n")
        
        # Load model
        model_path = self.models_dir / f"{model_name}_model.pkl"
        self.model = joblib.load(model_path)
        print(f"✓ Loaded model: {model_path.name}")
        
        # Load scaler
        scaler_path = self.models_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Loaded scaler: {scaler_path.name}")
        
        # Load label encoder
        encoder_path = self.models_dir / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path)
        print(f"✓ Loaded label encoder: {encoder_path.name}")
        
        print(f"\nTrained on: {', '.join(self.label_encoder.classes_)}")
        print()
    
    def evaluate_file(self, file_path: Path) -> Dict:
        """
        Evaluate model on a single test file
        
        Args:
            file_path: Path to HTF or LD file
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nProcessing: {file_path.name}")
        print(f"{'─'*60}")
        
        # Parse file based on extension
        if file_path.suffix == '.htf':
            parser = HTFParser(str(file_path))
            telemetry_df = parser.parse()
            true_driver = parser.header.get('driver', 'UNKNOWN')
        elif file_path.suffix == '.ld':
            parser = LDParser(str(file_path))
            header, telemetry_df = parser.parse()
            # Extract driver from filename
            parts = file_path.stem.split('_&_')
            true_driver = f"_{parts[2]}_" if len(parts) > 2 else "UNKNOWN"
        else:
            return {'success': False, 'error': f"Unknown file type: {file_path.suffix}"}
        
        if telemetry_df is None or len(telemetry_df) == 0:
            return {'success': False, 'error': 'Failed to parse file'}
        
        print(f"True driver: {true_driver}")
        print(f"Samples: {len(telemetry_df):,}")
        
        # Extract features
        engineer = FeatureEngineer(segment_size=500)
        features_df = engineer.extract_all_features(telemetry_df)
        
        if features_df is None or len(features_df) == 0:
            return {'success': False, 'error': 'No features extracted'}
        
        print(f"Segments: {len(features_df)}")
        
        # Prepare features
        columns_to_drop = []
        for col in features_df.columns:
            if col == 'driver_id' or '_id' in col or 'index' in col or 'segment' in col:
                columns_to_drop.append(col)
            elif features_df[col].dtype == 'object':
                columns_to_drop.append(col)
        
        X = features_df.drop(columns_to_drop, axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            confidences = probabilities.max(axis=1)
            avg_confidence = confidences.mean() * 100
        else:
            avg_confidence = None
        
        # Decode predictions
        predicted_drivers = [self.label_encoder.inverse_transform([pred])[0] for pred in predictions]
        
        # Calculate accuracy for this file
        from collections import Counter
        vote_counts = Counter(predicted_drivers)
        predicted_driver = vote_counts.most_common(1)[0][0]
        agreement = vote_counts[predicted_driver] / len(predicted_drivers) * 100
        
        # Check if prediction is correct
        is_correct = (predicted_driver == true_driver)
        
        print(f"Predicted: {predicted_driver} ({agreement:.1f}% agreement)")
        if avg_confidence:
            print(f"Confidence: {avg_confidence:.1f}%")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
        return {
            'success': True,
            'file': file_path.name,
            'true_driver': true_driver,
            'predicted_driver': predicted_driver,
            'is_correct': is_correct,
            'segments': len(features_df),
            'agreement': agreement,
            'confidence': avg_confidence,
            'vote_distribution': dict(vote_counts)
        }
    
    def evaluate_all_test_data(self) -> Dict:
        """
        Evaluate model on all test data
        
        Returns:
            Dictionary with overall evaluation results
        """
        # Get test files
        htf_files = list_test_htf_files()
        ld_files = list_test_ld_files()
        all_files = htf_files + ld_files
        
        print(f"{'='*60}")
        print(f"TEST DATA EVALUATION")
        print(f"{'='*60}")
        print(f"HTF files: {len(htf_files)}")
        print(f"LD files:  {len(ld_files)}")
        print(f"Total:     {len(all_files)}\n")
        
        if len(all_files) == 0:
            print("❌ No test files found!")
            return {'success': False, 'error': 'No test files'}
        
        # Evaluate each file
        results = []
        for file_path in all_files:
            result = self.evaluate_file(file_path)
            if result['success']:
                results.append(result)
        
        # Calculate overall statistics
        if len(results) == 0:
            print("\n❌ No successful evaluations")
            return {'success': False, 'error': 'No successful evaluations'}
        
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        accuracy = correct / total * 100
        
        avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] is not None])
        avg_agreement = np.mean([r['agreement'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"OVERALL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Files evaluated: {total}")
        print(f"Correct predictions: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Average agreement: {avg_agreement:.1f}%")
        print()
        
        # Show per-file results
        print("Per-file results:")
        for r in results:
            status = "✓" if r['is_correct'] else "✗"
            print(f"  {status} {r['file'][:40]:40s} | True: {r['true_driver']:12s} | Pred: {r['predicted_driver']:12s}")
        
        return {
            'success': True,
            'model_name': self.model_name,
            'total_files': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_agreement': avg_agreement,
            'results': results
        }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models on test data')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['random_forest', 'svm', 'xgboost'],
                       help='Model to evaluate (default: xgboost)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = TestEvaluator(model_name=args.model)
    results = evaluator.evaluate_all_test_data()
    
    if results['success']:
        # Save results
        results_path = get_results_path()
        results_path.mkdir(exist_ok=True)
        
        output_file = results_path / f"test_evaluation_{args.model}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"TEST DATA EVALUATION - {args.model.upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Files evaluated: {results['total_files']}\n")
            f.write(f"Correct predictions: {results['correct']}/{results['total_files']}\n")
            f.write(f"Accuracy: {results['accuracy']:.1f}%\n")
            f.write(f"Average confidence: {results['avg_confidence']:.1f}%\n")
            f.write(f"Average agreement: {results['avg_agreement']:.1f}%\n\n")
            
            f.write("Per-file results:\n")
            f.write(f"{'─'*60}\n")
            for r in results['results']:
                status = "✓" if r['is_correct'] else "✗"
                f.write(f"{status} {r['file']}\n")
                f.write(f"  True: {r['true_driver']}, Predicted: {r['predicted_driver']}\n")
                f.write(f"  Segments: {r['segments']}, Agreement: {r['agreement']:.1f}%, Confidence: {r['confidence']:.1f}%\n\n")
        
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
