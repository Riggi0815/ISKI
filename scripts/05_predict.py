"""
Step 5: Predict Driver from New Telemetry Data
Loads trained model and predicts driver identity from new HTF files
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple
import argparse
import importlib.util

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_models_path, get_results_path,
    print_dataframe_info
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

# Import HTFParser and FeatureEngineer
parse_htf = import_module_from_path("parse_htf", script_dir / "01_parse_htf.py")
feature_engineering = import_module_from_path("feature_engineering", script_dir / "03_feature_engineering.py")

HTFParser = parse_htf.HTFParser
FeatureEngineer = feature_engineering.FeatureEngineer


class DriverPredictor:
    """Predict driver identity from telemetry data"""
    
    def __init__(self, model_name: str = 'svm'):
        """
        Initialize predictor with trained model
        
        Args:
            model_name: Name of model to use ('random_forest', 'svm', 'xgboost')
        """
        self.model_name = model_name
        self.project_root = get_project_root()
        self.models_dir = get_models_path()
        
        print(f"{'='*60}")
        print(f"DRIVER PREDICTOR - Loading {model_name.upper()} Model")
        print(f"{'='*60}\n")
        
        # Load model
        model_path = self.models_dir / f"{model_name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
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
        
        print(f"\nKnown drivers: {', '.join(self.label_encoder.classes_)}")
        print()
    
    def predict_from_htf_file(self, htf_file_path: str, 
                              confidence_threshold: float = 0.6) -> Dict:
        """
        Predict driver from HTF file
        
        Args:
            htf_file_path: Path to HTF file
            confidence_threshold: Minimum confidence to consider prediction valid
        
        Returns:
            Dictionary with prediction results
        """
        htf_path = Path(htf_file_path)
        
        if not htf_path.exists():
            raise FileNotFoundError(f"HTF file not found: {htf_path}")
        
        print(f"{'='*60}")
        print(f"Processing: {htf_path.name}")
        print(f"{'='*60}\n")
        
        # Parse HTF file
        print("Parsing telemetry data...")
        parser = HTFParser(htf_path)
        header, telemetry_df = parser.parse()
        
        if telemetry_df is None or len(telemetry_df) == 0:
            return {
                'success': False,
                'error': 'Failed to parse HTF file',
                'file': str(htf_path)
            }
        
        print(f"  ✓ Parsed {len(telemetry_df)} telemetry samples")
        
        # Extract features
        print("\nExtracting features from telemetry segments...")
        feature_engineer = FeatureEngineer(telemetry_df, segment_size=500)
        features_df = feature_engineer.extract_all_features()
        
        if len(features_df) == 0:
            return {
                'success': False,
                'error': 'No features extracted (file may be too short)',
                'file': str(htf_path)
            }
        
        print(f"  ✓ Extracted {len(features_df)} feature sets")
        
        # Prepare features for prediction
        X = features_df.drop(['driver_id', 'segment_idx'], axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Predict for each segment
        print("\nMaking predictions...")
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            max_probas = probabilities.max(axis=1)
        else:
            # For models without predict_proba, use decision function
            max_probas = np.ones(len(predictions))
        
        # Decode predictions
        predicted_drivers = self.label_encoder.inverse_transform(predictions)
        
        # Aggregate predictions across segments
        results = self._aggregate_predictions(
            predicted_drivers, max_probas, confidence_threshold
        )
        
        results['file'] = str(htf_path)
        results['n_segments'] = len(features_df)
        results['n_samples'] = len(telemetry_df)
        results['success'] = True
        
        return results
    
    def _aggregate_predictions(self, predictions: np.ndarray, 
                               probabilities: np.ndarray,
                               confidence_threshold: float) -> Dict:
        """Aggregate predictions from multiple segments"""
        
        # Count predictions per driver
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_counts = dict(zip(unique, counts))
        
        # Find most common prediction
        most_common_driver = max(prediction_counts, key=prediction_counts.get)
        vote_count = prediction_counts[most_common_driver]
        vote_percentage = (vote_count / len(predictions)) * 100
        
        # Calculate average confidence for most common prediction
        mask = predictions == most_common_driver
        avg_confidence = probabilities[mask].mean()
        
        # Determine if driver is known or unknown
        is_known = avg_confidence >= confidence_threshold
        
        return {
            'predicted_driver': most_common_driver,
            'is_known': is_known,
            'confidence': float(avg_confidence),
            'vote_percentage': float(vote_percentage),
            'vote_count': int(vote_count),
            'total_segments': len(predictions),
            'all_predictions': prediction_counts
        }
    
    def print_prediction_result(self, result: Dict):
        """Pretty print prediction result"""
        
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        
        if not result.get('success', False):
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return
        
        print(f"File: {Path(result['file']).name}")
        print(f"Samples: {result['n_samples']:,} telemetry points")
        print(f"Segments: {result['n_segments']} (10-second windows)")
        print()
        
        # Main prediction
        if result['is_known']:
            print(f"✓ KNOWN DRIVER DETECTED")
            print(f"  Driver ID: {result['predicted_driver']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Agreement: {result['vote_percentage']:.1f}% of segments ({result['vote_count']}/{result['total_segments']})")
        else:
            print(f"⚠ UNKNOWN DRIVER")
            print(f"  Best match: {result['predicted_driver']}")
            print(f"  Confidence: {result['confidence']:.1%} (below threshold)")
            print(f"  This driver's pattern doesn't match known drivers well")
        
        # Show vote distribution if there are multiple predictions
        if len(result['all_predictions']) > 1:
            print(f"\n  Vote distribution across all segments:")
            for driver, count in sorted(result['all_predictions'].items(), 
                                       key=lambda x: x[1], reverse=True):
                pct = (count / result['total_segments']) * 100
                print(f"    {driver}: {count:3d} votes ({pct:5.1f}%)")
        
        print(f"{'='*60}\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Predict driver identity from telemetry data"
    )
    parser.add_argument(
        'htf_file',
        type=str,
        help='Path to HTF file to predict'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='svm',
        choices=['random_forest', 'svm', 'xgboost'],
        help='Model to use for prediction (default: svm)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for known vs unknown driver (default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DriverPredictor(model_name=args.model)
    
    # Make prediction
    result = predictor.predict_from_htf_file(
        args.htf_file,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print result
    predictor.print_prediction_result(result)
    
    # Save result to file
    results_dir = get_results_path()
    result_file = results_dir / f"prediction_{Path(args.htf_file).stem}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Prediction Result\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input File: {args.htf_file}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Confidence Threshold: {args.confidence_threshold}\n\n")
        f.write(f"Predicted Driver: {result.get('predicted_driver', 'N/A')}\n")
        f.write(f"Status: {'KNOWN' if result.get('is_known', False) else 'UNKNOWN'}\n")
        f.write(f"Confidence: {result.get('confidence', 0):.1%}\n")
        f.write(f"Vote Percentage: {result.get('vote_percentage', 0):.1f}%\n")
        f.write(f"\nVote Distribution:\n")
        for driver, count in result.get('all_predictions', {}).items():
            f.write(f"  {driver}: {count} votes\n")
    
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
