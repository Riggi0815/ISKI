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

# Import HTFParser, LDParser and FeatureEngineer
parse_htf = import_module_from_path("parse_htf", script_dir / "01_parse_htf.py")
parse_ld = import_module_from_path("parse_ld", script_dir / "02_parse_ld.py")
feature_engineering = import_module_from_path("feature_engineering", script_dir / "03b_feature_engineering_combined.py")

HTFParser = parse_htf.HTFParser
LDParser = parse_ld.LDParser
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
        # Use combined models directory
        self.models_dir = get_models_path() / "combined"
        
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
    
    def predict_from_file(self, file_path: str, 
                         confidence_threshold: float = 0.6) -> Dict:
        """
        Predict driver from telemetry file (HTF or LD)
        Automatically detects file type based on extension
        
        Args:
            file_path: Path to HTF or LD file
            confidence_threshold: Minimum confidence to consider prediction valid
        
        Returns:
            Dictionary with prediction results
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Detect file type by extension
        file_ext = path.suffix.lower()
        
        if file_ext == '.htf':
            return self.predict_from_htf_file(str(path), confidence_threshold)
        elif file_ext == '.ld':
            return self.predict_from_ld_file(str(path), confidence_threshold)
        else:
            return {
                'success': False,
                'error': f'Unsupported file type: {file_ext}. Use .htf or .ld files.',
                'file': str(path)
            }
    
    def predict_from_ld_file(self, ld_file_path: str, 
                            confidence_threshold: float = 0.6) -> Dict:
        """
        Predict driver from LD file
        
        Args:
            ld_file_path: Path to LD file
            confidence_threshold: Minimum confidence to consider prediction valid
        
        Returns:
            Dictionary with prediction results
        """
        ld_path = Path(ld_file_path)
        
        if not ld_path.exists():
            raise FileNotFoundError(f"LD file not found: {ld_path}")
        
        print(f"{'='*60}")
        print(f"Processing: {ld_path.name}")
        print(f"{'='*60}\n")
        
        # Parse LD file
        print("Parsing telemetry data...")
        parser = LDParser(ld_path)
        header, telemetry_df = parser.parse()
        
        if telemetry_df is None or len(telemetry_df) == 0:
            return {
                'success': False,
                'error': 'Failed to parse LD file',
                'file': str(ld_path)
            }
        
        print(f"  ✓ Parsed {len(telemetry_df)} telemetry samples")
        
        # Standardize LD column names to match HTF format
        print("  Standardizing column names...")
        column_mapping = {
            'throttle': 'percent_throttle',
            'brake': 'p_brakeF',
            'steering': 'a_steering',
            'speed': 'v_car',
            'rpm': 'n_engine',
            'tire_FL_pressure': 'p_tyreFL',
            'tire_FR_pressure': 'p_tyreFR',
            'tire_RL_pressure': 'p_tyreRL',
            'tire_RR_pressure': 'p_tyreRR',
            'tire_FL_temp': 't_tyreFL',
            'tire_FR_temp': 't_tyreFR',
            'tire_RL_temp': 't_tyreRL',
            'tire_RR_temp': 't_tyreRR',
            'tire_FL_vel': 'v_tyreFL',
            'tire_FR_vel': 'v_tyreFR',
            'tire_RL_vel': 'v_tyreRL',
            'tire_RR_vel': 'v_tyreRR',
        }
        telemetry_df = telemetry_df.rename(columns=column_mapping)
        print(f"  ✓ Standardized columns to HTF format")
        
        # Extract features
        print("\nExtracting features from telemetry segments...")
        feature_engineer = FeatureEngineer(telemetry_df, segment_size=500)
        features_df = feature_engineer.extract_all_features()
        
        if len(features_df) == 0:
            return {
                'success': False,
                'error': 'No features extracted (file may be too short)',
                'file': str(ld_path)
            }
        
        print(f"  ✓ Extracted {len(features_df)} feature sets")
        
        # Prepare features for prediction - drop all non-numeric columns
        columns_to_drop = []
        for col in features_df.columns:
            if col == 'driver_id' or '_id' in col or 'index' in col or 'segment' in col:
                columns_to_drop.append(col)
            elif features_df[col].dtype == 'object':
                columns_to_drop.append(col)
        
        X = features_df.drop(columns_to_drop, axis=1, errors='ignore')
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
        
        results['file'] = str(ld_path)
        results['n_segments'] = len(features_df)
        results['n_samples'] = len(telemetry_df)
        results['success'] = True
        
        return results
    
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
        
        # Prepare features for prediction - drop all non-numeric columns
        columns_to_drop = []
        for col in features_df.columns:
            if col == 'driver_id' or '_id' in col or 'index' in col or 'segment' in col:
                columns_to_drop.append(col)
            elif features_df[col].dtype == 'object':
                columns_to_drop.append(col)
        
        X = features_df.drop(columns_to_drop, axis=1, errors='ignore')
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
        description="Predict driver identity from telemetry data (HTF or LD)"
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to telemetry file (.htf or .ld)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='svm',
        choices=['random_forest', 'svm', 'xgboost'],
        help='Model to use for prediction (default: svm)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Model directory (default: models/ or models/combined/ if exists)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for known vs unknown driver (default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        # Check if combined models exist, otherwise use default
        combined_dir = get_models_path() / "combined"
        if combined_dir.exists() and (combined_dir / f"{args.model}_model.pkl").exists():
            model_dir = combined_dir
            print(f"Using combined model directory: {combined_dir}")
        else:
            model_dir = get_models_path()
            print(f"Using default model directory: {model_dir}")
    
    # Create predictor with custom model directory
    predictor = DriverPredictor(model_name=args.model)
    predictor.models_dir = model_dir
    
    # Reload models from correct directory
    predictor.model = joblib.load(model_dir / f"{args.model}_model.pkl")
    predictor.scaler = joblib.load(model_dir / "scaler.pkl")
    predictor.label_encoder = joblib.load(model_dir / "label_encoder.pkl")
    
    # Make prediction (auto-detect file type)
    result = predictor.predict_from_file(
        args.file,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print result
    predictor.print_prediction_result(result)
    
    # Save result to file
    results_dir = get_results_path()
    result_file = results_dir / f"prediction_{Path(args.file).stem}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Prediction Result\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input File: {args.file}\n")
        f.write(f"File Type: {Path(args.file).suffix.upper()}\n")
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
