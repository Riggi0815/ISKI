"""
Step 5: Predict Driver from New Telemetry Data
Loads trained model and predicts driver identity from HTF files
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple
import argparse
import importlib.util

sys.stdout.reconfigure(encoding='utf-8')

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_models_path, get_results_path,
    print_dataframe_info, extract_driver_from_filename
)
from ldparser import ldData

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
feature_engineering = import_module_from_path("feature_engineering", script_dir / "03b_feature_engineering_combined.py")

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
        
        # Load metadata to get feature names
        import json
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.feature_names = metadata['feature_names']
        print(f"✓ Loaded feature names: {len(self.feature_names)} features")
        
        print(f"\nKnown drivers: {', '.join(self.label_encoder.classes_)}")
        print()
    
    def predict_from_file(self, file_path: str, 
                         confidence_threshold: float = 0.6) -> Dict:
        """
        Predict driver from HTF telemetry file
        
        Args:
            file_path: Path to HTF file
            confidence_threshold: Minimum confidence to consider prediction valid
        
        Returns:
            Dictionary with prediction results
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check file type
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
        parser = HTFParser(str(htf_path))
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
        
        # Select only the features used during training
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        if missing_features:
            print(f"⚠ Warning: {len(missing_features)} features missing: {missing_features[:5]}...")
            return {
                'success': False,
                'error': f'Missing features: {missing_features}',
                'file': str(htf_path)
            }
        
        X = features_df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
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
    
    def predict_from_ld_file(self, ld_file_path: str,
                             confidence_threshold: float = 0.6) -> Dict:
        """Predict driver from MoTeC .ld file"""
        ld_path = Path(ld_file_path)

        print(f"{'='*60}")
        print(f"Processing: {ld_path.name}")
        print(f"{'='*60}\n")

        ld = ldData.fromfile(str(ld_path))
        available = list(ld)

        CHANNEL_MAP = {
            'Ground Speed': 'v_car', 'Throttle Pos': 'percent_throttle',
            'Brake Pos': 'percent_brake', 'Steering Angle': 'steering_angle',
            'CG Accel Lateral': 'g_lat', 'CG Accel Longitudinal': 'g_long',
            'CG Accel Vertical': 'g_vert', 'Engine RPM': 'n_engine',
            'Tire Temp Core FR': 't_tyreFR', 'Tire Temp Core FL': 't_tyreFL',
            'Tire Temp Core RR': 't_tyreRR', 'Tire Temp Core RL': 't_tyreRL',
            'Tire Pressure FR': 'p_tyreFR', 'Tire Pressure FL': 'p_tyreFL',
            'Tire Pressure RR': 'p_tyreRR', 'Tire Pressure RL': 'p_tyreRL',
            'Chassis Velocity X': 'v_x', 'Chassis Velocity Z': 'v_z',
            'Gear': 'gear',
        }

        rows = {}
        for ld_name, htf_name in CHANNEL_MAP.items():
            if ld_name in available:
                try:
                    rows[htf_name] = ld[ld_name].data
                except Exception:
                    pass

        if not rows:
            return {'success': False, 'error': 'No channels readable', 'file': str(ld_path)}

        min_len = min(len(v) for v in rows.values())
        rows = {k: v[:min_len] for k, v in rows.items()}
        driver_id = extract_driver_from_filename(ld_path.name)
        telemetry_df = pd.DataFrame(rows)
        telemetry_df.insert(0, 'driver_id', driver_id)

        print(f"  ✓ Parsed {len(telemetry_df):,} telemetry samples")

        print("\nExtracting features...")
        feature_engineer = FeatureEngineer(telemetry_df, segment_size=500)
        features_df = feature_engineer.extract_all_features()

        if len(features_df) == 0:
            return {'success': False, 'error': 'No features extracted (file too short)', 'file': str(ld_path)}

        print(f"  ✓ Extracted {len(features_df)} feature sets")

        missing = [f for f in self.feature_names if f not in features_df.columns]
        if missing:
            print(f"  Warning: {len(missing)} features missing, filling with 0")
            for f in missing:
                features_df[f] = 0

        # Apply round-based split if test_rounds_only requested
        if getattr(self, 'test_rounds_only', False):
            n_segs = len(features_df)
            n_rounds = 8
            test_rounds = [6, 7]
            segs_per_round = max(1, n_segs // n_rounds)
            test_mask = []
            for pos in range(n_segs):
                round_num = min(pos // segs_per_round + 1, n_rounds)
                test_mask.append(round_num in test_rounds)
            features_df = features_df[test_mask].reset_index(drop=True)
            n_test = len(features_df)
            print(f"  Using test rounds (6+7) only: {n_test}/{n_segs} segments")
            if n_test == 0:
                return {'success': False, 'error': 'No test segments after round split', 'file': str(ld_path)}

        X = features_df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_values = X.values

        # Random Forest was trained on unscaled features — skip scaler
        if self.model_name == 'random_forest':
            X_input = X_values
        else:
            X_input = self.scaler.transform(X_values)

        predictions = self.model.predict(X_input)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_input)
            max_probas = probabilities.max(axis=1)
        else:
            max_probas = np.ones(len(predictions))

        predicted_drivers = self.label_encoder.inverse_transform(predictions)
        results = self._aggregate_predictions(predicted_drivers, max_probas, confidence_threshold)
        results['file'] = str(ld_path)
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


ZONE_NAMES   = {0: 'straight', 1: 'eingang', 2: 'mitte', 3: 'apex'}
# Corner segments count double — straights carry less driver-specific signal
ZONE_WEIGHTS = {0: 1.0, 1: 2.0, 2: 2.0, 3: 2.0}


def run_test_round_evaluation(model_name: str = 'random_forest'):
    """
    For each driver: predict using their test-round segments (rounds 6 & 7),
    with corner segments weighted higher. Saves a per-driver .txt and summary.
    """
    import json
    from utils import get_features_path

    features_path = get_features_path()
    results_path  = get_results_path()
    models_dir    = get_models_path() / "combined"
    results_path.mkdir(exist_ok=True)

    print('=' * 60)
    print('DRIVER PREDICTION — Test Rounds 6 & 7 (per driver)')
    print(f'Model: {model_name}  |  Corner weight: {ZONE_WEIGHTS[1]}x  Straight weight: {ZONE_WEIGHTS[0]}x')
    print('=' * 60)

    pkl = features_path / 'driver_features_combined.pkl'
    if not pkl.exists():
        print("ERROR: features/driver_features_combined.pkl not found.")
        print("Run: py -3 scripts\\03b_feature_engineering_combined.py first")
        return
    features_df = pd.read_pickle(pkl)
    print(f'\nLoaded {len(features_df)} segments for {features_df["driver_id"].nunique()} drivers.\n')

    model = joblib.load(models_dir / f"{model_name}_model.pkl")
    le    = joblib.load(models_dir / "label_encoder.pkl")
    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']

    TEST_ROUNDS = [6, 7]
    N_ROUNDS    = 8

    all_results = []

    for driver_id in sorted(features_df['driver_id'].unique()):
        mask       = features_df['driver_id'] == driver_id
        driver_idx = features_df.index[mask].tolist()
        n_segs     = len(driver_idx)
        segs_per_round = max(1, n_segs // N_ROUNDS)

        test_indices = [
            idx for pos, idx in enumerate(driver_idx)
            if min(pos // segs_per_round + 1, N_ROUNDS) in TEST_ROUNDS
        ]
        if not test_indices:
            print(f"  {driver_id}: no test segments found")
            continue

        test_df = features_df.loc[test_indices].copy()
        print(f"    {driver_id}: {len(test_df)} test segments")

        for feat in feature_names:
            if feat not in test_df.columns:
                test_df[feat] = 0

        X = test_df[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        float32_max = np.finfo(np.float32).max * 0.9
        X = X.clip(lower=-float32_max, upper=float32_max)

        preds       = model.predict(X.values)
        pred_labels = le.inverse_transform(preds)

        if hasattr(model, 'predict_proba'):
            proba          = model.predict_proba(X.values)
            seg_confidence = proba.max(axis=1) * 100
            avg_confidence = seg_confidence.mean()
        else:
            seg_confidence = np.full(len(preds), 100.0)
            avg_confidence = 100.0

        # Corner segments count double — detected via bool features
        is_corner = (
            test_df.get('is_eingang', pd.Series(0, index=test_df.index)) |
            test_df.get('is_mitte',   pd.Series(0, index=test_df.index)) |
            test_df.get('is_apex',    pd.Series(0, index=test_df.index))
        ).values.astype(bool)
        weights = np.where(is_corner, ZONE_WEIGHTS[1], ZONE_WEIGHTS[0])

        weighted_votes: dict = {}
        for label, w in zip(pred_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0.0) + w
        total_weight = weights.sum()

        top_driver = max(weighted_votes, key=weighted_votes.get)
        agreement  = weighted_votes[top_driver] / total_weight * 100
        is_correct = (top_driver == driver_id)

        result = {
            'driver_id':        driver_id,
            'n_segments':       len(test_df),
            'predicted_driver': top_driver,
            'is_correct':       is_correct,
            'agreement':        agreement,
            'avg_confidence':   avg_confidence,
            'weighted_votes':   weighted_votes,
            'total_weight':     total_weight,
            'pred_labels':      list(pred_labels),
            'seg_confidence':   list(seg_confidence),
            'is_corner':        list(is_corner),
            'weights':          list(weights),
        }
        all_results.append(result)

        # --- Per-driver text file ---
        tag      = driver_id.strip('_')
        out_file = results_path / f"predict_{tag}.txt"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('DRIVER PREDICTION REPORT\n')
            f.write('=' * 60 + '\n\n')
            f.write(f"True driver:        {driver_id}\n")
            f.write(f"Model:              {model_name}\n")
            f.write(f"Test rounds:        6 & 7\n")
            f.write(f"Segments evaluated: {len(test_df)}\n")
            f.write(f"Corner weight:      {ZONE_WEIGHTS[1]}x  |  Straight weight: {ZONE_WEIGHTS[0]}x\n\n")
            f.write(f"Predicted driver:   {top_driver}\n")
            f.write(f"Result:             {'CORRECT' if is_correct else 'WRONG'}\n")
            f.write(f"Weighted agreement: {agreement:.1f}%  "
                    f"(weight {weighted_votes[top_driver]:.1f} / {total_weight:.1f})\n")
            f.write(f"Avg. confidence:    {avg_confidence:.1f}%\n\n")

            f.write('Weighted vote distribution:\n')
            f.write('-' * 40 + '\n')
            for drv, w in sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True):
                pct = w / total_weight * 100
                bar = '#' * int(pct / 5)
                f.write(f"  {drv:15s}: {w:6.1f} pts  ({pct:5.1f}%)  {bar}\n")

            f.write('\nPer-segment predictions:\n')
            f.write('-' * 40 + '\n')
            f.write(f"  {'#':>3}  {'Type':<8} {'Weight':>6}  {'Predicted':<15}  {'Conf':>6}  OK?\n")
            f.write(f"  {'-'*3}  {'-'*8} {'-'*6}  {'-'*15}  {'-'*6}  ---\n")
            for i, (pred, conf, corner, w) in enumerate(
                    zip(pred_labels, seg_confidence, is_corner, weights), 1):
                mark      = 'yes' if pred == driver_id else 'no'
                seg_type  = 'corner' if corner else 'straight'
                f.write(f"  {i:3d}  {seg_type:<8} {w:6.1f}  {pred:<15}  {conf:5.1f}%  {mark}\n")

        status = 'OK' if is_correct else 'WRONG'
        print(f"  [{status}] {driver_id}: predicted={top_driver}  "
              f"({agreement:.1f}% weighted, {len(test_df)} segs, conf {avg_confidence:.1f}%)  "
              f"-> {out_file.name}")

    # --- Summary ---
    correct = sum(1 for r in all_results if r['is_correct'])
    total   = len(all_results)
    print(f"\n{'='*60}")
    print(f"TOTAL: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    summary_file = results_path / "05_predict_test_rounds_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('TEST ROUND EVALUATION — SUMMARY\n')
        f.write(f"Model:         {model_name}\n")
        f.write(f"Test rounds:   6 & 7\n")
        f.write(f"Corner weight: {ZONE_WEIGHTS[1]}x  |  Straight weight: {ZONE_WEIGHTS[0]}x\n")
        f.write('=' * 60 + '\n\n')
        f.write(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)\n\n")
        f.write('-' * 60 + '\n')
        for r in all_results:
            status = 'OK   ' if r['is_correct'] else 'WRONG'
            f.write(f"[{status}] {r['driver_id']:15s} -> {r['predicted_driver']:15s} "
                    f"({r['agreement']:.1f}% weighted agreement, "
                    f"{r['n_segments']} segs, conf {r['avg_confidence']:.1f}%)\n")

    print(f"\nSummary:          {summary_file.name}")
    print(f"Per-driver files: {results_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Predict driver identity from telemetry data"
    )
    parser.add_argument(
        'file',
        nargs='?',
        type=str,
        help='Pfad zur Telemetrie-Datei (.ld oder .htf). '
             'Ohne Datei: Auswertung aller Fahrer auf Test-Runden 6 & 7.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'svm', 'xgboost'],
        help='Modell (default: random_forest)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Konfidenz-Schwelle (default: 0.6)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Nur Test-Runden (6+7) verwenden'
    )

    args = parser.parse_args()

    # No file given → batch evaluation over all drivers' test rounds
    if args.file is None:
        run_test_round_evaluation(model_name=args.model)
        return

    # Single-file prediction (existing behaviour)
    models_dir = get_models_path() / "combined"
    predictor  = DriverPredictor(model_name=args.model)
    predictor.test_rounds_only = args.test_only

    result = predictor.predict_from_file(
        args.file,
        confidence_threshold=args.confidence_threshold
    )
    predictor.print_prediction_result(result)

    results_dir = get_results_path()
    result_file = results_dir / f"prediction_{Path(args.file).stem}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Prediction Result\n")
        f.write('=' * 60 + '\n\n')
        f.write(f"Input File: {args.file}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Predicted Driver: {result.get('predicted_driver', 'N/A')}\n")
        f.write(f"Status: {'KNOWN' if result.get('is_known', False) else 'UNKNOWN'}\n")
        f.write(f"Confidence: {result.get('confidence', 0):.1%}\n")
        f.write(f"Vote Percentage: {result.get('vote_percentage', 0):.1f}%\n")
        f.write("\nVote Distribution:\n")
        for driver, count in result.get('all_predictions', {}).items():
            f.write(f"  {driver}: {count} votes\n")
    print(f"Result saved to: {result_file}")


if __name__ == "__main__":
    main()
