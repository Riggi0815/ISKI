"""
Leave-One-Driver-Out Evaluation
================================
Trainiert Modell ohne einen Fahrer und testet dann mit diesem unbekannten Fahrer
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_processed_data_path, get_features_path, get_models_path,
    get_results_path, load_dataframe, save_dataframe
)


class LeaveOneOutEvaluator:
    """Leave-One-Driver-Out Cross-Validation"""
    
    def __init__(self, holdout_driver=None):
        self.project_root = get_project_root()
        self.features_path = get_features_path()
        self.models_path = get_models_path() / "leave_one_out"
        self.results_path = get_results_path() / "leave_one_out"
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.holdout_driver = holdout_driver
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        
    def load_and_split_data(self):
        """Lade Features und splitte nach Fahrer"""
        print(f"{'='*70}")
        print("LOADING DATA")
        print(f"{'='*70}\n")
        
        # Load combined features from features/ directory
        df = load_dataframe(self.features_path / "driver_features_combined")
        
        if df is None:
            raise FileNotFoundError("driver_features_combined.pkl not found! Run 03b first.")
        
        print(f"Total features: {len(df)}")
        print(f"Drivers: {df['driver_id'].nunique()}")
        print(f"\nDriver distribution:")
        for driver in sorted(df['driver_id'].unique()):
            count = len(df[df['driver_id'] == driver])
            print(f"  {driver}: {count} feature sets")
        
        # Select holdout driver if not specified
        if self.holdout_driver is None:
            # Use smallest dataset to minimize training loss
            driver_counts = df['driver_id'].value_counts()
            self.holdout_driver = driver_counts.idxmin()
            print(f"\n💡 Auto-selected holdout driver: {self.holdout_driver} ({driver_counts.min()} samples)")
        
        if self.holdout_driver not in df['driver_id'].values:
            available = sorted(df['driver_id'].unique())
            raise ValueError(f"Driver {self.holdout_driver} not found! Available: {available}")
        
        # Split data
        self.holdout_data = df[df['driver_id'] == self.holdout_driver].copy()
        self.training_data = df[df['driver_id'] != self.holdout_driver].copy()
        
        print(f"\n{'='*70}")
        print(f"SPLIT: HOLDOUT = {self.holdout_driver}")
        print(f"{'='*70}")
        print(f"Training: {len(self.training_data)} samples from {self.training_data['driver_id'].nunique()} drivers")
        print(f"Holdout:  {len(self.holdout_data)} samples (UNSEEN during training)")
        
        return self.training_data, self.holdout_data
    
    def train_models(self):
        """Trainiere Modelle ohne holdout driver"""
        print(f"\n{'='*70}")
        print("TRAINING MODELS (WITHOUT HOLDOUT DRIVER)")
        print(f"{'='*70}\n")
        
        # Prepare features
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['driver_id', 'segment_id']]
        
        X_train = self.training_data[feature_cols].values
        y_train = self.training_data['driver_id'].values
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Classes: {len(np.unique(y_train))}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train_encoded, 
            test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        print(f"\nTrain set: {len(X_tr)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train Random Forest
        print(f"\n{'─'*70}")
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_tr, y_tr)
        val_acc_rf = accuracy_score(y_val, rf.predict(X_val))
        print(f"✓ Validation Accuracy: {val_acc_rf*100:.2f}%")
        self.models['random_forest'] = rf
        
        # Train XGBoost
        print(f"\n{'─'*70}")
        print("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_tr, y_tr)
        val_acc_xgb = accuracy_score(y_val, xgb.predict(X_val))
        print(f"✓ Validation Accuracy: {val_acc_xgb*100:.2f}%")
        self.models['xgboost'] = xgb
        
        # Train SVM (smaller for speed)
        print(f"\n{'─'*70}")
        print("Training SVM (Linear)...")
        svm = SVC(
            kernel='linear',
            C=1.0,
            random_state=42,
            verbose=0
        )
        svm.fit(X_tr, y_tr)
        val_acc_svm = accuracy_score(y_val, svm.predict(X_val))
        print(f"✓ Validation Accuracy: {val_acc_svm*100:.2f}%")
        self.models['svm'] = svm
        
        print(f"\n{'='*70}")
        print("✓ ALL MODELS TRAINED")
        print(f"{'='*70}")
    
    def evaluate_on_holdout(self):
        """Evaluiere auf unbekanntem Fahrer"""
        print(f"\n{'='*70}")
        print(f"EVALUATING ON UNSEEN DRIVER: {self.holdout_driver}")
        print(f"{'='*70}\n")
        
        # Prepare holdout data
        feature_cols = [col for col in self.holdout_data.columns 
                       if col not in ['driver_id', 'segment_id']]
        
        X_holdout = self.holdout_data[feature_cols].values
        y_holdout = self.holdout_data['driver_id'].values  # True label (but unseen)
        
        X_holdout_scaled = self.scaler.transform(X_holdout)
        
        print(f"Holdout data: {len(X_holdout)} samples")
        print(f"True driver: {self.holdout_driver}")
        print(f"\n⚠ This driver was NEVER seen during training!")
        print(f"   The model cannot correctly identify them.")
        print(f"   We measure: Which known drivers is this driver confused with?\n")
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'─'*70}")
            print(f"Evaluating {model_name.upper()}...")
            
            # Predict (will be one of the KNOWN drivers)
            y_pred = model.predict(X_holdout_scaled)
            
            # Decode predictions to driver names
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_holdout_scaled)
                # Average confidence across all predictions
                avg_confidence = np.mean(np.max(y_proba, axis=1))
                print(f"✓ Average Prediction Confidence: {avg_confidence*100:.1f}%")
                print(f"  (Low confidence = model is 'uncertain' = GOOD for unknown driver)")
            
            # Since true label is unknown, we can't measure "accuracy"
            # Instead, show distribution of predictions
            unique, counts = np.unique(y_pred_labels, return_counts=True)
            
            print(f"\n  Predicted as (confused with):")
            for driver, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
                pct = count / len(y_pred_labels) * 100
                print(f"    {driver}: {count:>4} samples ({pct:5.1f}%)")
            
            # Most common prediction
            most_common_driver = unique[np.argmax(counts)]
            most_common_pct = np.max(counts) / len(y_pred_labels) * 100
            
            print(f"\n  Most confused with: {most_common_driver} ({most_common_pct:.1f}%)")
            
            results[model_name] = {
                'predictions': y_pred_labels,
                'true_driver': self.holdout_driver,
                'most_confused': most_common_driver,
                'confusion_pct': most_common_pct,
                'avg_confidence': avg_confidence if hasattr(model, 'predict_proba') else None
            }
        
        self.results = results
        return results
    
    def evaluate_known_drivers(self):
        """Evaluiere auch auf den Trainings-Fahrern (sollte gut sein)"""
        print(f"\n{'='*70}")
        print("EVALUATING ON KNOWN DRIVERS (Sanity Check)")
        print(f"{'='*70}\n")
        
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['driver_id', 'segment_id']]
        
        X_train = self.training_data[feature_cols].values
        y_train = self.training_data['driver_id'].values
        
        X_train_scaled = self.scaler.transform(X_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        
        print(f"Known drivers: {len(X_train)} samples")
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_train_scaled)
            accuracy = accuracy_score(y_train_encoded, y_pred)
            print(f"{model_name}: {accuracy*100:.2f}% (should be high)")
    
    def save_results(self):
        """Speichere Modelle und Results"""
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}\n")
        
        # Save models
        for model_name, model in self.models.items():
            model_file = self.models_path / f"{model_name}.pkl"
            joblib.dump(model, model_file)
            print(f"✓ Saved {model_file.name}")
        
        # Save scaler and encoder
        joblib.dump(self.scaler, self.models_path / "scaler.pkl")
        joblib.dump(self.label_encoder, self.models_path / "label_encoder.pkl")
        
        # Save metadata
        metadata = {
            'holdout_driver': self.holdout_driver,
            'training_drivers': list(self.training_data['driver_id'].unique()),
            'n_training_samples': len(self.training_data),
            'n_holdout_samples': len(self.holdout_data),
            'n_features': len([col for col in self.training_data.columns 
                             if col not in ['driver_id', 'segment_id']]),
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(self.models_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata.json")
        
        # Create report
        self.create_report()
    
    def create_report(self):
        """Erstelle ausführlichen Report"""
        report_file = self.results_path / f"evaluation_{self.holdout_driver}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LEAVE-ONE-DRIVER-OUT EVALUATION\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Holdout Driver: {self.holdout_driver} (UNSEEN during training)\n\n")
            
            f.write("="*70 + "\n")
            f.write("DATA SPLIT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Training Drivers ({self.training_data['driver_id'].nunique()}):\n")
            for driver in sorted(self.training_data['driver_id'].unique()):
                count = len(self.training_data[self.training_data['driver_id'] == driver])
                f.write(f"  {driver}: {count} samples\n")
            
            f.write(f"\nHoldout Driver:\n")
            f.write(f"  {self.holdout_driver}: {len(self.holdout_data)} samples (NEVER seen during training)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RESULTS ON UNSEEN DRIVER (Open-Set Recognition)\n")
            f.write("="*70 + "\n\n")
            
            f.write("⚠ IMPORTANT: The holdout driver was NEVER seen during training.\n")
            f.write("   The model CANNOT correctly identify them.\n")
            f.write("   We measure: Which KNOWN drivers is the UNKNOWN driver confused with?\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"{model_name.upper()}:\n")
                
                if result.get('avg_confidence'):
                    f.write(f"  Average Confidence: {result['avg_confidence']*100:.1f}%\n")
                    f.write(f"  (Low = model uncertain = good for unknown driver)\n\n")
                
                f.write(f"  Most Confused With: {result['most_confused']} ({result['confusion_pct']:.1f}%)\n")
                
                # Prediction distribution
                unique, counts = np.unique(result['predictions'], return_counts=True)
                f.write(f"\n  Full Distribution:\n")
                for driver, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
                    pct = count / len(result['predictions']) * 100
                    f.write(f"    {driver}: {count:>4} ({pct:5.1f}%)\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*70 + "\n\n")
            
            # Best model = lowest average confidence (ignoring None values)
            valid_results = {k: v for k, v in self.results.items() if v.get('avg_confidence') is not None}
            if valid_results:
                best_model = min(valid_results.items(), 
                               key=lambda x: x[1]['avg_confidence'])
            else:
                # Fallback: use first model
                best_model = list(self.results.items())[0]
            best_name, best_result = best_model
            
            f.write(f"Most Uncertain Model: {best_name.upper()}\n")
            if best_result.get('avg_confidence'):
                f.write(f"  Average Confidence: {best_result['avg_confidence']*100:.1f}%\n\n")
            
            f.write("What does this mean?\n\n")
            
            avg_conf = best_result.get('avg_confidence', 1.0)
            if avg_conf < 0.5:
                f.write("✓ EXCELLENT: Model shows high uncertainty for unknown driver!\n")
                f.write("  This is GOOD - it recognizes the driver is different from training.\n")
                f.write("  Low confidence suggests the model could detect 'outliers'.\n\n")
            elif avg_conf < 0.7:
                f.write("✓ GOOD: Model shows moderate uncertainty.\n")
                f.write("  The unknown driver is somewhat similar to known drivers.\n\n")
            else:
                f.write("⚠ HIGH CONFIDENCE: Model is very confident in wrong predictions.\n")
                f.write(f"  The unknown driver's style is very similar to: {best_result['most_confused']}\n")
                f.write("  This suggests similar driving characteristics.\n\n")
            
            # Most confused drivers
            f.write("Confusion Analysis:\n")
            confusion_pct = best_result['confusion_pct']
            
            if confusion_pct > 70:
                f.write(f"  Strongly confused with {best_result['most_confused']} ({confusion_pct:.1f}%)\n")
                f.write(f"  → These drivers likely have very similar driving styles\n")
            elif confusion_pct > 40:
                f.write(f"  Primarily confused with {best_result['most_confused']} ({confusion_pct:.1f}%)\n")
                f.write(f"  → Some similarity, but also distributed across others\n")
            else:
                f.write(f"  Distributed confusion (max {confusion_pct:.1f}%)\n")
                f.write(f"  → Unknown driver doesn't match any single known driver well\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            if avg_conf > 0.7:
                f.write("1. Unknown driver style is similar to known drivers\n")
                f.write(f"   → Especially {best_result['most_confused']}\n")
                f.write("2. For better unknown detection, consider:\n")
                f.write("   - One-class SVM or Isolation Forest for outlier detection\n")
                f.write("   - Confidence thresholding (reject predictions < threshold)\n")
                f.write("   - Deep metric learning (face recognition style)\n\n")
            else:
                f.write("Model shows promise for detecting unknown drivers!\n")
                f.write("1. Use confidence scores to reject uncertain predictions\n")
                f.write("2. Set threshold (e.g., reject if confidence < 60%)\n")
                f.write("3. Consider this for 'new driver' detection\n\n")
            
            f.write("Scientific Context:\n")
            f.write("This is an 'Open-Set Recognition' problem:\n")
            f.write("- Closed-Set: All test classes seen in training (normal classification)\n")
            f.write("- Open-Set: Test samples may be from UNKNOWN classes\n")
            f.write("- Real-world scenario: New drivers not in training set\n")
            f.write("- Solution: Use confidence/probability thresholds or specialized models\n")
        
        print(f"✓ Saved report: {report_file.name}")
        
        # Create confusion visualization for best model
        self.plot_confusion_matrix(best_name, best_result)
    
    def plot_confusion_matrix(self, model_name, result):
        """Plot confusion distribution for unseen driver"""
        print(f"\nCreating prediction distribution plot for {model_name}...")
        
        # Get predictions (which known drivers was holdout confused with)
        y_pred = result['predictions']
        
        # Count predictions
        from collections import Counter
        pred_counts = Counter(y_pred)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        drivers = sorted(pred_counts.keys())
        counts = [pred_counts[d] for d in drivers]
        
        plt.bar(range(len(drivers)), counts, color='steelblue', alpha=0.7)
        plt.xticks(range(len(drivers)), drivers, rotation=45, ha='right')
        plt.ylabel('Number of Predictions')
        plt.xlabel('Predicted Driver (Known)')
        plt.title(f'Prediction Distribution - {model_name.upper()}\n(Unseen Driver: {self.holdout_driver})')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = len(y_pred)
        for i, count in enumerate(counts):
            pct = count / total * 100
            plt.text(i, count + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        plot_file = self.results_path / f"prediction_distribution_{model_name}_{self.holdout_driver}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {plot_file.name}")
    
    def run(self):
        """Complete evaluation workflow"""
        print(f"\n{'#'*70}")
        print("# LEAVE-ONE-DRIVER-OUT EVALUATION")
        print(f"{'#'*70}\n")
        
        # Load and split
        self.load_and_split_data()
        
        # Train
        self.train_models()
        
        # Evaluate on unseen
        self.evaluate_on_holdout()
        
        # Sanity check on known drivers
        self.evaluate_known_drivers()
        
        # Save everything
        self.save_results()
        
        print(f"\n{'#'*70}")
        print("✓ EVALUATION COMPLETE")
        print(f"{'#'*70}")
        print(f"\nResults saved to: {self.results_path}")
        print(f"Models saved to: {self.models_path}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Leave-One-Driver-Out Evaluation"
    )
    parser.add_argument(
        '--driver',
        type=str,
        default=None,
        help='Driver to hold out (default: auto-select smallest dataset)'
    )
    
    args = parser.parse_args()
    
    evaluator = LeaveOneOutEvaluator(holdout_driver=args.driver)
    evaluator.run()


if __name__ == "__main__":
    main()
