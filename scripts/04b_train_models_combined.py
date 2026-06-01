"""
Step 4b: Train Models on Combined HTF+LD Data
Train ML models using unified dataset with all drivers
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_features_path, get_models_path, 
    get_results_path, load_dataframe
)


class ModelTrainer:
    """Train and evaluate ML models for driver identification"""
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize trainer with features dataframe
        
        Args:
            features_df: DataFrame with features and driver_id column
        """
        self.features_df = features_df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, test_size=0.3, random_state=42):
        """Prepare data for training"""
        print(f"\n{'='*60}")
        print("PREPARING DATA")
        print(f"{'='*60}\n")
        
        # Print all columns to debug
        print(f"All columns: {list(self.features_df.columns)[:10]}...")  # First 10 columns
        
        # Separate features and labels - drop ALL non-numeric columns
        self.y = self.features_df['driver_id']
        
        # Drop non-feature columns (driver_id and any index-like columns)
        columns_to_drop = []
        for col in self.features_df.columns:
            if col == 'driver_id' or col.endswith('_index') or 'sample_index' in col:
                columns_to_drop.append(col)
            # Also check if column contains string values
            elif self.features_df[col].dtype == 'object':
                columns_to_drop.append(col)
        
        print(f"Dropping columns: {columns_to_drop}")
        self.X = self.features_df.drop(columns_to_drop, axis=1)
        
        print(f"Features shape: {self.X.shape}")
        print(f"Number of features: {self.X.shape[1]}")
        print(f"Number of samples: {len(self.X)}")
        print(f"Number of drivers: {self.y.nunique()}")
        
        # Encode labels
        self.y = self.label_encoder.fit_transform(self.y)
        
        # Train/test split (stratified by driver)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTrain set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        print(f"Split: {(1-test_size)*100:.0f}% train / {test_size*100:.0f}% test")
        
    def train_random_forest(self, n_estimators=100, max_depth=10, random_state=42):
        """Train Random Forest classifier"""
        print(f"\n{'='*60}")
        print("TRAINING: Random Forest")
        print(f"{'='*60}")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}\n")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        rf.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = rf.predict(self.X_train)
        y_test_pred = rf.predict(self.X_test)
        
        # Metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        print(f"✓ Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"✓ Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth}
        }
        
    def train_svm(self, C=1.0, kernel='rbf', random_state=42):
        """Train SVM classifier"""
        print(f"\n{'='*60}")
        print("TRAINING: Support Vector Machine (SVM)")
        print(f"{'='*60}")
        print(f"Parameters: C={C}, kernel={kernel}\n")
        
        svm = SVC(
            C=C,
            kernel=kernel,
            random_state=random_state
        )
        
        svm.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = svm.predict(self.X_train)
        y_test_pred = svm.predict(self.X_test)
        
        # Metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        print(f"✓ Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"✓ Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        self.models['svm'] = svm
        self.results['svm'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'params': {'C': C, 'kernel': kernel}
        }
        
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """Train XGBoost classifier"""
        print(f"\n{'='*60}")
        print("TRAINING: XGBoost")
        print(f"{'='*60}")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}\n")
        
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
        
        xgb.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = xgb.predict(self.X_train)
        y_test_pred = xgb.predict(self.X_test)
        
        # Metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        print(f"✓ Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"✓ Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        self.models['xgboost'] = xgb
        self.results['xgboost'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        }
        
    def _create_comparison_report(self):
        """Create model comparison report"""
        results_path = get_results_path()
        report_file = results_path / "04_model_comparison.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*50 + "\n")
                f.write(f"Train Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)\n")
                f.write(f"Test Accuracy:  {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)\n")
                f.write(f"Parameters: {metrics['params']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("SUMMARY\n")
            f.write(f"{'='*70}\n")
            
            best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
            f.write(f"\nBest Model: {best_model[0].upper()}\n")
            f.write(f"Test Accuracy: {best_model[1]['test_accuracy']:.4f} ({best_model[1]['test_accuracy']*100:.2f}%)\n")


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
