"""
ML Model Training for Driver Identification
Trains Random Forest, SVM, and XGBoost classifiers
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import json
import sys

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_dataframe,
    get_models_path,
    get_results_path,
    save_json
)


class ModelTrainer:
    """Train and evaluate ML models for driver identification"""
    
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        
    def prepare_data(self, test_size=0.3, random_state=42):
        """
        Prepare features and labels for training
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
        """
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Separate features and labels
        self.X = self.features_df.drop(['driver_id', 'segment_idx'], axis=1, errors='ignore')
        self.y = self.features_df['driver_id']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Unique drivers: {self.y.nunique()}")
        print(f"Driver distribution:\n{self.y.value_counts()}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"\nLabel encoding:")
        for i, driver in enumerate(self.label_encoder.classes_):
            print(f"  {driver} -> {i}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded  # Maintain class distribution
        )
        
        print(f"\nTrain set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Scale features (important for SVM)
        print("\nScaling features...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✓ Data preparation complete")
        
    def train_random_forest(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Train Random Forest classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        print("\n" + "="*60)
        print("TRAINING: Random Forest")
        print("="*60)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        
        rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = rf_model.score(self.X_train, self.y_train)
        test_acc = rf_model.score(self.X_test, self.y_test)
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=3, n_jobs=-1)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = rf_model.predict(self.X_test)
        
        # Store results
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'predictions': y_pred.tolist(),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features_df.drop(['driver_id', 'segment_idx'], axis=1, errors='ignore').columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.results['random_forest']['top_features'] = feature_importance.head(20).to_dict('records')
        
        print("\n✓ Random Forest training complete")
        
    def train_svm(self, C=1.0, kernel='rbf', random_state=42):
        """
        Train SVM classifier
        
        Args:
            C: Regularization parameter
            kernel: Kernel type
            random_state: Random seed
        """
        print("\n" + "="*60)
        print("TRAINING: Support Vector Machine")
        print("="*60)
        
        # Train model
        svm_model = SVC(
            C=C,
            kernel=kernel,
            random_state=random_state,
            verbose=True
        )
        
        print(f"Parameters: C={C}, kernel={kernel}")
        
        svm_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = svm_model.score(self.X_train, self.y_train)
        test_acc = svm_model.score(self.X_test, self.y_test)
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Cross-validation (smaller CV for speed)
        print("\nRunning cross-validation...")
        cv_scores = cross_val_score(svm_model, self.X_train, self.y_train, cv=2)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = svm_model.predict(self.X_test)
        
        # Store results
        self.models['svm'] = svm_model
        self.results['svm'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'predictions': y_pred.tolist(),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        print("\n✓ SVM training complete")
        
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Train XGBoost classifier
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
        """
        print("\n" + "="*60)
        print("TRAINING: XGBoost")
        print("="*60)
        
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=1,
            eval_metric='mlogloss'
        )
        
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
        
        xgb_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = xgb_model.score(self.X_train, self.y_train)
        test_acc = xgb_model.score(self.X_test, self.y_test)
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=3, n_jobs=-1)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = xgb_model.predict(self.X_test)
        
        # Store results
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'predictions': y_pred.tolist(),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features_df.drop(['driver_id', 'segment_idx'], axis=1, errors='ignore').columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.results['xgboost']['top_features'] = feature_importance.head(20).to_dict('records')
        
        print("\n✓ XGBoost training complete")
        
    def save_models(self):
        """Save trained models and metadata"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        models_path = get_models_path()
        models_path.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            model_file = models_path / f"{model_name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"✓ Saved: {model_file}")
        
        # Save scaler and encoder
        joblib.dump(self.scaler, models_path / "scaler.pkl")
        joblib.dump(self.label_encoder, models_path / "label_encoder.pkl")
        print(f"✓ Saved: scaler.pkl and label_encoder.pkl")
        
        # Save results
        results_path = get_results_path()
        results_file = results_path / "04_training_results.json"
        
        # Add metadata
        self.results['metadata'] = {
            'n_features': self.X.shape[1],
            'n_drivers': len(self.label_encoder.classes_),
            'drivers': self.label_encoder.classes_.tolist(),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'feature_names': self.features_df.drop(['driver_id', 'segment_idx'], axis=1, errors='ignore').columns.tolist()
        }
        
        save_json(self.results, "training_results", directory="results")
        print(f"✓ Saved: {results_file}")
        
        # Create comparison report
        self._create_comparison_report()
        
    def _create_comparison_report(self):
        """Create model comparison report"""
        results_path = get_results_path()
        report_file = results_path / "04_model_comparison.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset:\n")
            f.write(f"  Train samples: {len(self.X_train)}\n")
            f.write(f"  Test samples: {len(self.X_test)}\n")
            f.write(f"  Features: {self.X.shape[1]}\n")
            f.write(f"  Drivers: {len(self.label_encoder.classes_)}\n\n")
            
            f.write("Model Performance:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12}\n")
            f.write("-"*70 + "\n")
            
            for model_name in ['random_forest', 'svm', 'xgboost']:
                if model_name in self.results:
                    res = self.results[model_name]
                    f.write(f"{model_name:<20} {res['train_accuracy']:<12.4f} "
                           f"{res['test_accuracy']:<12.4f} {res['cv_mean']:<12.4f}\n")
            
            f.write("-"*70 + "\n\n")
            
            # Best model
            best_model = max(self.results.keys(), 
                           key=lambda k: self.results[k].get('test_accuracy', 0) if k != 'metadata' else 0)
            
            if best_model != 'metadata':
                f.write(f"Best Model: {best_model}\n")
                f.write(f"Test Accuracy: {self.results[best_model]['test_accuracy']:.4f}\n\n")
            
            # Confusion matrices
            for model_name in ['random_forest', 'svm', 'xgboost']:
                if model_name in self.results:
                    f.write(f"\n{model_name.upper()} - Confusion Matrix:\n")
                    cm = np.array(self.results[model_name]['confusion_matrix'])
                    f.write(str(cm) + "\n")
        
        print(f"✓ Saved: {report_file}")


def main():
    """Main execution"""
    print("="*70)
    print("ML MODEL TRAINING - Driver Identification")
    print("="*70)
    
    # Load features
    print("\nLoading features...")
    features_df = load_dataframe("driver_features", directory="features")
    
    if features_df.empty:
        print("ERROR: No features found. Run 03_feature_engineering.py first.")
        return
    
    print(f"✓ Loaded features for {len(features_df)} drivers")
    print(f"  Features per driver: {len(features_df.columns) - 1}")
    
    # Initialize trainer
    trainer = ModelTrainer(features_df)
    
    # Prepare data
    trainer.prepare_data(test_size=0.3, random_state=42)
    
    # Train all models
    trainer.train_random_forest(n_estimators=100, max_depth=10)
    trainer.train_svm(C=10.0, kernel='rbf')
    trainer.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
    
    # Save everything
    trainer.save_models()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved in: models/")
    print(f"Results saved in: results/")
    print()


if __name__ == "__main__":
    main()
