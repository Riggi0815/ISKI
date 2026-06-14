"""
Step 4b: Train Random Forest on Combined HTF+LD Data
Driver identification: predict which of the 5 drivers is at the wheel.

Train/Test Split:
  - Training: Rounds 1, 2, 3, 4, 5, 8  (inferred from sequential segment order)
  - Test:     Rounds 6, 7

Evaluation:
  - Accuracy
  - Confusion Matrix
  - Feature Importance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import json

sys.path.append(str(Path(__file__).parent))
from utils import get_project_root, get_features_path, get_models_path, get_results_path


# ---------------------------------------------------------------------------
# Feature columns used for training
# Mapped to actual column names present in driver_features_combined
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # Zone context — one-hot booleans (dominant zone in the 10s window)
    'is_straight', 'is_eingang', 'is_mitte', 'is_apex',

    # Vehicle speed before / in / after corner
    'v_car_mean', 'v_car_std', 'v_car_min', 'v_car_max',
    # Speed variation (how much the driver varies speed through a segment)
    'speed_cv',

    # Lateral G-force (Querbeschleunigung)
    'g_lat_mean', 'g_lat_std', 'g_lat_min', 'g_lat_max',
    'g_lat_extreme_pct',    # fraction of time with >1G lateral
    'corner_count',          # how many corners per segment

    # Longitudinal G / Brake behaviour (wann und wie hart gebremst wird)
    'g_long_mean', 'g_long_std', 'g_long_min', 'g_long_max',
    'jerk_mean', 'jerk_max',   # rate of change of longitudinal G

    # Throttle application (aggressiv vs. smooth aus der Kurve)
    'percent_throttle_mean', 'percent_throttle_std',
    'throttle_changes',      # number of throttle transitions
    'throttle_smoothness',   # std of throttle change rate
    'throttle_aggressive',   # large, rapid throttle inputs

    # Engine RPM behaviour (jeder Fahrer schaltet bei anderen RPM)
    'n_engine_mean', 'n_engine_std', 'n_engine_max',

    # Gear-ratio proxy: speed / RPM → encodes driving-line and shift style
    'gear_ratio_mean', 'gear_ratio_std',

    # Tyre temperature differential (front vs rear — driving style signature)
    'tire_temp_diff_fr',
    # Tyre pressure differential (left vs right — cornering asymmetry)
    'tire_pressure_diff_lr',
]


def split_by_rounds(
    features_df: pd.DataFrame,
    train_rounds: list = [1, 2, 3, 4, 5, 8],
    test_rounds: list = [6, 7],
    n_rounds: int = 8,
) -> tuple:
    """
    Divide each driver's segments into n_rounds equal-sized groups,
    then assign groups to train or test according to round numbers.

    Round numbering is 1-based; segments are ordered sequentially as
    recorded (the order already present in the DataFrame for each driver).

    Returns:
        (train_indices, test_indices) — list of DataFrame index values
    """
    train_idx, test_idx = [], []

    for driver_id in features_df['driver_id'].unique():
        mask = features_df['driver_id'] == driver_id
        driver_idx = features_df.index[mask].tolist()
        n_segs = len(driver_idx)

        segs_per_round = max(1, n_segs // n_rounds)

        for pos, idx in enumerate(driver_idx):
            # Clamp to n_rounds so the tail goes into round 8
            round_num = min(pos // segs_per_round + 1, n_rounds)
            if round_num in train_rounds:
                train_idx.append(idx)
            else:
                test_idx.append(idx)

    return train_idx, test_idx


def prepare_features(features_df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Select and clean the feature matrix.
    Returns (X, valid_feature_cols) — only columns that actually exist.
    Values are clipped to float32-safe range to avoid sklearn overflow errors.
    """
    available = [c for c in feature_cols if c in features_df.columns]
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        print(f"  Note: {len(missing)} requested features not in data: {missing}")
    X = features_df[available].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Clip to float32 range so sklearn doesn't overflow during tree building
    float32_max = np.finfo(np.float32).max * 0.9
    X = X.clip(lower=-float32_max, upper=float32_max)
    return X, available


def train_random_forest(X_train, y_train, n_estimators: int = 200, random_state: int = 42,
                        sample_weight=None):
    """Train Random Forest — no scaling needed (tree-based model)."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,        # let trees grow fully for expressive power
        min_samples_leaf=2,    # slight regularisation against overfitting
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    return rf


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: Path):
    """Save a labelled confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix - Random Forest\n(Test: Rounds 6 & 7)', fontsize=13)
    # Rotate x-axis labels 90 degrees, right-aligned so they don't overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_feature_importance(rf_model, feature_names: list, save_path: Path, top_n: int = 20):
    """Save a horizontal bar chart of the top-N most important features."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_importances[::-1], color='steelblue', edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)')
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest\n(Which features best identify the driver?)',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

    return list(zip(top_features, top_importances))


def save_text_report(report_lines: list, save_path: Path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {save_path.name}")


def main():
    project_root = get_project_root()
    features_path = get_features_path()
    models_path = get_models_path()
    results_path = get_results_path()

    print('=' * 70)
    print('RANDOM FOREST — Driver Identification (Combined HTF+LD)')
    print('Train: Rounds 1,2,3,4,5,8  |  Test: Rounds 6,7')
    print('=' * 70)

    # -----------------------------------------------------------------------
    # 1. Load features
    # -----------------------------------------------------------------------
    print('\n[1] Loading features...')
    features_file = features_path / 'driver_features_combined'

    pkl = features_path / 'driver_features_combined.pkl'
    csv = features_path / 'driver_features_combined.csv'
    if pkl.exists():
        features_df = pd.read_pickle(pkl)
        print(f'  Loaded from pickle: {pkl.name}')
    elif csv.exists():
        features_df = pd.read_csv(csv)
        print(f'  Loaded from CSV: {csv.name}')
    else:
        print('ERROR: No combined feature file found.')
        print('Run: py -3 scripts\\03b_feature_engineering_combined.py first')
        return

    print(f'  Rows: {len(features_df):,}  |  Columns: {len(features_df.columns)}')
    print(f'  Drivers ({features_df["driver_id"].nunique()}): {sorted(features_df["driver_id"].unique())}')
    print()
    for drv, cnt in features_df['driver_id'].value_counts().items():
        print(f'    {drv}: {cnt} segments')

    # -----------------------------------------------------------------------
    # 2. Round-based train / test split
    # -----------------------------------------------------------------------
    print('\n[2] Splitting by rounds (8 rounds per driver)...')
    TRAIN_ROUNDS = [1, 2, 3, 4, 5, 8]
    TEST_ROUNDS  = [6, 7]

    train_idx, test_idx = split_by_rounds(
        features_df,
        train_rounds=TRAIN_ROUNDS,
        test_rounds=TEST_ROUNDS,
        n_rounds=8,
    )

    train_df = features_df.loc[train_idx]
    test_df  = features_df.loc[test_idx]

    print(f'  Train segments: {len(train_df):,}  ({len(train_df)/len(features_df)*100:.1f}%)')
    print(f'  Test  segments: {len(test_df):,}  ({len(test_df)/len(features_df)*100:.1f}%)')

    # Per-driver breakdown
    print()
    for drv in sorted(features_df['driver_id'].unique()):
        n_tr = (train_df['driver_id'] == drv).sum()
        n_te = (test_df['driver_id'] == drv).sum()
        print(f'    {drv}: train={n_tr}  test={n_te}')

    # -----------------------------------------------------------------------
    # 3. Prepare feature matrices
    # -----------------------------------------------------------------------
    print('\n[3] Preparing features...')
    X_train, valid_features = prepare_features(train_df, FEATURE_COLUMNS)
    X_test,  _              = prepare_features(test_df,  valid_features)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['driver_id'])
    y_test  = le.transform(test_df['driver_id'])
    class_names = list(le.classes_)

    # Fit scaler on training data (needed for test evaluation script)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    print(f'  Using {len(valid_features)} features')
    print(f'  Classes: {class_names}')

    # -----------------------------------------------------------------------
    # 4. Train Random Forest
    # -----------------------------------------------------------------------
    print('\n[4] Training Random Forest (n_estimators=200)...')
    rf = train_random_forest(X_train.values, y_train, n_estimators=200)

    y_train_pred = rf.predict(X_train.values)
    y_test_pred  = rf.predict(X_test.values)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)

    print(f'  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)')
    print(f'  Test  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

    # -----------------------------------------------------------------------
    # 5. Detailed evaluation
    # -----------------------------------------------------------------------
    print('\n[5] Evaluation:')
    clf_report = classification_report(y_test, y_test_pred, target_names=class_names)
    print(clf_report)

    # -----------------------------------------------------------------------
    # 6. Save plots
    # -----------------------------------------------------------------------
    print('\n[6] Saving plots...')
    results_path.mkdir(exist_ok=True)

    # Confusion Matrix
    cm_path = results_path / 'confusion_matrix_rf.png'
    plot_confusion_matrix(y_test, y_test_pred, class_names, cm_path)

    # Feature Importance
    fi_path = results_path / 'feature_importance_rf.png'
    top_features = plot_feature_importance(rf, valid_features, fi_path, top_n=20)

    # -----------------------------------------------------------------------
    # 7. Save model
    # -----------------------------------------------------------------------
    print('\n[7] Saving model...')
    combined_models_path = models_path / 'combined'
    combined_models_path.mkdir(exist_ok=True)

    joblib.dump(rf, combined_models_path / 'random_forest_model.pkl')
    joblib.dump(le, combined_models_path / 'label_encoder.pkl')
    joblib.dump(scaler, combined_models_path / 'scaler.pkl')

    metadata = {
        'feature_names': valid_features,
        'driver_names': class_names,
        'n_features': len(valid_features),
        'n_drivers': len(class_names),
        'data_source': 'HTF + LD combined',
        'train_rounds': TRAIN_ROUNDS,
        'test_rounds': TEST_ROUNDS,
        'train_accuracy': round(train_acc, 4),
        'test_accuracy': round(test_acc, 4),
    }
    with open(combined_models_path / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  Saved model + metadata to: {combined_models_path}')

    # -----------------------------------------------------------------------
    # 8. Text report
    # -----------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_test_pred)
    report_lines = [
        '=' * 70,
        'RANDOM FOREST — Driver Identification Report',
        '=' * 70,
        '',
        f'Train/Test Split:',
        f'  Training rounds : {TRAIN_ROUNDS}',
        f'  Test rounds     : {TEST_ROUNDS}',
        f'  Train segments  : {len(train_df):,}',
        f'  Test  segments  : {len(test_df):,}',
        '',
        f'Model: RandomForestClassifier (n_estimators=200, balanced class weights)',
        f'Features: {len(valid_features)}',
        '',
        f'Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)',
        f'Test  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)',
        '',
        'Classification Report (Test Set):',
        '-' * 50,
        clf_report,
        '',
        'Confusion Matrix (Test Set):',
        '  Rows = True Driver, Columns = Predicted Driver',
        f'  Classes: {class_names}',
        '',
        str(cm),
        '',
        'Top 20 Feature Importances:',
        '-' * 50,
    ]
    for i, (feat, imp) in enumerate(top_features, 1):
        report_lines.append(f'  {i:2d}. {feat:<45s} {imp:.4f}')

    report_lines += ['', '=' * 70]

    report_path = results_path / '04b_rf_driver_identification_report.txt'
    save_text_report(report_lines, report_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print('=' * 70)
    print('DONE')
    print('=' * 70)
    print(f'Test Accuracy  : {test_acc:.2%}')
    print(f'Confusion Matrix   -> {cm_path.name}')
    print(f'Feature Importance -> {fi_path.name}')
    print(f'Full Report        -> {report_path.name}')
    print()
    print('Top 5 most identifying features:')
    for feat, imp in top_features[:5]:
        print(f'  {feat}: {imp:.4f}')


if __name__ == '__main__':
    main()
