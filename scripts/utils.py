"""
Utility functions for the Sim Racing Driver Identification Project
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_raw_data_path() -> Path:
    """Get the raw_data directory path"""
    return get_project_root() / "raw_data"


def get_training_data_path() -> Path:
    """Get the training_data directory path"""
    return get_project_root() / "training_data"


def get_test_data_path() -> Path:
    """Get the test_data directory path"""
    return get_project_root() / "test_data"


def get_processed_data_path() -> Path:
    """Get the processed_data directory path"""
    return get_project_root() / "processed_data"


def get_features_path() -> Path:
    """Get the features directory path"""
    return get_project_root() / "features"


def get_models_path() -> Path:
    """Get the models directory path"""
    return get_project_root() / "models"


def get_results_path() -> Path:
    """Get the results directory path"""
    return get_project_root() / "results"


def list_htf_files() -> List[Path]:
    """List all .htf files in raw_data directory"""
    raw_data_path = get_raw_data_path()
    return sorted(raw_data_path.glob("*.htf"))


def list_ld_files() -> List[Path]:
    """List all .ld files in raw_data directory"""
    raw_data_path = get_raw_data_path()
    return sorted(raw_data_path.glob("*.ld"))


def list_training_htf_files() -> List[Path]:
    """List all .htf files in training_data directory"""
    training_data_path = get_training_data_path()
    return sorted(training_data_path.glob("*.htf"))


def list_training_ld_files() -> List[Path]:
    """List all .ld files in training_data directory"""
    training_data_path = get_training_data_path()
    return sorted(training_data_path.glob("*.ld"))


def list_test_htf_files() -> List[Path]:
    """List all .htf files in test_data directory"""
    test_data_path = get_test_data_path()
    return sorted(test_data_path.glob("*.htf"))


def list_test_ld_files() -> List[Path]:
    """List all .ld files in test_data directory"""
    test_data_path = get_test_data_path()
    return sorted(test_data_path.glob("*.ld"))


def save_dataframe(df: pd.DataFrame, filename: str, directory: str = "processed_data"):
    """
    Save DataFrame to CSV and Pickle for backup
    
    Args:
        df: DataFrame to save
        filename: Name of the file (without extension)
        directory: Target directory (default: processed_data)
    """
    base_path = get_project_root() / directory
    base_path.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = base_path / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save as Pickle for faster loading
    pkl_path = base_path / f"{filename}.pkl"
    df.to_pickle(pkl_path)
    print(f"✓ Saved Pickle: {pkl_path}")


def load_dataframe(filename: str, directory: str = "processed_data") -> pd.DataFrame:
    """
    Load DataFrame from Pickle (faster) or CSV (fallback)
    
    Args:
        filename: Name of the file (without extension)
        directory: Source directory (default: processed_data)
    
    Returns:
        Loaded DataFrame
    """
    base_path = get_project_root() / directory
    
    # Try Pickle first
    pkl_path = base_path / f"{filename}.pkl"
    if pkl_path.exists():
        print(f"Loading from Pickle: {pkl_path}")
        return pd.read_pickle(pkl_path)
    
    # Fallback to CSV
    csv_path = base_path / f"{filename}.csv"
    if csv_path.exists():
        print(f"Loading from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    raise FileNotFoundError(f"File {filename} not found in {base_path}")


def save_json(data: Dict, filename: str, directory: str = "processed_data"):
    """
    Save dictionary as JSON
    
    Args:
        data: Dictionary to save
        filename: Name of the file (without extension)
        directory: Target directory
    """
    base_path = get_project_root() / directory
    base_path.mkdir(exist_ok=True)
    
    json_path = base_path / f"{filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")


def load_json(filename: str, directory: str = "processed_data") -> Dict:
    """
    Load JSON file
    
    Args:
        filename: Name of the file (without extension)
        directory: Source directory
    
    Returns:
        Loaded dictionary
    """
    base_path = get_project_root() / directory
    json_path = base_path / f"{filename}.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_driver_distribution(df: pd.DataFrame, driver_col: str = "driver_id", 
                             save_path: str = None):
    """
    Plot the distribution of samples per driver
    
    Args:
        df: DataFrame with driver information
        driver_col: Column name containing driver IDs
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    driver_counts = df[driver_col].value_counts().sort_index()
    
    ax = driver_counts.plot(kind='bar', color='steelblue')
    plt.title('Sample Distribution per Driver', fontsize=16, fontweight='bold')
    plt.xlabel('Driver ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(driver_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print comprehensive information about a DataFrame
    
    Args:
        df: DataFrame to analyze
        name: Name for the report
    """
    print(f"\n{'='*60}")
    print(f"{name} - Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if df.isnull().sum().sum() == 0:
        print("✓ No missing values!")
    
    print(f"\nFirst few rows:")
    print(df.head(3))
    print(f"{'='*60}\n")


def extract_driver_from_filename(filename: str) -> str:
    """
    Extract driver code from .ld filename
    Expected format: track_&_car_&_DRIVER_&_stint_X.ld
    
    Args:
        filename: Filename string
    
    Returns:
        Driver code
    """
    parts = filename.split('&')
    if len(parts) >= 3:
        return parts[2].strip()
    return None


def create_summary_report(data_dict: Dict[str, Any], save_path: str = None) -> str:
    """
    Create a summary report from data dictionary
    
    Args:
        data_dict: Dictionary with summary information
        save_path: Optional path to save the report
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("="*70)
    report.append("DATA SUMMARY REPORT")
    report.append("="*70)
    
    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)):
            report.append(f"\n{key}:")
            for item in value:
                report.append(f"  - {item}")
        elif isinstance(value, dict):
            report.append(f"\n{key}:")
            for k, v in value.items():
                report.append(f"  {k}: {v}")
        else:
            report.append(f"{key}: {value}")
    
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Report saved: {save_path}")
    
    return report_text


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    print(f"\nProject Root: {get_project_root()}")
    print(f"Raw Data Path: {get_raw_data_path()}")
    
    htf_files = list_htf_files()
    print(f"\nFound {len(htf_files)} HTF files")
    
    ld_files = list_ld_files()
    print(f"Found {len(ld_files)} LD files")
    
    # Test driver extraction from LD filenames
    if ld_files:
        print("\nDriver codes from LD files:")
        for ld_file in ld_files:
            driver = extract_driver_from_filename(ld_file.name)
            print(f"  {ld_file.name} -> {driver}")
    
    print("\n✓ All utility functions loaded successfully!")
