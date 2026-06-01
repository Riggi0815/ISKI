"""
Step 0: Split Raw Data into Training and Test Sets
Copies HTF and LD files from raw_data/ into training_data/ and test_data/
Uses 80/20 split with deterministic ordering
"""

import sys
from pathlib import Path
import shutil
from typing import List, Tuple

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import get_project_root

def split_files(files: List[Path], train_ratio: float = 0.8) -> Tuple[List[Path], List[Path]]:
    """
    Split files into training and test sets
    
    Args:
        files: List of file paths
        train_ratio: Fraction of files for training (default 0.8 = 80%)
    
    Returns:
        Tuple of (training_files, test_files)
    """
    # Sort files by name for deterministic split
    sorted_files = sorted(files, key=lambda x: x.name)
    
    split_idx = int(len(sorted_files) * train_ratio)
    train_files = sorted_files[:split_idx]
    test_files = sorted_files[split_idx:]
    
    return train_files, test_files


def copy_files(files: List[Path], source_dir: Path, target_dir: Path) -> int:
    """
    Copy files from source to target directory
    
    Args:
        files: List of file paths
        source_dir: Source directory
        target_dir: Target directory
    
    Returns:
        Number of files copied
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    
    for file in files:
        source_path = source_dir / file.name
        target_path = target_dir / file.name
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            copied += 1
    
    return copied


def main():
    project_root = get_project_root()
    raw_data_dir = project_root / "raw_data"
    training_dir = project_root / "training_data"
    test_dir = project_root / "test_data"
    
    print(f"{'='*60}")
    print(f"SPLIT RAW DATA INTO TRAINING AND TEST SETS")
    print(f"{'='*60}\n")
    
    # Check if raw_data exists
    if not raw_data_dir.exists():
        print(f"❌ Error: raw_data directory not found at {raw_data_dir}")
        return
    
    # Find all HTF and LD files
    htf_files = list(raw_data_dir.glob("*.htf"))
    ld_files = list(raw_data_dir.glob("*.ld"))
    
    print(f"Found in raw_data/:")
    print(f"  HTF files: {len(htf_files)}")
    print(f"  LD files:  {len(ld_files)}\n")
    
    if len(htf_files) == 0 and len(ld_files) == 0:
        print("❌ No HTF or LD files found in raw_data/")
        return
    
    # Split HTF files (80/20)
    if htf_files:
        htf_train, htf_test = split_files(htf_files, train_ratio=0.8)
        
        print(f"{'─'*60}")
        print(f"HTF Files Split:")
        print(f"{'─'*60}")
        print(f"  Training: {len(htf_train)} files ({len(htf_train)/len(htf_files)*100:.1f}%)")
        print(f"  Test:     {len(htf_test)} files ({len(htf_test)/len(htf_files)*100:.1f}%)")
        print()
        
        # Copy HTF files
        print("Copying HTF files to training_data/...")
        copied = copy_files(htf_train, raw_data_dir, training_dir)
        print(f"  ✓ Copied {copied} files\n")
        
        print("Copying HTF files to test_data/...")
        copied = copy_files(htf_test, raw_data_dir, test_dir)
        print(f"  ✓ Copied {copied} files\n")
        
        # Show which files went where
        print("Training files:")
        for f in htf_train[:5]:  # Show first 5
            print(f"  • {f.name}")
        if len(htf_train) > 5:
            print(f"  ... and {len(htf_train) - 5} more")
        
        print("\nTest files:")
        for f in htf_test:
            print(f"  • {f.name}")
        print()
    
    # Split LD files (80/20)
    if ld_files:
        ld_train, ld_test = split_files(ld_files, train_ratio=0.8)
        
        print(f"{'─'*60}")
        print(f"LD Files Split:")
        print(f"{'─'*60}")
        print(f"  Training: {len(ld_train)} files ({len(ld_train)/len(ld_files)*100:.1f}%)")
        print(f"  Test:     {len(ld_test)} files ({len(ld_test)/len(ld_files)*100:.1f}%)")
        print()
        
        # Copy LD files
        print("Copying LD files to training_data/...")
        copied = copy_files(ld_train, raw_data_dir, training_dir)
        print(f"  ✓ Copied {copied} files\n")
        
        print("Copying LD files to test_data/...")
        copied = copy_files(ld_test, raw_data_dir, test_dir)
        print(f"  ✓ Copied {copied} files\n")
        
        # Show which files went where
        print("Training files:")
        for f in ld_train[:5]:  # Show first 5
            print(f"  • {f.name}")
        if len(ld_train) > 5:
            print(f"  ... and {len(ld_train) - 5} more")
        
        print("\nTest files:")
        for f in ld_test:
            print(f"  • {f.name}")
        print()
    
    print(f"{'='*60}")
    print("✓ DATA SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"\nTraining data: {training_dir}")
    print(f"Test data:     {test_dir}")
    print("\n💡 Next steps:")
    print("   1. Run 01_parse_htf.py (uses training_data/)")
    print("   2. Run 02_parse_ld.py (uses training_data/)")
    print("   3. Continue with pipeline...")


if __name__ == "__main__":
    main()
