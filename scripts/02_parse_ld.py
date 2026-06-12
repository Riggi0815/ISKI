"""
Step 2: Parse Binary LD (Lap Data) Files
Parse Assetto Corsa MoTeC LD files into pandas DataFrames using ldparser.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from ldparser import ldData
from utils import (
    get_project_root, get_raw_data_path, get_processed_data_path,
    get_results_path, save_dataframe, extract_driver_from_filename
)

# Mapping: LD channel name → HTF column name (expected by feature engineering)
CHANNEL_MAP = {
    'Ground Speed':              'v_car',
    'Throttle Pos':              'percent_throttle',
    'Brake Pos':                 'percent_brake',
    'Steering Angle':            'steering_angle',
    'CG Accel Lateral':          'g_lat',
    'CG Accel Longitudinal':     'g_long',
    'CG Accel Vertical':         'g_vert',
    'Engine RPM':                'n_engine',
    'Tire Temp Core FR':         't_tyreFR',
    'Tire Temp Core FL':         't_tyreFL',
    'Tire Temp Core RR':         't_tyreRR',
    'Tire Temp Core RL':         't_tyreRL',
    'Tire Pressure FR':          'p_tyreFR',
    'Tire Pressure FL':          'p_tyreFL',
    'Tire Pressure RR':          'p_tyreRR',
    'Tire Pressure RL':          'p_tyreRL',
    'Chassis Velocity X':        'v_x',
    'Chassis Velocity Z':        'v_z',
    'Gear':                      'gear',
}


def parse_ld_file(file_path: Path) -> pd.DataFrame:
    """Parse a single .ld file into a DataFrame with HTF-compatible column names."""
    print(f"\nParsing: {file_path.name}")

    driver_id = extract_driver_from_filename(file_path.name)
    filename_parts = file_path.stem.split('_&_')
    circuit = filename_parts[0].replace('_', ' ').title() if len(filename_parts) >= 1 else 'Unknown'
    vehicle = filename_parts[1].replace('_', ' ').title() if len(filename_parts) >= 2 else 'Unknown'

    ld = ldData.fromfile(str(file_path))
    available = list(ld)

    print(f"  Driver: {driver_id} | Circuit: {circuit}")
    print(f"  Total channels in file: {len(available)}")

    rows = {}
    for ld_name, htf_name in CHANNEL_MAP.items():
        if ld_name in available:
            try:
                rows[htf_name] = ld[ld_name].data
            except Exception as e:
                print(f"  Warning: could not read '{ld_name}': {e}")
        else:
            print(f"  Warning: channel '{ld_name}' not found in file")

    if not rows:
        print("  Error: no channels could be read!")
        return pd.DataFrame()

    # Align all channels to the same length (shortest wins)
    min_len = min(len(v) for v in rows.values())
    rows = {k: v[:min_len] for k, v in rows.items()}

    df = pd.DataFrame(rows)
    df.insert(0, 'driver_id', driver_id)
    df.insert(1, 'circuit', circuit)
    df.insert(2, 'vehicle', vehicle)
    df.insert(3, 'sample_index', range(len(df)))

    print(f"  ✓ Parsed {len(df):,} samples | {len(df.columns)-4} telemetry channels")
    return df


def main():
    raw_data_path = get_raw_data_path()
    processed_data_path = get_processed_data_path()
    results_path = get_results_path()

    print(f"{'='*60}")
    print(f"PARSING LD FILES (MoTeC format via ldparser)")
    print(f"{'='*60}")
    print(f"Raw data path: {raw_data_path}\n")

    ld_files = sorted(raw_data_path.glob("*.ld"))
    if not ld_files:
        print("No .ld files found in raw_data/")
        return

    print(f"Found {len(ld_files)} .ld files\n")

    all_dfs = []
    for ld_file in ld_files:
        df = parse_ld_file(ld_file)
        if not df.empty:
            all_dfs.append(df)

    print(f"\n{'='*60}")
    print(f"PARSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully parsed: {len(all_dfs)}/{len(ld_files)} files")

    if not all_dfs:
        print("No data extracted!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nCombined telemetry:")
    print(f"  Total samples : {len(combined_df):,}")
    print(f"  Unique drivers: {combined_df['driver_id'].nunique()}")
    for driver, count in combined_df['driver_id'].value_counts().items():
        print(f"    {driver}: {count:,} samples")

    print(f"\n{'='*60}")
    print("SAVING DATA")
    print(f"{'='*60}")

    output_file = processed_data_path / "telemetry_ld"
    save_dataframe(combined_df, output_file)

    import json
    meta = {
        'n_files': len(ld_files),
        'n_successful': len(all_dfs),
        'n_samples': len(combined_df),
        'n_drivers': combined_df['driver_id'].nunique(),
        'drivers': combined_df['driver_id'].unique().tolist(),
        'channels': [c for c in combined_df.columns if c not in ['driver_id','circuit','vehicle','sample_index']],
    }
    with open(processed_data_path / "telemetry_ld_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("✓ Saved metadata")

    summary_file = results_path / "02_ld_parsing_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LD FILE PARSING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Files: {meta['n_successful']}/{meta['n_files']}\n")
        f.write(f"Samples: {meta['n_samples']:,}\n")
        f.write(f"Drivers: {meta['n_drivers']}\n\n")
        f.write("Channels mapped:\n")
        for ch in meta['channels']:
            f.write(f"  - {ch}\n")
    print(f"✓ Saved summary")

    print(f"\n{'='*60}")
    print("✓ LD PARSING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
