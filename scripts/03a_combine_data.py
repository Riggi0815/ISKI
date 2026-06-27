"""
Step 3a: Combine HTF and LD Telemetry Data
Merges processed_data/telemetry_all (HTF) and processed_data/telemetry_ld (LD)
into a single processed_data/telemetry_combined for feature engineering.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from utils import get_processed_data_path, get_results_path, load_dataframe, save_dataframe


def main():
    processed_path = get_processed_data_path()
    results_path = get_results_path()

    print('=' * 60)
    print('COMBINE HTF + LD TELEMETRY')
    print('=' * 60)

    # Load HTF data (from 01_parse_htf.py)
    print('\nLoading HTF data (telemetry_all)...')
    htf_df = load_dataframe(processed_path / 'telemetry_all')
    if htf_df is not None and not htf_df.empty:
        htf_df['data_source'] = 'HTF'
        print(f'  ✓ {len(htf_df):,} samples, {htf_df["driver_id"].nunique()} drivers')
    else:
        print('  ⚠ No HTF data found — skipping')
        htf_df = pd.DataFrame()

    # Load LD data (from 02_parse_ld.py)
    print('\nLoading LD data (telemetry_ld)...')
    ld_df = load_dataframe(processed_path / 'telemetry_ld')
    if ld_df is not None and not ld_df.empty:
        ld_df['data_source'] = 'LD'
        print(f'  ✓ {len(ld_df):,} samples, {ld_df["driver_id"].nunique()} drivers')
    else:
        print('  ⚠ No LD data found — skipping')
        ld_df = pd.DataFrame()

    if htf_df.empty and ld_df.empty:
        print('\n❌ No data found at all. Run 01_parse_htf.py and/or 02_parse_ld.py first.')
        return

    # Combine — pandas aligns on column names, missing columns become NaN
    parts = [df for df in [htf_df, ld_df] if not df.empty]
    combined_df = pd.concat(parts, ignore_index=True)
    combined_df['sample_index'] = range(len(combined_df))

    print(f'\n✓ Combined: {len(combined_df):,} samples, {combined_df["driver_id"].nunique()} drivers')
    print(f'  Drivers: {sorted(combined_df["driver_id"].unique())}')

    # Save
    output_file = processed_path / 'telemetry_combined'
    save_dataframe(combined_df, output_file)

    # Summary report
    summary_file = results_path / '03a_combine_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('COMBINE HTF + LD SUMMARY\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'HTF samples : {len(htf_df):,}\n')
        f.write(f'LD  samples : {len(ld_df):,}\n')
        f.write(f'Total       : {len(combined_df):,}\n')
        f.write(f'Drivers     : {combined_df["driver_id"].nunique()}\n\n')
        for drv, cnt in combined_df['driver_id'].value_counts().items():
            src = combined_df[combined_df['driver_id'] == drv]['data_source'].iloc[0]
            f.write(f'  {drv}: {cnt:,} ({src})\n')
    print(f'✓ Summary: {summary_file.name}')

    print('\nNext step:')
    print('  python scripts/03b_feature_engineering_combined.py')


if __name__ == '__main__':
    main()
