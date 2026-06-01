"""
HTF File Parser for Sim Racing Telemetry Data
Parses .htf files and converts them to pandas DataFrames
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    list_training_htf_files, 
    save_dataframe, 
    save_json, 
    print_dataframe_info,
    get_results_path,
    create_summary_report
)


class HTFParser:
    """Parser for HTF telemetry files"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.header_data = {}
        self.channel_data = {}
        
    def parse(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Parse HTF file into header metadata and telemetry DataFrame
        
        Returns:
            Tuple of (header_dict, telemetry_dataframe)
        """
        print(f"\nParsing: {self.file_path.name}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse header
        self._parse_header(content)
        
        # Parse telemetry channels
        self._parse_channels(content)
        
        # Convert to DataFrame
        df = self._create_dataframe()
        
        return self.header_data, df
    
    def _parse_header(self, content: str):
        """Extract header information"""
        # Pattern: [key;]value
        header_pattern = r'\[(.*?);](.*?)(?=\n|\r)'
        
        for match in re.finditer(header_pattern, content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            self.header_data[key] = value
        
        print(f"  Header: Driver={self.header_data.get('driver_pseudonym_code', 'Unknown')}, "
              f"Circuit={self.header_data.get('circuit_name', 'Unknown')}")
    
    def _parse_channels(self, content: str):
        """Parse all telemetry channels"""
        # Pattern: (channel_name;unit;sampling_rate;data_count)data
        channel_pattern = r'\((.*?);(.*?);(.*?);(.*?)\)(.*?)(?=\(|$)'
        
        for match in re.finditer(channel_pattern, content, re.DOTALL):
            channel_name = match.group(1).strip()
            unit = match.group(2).strip()
            sampling_rate = int(match.group(3))
            data_count = int(match.group(4))
            data_str = match.group(5).strip()
            
            # Parse data values
            values = self._parse_channel_data(data_str, data_count)
            
            self.channel_data[channel_name] = {
                'unit': unit,
                'sampling_rate': sampling_rate,
                'data_count': data_count,
                'values': values
            }
        
        print(f"  Channels: {len(self.channel_data)} parsed")
    
    def _parse_channel_data(self, data_str: str, expected_count: int) -> np.ndarray:
        """
        Parse channel data string into numpy array
        Format: 0=value1;1=value2;2=value3;...
        """
        # Initialize array with NaN
        values = np.full(expected_count, np.nan, dtype=np.float64)
        
        # Parse index=value pairs
        pairs = data_str.split(';')
        
        for pair in pairs:
            if '=' in pair:
                try:
                    idx_str, val_str = pair.split('=', 1)
                    idx = int(idx_str)
                    
                    # Handle different value types
                    if val_str.strip() == '':
                        val = np.nan
                    else:
                        val = float(val_str)
                    
                    if 0 <= idx < expected_count:
                        values[idx] = val
                except (ValueError, IndexError):
                    # Skip malformed pairs
                    continue
        
        # Forward fill NaN values (telemetry often skips unchanged values)
        # Use pandas for efficient forward fill
        series = pd.Series(values)
        series = series.ffill()  # Forward fill
        series = series.fillna(0)  # Remaining NaN at start -> 0
        
        return series.values
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert parsed channel data to DataFrame"""
        df_dict = {}
        
        # Add each channel as a column
        for channel_name, channel_info in self.channel_data.items():
            df_dict[channel_name] = channel_info['values']
        
        df = pd.DataFrame(df_dict)
        
        # Add metadata columns
        df.insert(0, 'driver_id', self.header_data.get('driver_pseudonym_code', 'Unknown'))
        df.insert(1, 'circuit', self.header_data.get('circuit_name', 'Unknown'))
        df.insert(2, 'vehicle', self.header_data.get('vehicle_label', 'Unknown'))
        df.insert(3, 'recording_date', self.header_data.get('recording_date', 'Unknown'))
        df.insert(4, 'simulation_setup', self.header_data.get('simulation_setup_label', 'Unknown'))
        
        # Add sample index
        df.insert(5, 'sample_index', np.arange(len(df)))
        
        return df


def parse_all_htf_files() -> pd.DataFrame:
    """
    Parse all HTF files from training_data and combine into single DataFrame
    
    Returns:
        Combined DataFrame with all telemetry data
    """
    htf_files = list_training_htf_files()
    
    if not htf_files:
        print("No HTF files found in training_data!")
        return pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"HTF PARSER - Processing {len(htf_files)} files")
    print(f"{'='*60}")
    
    all_dataframes = []
    metadata_list = []
    
    for htf_file in htf_files:
        try:
            parser = HTFParser(htf_file)
            header, df = parser.parse()
            
            all_dataframes.append(df)
            
            # Store metadata
            metadata_list.append({
                'filename': htf_file.name,
                'driver_id': header.get('driver_pseudonym_code'),
                'circuit': header.get('circuit_name'),
                'vehicle': header.get('vehicle_label'),
                'recording_date': header.get('recording_date'),
                'samples': len(df)
            })
            
        except Exception as e:
            print(f"  ERROR parsing {htf_file.name}: {e}")
            continue
    
    # Combine all DataFrames
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"COMBINED DATASET")
        print(f"{'='*60}")
        print(f"Total samples: {len(combined_df):,}")
        print(f"Total drivers: {combined_df['driver_id'].nunique()}")
        print(f"Columns: {len(combined_df.columns)}")
        
        # Driver distribution
        print(f"\nDriver Distribution:")
        driver_counts = combined_df['driver_id'].value_counts()
        for driver, count in driver_counts.items():
            print(f"  {driver}: {count:,} samples")
        
        return combined_df, metadata_list
    else:
        print("No data parsed successfully!")
        return pd.DataFrame(), []


def main():
    """Main execution"""
    print("="*70)
    print("HTF PARSER - Sim Racing Telemetry Data")
    print("="*70)
    
    # Parse all HTF files
    combined_df, metadata_list = parse_all_htf_files()
    
    if combined_df.empty:
        print("\nNo data to save. Exiting.")
        return
    
    # Save combined dataset
    print(f"\n{'='*60}")
    print("SAVING DATA")
    print(f"{'='*60}")
    
    save_dataframe(combined_df, "telemetry_all", directory="processed_data")
    
    # Save metadata
    save_json(metadata_list, "telemetry_metadata", directory="processed_data")
    
    # Print detailed info
    print_dataframe_info(combined_df, "Combined Telemetry Dataset")
    
    # Create summary report
    summary = {
        "Total Files Processed": len(metadata_list),
        "Total Samples": len(combined_df),
        "Unique Drivers": combined_df['driver_id'].nunique(),
        "Unique Circuits": combined_df['circuit'].nunique(),
        "Unique Vehicles": combined_df['vehicle'].nunique(),
        "Date Range": f"{combined_df['recording_date'].min()} to {combined_df['recording_date'].max()}",
        "Total Columns": len(combined_df.columns),
        "Telemetry Channels": len(combined_df.columns) - 6,  # Exclude metadata columns
        "Drivers": combined_df['driver_id'].value_counts().to_dict(),
        "Column Names": combined_df.columns.tolist()
    }
    
    results_path = get_results_path()
    results_path.mkdir(exist_ok=True)
    report_path = results_path / "01_htf_parsing_summary.txt"
    
    create_summary_report(summary, save_path=report_path)
    
    print(f"\n{'='*60}")
    print("✓ HTF PARSING COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset saved: processed_data/telemetry_all.csv")
    print(f"Metadata saved: processed_data/telemetry_metadata.json")
    print(f"Summary report: {report_path}")
    print()


if __name__ == "__main__":
    main()
