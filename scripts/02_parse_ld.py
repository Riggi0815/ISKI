"""
Step 2: Parse Binary LD (Lap Data) Files
Parse Assetto Corsa binary lap data files into pandas DataFrames
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import struct
from typing import Dict, Tuple, List
from datetime import datetime

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_raw_data_path, get_processed_data_path,
    get_results_path, save_dataframe, extract_driver_from_filename
)


class LDParser:
    """Parse Assetto Corsa LD (Lap Data) binary files"""
    
    def __init__(self, file_path: str):
        """
        Initialize parser with file path
        
        Args:
            file_path: Path to .ld file
        """
        self.file_path = Path(file_path)
        self.header_data = {}
        self.telemetry_data = []
        
        # Extract driver from filename (format: track_&_car_&_DRIVER_&_stint_X.ld)
        self.driver_id = extract_driver_from_filename(self.file_path.name)
    
    def parse(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Parse LD file into header metadata and telemetry DataFrame
        
        Returns:
            Tuple of (header_dict, telemetry_dataframe)
        """
        print(f"\nParsing: {self.file_path.name}")
        
        with open(self.file_path, 'rb') as f:
            content = f.read()
        
        # Parse header
        self._parse_header(content)
        
        # Parse telemetry data
        df = self._parse_telemetry(content)
        
        return self.header_data, df
    
    def _parse_header(self, content: bytes):
        """Extract header information from binary data"""
        
        # Try to find ASCII strings in first 1024 bytes (likely header)
        header_section = content[:1024]
        
        # Look for driver ID (from filename)
        self.header_data['driver_pseudonym_code'] = self.driver_id
        
        # Try to extract date/time (often in ASCII near start)
        try:
            # Look for date pattern DD/MM/YYYY
            header_str = header_section.decode('latin-1', errors='ignore')
            
            # Extract any readable strings
            import re
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', header_str)
            time_match = re.search(r'(\d{2}:\d{2})', header_str)
            
            if date_match:
                self.header_data['recording_date'] = date_match.group(1)
            
            if time_match:
                self.header_data['recording_time'] = time_match.group(1)
        
        except Exception as e:
            print(f"  Warning: Could not fully parse header: {e}")
        
        # Extract track and car from filename
        filename_parts = self.file_path.stem.split('_&_')
        if len(filename_parts) >= 2:
            self.header_data['circuit_name'] = filename_parts[0].replace('_', ' ').title()
            self.header_data['vehicle_label'] = filename_parts[1].replace('_', ' ').title()
        
        print(f"  Header: Driver={self.driver_id}, "
              f"Circuit={self.header_data.get('circuit_name', 'Unknown')}")
    
    def _parse_telemetry(self, content: bytes) -> pd.DataFrame:
        """
        Parse telemetry data from binary content
        
        Note: This is a simplified parser that attempts to extract basic telemetry.
        The exact binary format of AC .ld files is not publicly documented,
        so this implementation uses pattern recognition and heuristics.
        """
        
        # Skip header (first 1024 bytes assumed to be header)
        data_start = 1024
        data_section = content[data_start:]
        
        # Try to parse as array of floats (common telemetry format)
        # Assetto Corsa typically uses 4-byte floats
        
        try:
            # Calculate number of float values
            n_floats = len(data_section) // 4
            
            # Unpack as floats
            float_data = struct.unpack(f'>{n_floats}f', data_section[:n_floats*4])
            
            # Heuristic: Organize into channels
            # Typical AC telemetry has ~40-50 channels at 50Hz
            # We'll try to detect the pattern
            
            # Assume 43 channels (same as HTF) for consistency
            n_channels = 43
            
            # Calculate samples per channel
            n_samples = n_floats // n_channels
            
            if n_samples < 10:
                print(f"  Warning: Too few samples detected ({n_samples})")
                return self._create_empty_dataframe()
            
            # Reshape into (samples, channels)
            try:
                data_array = np.array(float_data[:n_samples * n_channels])
                data_array = data_array.reshape(n_samples, n_channels)
            except ValueError:
                # If reshape fails, try different approach
                print(f"  Warning: Could not reshape data, using partial extraction")
                # Take whatever we can fit
                usable_floats = (n_floats // n_channels) * n_channels
                data_array = np.array(float_data[:usable_floats])
                n_samples = usable_floats // n_channels
                data_array = data_array.reshape(n_samples, n_channels)
            
            # Create DataFrame with generic channel names
            # (We don't know the exact channel mapping for .ld files)
            channel_names = [
                'throttle', 'brake', 'steering', 'gear', 'speed',
                'rpm', 'g_long', 'g_lat', 'g_vert',
                'tire_FL_pressure', 'tire_FR_pressure', 
                'tire_RL_pressure', 'tire_RR_pressure',
                'tire_FL_temp', 'tire_FR_temp',
                'tire_RL_temp', 'tire_RR_temp',
                'tire_FL_vel', 'tire_FR_vel',
                'tire_RL_vel', 'tire_RR_vel',
            ]
            
            # Pad with generic names if needed
            while len(channel_names) < n_channels:
                channel_names.append(f'channel_{len(channel_names)}')
            
            # Create DataFrame
            df = pd.DataFrame(data_array, columns=channel_names[:n_channels])
            
            # Add metadata columns
            df.insert(0, 'driver_id', self.driver_id)
            df.insert(1, 'circuit', self.header_data.get('circuit_name', 'Unknown'))
            df.insert(2, 'vehicle', self.header_data.get('vehicle_label', 'Unknown'))
            df.insert(3, 'recording_date', self.header_data.get('recording_date', 'Unknown'))
            df.insert(4, 'sample_index', range(len(df)))
            
            print(f"  Channels: {n_channels} detected")
            print(f"  ✓ Parsed {len(df)} telemetry samples")
            
            return df
        
        except Exception as e:
            print(f"  Error parsing telemetry data: {e}")
            return self._create_empty_dataframe()
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with minimal structure"""
        return pd.DataFrame({
            'driver_id': [self.driver_id],
            'circuit': [self.header_data.get('circuit_name', 'Unknown')],
            'vehicle': [self.header_data.get('vehicle_label', 'Unknown')],
            'recording_date': [self.header_data.get('recording_date', 'Unknown')],
            'sample_index': [0]
        })


def parse_all_ld_files() -> Tuple[pd.DataFrame, Dict]:
    """
    Parse all .ld files in raw_data directory
    
    Returns:
        Tuple of (combined_dataframe, metadata_dict)
    """
    project_root = get_project_root()
    raw_data_path = get_raw_data_path()
    
    print(f"{'='*60}")
    print(f"PARSING LD FILES")
    print(f"{'='*60}")
    print(f"Raw data path: {raw_data_path}\n")
    
    # Find all .ld files
    ld_files = list(raw_data_path.glob("*.ld"))
    
    if not ld_files:
        print("No .ld files found!")
        return pd.DataFrame(), {}
    
    print(f"Found {len(ld_files)} .ld files\n")
    
    # Parse each file
    all_dataframes = []
    all_headers = []
    successful_parses = 0
    
    for ld_file in ld_files:
        try:
            parser = LDParser(ld_file)
            header, df = parser.parse()
            
            if df is not None and len(df) > 1:  # More than just header row
                all_dataframes.append(df)
                all_headers.append(header)
                successful_parses += 1
        
        except Exception as e:
            print(f"  Error parsing {ld_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"PARSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully parsed: {successful_parses}/{len(ld_files)} files")
    
    if not all_dataframes:
        print("No telemetry data extracted!")
        return pd.DataFrame(), {}
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Create metadata
    metadata = {
        'n_files': len(ld_files),
        'n_successful': successful_parses,
        'n_samples': len(combined_df),
        'n_drivers': combined_df['driver_id'].nunique(),
        'drivers': combined_df['driver_id'].unique().tolist(),
        'circuits': combined_df['circuit'].unique().tolist(),
        'vehicles': combined_df['vehicle'].unique().tolist()
    }
    
    print(f"\nCombined telemetry:")
    print(f"  Total samples: {len(combined_df):,}")
    print(f"  Unique drivers: {metadata['n_drivers']}")
    print(f"  Driver IDs: {', '.join(metadata['drivers'])}")
    
    return combined_df, metadata


def main():
    """Main execution function"""
    project_root = get_project_root()
    processed_data_path = get_processed_data_path()
    results_path = get_results_path()
    
    # Parse all LD files
    telemetry_df, metadata = parse_all_ld_files()
    
    if telemetry_df.empty:
        print("\n⚠ No data to save!")
        return
    
    # Save combined telemetry
    print(f"\n{'='*60}")
    print("SAVING DATA")
    print(f"{'='*60}")
    
    output_prefix = "telemetry_ld"
    save_dataframe(telemetry_df, processed_data_path / output_prefix)
    
    # Save metadata
    import json
    metadata_file = processed_data_path / f"{output_prefix}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file.name}")
    
    # Create summary report
    summary_file = results_path / "02_ld_parsing_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LD FILE PARSING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Files processed: {metadata['n_successful']}/{metadata['n_files']}\n")
        f.write(f"Total samples: {metadata['n_samples']:,}\n")
        f.write(f"Unique drivers: {metadata['n_drivers']}\n\n")
        
        f.write("Drivers:\n")
        for driver in metadata['drivers']:
            count = len(telemetry_df[telemetry_df['driver_id'] == driver])
            f.write(f"  {driver}: {count:,} samples\n")
        
        f.write(f"\nCircuits: {', '.join(metadata['circuits'])}\n")
        f.write(f"Vehicles: {', '.join(metadata['vehicles'])}\n")
    
    print(f"✓ Saved summary: {summary_file.name}")
    
    print(f"\n{'='*60}")
    print("✓ LD PARSING COMPLETE")
    print(f"{'='*60}")
    print(f"Data saved to: {processed_data_path}")


if __name__ == "__main__":
    main()
