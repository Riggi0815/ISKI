"""Konvertiert ALLE .htf Dateien zu CSV"""

from pathlib import Path
import pandas as pd
import re
import numpy as np
from parse_htf import parse_htf_file, save_to_csv

def extract_driver_from_metadata(metadata):
    """Extrahiert Fahrer-Code aus Metadata"""
    if 'driver_pseudonym_code' in metadata:
        return metadata['driver_pseudonym_code']
    return "UNKNOWN"

def convert_all_htf_files(input_dir="raw_data", output_dir="processed_data"):
    """Konvertiert alle .htf Dateien zu CSV"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Finde alle .htf Dateien
    htf_files = list(input_path.glob("*.htf"))
    
    print(f"\n{'='*70}")
    print(f"HTF BATCH CONVERTER")
    print(f"{'='*70}")
    print(f"Gefunden: {len(htf_files)} .htf Dateien\n")
    
    results = []
    errors = []
    
    for i, htf_file in enumerate(htf_files, 1):
        print(f"\n[{i}/{len(htf_files)}] Processing: {htf_file.name}")
        print("-" * 70)
        
        try:
            # Parse
            df, metadata = parse_htf_file(str(htf_file))
            
            # Extrahiere Fahrer
            driver = extract_driver_from_metadata(metadata)
            
            # Output-Dateiname: driver_originalname.csv
            output_filename = f"{driver}_{htf_file.stem}.csv"
            output_file = output_path / output_filename
            
            # Speichere
            save_to_csv(df, metadata, str(output_file))
            
            results.append({
                'file': htf_file.name,
                'driver': driver,
                'rows': len(df),
                'columns': len(df.columns),
                'output': output_filename,
                'status': '✓'
            })
            
        except Exception as e:
            print(f"❌ FEHLER: {e}")
            errors.append({
                'file': htf_file.name,
                'error': str(e)
            })
            results.append({
                'file': htf_file.name,
                'driver': 'ERROR',
                'rows': 0,
                'columns': 0,
                'output': '-',
                'status': '✗'
            })
    
    # Zusammenfassung
    print(f"\n{'='*70}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*70}\n")
    
    print(f"Erfolgreich: {len([r for r in results if r['status'] == '✓'])}/{len(htf_files)}")
    print(f"Fehler:      {len(errors)}\n")
    
    # Gruppiere nach Fahrer
    drivers = {}
    for r in results:
        if r['status'] == '✓':
            driver = r['driver']
            if driver not in drivers:
                drivers[driver] = []
            drivers[driver].append(r)
    
    print("Dateien pro Fahrer:")
    for driver, files in drivers.items():
        print(f"\n  {driver}: {len(files)} Dateien")
        total_rows = sum(f['rows'] for f in files)
        print(f"    Total Datenpunkte: {total_rows:,}")
    
    if errors:
        print("\n❌ Fehlerhafte Dateien:")
        for err in errors:
            print(f"  - {err['file']}: {err['error']}")
    
    print(f"\nAlle CSV-Dateien in: {output_path}/")

if __name__ == "__main__":
    convert_all_htf_files()