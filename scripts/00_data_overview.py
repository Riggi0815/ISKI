"""
Data Overview - Zeigt welche Fahrer verfügbar und trainiert sind
"""
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json

sys.path.append(str(Path(__file__).parent))
from utils import (
    get_project_root, get_processed_data_path, get_models_path,
    get_results_path, load_dataframe
)


class DataOverview:
    """Erstellt Übersicht über verfügbare und trainierte Fahrer"""
    
    def __init__(self):
        self.project_root = get_project_root()
        self.raw_data_path = self.project_root / "raw_data"
        self.processed_path = get_processed_data_path()
        self.models_path = get_models_path()
        self.results_path = get_results_path()
        
        self.raw_drivers = defaultdict(lambda: {'htf': [], 'ld': [], 'samples': 0})
        self.trained_drivers = {}
        
    def scan_raw_data(self):
        """Scanne raw_data/ für HTF und LD Dateien - nutze geparste Daten für Driver IDs"""
        print(f"{'='*70}")
        print("SCANNING RAW DATA FILES")
        print(f"{'='*70}\n")
        
        # HTF Dateien - Map filenames zu Driver IDs via geparste Daten
        htf_files = list(self.raw_data_path.glob("*.htf"))
        print(f"Found {len(htf_files)} HTF files")
        
        # Load HTF parsed data to get real driver IDs
        htf_data = load_dataframe(self.processed_path / "telemetry_all")
        if htf_data is not None and 'driver_id' in htf_data.columns:
            for driver_id in htf_data['driver_id'].unique():
                # Count files (assumption: each driver has files in raw_data)
                self.raw_drivers[driver_id]['htf'] = [f"HTF files (parsed)"]
                self.raw_drivers[driver_id]['samples'] = len(htf_data[htf_data['driver_id'] == driver_id])
        else:
            # Fallback: scan files directly but warn
            print("  ⚠ Could not load parsed HTF data, using file count only")
            for htf_file in htf_files:
                # Use generic placeholder
                driver_id = f"HTF_{htf_file.stem[:8]}"
                self.raw_drivers[driver_id]['htf'].append(htf_file.name)
        
        # LD Dateien - Map filenames zu Driver IDs
        ld_files = list(self.raw_data_path.glob("*.ld"))
        print(f"Found {len(ld_files)} LD files")
        
        # Load LD parsed data
        ld_data = load_dataframe(self.processed_path / "telemetry_ld")
        if ld_data is not None and 'driver_id' in ld_data.columns:
            for driver_id in ld_data['driver_id'].unique():
                self.raw_drivers[driver_id]['ld'] = [f"LD files (parsed)"]
                if 'samples' not in self.raw_drivers[driver_id] or self.raw_drivers[driver_id]['samples'] == 0:
                    self.raw_drivers[driver_id]['samples'] = len(ld_data[ld_data['driver_id'] == driver_id])
        else:
            # Fallback: extract from LD filenames
            print("  ⚠ Could not load parsed LD data, using filename extraction")
            for ld_file in ld_files:
                parts = ld_file.stem.split('_&_')
                if len(parts) >= 3:
                    driver_id = parts[2]
                    self.raw_drivers[driver_id]['ld'].append(ld_file.name)
        
        print(f"\nUnique drivers in raw_data/: {len(self.raw_drivers)}")
        for driver_id in sorted(self.raw_drivers.keys()):
            files = self.raw_drivers[driver_id]
            htf_count = len([f for f in files['htf'] if f]) if files['htf'] else 0
            ld_count = len([f for f in files['ld'] if f]) if files['ld'] else 0
            samples = files.get('samples', 0)
            print(f"  {driver_id}: {htf_count} HTF + {ld_count} LD | {samples:,} samples")
    
    def load_training_data(self):
        """Lade trainierte Fahrer aus Models"""
        print(f"\n{'='*70}")
        print("LOADING TRAINED MODELS")
        print(f"{'='*70}\n")
        
        # Check combined models
        combined_models = self.models_path / "combined"
        if (combined_models / "model_metadata.json").exists():
            with open(combined_models / "model_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            print(f"✓ Found combined model with {metadata['n_drivers']} drivers")
            print(f"  Data source: {metadata.get('data_source', 'Unknown')}")
            print(f"  Features: {metadata['n_features']}")
            
            # Load actual training data stats
            combined_data = load_dataframe(self.processed_path / "telemetry_combined")
            if combined_data is not None:
                for driver_id in metadata['driver_names']:
                    driver_samples = len(combined_data[combined_data['driver_id'] == driver_id])
                    self.trained_drivers[driver_id] = {
                        'samples': driver_samples,
                        'source': 'HTF+LD Combined'
                    }
                    
                    # Zeige Datenquelle
                    if 'data_source' in combined_data.columns:
                        sources = combined_data[combined_data['driver_id'] == driver_id]['data_source'].unique()
                        self.trained_drivers[driver_id]['data_sources'] = list(sources)
            else:
                # Fallback: nur aus metadata
                for driver_id in metadata['driver_names']:
                    self.trained_drivers[driver_id] = {
                        'samples': 0,
                        'source': 'Combined Model'
                    }
        
        # Check HTF-only models (fallback)
        elif (self.models_path / "model_metadata.json").exists():
            with open(self.models_path / "model_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            print(f"✓ Found HTF-only model with {metadata['n_drivers']} drivers")
            
            for driver_id in metadata['driver_names']:
                self.trained_drivers[driver_id] = {
                    'samples': 0,
                    'source': 'HTF Only'
                }
        else:
            print("⚠ No trained models found!")
            print("  Run: py -3 scripts\\04b_train_models_combined.py")
    
    def create_comparison_report(self):
        """Erstelle Vergleichsreport"""
        print(f"\n{'='*70}")
        print("DATA COMPARISON REPORT")
        print(f"{'='*70}\n")
        
        # Alle Fahrer (roh + trainiert)
        all_drivers = set(self.raw_drivers.keys()) | set(self.trained_drivers.keys())
        
        print(f"Total unique drivers: {len(all_drivers)}\n")
        
        # Kategorisierung
        trained_only = set(self.trained_drivers.keys()) - set(self.raw_drivers.keys())
        raw_only = set(self.raw_drivers.keys()) - set(self.trained_drivers.keys())
        both = set(self.raw_drivers.keys()) & set(self.trained_drivers.keys())
        
        # Report-Datei
        report_file = self.results_path / "00_data_overview.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DRIVER DATA OVERVIEW\n")
            f.write("="*70 + "\n\n")
            
            # Summary
            f.write(f"Total Unique Drivers: {len(all_drivers)}\n")
            f.write(f"  ✓ In Training Data: {len(self.trained_drivers)}\n")
            f.write(f"  ✓ In Raw Data: {len(self.raw_drivers)}\n")
            f.write(f"  ✓ Both: {len(both)}\n")
            f.write(f"  ⚠ Only Raw (Not Trained): {len(raw_only)}\n")
            f.write(f"  ⚠ Only Trained (No Raw): {len(trained_only)}\n\n")
            
            # Trained Drivers
            f.write("="*70 + "\n")
            f.write("TRAINED DRIVERS (In Model)\n")
            f.write("="*70 + "\n\n")
            
            for driver_id in sorted(self.trained_drivers.keys()):
                info = self.trained_drivers[driver_id]
                f.write(f"{driver_id}:\n")
                f.write(f"  Samples: {info['samples']:,}\n")
                f.write(f"  Source: {info.get('source', 'Unknown')}\n")
                if 'data_sources' in info:
                    f.write(f"  File Types: {', '.join(info['data_sources'])}\n")
                
                # Check if in raw data
                if driver_id in self.raw_drivers:
                    raw = self.raw_drivers[driver_id]
                    f.write(f"  Raw Files: {len(raw['htf'])} HTF + {len(raw['ld'])} LD\n")
                    f.write(f"  ✓ Can add more data\n")
                else:
                    f.write(f"  ⚠ No raw files (already processed)\n")
                f.write("\n")
            
            # Raw Only Drivers (Not Trained)
            if raw_only:
                f.write("="*70 + "\n")
                f.write("AVAILABLE BUT NOT TRAINED (Add to Training!)\n")
                f.write("="*70 + "\n\n")
                
                for driver_id in sorted(raw_only):
                    raw = self.raw_drivers[driver_id]
                    f.write(f"{driver_id}:\n")
                    f.write(f"  HTF Files: {len(raw['htf'])}\n")
                    for htf in raw['htf']:
                        f.write(f"    - {htf}\n")
                    f.write(f"  LD Files: {len(raw['ld'])}\n")
                    for ld in raw['ld']:
                        f.write(f"    - {ld}\n")
                    f.write(f"  💡 ACTION: Add to training by running full pipeline\n")
                    f.write("\n")
            
            # Statistics
            f.write("="*70 + "\n")
            f.write("STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            if self.trained_drivers:
                total_samples = sum(d['samples'] for d in self.trained_drivers.values())
                f.write(f"Total Training Samples: {total_samples:,}\n")
                f.write(f"Average Samples/Driver: {total_samples / len(self.trained_drivers):,.0f}\n\n")
                
                f.write("Sample Distribution:\n")
                for driver_id in sorted(self.trained_drivers.keys(), 
                                       key=lambda x: self.trained_drivers[x]['samples'], 
                                       reverse=True):
                    samples = self.trained_drivers[driver_id]['samples']
                    pct = samples / total_samples * 100 if total_samples > 0 else 0
                    f.write(f"  {driver_id}: {samples:>8,} samples ({pct:5.1f}%)\n")
            
            # Recommendations
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            if raw_only:
                f.write(f"1. {len(raw_only)} driver(s) available but not trained:\n")
                for driver_id in sorted(raw_only):
                    f.write(f"   - {driver_id}\n")
                f.write(f"\n   Run full pipeline to include them:\n")
                f.write(f"   py -3 scripts\\01_parse_htf.py\n")
                f.write(f"   py -3 scripts\\02_parse_ld.py\n")
                f.write(f"   py -3 scripts\\03a_combine_data.py\n")
                f.write(f"   py -3 scripts\\03b_feature_engineering_combined.py\n")
                f.write(f"   py -3 scripts\\04b_train_models_combined.py\n\n")
            
            if self.trained_drivers:
                min_samples = min(d['samples'] for d in self.trained_drivers.values())
                max_samples = max(d['samples'] for d in self.trained_drivers.values())
                ratio = max_samples / min_samples if min_samples > 0 else 0
                
                if ratio > 3:
                    f.write(f"2. Imbalanced dataset detected (ratio: {ratio:.1f}:1)\n")
                    f.write(f"   Consider collecting more data for underrepresented drivers:\n")
                    for driver_id in sorted(self.trained_drivers.keys(), 
                                           key=lambda x: self.trained_drivers[x]['samples'])[:3]:
                        samples = self.trained_drivers[driver_id]['samples']
                        f.write(f"   - {driver_id}: {samples:,} samples (needs more)\n")
                    f.write("\n")
            
            if both:
                f.write(f"3. Data collection progress:\n")
                f.write(f"   {len(both)} driver(s) have both raw files and training data\n")
                f.write(f"   Consider recording more sessions for better accuracy\n")
                f.write(f"   Target: 20-50 laps per driver\n")
        
        print(f"✓ Saved report: {report_file.name}")
        
        # Console summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if both:
            print(f"\n✓ Trained & Available ({len(both)} drivers):")
            for driver_id in sorted(both):
                samples = self.trained_drivers[driver_id]['samples']
                raw = self.raw_drivers[driver_id]
                print(f"  {driver_id}: {samples:,} samples | {len(raw['htf'])} HTF + {len(raw['ld'])} LD files")
        
        if raw_only:
            print(f"\n⚠ Available but NOT Trained ({len(raw_only)} drivers):")
            for driver_id in sorted(raw_only):
                raw = self.raw_drivers[driver_id]
                print(f"  {driver_id}: {len(raw['htf'])} HTF + {len(raw['ld'])} LD files")
            print(f"\n  💡 Run full pipeline to include these drivers!")
        
        if trained_only:
            print(f"\n⚠ Trained but NO Raw Files ({len(trained_only)} drivers):")
            for driver_id in sorted(trained_only):
                samples = self.trained_drivers[driver_id]['samples']
                print(f"  {driver_id}: {samples:,} samples (already processed)")
    
    def run(self):
        """Run complete overview"""
        self.scan_raw_data()
        self.load_training_data()
        self.create_comparison_report()
        
        print(f"\n{'='*70}")
        print("✓ OVERVIEW COMPLETE")
        print(f"{'='*70}")
        print(f"Report: {self.results_path / '00_data_overview.txt'}")


def main():
    """Main execution"""
    overview = DataOverview()
    overview.run()


if __name__ == "__main__":
    main()
