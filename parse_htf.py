"""HTF Parser - Text-basiertes Format (Fixed Version)"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

def parse_htf_file(filepath):
    """Parst eine .htf Datei zu DataFrame"""
    
    print(f"\n{'='*70}")
    print(f"Parsing: {Path(filepath).name}")
    print(f"{'='*70}\n")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 1. Extrahiere Metadata (alles zwischen [...;])
    metadata = {}
    metadata_pattern = r'\[([^;]+);?\]([^\r\n\[]+)'
    
    for match in re.finditer(metadata_pattern, content):
        key = match.group(1).strip()
        value = match.group(2).strip()
        metadata[key] = value
        print(f"📋 {key}: {value}")
    
    # 2. Extrahiere Channel-Daten
    channel_pattern = r'\(([^;]+);([^;]*);([^;]*);([^)]+)\)(.+?)(?=\(|$)'
    
    channels_data = {}
    max_length = 0
    
    for match in re.finditer(channel_pattern, content, re.DOTALL):
        channel_name = match.group(1).strip()
        unit = match.group(2).strip()
        decimals = match.group(3).strip()
        count = match.group(4).strip()
        data_string = match.group(5).strip()
        
        # Parse die Werte: "0=123.45;1=123.67;..."
        values = []
        for value_match in re.finditer(r'\d+=([^;]+)', data_string):
            try:
                val = float(value_match.group(1))
                values.append(val)
            except ValueError:
                pass
        
        channels_data[channel_name] = {
            'unit': unit,
            'data': values,
            'count': len(values)
        }
        
        # Track max length
        if len(values) > max_length:
            max_length = len(values)
        
        print(f"\n📊 Channel: {channel_name}")
        print(f"   Unit: {unit}, Datenpunkte: {len(values)}")
        if len(values) > 0:
            print(f"   Min: {min(values):.2f}, Max: {max(values):.2f}, Mean: {sum(values)/len(values):.2f}")
    
    print(f"\n{'='*70}")
    print(f"📏 Maximale Länge: {max_length} Datenpunkte")
    print(f"{'='*70}\n")
    
    # 3. Baue DataFrame - fülle kürzere Arrays mit dem letzten Wert auf
    df_data = {}
    
    for channel_name, channel_info in channels_data.items():
        data = channel_info['data']
        
        if len(data) == 0:
            # Keine Daten -> fülle mit NaN
            df_data[channel_name] = [np.nan] * max_length
        elif len(data) == 1:
            # Konstanter Wert -> wiederhole ihn
            df_data[channel_name] = [data[0]] * max_length
        elif len(data) < max_length:
            # Fülle mit letztem Wert auf (Forward Fill)
            padded = data + [data[-1]] * (max_length - len(data))
            df_data[channel_name] = padded
        else:
            # Volle Länge
            df_data[channel_name] = data
    
    df = pd.DataFrame(df_data)
    
    print(f"\n{'='*70}")
    print(f"✓ DataFrame erstellt: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    print(f"{'='*70}\n")
    
    return df, metadata

def save_to_csv(df, metadata, output_path):
    """Speichert DataFrame als CSV mit Metadata im Header"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Schreibe Metadata als Kommentar-Zeilen
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# HTF Telemetry Data\n")
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")
    
    # Append DataFrame
    df.to_csv(output_path, mode='a', index=False)
    
    print(f"✓ Gespeichert: {output_path}")
    print(f"  Größe: {output_path.stat().st_size / (1024*1024):.2f} MB")

# MAIN
if __name__ == "__main__":
    print("\n🔧 HTF PARSER - ISKI Project\n")
    
    # Parse eine .htf Datei
    htf_file = "raw_data/00f946d7-504b-4a0d-8314-fdbe1d58d4c8.htf"
    
    if not Path(htf_file).exists():
        print(f"❌ Datei nicht gefunden: {htf_file}")
        print("\nVerfügbare .htf Dateien:")
        for f in Path("raw_data").glob("*.htf"):
            print(f"  - {f.name}")
        exit()
    
    # Parse
    df, metadata = parse_htf_file(htf_file)
    
    # Zeige erste Zeilen
    print("\n📊 Erste 5 Zeilen:")
    print(df.head())
    
    print("\n📊 Wichtige Spalten:")
    important_cols = ['t_time', 'v_car', 'n_engine', 'n_gear', 'percent_throttle', 
                      'p_brakeF', 'a_steering', 'g_lat', 'g_long']
    available_cols = [col for col in important_cols if col in df.columns]
    print(df[available_cols].head(10))
    
    # Speichere als CSV
    output_file = "processed_data/htf_sample.csv"
    save_to_csv(df, metadata, output_file)
    
    print("\n📊 Statistiken:")
    print(f"  Dauer: {df['t_time'].max() - df['t_time'].min():.1f} Sekunden")
    print(f"  Max Speed: {df['v_car'].max():.1f} km/h")
    print(f"  Max RPM: {df['n_engine'].max():.0f} rpm")
    print(f"  Max Lateral G: {df['g_lat'].max():.2f} G")
    
    print("\n✅ FERTIG! CSV gespeichert in: processed_data/htf_sample.csv")