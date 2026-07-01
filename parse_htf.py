"""HTF Parser - Text-basiertes Format (Fixed Version v2 -- Index-Bug behoben)"""

from pathlib import Path
import re
import pandas as pd
import numpy as np


def parse_htf_file(filepath, fill_method="ffill"):
    """Parst eine .htf Datei zu DataFrame.

    fill_method: wie Luecken (fehlende Indizes) innerhalb eines Kanals
        aufgefuellt werden.
        - "ffill"        : letzten bekannten Wert fortschreiben (Standard,
                            passt zum Rest der bestehenden Pipeline)
        - "interpolate"  : linear zwischen den bekannten Nachbarwerten
                            interpolieren (glatter, da es echte Messsignale sind)
        - None           : Luecken bleiben NaN
    """

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
        count_raw = match.group(4).strip()
        data_string = match.group(5).strip()

        # Deklarierte Soll-Laenge aus dem Header (z.B. "750")
        count_digits = re.search(r'\d+', count_raw)
        declared_count = int(count_digits.group()) if count_digits else 0

        # --- DER FIX: Index UND Wert erfassen, statt nur den Wert ---
        # Vorher: r'\d+=([^;]+)'  -> Index war keine Capture-Group und wurde verworfen,
        #         dadurch wurden fehlende Indizes (Luecken) einfach uebersprungen
        #         und die Liste kompakt (zu kurz) aufgebaut.
        idx_val_pairs = []
        max_seen_idx = -1
        for value_match in re.finditer(r'(\d+)=([^;]+)', data_string):
            try:
                idx = int(value_match.group(1))
                val = float(value_match.group(2))
                idx_val_pairs.append((idx, val))
                if idx > max_seen_idx:
                    max_seen_idx = idx
            except ValueError:
                pass

        # Ziel-Laenge: das Maximum aus deklariertem Count und hoechstem gesehenen Index + 1
        # (schuetzt davor, Werte zu verlieren, falls der Header mal nicht stimmt)
        channel_length = max(declared_count, max_seen_idx + 1)

        # Werte an ihrer ECHTEN Position platzieren, Luecken zunaechst als NaN
        values = np.full(channel_length, np.nan)
        for idx, val in idx_val_pairs:
            values[idx] = val

        gefunden = len(idx_val_pairs)
        luecken = channel_length - gefunden

        channels_data[channel_name] = {
            'unit': unit,
            'data': values,
            'count_declared': declared_count,
            'count_found': gefunden,
        }

        if channel_length > max_length:
            max_length = channel_length

        print(f"\n📊 Channel: {channel_name}")
        print(f"   Unit: {unit}, deklarierte Länge: {declared_count}, gefundene Werte: {gefunden}, Lücken: {luecken}")
        if gefunden > 0:
            vals_only = [v for _, v in idx_val_pairs]
            print(f"   Min: {min(vals_only):.2f}, Max: {max(vals_only):.2f}, Mean: {sum(vals_only)/len(vals_only):.2f}")

    print(f"\n{'='*70}")
    print(f"📏 Maximale Länge: {max_length} Datenpunkte")
    print(f"{'='*70}\n")

    # 3. Baue DataFrame
    #    a) Luecken INNERHALB eines Kanals schliessen (fill_method)
    #    b) Kanaele, die insgesamt kuerzer als max_length sind, ans Ende auffuellen
    df_data = {}

    for channel_name, channel_info in channels_data.items():
        series = pd.Series(channel_info['data'])

        if fill_method == "ffill":
            series = series.ffill().bfill()
        elif fill_method == "interpolate":
            series = series.interpolate(limit_direction="both")

        if len(series) == 0:
            series = pd.Series([np.nan] * max_length)
        elif len(series) < max_length:
            pad_value = series.iloc[-1] if not pd.isna(series.iloc[-1]) else np.nan
            series = pd.concat(
                [series, pd.Series([pad_value] * (max_length - len(series)))],
                ignore_index=True,
            )

        df_data[channel_name] = series.values

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
