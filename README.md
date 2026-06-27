# Sim Racing Driver Identification

Fahrererkennung anhand von Assetto Corsa Telemetriedaten (MoTeC `.ld` Format). Das System extrahiert Fahrverhaltensmuster aus Telemetrie-Segmenten und klassifiziert Fahrer mit einem Random Forest.

**Aktuelles Ergebnis: 87.93% Test Accuracy (5 Fahrer)**

---

## Installation

```
pip install -r requirements.txt
```

---

## Pipeline

`.ld` Dateien in `raw_data/` ablegen, dann der Reihe nach ausführen:

### Schritt 1 — LD-Dateien parsen

```
python scripts/02_parse_ld.py
```

Liest die `.ld` Binärdateien aus `raw_data/` via `ldparser.py` und speichert die Telemetrie als `processed_data/telemetry_ld.pkl`. Mappt MoTeC-Kanalnamen auf interne Namen (z.B. `Ground Speed` → `v_car`).

### Schritt 2 — Features extrahieren

```
python scripts/03b_feature_engineering_combined.py
```

Teilt die Telemetrie in **10-Sekunden-Segmente** mit **5-Sekunden-Versatz** (50% Overlap, stride=250 bei 50 Hz) auf und berechnet pro Segment ~124 Features:

- Statistische Features: Mittelwert, Std, Min, Max, Skewness, Kurtosis je Kanal
- Verhaltensfeatures: Jerk, Lenkrate, Throttle-Smoothness, Bremsereignisse, Trail-Braking
- Frequenzfeatures: Dominante FFT-Frequenz für Lenkung, Gas, Querbeschleunigung
- Reifenfeatures: Temperatur- und Druckdifferenzen zwischen Achsen/Seiten
- Zonenfeatures: Is-Straight / Is-Eingang / Is-Mitte / Is-Apex (One-Hot)

Speichert als `features/driver_features_combined.pkl`.

### Schritt 3 — Modell trainieren

```
python scripts/04b_train_models_combined.py
```

Trainiert einen Random Forest auf den extrahierten Features.

**Train/Test-Split (round-basiert):**
- Training: Runden 1, 2, 3, 4, 5, 8 (~77% der Segmente)
- Test: Runden 6, 7 (~23% der Segmente)

Die Segmente eines Fahrers werden gleichmäßig in 8 Runden aufgeteilt. Runden 6 und 7 wurden während des gesamten Trainings nie gesehen.

Gibt aus: Train/Test Accuracy, Confusion Matrix, Feature Importance.
Speichert Modell in `models/combined/`.

### Schritt 4 — Auswertung / Vorhersage

**Alle Fahrer auf Test-Runden 6 & 7 auswerten (Hauptauswertung):**

```
python scripts/05_predict.py --model random_forest
```

Wertet für jeden Fahrer seine Test-Segmente (Runden 6+7) aus mit zonengewichtetem Majority Voting (Apex-Segmente zählen 3x, Geraden 1x). Speichert pro Fahrer eine Ergebnisdatei in `results/`.

**Einzelne .ld Datei vorhersagen:**

```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest
```

Parst eine `.ld` Datei, extrahiert Features und sagt den Fahrer via Majority Voting vorher.

```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest --test-only
```

Mit `--test-only` werden nur die Test-Segmente (Runden 6+7) verwendet — das ist der faire Test auf ungesehenen Daten.

---

## Optionale Auswertung

### Leave-One-Out Test

```
python scripts/06_leave_one_out_evaluation.py
```

Trainiert das Modell ohne einen Fahrer und testet dann auf genau diesem. Misst, ob ein unbekannter Fahrer erkannt wird (niedrige Confidence = gut). Ergebnis in `results/leave_one_out/`.

---

## Projektstruktur

```
raw_data/                           ← .ld Eingabedateien (Assetto Corsa / MoTeC)
processed_data/                     ← geparste Telemetrie (wird nicht gepusht)
features/                           ← extrahierte Feature-Vektoren + corner_zones.json
models/combined/                    ← trainiertes Modell (wird nicht gepusht)
results/                            ← Auswertungs-Outputs
scripts/
  utils.py                          ← Pfad-Hilfsfunktionen, CHANNEL_MAP
  corner_zones.py                   ← Kurven-Zonenerkennung (Eingang/Mitte/Apex)
  02_parse_ld.py                    ← Schritt 1: .ld Dateien parsen
  03b_feature_engineering_combined.py ← Schritt 2: Feature Extraktion
  04b_train_models_combined.py      ← Schritt 3: Modell trainieren
  05_predict.py                     ← Schritt 4: Vorhersage & Auswertung
  06_leave_one_out_evaluation.py    ← Optional: Open-Set Evaluation
ldparser.py                         ← MoTeC Binary Parser (Abhängigkeit von Schritt 1)
requirements.txt
```

---

## Nicht benötigte Dateien (Legacy)

Die folgenden Dateien sind **nicht Teil der Pipeline** und werden für die Abgabe nicht benötigt. Sie entstammen früheren Entwicklungsstufen:

| Datei | Warum nicht benötigt |
|---|---|
| `scripts/00_split_raw_data.py` | Alter Ansatz: Dateien manuell in `training_data/` und `test_data/` kopieren. Ersetzt durch round-basiertes Splitting direkt in `04b_train_models_combined.py`. |
| `scripts/00_data_overview.py` | Debug-Hilfstool, kein Schritt der Pipeline. |
| `scripts/01_parse_htf.py` | Parser für das HTF-Format (ein anderes Telemetrieformat). Es gibt keine `.htf` Dateien im Projekt — nur `.ld`. |
| `scripts/03a_combine_data.py` | Alter Schritt zum Zusammenführen von HTF- und LD-Daten. Nicht mehr relevant, da ausschließlich `.ld` Daten verwendet werden. |
| `scripts/07_test_evaluation.py` | Redundant mit `05_predict.py` (ohne Dateiargument). Liest aus `test_data/` (leer) und fällt auf `raw_data/` zurück — andere Logik als der round-basierte Split. |
| `scripts/plot_track.py` | Strecken-Visualisierung, kein Pipeline-Schritt. |
| `scripts/plot_track_corners.py` | Kurven-Visualisierung, kein Pipeline-Schritt. |
| `training_data/` | Aus altem file-basierten Split-Ansatz, wird nicht mehr verwendet. |
| `test_data/` | Aus altem file-basierten Split-Ansatz, leer und nicht verwendet. |

---

## Hinweise

- **Random Forest braucht keinen Scaler** — `05_predict.py` skaliert für RF nicht (wird intern korrekt behandelt)
- **Confidence vs. Agreement**: Agreement = wie oft der richtige Fahrer pro Segment erkannt wurde. Confidence = zonengewichteter Anteil der Stimmen für den Gewinner
- **Fairer Test**: Nur Runden 6+7 (`--test-only` oder Auswertung ohne Dateiargument) sind valide — alles andere testet auf Trainingsdaten
- **corner_zones.json**: Wird automatisch aus den LD-Dateien gebaut, wenn die Datei fehlt. Liegt in `features/corner_zones.json`
