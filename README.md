# Sim Racing Driver Identification

Fahrererkennung anhand von Assetto Corsa Telemetriedaten (MoTeC `.ld` Format). Das System extrahiert Fahrverhaltensmuster aus Telemetrie-Segmenten und klassifiziert Fahrer mit einem Random Forest.

## Installation

```
pip install -r requirements.txt
```

## Pipeline

`.ld` Dateien in `raw_data/` ablegen, dann der Reihe nach ausführen:

### 1. LD-Dateien parsen

```
python scripts/02_parse_ld.py
```

Liest die `.ld` Binärdateien aus `raw_data/` via MoTeC-Parser (`ldparser.py`) und speichert die Telemetrie als `processed_data/telemetry_ld.pkl`. Mappt MoTeC-Kanalnamen auf interne Namen (z.B. `Ground Speed` → `v_car`).

### 2. Features extrahieren

```
python scripts/03b_feature_engineering_combined.py
```

Teilt die Telemetrie in 10-Sekunden-Segmente auf und berechnet pro Segment ~124 Features (Mittelwert, Standardabweichung, Throttle-Smoothness, Jerk, Reifentemperaturen, etc.). Speichert als `features/driver_features_combined.pkl`.

### 3. Modell trainieren

```
python scripts/04b_train_models_combined.py
```

Trainiert einen Random Forest auf den extrahierten Features. Split: Runden 1,2,3,4,5,8 als Training (~77%), Runden 6,7 als Test (~23%). Speichert Modell in `models/combined/`. Gibt Train/Test Accuracy, Confusion Matrix und Feature Importance aus.

**Aktuelles Ergebnis: 87.93% Test Accuracy (5 Fahrer)**

### 4. Einzelne Datei vorhersagen

```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest
```

Parst eine `.ld` Datei, extrahiert Features und sagt segmentweise den Fahrer vorher (Majority Voting). Mit `--test-only` werden nur die zurückgehaltenen Segmente (Runden 6+7) verwendet — das ist der faire Test auf ungesehenen Daten.

```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest --test-only
```

Ergebnis wird in `results/prediction_*.txt` gespeichert.

### 5. Alle Fahrer evaluieren

```
python scripts/07_test_evaluation.py --model random_forest
```

Läuft alle `.ld` Dateien durch und gibt True/Predicted, Agreement und Confidence pro Fahrer aus. Ergebnis in `results/test_evaluation_random_forest.txt`.

### 6. Leave-One-Out Test (optional)

```
python scripts/06_leave_one_out_evaluation.py
```

Trainiert das Modell ohne einen Fahrer und testet dann auf genau diesem. Misst ob das Modell einen unbekannten Fahrer erkennt (niedrige Confidence = gut). Ergebnis in `results/leave_one_out/`.

## Projektstruktur

```
raw_data/           ← .ld Dateien (Eingabe)
processed_data/     ← geparste Telemetrie (wird nicht gepusht)
features/           ← extrahierte Feature-Vektoren (wird nicht gepusht)
models/combined/    ← trainiertes Modell (wird nicht gepusht)
results/            ← Evaluation-Outputs
scripts/            ← Pipeline-Scripts
ldparser.py         ← MoTeC Binary Parser (Abhängigkeit von 02_parse_ld.py)
```

## Hinweise

- **Random Forest braucht keinen Scaler** — `05_predict.py` und `07_test_evaluation.py` skalieren nicht für RF
- **Confidence vs. Agreement**: Agreement = wie oft der richtige Fahrer erkannt wurde. Confidence = wie sicher das Modell pro Segment war (RF gibt generell niedrigere Confidence als SVM/XGBoost)
- **Fairer Test**: Nur `--test-only` oder die `04b` Ergebnisse sind valide — alles andere testet auf Trainingsdaten
