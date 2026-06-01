# Sim Racing Driver Identification - Machine Learning Project

Dieses Projekt verwendet Machine Learning, um Sim-Racing-Fahrer anhand ihrer Telemetriedaten zu identifizieren. Das System analysiert Fahrverhaltensmerkmale wie Brems-, Lenk- und Beschleunigungsmuster, um Fahrer mit hoher Genauigkeit zu erkennen.

## 📋 Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Projektstruktur](#projektstruktur)
- [Anforderungen](#anforderungen)
- [Installation](#installation)
- [Schnellstart - Komplette Pipeline](#schnellstart---komplette-pipeline)
- [Detaillierte Anleitung](#detaillierte-anleitung)
- [Script-Übersicht](#script-übersicht)
- [Verwendung der Prediction](#verwendung-der-prediction)
- [Ergebnisse](#ergebnisse)

---

## 🎯 Projektübersicht

### Was macht dieses Projekt?

Das System kann Sim-Racing-Fahrer anhand ihrer Telemetriedaten identifizieren:

- **Datenquellen**: HTF (text-basiert) und LD (binär, Assetto Corsa) Telemetriedateien
- **Fahrer**: Aktuell 11 verschiedene Fahrer trainiert
- **Accuracy**: ~71% bei 11 Fahrern (XGBoost), ~96% bei 6 Fahrern (SVM)
- **Features**: 171 extrahierte Verhaltensmerkmale pro 10-Sekunden-Segment
- **Modelle**: Random Forest, SVM, XGBoost

### Hauptmerkmale

✅ **Unterstützt zwei Dateiformate**: HTF und LD  
✅ **Feature Engineering**: 171 Features aus Fahrverhalten extrahiert  
✅ **Mehrere ML-Modelle**: Vergleich von RF, SVM und XGBoost  
✅ **Segmentierung**: 10-Sekunden-Fenster (500 Samples @ 50Hz)  
✅ **Prediction**: Voting-basierte Fahrererkennung mit Konfidenz-Score

---

## 📁 Projektstruktur

```
ISKI/
├── raw_data/                          # Roh-Telemetriedaten
│   ├── *.htf                          # HTF-Dateien (10 Dateien, 6 Fahrer)
│   └── *.ld                           # LD-Dateien (5 Dateien, 5 Fahrer)
│
├── processed_data/                    # Verarbeitete Daten
│   ├── telemetry_all.pkl/csv         # HTF Telemetrie (1M Samples)
│   ├── telemetry_ld.pkl/csv          # LD Telemetrie (249K Samples)
│   └── telemetry_combined.pkl/csv    # Kombiniert HTF+LD (1.3M Samples)
│
├── features/                          # Extrahierte Features
│   ├── driver_features.pkl/csv       # HTF-only Features (2056 Sets)
│   └── driver_features_combined.pkl  # HTF+LD Features (2552 Sets)
│
├── models/                            # Trainierte Modelle
│   ├── *.pkl                          # HTF-only Modelle (6 Fahrer)
│   └── combined/                      # HTF+LD Modelle (11 Fahrer)
│       ├── random_forest_model.pkl
│       ├── svm_model.pkl
│       ├── xgboost_model.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       └── model_metadata.json
│
├── results/                           # Ergebnisse & Reports
│   ├── *_summary.txt                  # Pipeline-Zusammenfassungen
│   ├── *_combined.txt                 # Combined-Model Reports
│   ├── *.png                          # Visualisierungen
│   └── prediction_*.txt               # Prediction-Ergebnisse
│
├── scripts/                           # Python-Scripts
│   ├── utils.py                       # Helper-Funktionen
│   ├── 01_parse_htf.py               # HTF Parser
│   ├── 02_parse_ld.py                # LD Parser
│   ├── 03a_combine_data.py           # Daten kombinieren
│   ├── 03b_feature_engineering_combined.py  # Features extrahieren
│   ├── 03_feature_engineering.py     # (Original, für HTF-only)
│   ├── 04b_train_models_combined.py  # Modelle trainieren
│   ├── 04_train_models.py            # (Original, für HTF-only)
│   ├── 05_predict.py                 # Prediction (HTF + LD)
│   └── 06_evaluate.py                # Model-Evaluation
│
└── README.md                          # Diese Datei
```

---

## 💻 Anforderungen

### Software

- **Python 3.13** (oder 3.9+)
- **PowerShell** oder **cmd** (Windows)

### Python-Packages

```bash
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
seaborn
joblib
```

---

## 🚀 Installation

### 1. Python installieren

Stelle sicher, dass Python 3 installiert ist:

```powershell
py -3 --version
```

### 2. Dependencies installieren

Installiere alle benötigten Packages:

```powershell
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib
```

**Oder mit Requirements-Datei** (falls erstellt):

```powershell
pip install -r requirements.txt
```

### 3. Projekt-Ordner prüfen

Stelle sicher, dass `raw_data/` deine Telemetriedateien enthält:

```powershell
dir raw_data
```

Du solltest `.htf` und `.ld` Dateien sehen.

---

## ⚡ Schnellstart - Komplette Pipeline

Führe alle Scripts der Reihe nach aus, um das komplette System zu trainieren:

```powershell
# 0. Navigiere ins Projektverzeichnis
cd E:\Master\ISKI\ISKI

# 1. HTF-Dateien parsen
py -3 scripts\01_parse_htf.py

# 2. LD-Dateien parsen
py -3 scripts\02_parse_ld.py

# 3. HTF + LD Daten kombinieren
py -3 scripts\03a_combine_data.py

# 4. Features extrahieren (kombinierte Daten)
py -3 scripts\03b_feature_engineering_combined.py

# 5. Modelle trainieren (alle 11 Fahrer)
py -3 scripts\04b_train_models_combined.py

# 6. Prediction testen
py -3 scripts\05_predict.py "raw_data\00f946d7-504b-4a0d-8314-fdbe1d58d4c8.htf" --model svm

# 7. (Optional) Evaluation & Visualisierungen erstellen
py -3 scripts\06_evaluate.py
```

**⏱ Geschätzte Dauer**: 5-10 Minuten (je nach Hardware)

---

## 📖 Detaillierte Anleitung

### Schritt 1: HTF-Dateien parsen (01_parse_htf.py)

**Was macht es?**  
Liest text-basierte HTF-Telemetriedateien und konvertiert sie in strukturierte DataFrames.

**Befehl:**

```powershell
py -3 scripts\01_parse_htf.py
```

**Output:**

- `processed_data/telemetry_all.pkl` - Pickle-Format (schnell)
- `processed_data/telemetry_all.csv` - CSV-Format (lesbar)
- `results/01_htf_parsing_summary.txt` - Zusammenfassung

**Erwartete Ausgabe:**

```
Successfully parsed: 9/10 files
Total samples: 1,029,209
Unique drivers: 6
```

---

### Schritt 2: LD-Dateien parsen (02_parse_ld.py)

**Was macht es?**  
Liest binäre LD-Telemetriedateien (Assetto Corsa) und konvertiert sie.

**Befehl:**

```powershell
py -3 scripts\02_parse_ld.py
```

**Output:**

- `processed_data/telemetry_ld.pkl/csv`
- `results/02_ld_parsing_summary.txt`

**Erwartete Ausgabe:**

```
Successfully parsed: 5/5 files
Total samples: 248,922
Unique drivers: 5
```

---

### Schritt 3: Daten kombinieren (03a_combine_data.py)

**Was macht es?**  
Kombiniert HTF (6 Fahrer) + LD (5 Fahrer) = 11 Fahrer für umfassendes Training.

**Befehl:**

```powershell
py -3 scripts\03a_combine_data.py
```

**Output:**

- `processed_data/telemetry_combined.pkl/csv`
- `results/03b_data_combination_summary.txt`

**Erwartete Ausgabe:**

```
Combined dataset: 1,278,131 samples
Unique drivers: 11
Common telemetry channels: 19
```

---

### Schritt 4: Feature Engineering (03b_feature_engineering_combined.py)

**Was macht es?**  
Extrahiert 171 Verhaltensfeatures pro 10-Sekunden-Segment:

- Brems-/Lenkverhalten
- G-Kräfte
- Reifenmanagement
- FFT-Frequenzanalyse

**Befehl:**

```powershell
py -3 scripts\03b_feature_engineering_combined.py
```

**Output:**

- `features/driver_features_combined.pkl/csv`
- `results/03b_feature_engineering_combined_summary.txt`

**Erwartete Ausgabe:**

```
Total feature sets: 2552
Features per set: 171
Segments per driver: 84-545
```

⚠️ **Hinweis**: RuntimeWarnings für `skew`/`kurtosis` sind normal und werden automatisch behandelt.

---

### Schritt 5: Modelle trainieren (04b_train_models_combined.py)

**Was macht es?**  
Trainiert 3 ML-Modelle mit allen 11 Fahrern:

- Random Forest
- SVM (Support Vector Machine)
- XGBoost

**Befehl:**

```powershell
py -3 scripts\04b_train_models_combined.py
```

**Output:**

- `models/combined/random_forest_model.pkl`
- `models/combined/svm_model.pkl`
- `models/combined/xgboost_model.pkl`
- `models/combined/scaler.pkl` & `label_encoder.pkl`
- `results/training_results_combined.json`
- `results/04_model_comparison_combined.txt`

**Erwartete Ausgabe:**

```
Random Forest: 99% train, 87% test
SVM: 47% train, 38% test
XGBoost: 98% train, 71% test  ← Bestes Modell
```

**⏱ Dauer**: 2-5 Minuten (abhängig von CPU)

---

### Schritt 6: Prediction testen (05_predict.py)

**Was macht es?**  
Identifiziert Fahrer aus neuen Telemetriedateien (HTF oder LD).

**Befehle:**

#### HTF-Datei:

```powershell
py -3 scripts\05_predict.py "raw_data\00f946d7-504b-4a0d-8314-fdbe1d58d4c8.htf" --model svm
```

#### LD-Datei:

```powershell
py -3 scripts\05_predict.py "raw_data\ks_nurburgring_&_ks_porsche_911_gt3_rs_&_ALAD201_&_stint_1.ld" --model svm
```

**Optionen:**

- `--model`: `random_forest`, `svm`, oder `xgboost` (default: `svm`)
- `--confidence-threshold`: Mindest-Konfidenz (default: `0.6`)
- `--model-dir`: Custom model directory

**Output:**

- Konsolen-Ausgabe mit Prediction-Ergebnis
- `results/prediction_<filename>.txt`

**Beispiel-Ausgabe:**

```
✓ KNOWN DRIVER DETECTED
  Driver ID: _ALAD201_
  Confidence: 100.0%
  Agreement: 58.1% of segments (68/117)
```

---

### Schritt 7: Evaluation (06_evaluate.py) - Optional

**Was macht es?**  
Erstellt umfassende Visualisierungen und Reports.

**Befehl:**

```powershell
py -3 scripts\06_evaluate.py
```

**Output:**

- `results/06_confusion_matrices.png`
- `results/06_model_comparison.png`
- `results/06_per_driver_performance.png`
- `results/06_feature_importance.png`
- `results/06_classification_report.txt`
- `results/06_evaluation_summary.txt`

---

## 🔧 Script-Übersicht

| Script                                  | Zweck              | Input              | Output                                  |
| --------------------------------------- | ------------------ | ------------------ | --------------------------------------- |
| **utils.py**                            | Helper-Funktionen  | -                  | Importiert von allen anderen Scripts    |
| **01_parse_htf.py**                     | HTF Parser         | `raw_data/*.htf`   | `processed_data/telemetry_all.pkl`      |
| **02_parse_ld.py**                      | LD Parser          | `raw_data/*.ld`    | `processed_data/telemetry_ld.pkl`       |
| **03a_combine_data.py**                 | Daten kombinieren  | HTF + LD           | `processed_data/telemetry_combined.pkl` |
| **03b_feature_engineering_combined.py** | Feature-Extraktion | Combined telemetry | `features/driver_features_combined.pkl` |
| **04b_train_models_combined.py**        | ML Training        | Features           | `models/combined/*.pkl`                 |
| **05_predict.py**                       | Prediction         | HTF/LD Datei       | Konsole + `results/prediction_*.txt`    |
| **06_evaluate.py**                      | Evaluation         | Modelle + Features | `results/*.png` & `.txt`                |

---

## 🎮 Verwendung der Prediction

### Syntax

```powershell
py -3 scripts\05_predict.py <DATEI> [OPTIONS]
```

### Beispiele

#### 1. Mit SVM Model (empfohlen für wenige Fahrer)

```powershell
py -3 scripts\05_predict.py "raw_data\00f946d7-504b-4a0d-8314-fdbe1d58d4c8.htf" --model svm
```

#### 2. Mit XGBoost (empfohlen für viele Fahrer)

```powershell
py -3 scripts\05_predict.py "raw_data\ks_nurburgring_&_ks_porsche_911_gt3_rs_&_ALAD201_&_stint_1.ld" --model xgboost
```

#### 3. Mit Custom Confidence Threshold

```powershell
py -3 scripts\05_predict.py "raw_data\0afc3817-a5b6-4bbf-b6ae-79c6e5c4e881.htf" --model svm --confidence-threshold 0.8
```

### Parameter

| Parameter                | Beschreibung          | Default            | Optionen                          |
| ------------------------ | --------------------- | ------------------ | --------------------------------- |
| `<DATEI>`                | Pfad zur HTF/LD Datei | _erforderlich_     | `.htf` oder `.ld`                 |
| `--model`                | ML-Modell             | `svm`              | `random_forest`, `svm`, `xgboost` |
| `--confidence-threshold` | Mindest-Konfidenz     | `0.6`              | `0.0` - `1.0`                     |
| `--model-dir`            | Model Directory       | `models/combined/` | Beliebiger Pfad                   |

---

## 📊 Ergebnisse

### Trainierte Fahrer (11 total)

**HTF-Fahrer (6):**

- MAAKZ19001, CHIPZ26000, TOINZ27000, INBWZ11002, PASZZ20000, MAMCZ06001

**LD-Fahrer (5):**

- _ALAD201_, _NIMB230_, _RINE150_, _SOMD122_, _THTH312_

### Model Performance (11 Fahrer)

| Modell            | Train Accuracy | Test Accuracy | Cross-Val      | Empfehlung       |
| ----------------- | -------------- | ------------- | -------------- | ---------------- |
| **Random Forest** | 99.09%         | 87.36%        | 88.20% ± 0.41% | ⭐ Sehr gut      |
| **SVM**           | 46.59%         | 37.60%        | 37.63% ± 0.22% | ❌ Schlecht      |
| **XGBoost**       | 98.15%         | **70.76%**    | 66.58% ± 2.49% | ✅ Beste Balance |

**Empfehlung**: Verwende **Random Forest** oder **XGBoost** für beste Ergebnisse.

### Wichtigste Features

Top 10 Features für Fahrererkennung:

1. `corner_count` - Anzahl Kurven
2. `g_lat_extreme_pct` - Laterale G-Kräfte
3. `t_tyreFR_min` - Reifentemperatur
4. `n_engine_kurtosis` - Motor-Drehzahl-Verteilung
5. `trail_brake_pct` - Trail-Braking Prozent
6. `v_car_min` - Minimale Geschwindigkeit
7. `n_engine_mean` - Durchschnittliche Drehzahl
8. `speed_cv` - Geschwindigkeits-Variationskoeffizient
9. `percent_throttle_skew` - Gas-Asymmetrie
10. `t_tyreRL_kurtosis` - Reifentemperatur-Verteilung

---

## 🐛 Troubleshooting

### Problem: "Module not found"

**Lösung:**

```powershell
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib
```

### Problem: "File not found" bei Prediction

**Lösung:**  
Verwende Anführungszeichen bei Dateinamen mit `&`:

```powershell
py -3 scripts\05_predict.py "raw_data\datei_&_mit_&_und.ld" --model svm
```

### Problem: RuntimeWarnings bei Feature Engineering

**Lösung:**  
Diese Warnings sind normal und werden automatisch behandelt. Sie treten auf bei konstanten Werten (z.B. Reifendruck auf Geraden).

### Problem: Niedrige Accuracy

**Ursachen:**

- Zu wenige Daten pro Fahrer
- Zu viele ähnliche Fahrer
- Falsche Hyperparameter

**Lösung:**

1. Mehr Daten sammeln (mehr Runden)
2. Andere Strecke/Fahrzeug verwenden (mehr Variation)
3. Hyperparameter in `04b_train_models_combined.py` anpassen

---

## 📝 Anmerkungen

### Daten-Segmentierung

- **Segment-Größe**: 500 Samples = 10 Sekunden @ 50Hz
- **Min. Daten**: 80% der Segment-Größe (400 Samples)
- **Warum?**: Erfasst temporale Fahrverhaltens-Muster

### Train/Test Split

- **70% Training / 30% Test**
- **Stratified**: Gleichmäßige Verteilung pro Fahrer
- **Random State**: 42 (reproduzierbar)

### Dateiformate

**HTF (Text):**

- Human-readable
- Header mit Metadaten
- Sparse data representation (forward-fill)

**LD (Binary):**

- Assetto Corsa native format
- Kompakt
- Metadaten aus Filename extrahiert

---

## 🚀 Next Steps

### Mehr Daten sammeln

```powershell
# Neue Dateien zu raw_data/ hinzufügen, dann:
py -3 scripts\01_parse_htf.py    # Für neue HTF
py -3 scripts\02_parse_ld.py     # Für neue LD
py -3 scripts\03a_combine_data.py
py -3 scripts\03b_feature_engineering_combined.py
py -3 scripts\04b_train_models_combined.py
```

### Hyperparameter Tuning

Bearbeite `scripts/04b_train_models_combined.py`:

```python
# Random Forest
trainer.train_random_forest(n_estimators=200, max_depth=15)

# XGBoost
trainer.train_xgboost(n_estimators=150, max_depth=8, learning_rate=0.05)
```

### Andere Strecken/Fahrzeuge

Das System funktioniert mit beliebigen Strecken/Fahrzeugen - Features basieren auf Fahrverhalten, nicht auf Strecken-Layout.

---

## 📧 Support

Bei Fragen oder Problemen:

1. Prüfe `results/*_summary.txt` für Details
2. Lies Troubleshooting-Section
3. Kontaktiere Projekt-Maintainer

---

**Version**: 1.0  
**Letzte Aktualisierung**: Juni 2026  
**Python**: 3.13  
**Status**: ✅ Produktionsbereit
