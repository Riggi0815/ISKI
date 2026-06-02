# 🏎️ Sim Racing Driver Identification - Machine Learning Project

Machine Learning System zur Identifikation von Sim-Racing-Fahrern anhand ihrer Telemetriedaten. Das System analysiert Fahrverhaltensmerkmale wie Brems-, Lenk- und Beschleunigungsmuster, um Fahrer mit hoher Genauigkeit zu erkennen.

## 📋 Inhaltsverzeichnis

- [Projektübersicht](#-projektübersicht)
- [Projektstruktur](#-projektstruktur)
- [Installation](#-installation)
- [Komplette Pipeline](#-komplette-pipeline)
- [Script-Details](#-script-details)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Projektübersicht

### Kernfunktionen

- **HTF Format**: HTF (iRacing, text-basiert) Telemetriedaten
- **8 Fahrer**: ALAD201, CHIPZ26000, INBWZ11002, MAAKZ19001, MAMCZ06001, NIMB230, PASZZ20000, TOINZ27000 (~1.08M Telemetrie-Samples)
- **29 Features**: Statistische, Verhaltens- und Frequenz-Merkmale (aus 92 extrahierten)
- **Random Forest**: 96.24% Test-Accuracy (Runden-basierter Split)
- **Leave-One-Out Evaluation**: Test auf nie gesehenen Fahrern (Open-Set Recognition)
- **Real-time Prediction**: HTF Dateien automatisch verarbeiten und klassifizieren

### Wissenschaftlicher Ansatz

✅ Time-series Segmentierung (10 Sekunden = 500 Samples @ 50Hz)  
✅ Feature Engineering mit 92 Merkmalen pro Segment  
✅ Random Forest Classifier (kein Scaling nötig, eingebaute Feature Importance)  
✅ Runden-basierter Train/Test Split (Runden 1,2,3,4,5,8 / Runden 6,7)  
✅ Confusion Matrix + Feature Importance Visualisierung

---

## 📁 Projektstruktur

```
ISKI/
├── raw_data/                          # 🔴 Original Roh-Telemetriedaten
│   └── *.htf                          # HTF-Dateien (15 Dateien, 8 Fahrer)
│
├── training_data/                     # 🟣 Training Split (80%)
│   └── *.htf                          # 12 HTF Dateien
│
├── test_data/                         # 🟤 Test Split (20%)
│   └── *.htf                          # 3 HTF Dateien (RINE150, SOMD122, THTH312)
│
├── processed_data/                    # 🟡 Verarbeitete Telemetrie
│   ├── telemetry_all.pkl              # HTF Telemetrie (von training_data/)
│   └── telemetry_combined.pkl         # Vorbereitete HTF-Daten für Training
│
├── features/                          # 🟢 Extrahierte Features
│   └── driver_features_combined.pkl   # Feature-Sets (92 Features/Set)
│
├── models/                            # 🔵 Trainierte ML-Modelle
│   ├── combined/                      # Production Models (Training Data)
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── svm_model.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_metadata.json
│   └── leave_one_out/                 # Generalisierungs-Test (LOOCV)
│       └── [models + metadata]
│
├── results/                           # 📊 Evaluationsergebnisse
│   ├── 00_data_overview.txt
│   ├── test_evaluation_*.txt          # Test Data Performance
│   └── leave_one_out/
│
├── scripts/                           # 🟠 Python Pipeline
│   ├── 00_split_raw_data.py          # Train/Test Split
│   ├── 00_data_overview.py           # Daten-Inventar
│   ├── 01_parse_htf.py               # HTF Parser (training_data/)
│   ├── 03a_combine_data.py           # HTF Daten vorbereiten
│   ├── 03b_feature_engineering_combined.py  # Feature Extraction
│   ├── 04b_train_models_combined.py  # Model Training
│   ├── 05_predict.py                 # Prediction auf neuen HTF-Daten
│   ├── 06_leave_one_out_evaluation.py  # Unseen Driver Test
│   └── 07_test_evaluation.py         # Test Set Evaluation
│
├── Professoren_Fragen.md              # 📝 Fragen für Academic Review
└── requirements.txt                   # Python Dependencies
```

---

## 🔧 Installation

### Voraussetzungen

- **Python 3.9+** (getestet mit 3.13)
- **Windows** (PowerShell/cmd)

### Setup

```powershell
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Projektstruktur prüfen
dir raw_data      # Sollte .htf und .ld Dateien enthalten
```

**requirements.txt:**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 🚀 Komplette Pipeline

### Übersicht

```
RAW DATA → SPLIT → PARSING → PREPARATION → FEATURES → TRAINING → TEST
   ↓        ↓        ↓           ↓            ↓           ↓        ↓
  .htf    train/   .pkl     combined.pkl  features.pkl models/  results/
           test/
```

### Schritt-für-Schritt Anleitung

#### **Schritt 0a: Train/Test Split** ⚡ NEU

```powershell
py -3 scripts\00_split_raw_data.py
```

**Input:** `raw_data/*.htf` (15 Dateien)  
**Output:**

- `training_data/` (80%): 12 HTF Dateien
- `test_data/` (20%): 3 HTF Dateien (RINE150, SOMD122, THTH312)

**Dauer:** ~5 Sekunden  
**Wichtig:** Kopiert Dateien (raw_data bleibt intakt), deterministische Sortierung

---

#### **Schritt 0b: Daten-Übersicht** (Optional)

```powershell
py -3 scripts\00_data_overview.py
```

**Output:** `results/00_data_overview.txt`  
Zeigt alle verfügbaren Fahrer, Sample-Verteilung und Imbalance-Ratio.

---

#### **Schritt 1: HTF Parsing**

```powershell
py -3 scripts\01_parse_htf.py
```

**Input:** `training_data/*.htf` (12 Dateien)  
**Output:** `processed_data/telemetry_all.pkl` (~1.08M Training HTF Samples, 8 Fahrer)  
**Dauer:** ~20-30 Sekunden

---

#### **Schritt 3: Daten vorbereiten**

```powershell
py -3 scripts\03a_combine_data.py
```

**Input:** `telemetry_all.pkl`  
**Output:** `processed_data/telemetry_combined.pkl` (8 Fahrer, alle HTF Channels)  
**Dauer:** ~5-10 Sekunden

**Funktion:** Bereitet HTF-Daten für Feature Engineering vor, fügt data_source Spalte hinzu

---

#### **Schritt 4: Feature Engineering**

```powershell
py -3 scripts\03b_feature_engineering_combined.py
```

**Input:** `telemetry_combined.pkl`  
**Output:** `features/driver_features_combined.pkl` (2,157 Feature-Sets, 92 Features)  
**Dauer:** ~2-3 Minuten

**Segment-Größe:** 500 Samples (10 Sekunden @ 50Hz)

**Features (92 pro Segment):**

- **Geschwindigkeit**: mean, std, min, max, CV
- **Lateral/Longitudinal G**: mean, std, min, max, extreme_pct, corner_count
- **Throttle/Brake**: mean, std, changes, smoothness, aggressive
- **Motor**: mean, std, max RPM
- **Reifen**: Temperatur- und Druckdifferenzen (front/rear, left/right)
- **Gear Ratio**: Fahrweise-Signatur (speed/RPM)

---

#### **Schritt 5: Model Training**

```powershell
py -3 scripts\04b_train_models_combined.py
```

**Input:** `features/driver_features_combined.pkl`  
**Output:** `models/combined/` (Modell + Scaler + Encoder + Metadata)  
**Dauer:** ~1-2 Minuten

**Modell:**

- Random Forest (n_estimators=200, max_depth=None, class_weight=balanced)
- **29 Features** werden tatsächlich verwendet (wichtigste aus 92)

**Split:** Runden-basiert — Runden 1,2,3,4,5,8 Training (75.3%) / Runden 6,7 Test (24.7%)

**Output-Dateien:**

- `random_forest_model.pkl`
- `scaler.pkl` (StandardScaler für Test-Kompatibilität)
- `label_encoder.pkl` (8 Fahrer)
- `model_metadata.json` (Feature-Namen, Hyperparameter, Performance)

---

#### **Schritt 6: Prediction auf einzelnen Dateien**

```powershell
# Bekannter Fahrer aus Training-Daten testen
py -3 scripts\05_predict.py training_data\ALAD201.htf --model random_forest

# Unbekannter Fahrer aus Test-Daten (nie im Training gesehen)
py -3 scripts\05_predict.py test_data\RINE150.htf --model random_forest
py -3 scripts\05_predict.py test_data\SOMD122.htf --model random_forest
py -3 scripts\05_predict.py test_data\THTH312.htf --model random_forest

# Beliebige HTF-Datei (auch aus raw_data/)
py -3 scripts\05_predict.py raw_data\NIMB230.htf --model random_forest
```

**Verfügbare Modelle:**

- `--model random_forest` (Standard, beste Performance)
- `--model xgboost` (falls trainiert)
- `--model svm` (falls trainiert)

**Output:** `results/prediction_*.txt`

**Methodik:**

1. Parse HTF file → Telemetrie-Daten
2. Feature Extraction → 29 Features pro Segment (500 Samples)
3. Segment-wise Prediction → Jedes 10s-Fenster wird klassifiziert
4. Majority Voting → Häufigste Vorhersage gewinnt
5. Confidence Score → Durchschnittliche Modell-Sicherheit
6. Agreement % → Wie viele Segmente stimmen überein

**Interpretation:**

- **Confidence > 60%** + **Agreement > 80%**: Bekannter Fahrer (RICHTIG)
- **Confidence < 30%** + **Agreement > 95%**: Unbekannter Fahrer (wird zu ähnlichstem Fahrer zugeordnet)
- **Verteilte Predictions**: Datei enthält mehrere Fahrer oder schlechte Datenqualität

---

#### **Schritt 7: Test Set Evaluation** ⚡ NEU

```powershell
# Evaluiere XGBoost (default)
py -3 scripts\07_test_evaluation.py

# Andere Modelle
py -3 scripts\07_test_evaluation.py --model random_forest
py -3 scripts\07_test_evaluation.py --model svm
```

**Input:** `test_data/*.htf` (3 Dateien: RINE150, SOMD122, THTH312 — nie im Training gesehen)  
**Output:** `results/test_evaluation_*.txt`

**Zweck:** Performance auf **echten Holdout-Daten** messen (externe Validierung)

**Methodik:**

1. Parse alle Test-Dateien
2. Feature Extraction
3. Predict mit trainiertem Modell
4. Berechne Accuracy, Confidence, Agreement pro Datei
5. Overall Statistics

---

#### **Schritt 8: Leave-One-Driver-Out Evaluation**

```powershell
py -3 scripts\06_leave_one_out_evaluation.py
```

**Zweck:** Test auf **nie gesehenen Fahrern** (Open-Set Recognition)

**Methodik:**

1. Wähle kleinsten Fahrer als Holdout (_NIMB230_: 42 Segmente)
2. Trainiere auf 7 restlichen Fahrern (2,115 Segmente)
3. Evaluiere auf Holdout: **Confusion-Analyse** (nicht Accuracy!)
4. Measure: Welche bekannten Fahrer werden verwechselt? + Confidence
5. Sanity Check: Teste auf bekannten Fahrern (sollte >95% sein)

**Output:**

- `results/leave_one_out/evaluation_NIMB230.txt`
- `results/leave_one_out/prediction_distribution_random_forest_NIMB230.png`
- `models/leave_one_out/` (Modelle ohne Holdout-Fahrer: RF, XGBoost, SVM)
- `models/leave_one_out/metadata.json` (Holdout-Info, Performance)

**Interpretation:**

- **Niedrige Confidence** (z.B. 31.8%) = **GUT** → Modell erkennt "Outlier"
- **Hohe Confidence** (z.B. 80%) = Ähnlicher Fahrstil zu bekanntem Fahrer
- Verteilte Confusion = Fahrer passt zu keinem bekannten Fahrer

---

## 📊 Model Performance

### Closed-Set Classification (8 Fahrer, alle trainiert)

| Model         | Train Accuracy | Test Accuracy | Split              |
| ------------- | -------------- | ------------- | ------------------ |
| Random Forest | **99.69%**     | **96.24%**    | Runden 1-5,8 / 6,7 |

**Train/Test Split:**

- Training: 1,625 Segmente (75.3%)
- Test: 532 Segmente (24.7%)

**Hinweis:** Sehr gute Generalisierung auf ungesehene Runden derselben Fahrer.

**Top Features:**

1. tire_temp_diff_fr (10.0%)
2. n_engine_mean (9.9%)
3. n_engine_max (9.8%)
4. throttle_smoothness (9.5%)
5. throttle_changes (8.1%)

### Open-Set Recognition (Leave-One-Out)

**Holdout:** _NIMB230_ (42 Segmente, **nie** im Training)

| Model         | Avg Confidence | Most Confused | Confusion % | Known Accuracy |
| ------------- | -------------- | ------------- | ----------- | -------------- |
| Random Forest | **50.9%** ✅   | _MAAKZ19001_  | 100.0%      | 99.05%         |
| XGBoost       | **98.2%**      | _ALAD201_     | 100.0%      | 99.48%         |
| SVM           | N/A            | _ALAD201_     | 100.0%      | 98.39%         |

**Sanity Check (bekannte 7 Fahrer):** Alle Modelle >98% → korrekt trainiert

**Key Insight:**  
Random Forest zeigt **niedrige Confidence** bei unbekannten Fahrern → kann für "Neuer Fahrer"-Detektion verwendet werden (Threshold: < 60%)

---

## 🔬 Script-Details

### 00_data_overview.py

Erstellt Inventar aller verfügbaren Fahrer in `raw_data/`. Zeigt Sample-Verteilung, Imbalance-Ratio und Empfehlungen.

### 01_parse_htf.py

Parst text-basierte HTF-Dateien (iRacing). Extrahiert 21 Telemetrie-Channels pro Sample.

### 03a_combine_data.py

Bereitet HTF-Daten für Feature Engineering vor. Lädt `telemetry_all.pkl` und erstellt `telemetry_combined.pkl`.

### 03b_feature_engineering_combined.py

Extrahiert 92 Features pro 10-Sekunden-Segment (500 Samples @ 50Hz). Training nutzt die 29 wichtigsten Features.

### 04b_train_models_combined.py

Trainiert Random Forest (kein Scaler) mit runden-basiertem Split. Speichert Confusion Matrix und Feature Importance als PNG.

### 05_predict.py

Prediction auf neuen HTF-Dateien mit segment-wise voting. Unterstützt Random Forest, XGBoost und SVM.

**Wichtig:** Test-Dateien (RINE150, SOMD122, THTH312) waren NICHT im Training und werden als "unbekannt" erkannt (niedrige Confidence).

### 06_leave_one_out_evaluation.py

Leave-One-Driver-Out Cross-Validation für Generalisierungs-Test auf unseen drivers.

---

## 🛠️ Troubleshooting

### RuntimeWarning: invalid value encountered in skew/kurtosis

**Normal!** Tritt auf wenn Daten nahezu identisch (z.B. Reifendruck auf Geraden). Wird automatisch mit `nan_to_num()` behandelt.

### KeyError: 'v_car' / Column-Fehler

Column-Mapping zwischen HTF und LD nicht konsistent. Prüfe `03a_combine_data.py` → `ld_to_htf_mapping`.

### FileNotFoundError: driver_features_combined.pkl

Features noch nicht extrahiert. Führe zuerst Schritt 4 aus: `py -3 scripts\03b_feature_engineering_combined.py`

### Prediction Agreement < 50%

File enthält möglicherweise mehrere Fahrer oder unvollständige Daten. Prüfe Segment-Count in Output.

### ValueError: y contains previously unseen labels

In Leave-One-Out Evaluation: **Erwartet!** Holdout-Fahrer kann nicht encodiert werden (nicht im Training). Skript behandelt dies korrekt durch Confusion-Analyse statt Accuracy.

---

## 📚 Weitere Ressourcen

- **Professoren_Fragen.md**: 30 Fragen für Academic Review (Datenqualität, Methodik, Validierung)
- **Model Metadata**: `models/combined/model_metadata.json` (Hyperparameter, Performance)
- **Visualisierungen**: Confusion Matrices, Feature Importance, Per-Driver Performance

---

## 🎓 Für Professoren

### Wichtige Punkte für Academic Review

1. **Datenqualität**: ~1.08M Samples, 8 Fahrer (nur HTF-Daten, eine Datei nicht parsebar)
2. **Methodologie**: Time-series Segmentierung (10s), 29 ausgewählte Features, Random Forest
3. **Validierung**: Runden-basierter Train/Test Split (75/25) + Leave-One-Out + Test-Set Evaluation
4. **Open-Set Recognition**: Random Forest zeigt niedrige Confidence bei unbekannten Fahrern
5. **Limitationen**: Single Track/Vehicle, kein Temporal Modeling (LSTM), geschlossene Fahrer-Menge

### Nächste Schritte

- [ ] Full LOOCV (alle 8 Fahrer als Holdout testen)
- [ ] Class Balancing (SMOTE, Undersampling für kleinere Fahrer)
- [ ] Deep Learning (LSTM, 1D-CNN für Temporal Dependencies)
- [ ] Multi-Track/Vehicle Testing (verschiedene Strecken/Autos)
- [ ] Real-time Streaming Prediction
- [ ] Feature-Selection Optimierung (aktuell 29/92 Features)
- [ ] Ensemble-Methods (Voting/Stacking von RF, XGBoost, SVM)

---

**Viel Erfolg! 🚀**

Bei Fragen oder Problemen:

1. Prüfe `results/*_summary.txt` für Details
2. Lies Troubleshooting-Section
3. Kontaktiere Projekt-Maintainer

---

**Version**: 1.0  
**Letzte Aktualisierung**: Juni 2026  
**Python**: 3.13  
**Status**: ✅ Produktionsbereit
