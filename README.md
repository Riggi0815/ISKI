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

- **Multi-Format Support**: HTF (iRacing, text-basiert) und LD (Assetto Corsa, binär)
- **11 Fahrer**: 6 HTF + 5 LD Fahrer (~1.3M Telemetrie-Samples)
- **171 Features**: Statistische, Verhaltens- und Frequenz-Merkmale
- **3 ML-Modelle**: Random Forest (87.36%), XGBoost (70.76%), SVM (37.60%)
- **Leave-One-Out Evaluation**: Test auf nie gesehenen Fahrern (Open-Set Recognition)
- **Real-time Prediction**: HTF/LD Dateien automatisch erkennen und klassifizieren

### Wissenschaftlicher Ansatz

✅ Time-series Segmentierung (10 Sekunden = 500 Samples @ 50Hz)  
✅ Feature Engineering mit 171 Merkmalen pro Segment  
✅ Multi-model Ensemble (Random Forest, XGBoost, SVM)  
✅ Leave-One-Driver-Out Cross-Validation für Generalisierung  
✅ Open-Set Recognition (erkennt unbekannte Fahrer durch niedrige Confidence)

---

## 📁 Projektstruktur

```
ISKI/
├── raw_data/                          # 🔴 Roh-Telemetriedaten
│   ├── *.htf                          # HTF-Dateien (10 Dateien, 6 Fahrer)
│   └── *.ld                           # LD-Dateien (5 Dateien, 5 Fahrer)
│
├── processed_data/                    # 🟡 Verarbeitete Telemetrie
│   ├── telemetry_all.pkl              # HTF Telemetrie (1,029,209 Samples)
│   ├── telemetry_all.pkl              # HTF Telemetrie (1,029,209 Samples)
│   ├── telemetry_ld.pkl               # LD Telemetrie (248,922 Samples)
│   └── telemetry_combined.pkl         # Kombiniert HTF+LD (1,278,131 Samples)
│
├── features/                          # 🟢 Extrahierte Features
│   └── driver_features_combined.pkl   # 2,552 Feature-Sets (171 Features/Set)
│
├── models/                            # 🔵 Trainierte ML-Modelle
│   ├── combined/                      # Production Models (11 Fahrer)
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── svm_model.pkl
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_metadata.json
│   └── leave_one_out/                 # Generalisierungs-Test (LOOCV)
│       └── [models + metadata]
│
├── scripts/                           # 🟠 Python Pipeline
│   ├── 00_data_overview.py           # Daten-Inventar
│   ├── 01_parse_htf.py               # HTF Parser
│   ├── 02_parse_ld.py                # LD Parser
│   ├── 03a_combine_data.py           # HTF+LD kombinieren
│   ├── 03b_feature_engineering_combined.py  # Feature Extraction
│   ├── 04b_train_models_combined.py  # Model Training
│   ├── 05_predict.py                 # Prediction auf neuen Daten
│   └── 06_leave_one_out_evaluation.py  # Unseen Driver Test
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
RAW DATA → PARSING → COMBINATION → FEATURES → TRAINING → PREDICTION
   ↓          ↓           ↓            ↓           ↓          ↓
  .htf      .pkl     combined.pkl  features.pkl models/   results/
  .ld
```

### Schritt-für-Schritt Anleitung

#### **Schritt 0: Daten-Übersicht** (Optional)

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

**Input:** `raw_data/*.htf` (10 Dateien)  
**Output:** `processed_data/telemetry_all.pkl` (1,029,209 Samples, 6 Fahrer)  
**Dauer:** ~30 Sekunden

---

#### **Schritt 2: LD Parsing**

```powershell
py -3 scripts\02_parse_ld.py
```

**Input:** `raw_data/*.ld` (5 Dateien)  
**Output:** `processed_data/telemetry_ld.pkl` (248,922 Samples, 5 Fahrer)  
**Dauer:** ~10 Sekunden

---

#### **Schritt 3: Daten kombinieren**

```powershell
py -3 scripts\03a_combine_data.py
```

**Input:** `telemetry_all.pkl` + `telemetry_ld.pkl`  
**Output:** `processed_data/telemetry_combined.pkl` (1,278,131 Samples, 11 Fahrer, 19 gemeinsame Channels)  
**Dauer:** ~15 Sekunden

**Mapping:** LD → HTF Column Names (z.B., `speed` → `v_car`)

---

#### **Schritt 4: Feature Engineering**

```powershell
py -3 scripts\03b_feature_engineering_combined.py
```

**Input:** `telemetry_combined.pkl`  
**Output:** `features/driver_features_combined.pkl` (2,552 Feature-Sets)  
**Dauer:** ~2-3 Minuten

**Features (171 pro Segment):**

- **Statistisch**: mean, std, min, max, skew, kurtosis (17 Channels × 6 = 102)
- **Verhalten**: jerk, steering_rate, throttle_changes, brake_events (41)
- **Frequenz**: FFT dominant frequency (17)
- **Relativ**: v_car/gear_ratio, tire_temp_diff (11)

---

#### **Schritt 5: Model Training**

```powershell
py -3 scripts\04b_train_models_combined.py
```

**Input:** `features/driver_features_combined.pkl`  
**Output:** `models/combined/` (3 Modelle + Scaler + Encoder)  
**Dauer:** ~3-5 Minuten

**Modelle:**

- Random Forest (n_estimators=200, max_depth=30)
- XGBoost (n_estimators=200, max_depth=10, learning_rate=0.1)
- SVM (Linear kernel, C=1.0)

**Split:** 80% Training / 20% Test

---

#### **Schritt 6: Prediction**

```powershell
# HTF Datei
py -3 scripts\05_predict.py raw_data\00f946d7-504b-4a0d-8314-fdbe1d58d4c8.htf

# LD Datei
py -3 scripts\05_predict.py "raw_data\ks_nurburgring_&_ks_porsche_911_gt3_rs_&_ALAD201_&_stint_1.ld"
```

**Output:** `results/prediction_*.txt`

**Methodik:**

1. Auto-detect Format (.htf vs .ld)
2. Parse → Feature Extraction
3. Predict mit allen 3 Modellen
4. Majority Voting → Final Prediction
5. Agreement % + Confidence Score

---

#### **Schritt 7: Leave-One-Out Evaluation**

```powershell
py -3 scripts\06_leave_one_out_evaluation.py
```

**Zweck:** Test auf **nie gesehenen Fahrern** (Open-Set Recognition)

**Methodik:**

1. Wähle kleinsten Fahrer als Holdout (_NIMB230_: 84 Samples)
2. Trainiere auf 10 restlichen Fahrern (2,468 Samples)
3. Evaluiere auf Holdout: **Confusion-Analyse** (nicht Accuracy!)
4. Measure: Welche bekannten Fahrer werden verwechselt? + Confidence

**Output:**

- `results/leave_one_out/evaluation__NIMB230_.txt`
- `results/leave_one_out/prediction_distribution_*.png`
- `models/leave_one_out/` (Modelle ohne Holdout-Fahrer)

**Interpretation:**

- **Niedrige Confidence** (z.B. 31.8%) = **GUT** → Modell erkennt "Outlier"
- **Hohe Confidence** (z.B. 80%) = Ähnlicher Fahrstil zu bekanntem Fahrer
- Verteilte Confusion = Fahrer passt zu keinem bekannten Fahrer

---

## 📊 Model Performance

### Closed-Set Classification (11 Fahrer, alle trainiert)

| Model         | Test Accuracy | Training Time |
| ------------- | ------------- | ------------- |
| Random Forest | **87.36%**    | ~2 min        |
| XGBoost       | 70.76%        | ~3 min        |
| SVM (Linear)  | 37.60%        | ~1 min        |

**Class Distribution:** 6.5:1 Imbalance (MAAKZ19001: 272,790 vs _NIMB230_: 42,157)

### Open-Set Recognition (Leave-One-Out)

**Holdout:** _NIMB230_ (84 Samples, **nie** im Training)

| Model         | Avg Confidence | Most Confused | Confusion % |
| ------------- | -------------- | ------------- | ----------- |
| Random Forest | **31.8%** ✅   | _THTH312_     | 33.3%       |
| XGBoost       | 59.2%          | _SOMD122_     | 27.4%       |
| SVM           | N/A            | _ALAD201_     | 31.0%       |

**Sanity Check (bekannte Fahrer):** RF 92.67%, XGB 94.17%, SVM 64.47%

**Key Insight:**  
Random Forest zeigt **niedrige Confidence** bei unbekannten Fahrern → kann für "Neuer Fahrer"-Detektion verwendet werden (Threshold: < 60%)

---

## 🔬 Script-Details

### 00_data_overview.py

Erstellt Inventar aller verfügbaren Fahrer in `raw_data/`. Zeigt Sample-Verteilung, Imbalance-Ratio und Empfehlungen.

### 01_parse_htf.py

Parst text-basierte HTF-Dateien (iRacing). Extrahiert 21 Telemetrie-Channels pro Sample.

### 02_parse_ld.py

Parst binäre LD-Dateien (Assetto Corsa). Extrahiert 19 Channels + Driver-ID aus Dateinamen.

### 03a_combine_data.py

Kombiniert HTF und LD Daten mit standardisiertem Column-Mapping (19 gemeinsame Channels).

### 03b_feature_engineering_combined.py

Extrahiert 171 Features pro 10-Sekunden-Segment (500 Samples @ 50Hz).

### 04b_train_models_combined.py

Trainiert Random Forest, XGBoost, SVM mit StandardScaler und LabelEncoder.

### 05_predict.py

Prediction auf neuen HTF/LD Dateien mit Majority Voting über 3 Modelle.

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

1. **Datenqualität**: 1.3M Samples, 11 Fahrer, 6.5:1 Imbalance
2. **Methodologie**: Time-series Segmentierung (10s), 171 Features, 3 ML-Modelle
3. **Validierung**: 80/20 Train-Test Split + Leave-One-Out für Generalisierung
4. **Open-Set Recognition**: Random Forest erkennt unbekannte Fahrer (31.8% Confidence)
5. **Limitationen**: Class Imbalance, Single Track/Vehicle, kein Temporal Modeling (LSTM)

### Nächste Schritte

- [ ] Full LOOCV (alle 11 Fahrer als Holdout testen)
- [ ] Class Balancing (SMOTE, Undersampling)
- [ ] Deep Learning (LSTM, 1D-CNN für Temporal Dependencies)
- [ ] Multi-Track/Vehicle Testing
- [ ] Real-time Streaming Prediction

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
