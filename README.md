# Sim Racing Driver Identification

Fahrererkennung aus Assetto Corsa Telemetriedaten (MoTeC `.ld` Format). Der Random Forest klassifiziert Fahrer anhand von Fahrverhaltenssegmenten.

**Ergebnis: 24/25 Fahrer korrekt erkannt, Agreement 81.1%, Confidence 50.6%**

---

## Vorbereitung

1. Abhängigkeiten installieren:
   ```
   pip install -r requirements.txt
   ```

2. `.ld` Dateien in den Ordner `raw_data/` legen. Jede Datei steht für einen Fahrer. Die Dateien werden automatisch nach Fahrername gruppiert.

---

## Pipeline

Schritte der Reihe nach ausführen:

### Schritt 1 - LD-Dateien parsen

```
python scripts/02_parse_ld.py
```

Liest alle `.ld` Dateien aus `raw_data/` und speichert die Telemetrie als `processed_data/telemetry_ld.pkl`.

### Schritt 2 - Kurvenzonierung erstellen

```
python scripts/corner_zones.py
```

Wertet die Streckenkoordinaten aus und teilt die Strecke in Zonen auf (Eingang, Mitte, Apex, Gerade). Ergebnis liegt in `features/corner_zones.json`. Nur einmal noetig, solange sich die Strecke nicht aendert.

### Schritt 3 - Features extrahieren

```
python scripts/03b_feature_engineering_combined.py
```

Teilt die Telemetrie in 10-Sekunden-Segmente auf (5 Sekunden Versatz, 50% Overlap) und berechnet pro Segment ca. 45 Features aus Geschwindigkeit, Lenkung, Gas, Bremse, Reifentemperaturen und Zonenzugehoerigkeit. Stehende Segmente (unter 5 km/h) werden uebersprungen. Ergebnis: `features/driver_features_combined.pkl`.

### Schritt 4 - Modell trainieren

```
python scripts/04b_train_models_combined.py
```

Trainiert den Random Forest. Die Segmente jedes Fahrers werden in 8 gleiche Bloecke aufgeteilt:
- Training: Bloecke 1, 2, 3, 4, 5, 8 (ca. 77%)
- Test: Bloecke 6, 7 (ca. 23%)

Modell wird in `models/combined/` gespeichert.

### Schritt 5 - Auswertung

Alle Fahrer auf den Testdaten auswerten:
```
python scripts/05_predict.py --model random_forest
```

Einzelne Datei vorhersagen:
```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest
```

Nur auf ungesehenen Segmenten testen (Bloecke 6+7):
```
python scripts/05_predict.py "raw_data/<datei>.ld" --model random_forest --test-only
```

Ergebnisse werden in `results/` gespeichert.

---

## Optional: HTF-Daten einbinden

Falls zusaetzlich HTF-Telemetriedaten vorliegen, koennen diese vor Schritt 3 eingebunden werden.

### HTF-Dateien parsen

HTF-Dateien ebenfalls in `raw_data/` legen, dann:

```
python scripts/01_parse_htf.py
```

Speichert die HTF-Telemetrie als `processed_data/telemetry_htf.pkl`.

### HTF und LD zusammenfuehren

```
python scripts/03a_combine_data.py
```

Fuegt HTF- und LD-Daten zu `processed_data/telemetry_combined.pkl` zusammen. Schritt 3 (`03b_feature_engineering_combined.py`) laedt automatisch die kombinierte Datei, falls sie existiert, ansonsten nur die LD-Daten.

Danach normal mit Schritt 3 weitermachen.

---

## Projektstruktur

```
raw_data/                              <- .ld Dateien hier rein (eine pro Fahrer)
processed_data/                        <- geparste Telemetrie (nicht gepusht)
features/                              <- Feature-Vektoren und corner_zones.json
models/combined/                       <- trainiertes Modell (nicht gepusht)
results/                               <- Auswertungs-Outputs
scripts/
  02_parse_ld.py                       <- Schritt 1: .ld parsen
  corner_zones.py                      <- Schritt 2: Kurvenzonierung
  03b_feature_engineering_combined.py  <- Schritt 3: Features extrahieren
  04b_train_models_combined.py         <- Schritt 4: Modell trainieren
  05_predict.py                        <- Schritt 5: Vorhersage und Auswertung
  01_parse_htf.py                      <- Optional: HTF-Daten parsen
  03a_combine_data.py                  <- Optional: HTF + LD zusammenfuehren
  06_leave_one_out_evaluation.py       <- Optional: Open-Set Test
  utils.py                             <- Hilfsfunktionen
ldparser.py                            <- MoTeC Binary Parser
```

---

## Legacy-Dateien (nicht benoetigt)

| Datei | Grund |
|---|---|
| `scripts/00_split_raw_data.py` | Alter manueller Split-Ansatz, ersetzt durch round-basiertes Splitting in `04b`. |
| `scripts/00_data_overview.py` | Debug-Tool, kein Pipeline-Schritt. |
| `scripts/07_test_evaluation.py` | Redundant mit `05_predict.py`. |
| `scripts/plot_track.py` | Strecken-Visualisierung, kein Pipeline-Schritt. |
| `scripts/plot_track_corners.py` | Kurven-Visualisierung, kein Pipeline-Schritt. |
| `training_data/` | Alter Split-Ansatz, nicht mehr verwendet. |
| `test_data/` | Alter Split-Ansatz, leer. |
