# Experiment Log — Driver Identification (ISKI)

**Branch:** `Rick-Test`  
**Datum:** 2026-06-13

---

## Baseline — Runden-basiertes Splitting mit fixen 10s-Fenstern

**Ansatz:** Telemetrie in gleichmäßige 500-Sample-Fenster (~10s bei 50Hz), Aufteilung in Trainings-/Testrunden.  
**Modell:** Random Forest (n=200, class_weight='balanced')  
**Ergebnis:** 5/5 Fahrer korrekt identifiziert auf Trainingsdaten.  
**Problem:** Testgenauigkeit schwankte stark, da die "Runden" nicht sauber getrennt waren und das Modell Lap-Kontext lernte statt Fahrerstil.

---

## Experiment 1 — Corner-Phase Segmentierung (Eingang / Mitte / Apex)

**Was wurde geändert:**
- Strecke per Kurvenkrümmung in Kurven segmentiert (`corner_zones.py`)
- Jede Kurve in 3 Phasen aufgeteilt: **Eingang** (Bremszone), **Mitte** (Scheitelpunkt), **Apex** (Ausfahrt)
- Lap-Erkennung über `pos_norm`-Wrap-around (Abfall > 0.5 = neue Runde)
- Aufteilung: **5 Trainingsrunden / 2 Testrunden** pro Fahrer (`split_by_laps`)
- ~845 Segmente gesamt

**Ergebnis:** **73,33% Testgenauigkeit** (Segment-Level), alle 5 Fahrer korrekt identifiziert per Majority-Vote.  
**Erkenntnis:** Corner-Phasen sind aussagekräftiger als fixe Zeitfenster, weil jede Phase eine klare Fahrsituation abbildet. Das ist die Basis für alle weiteren Experimente.

---

## Experiment 2 — Sliding Windows (WINDOW_SIZE=30, STRIDE=15)

**Motivation:** Mehr Trainingsdaten erzeugen (~5× Multiplikation), um die Modellkonfidenz auf unseen Runden (6 & 7) zu verbessern.

**Was wurde geändert:**
- In `_create_corner_phase_segments`: Statt ein Segment pro Phase wird ein 30-Sample-Fenster mit Stride 15 über jede Phase geslided
- 4394 Segmente gesamt (vs. vorher 845)

**Ergebnis:** **68,57% Testgenauigkeit** — schlechter als die Baseline.

**Warum es nicht funktioniert hat:**
- 50% Überlappung der Fenster erzeugt hochkorrelierte Trainingssamples
- Der Random Forest lernt die überlappenden Muster auswendig (Overfitting auf Trainingsrunden)
- Mehr Daten mit gleicher Information bringt nichts — Qualität schlägt Quantität

---

## Experiment 3 — Zone als Feature (Eingang/Mitte/Apex = 1/2/3)

**Motivation:** Dem Modell mitteilen, in welcher Kurvenphase ein Segment liegt, damit es Eingang vs. Apex unterschiedlich gewichten kann.

**Was wurde geändert:**
- `'zone'` als Integer-Feature (1/2/3) in `FEATURE_COLUMNS` hinzugefügt

**Ergebnis:** **68,39% Testgenauigkeit** — noch schlechter.

**Warum es nicht funktioniert hat:**
- Random Forest braucht keine explizite Zone-Angabe; die statistischen Features (z.B. v_car_min, g_long_mean) kodieren bereits implizit, in welcher Phase man ist
- Falsches Signal: Zone 1 vs 3 kann sich über die gleiche Kurve aber verschiedene Runden unterschiedlich anfühlen

---

## Experiment 4 — Zone als One-Hot-Encoding (zone_1/zone_2/zone_3)

**Motivation:** Zone als ordinale Zahl könnte RF in die Irre führen (Zone 3 ≠ dreimal so viel wie Zone 1). One-Hot vermeidet das.

**Was wurde geändert:**
- `pd.get_dummies(zone, prefix='zone')` → Spalten `zone_1`, `zone_2`, `zone_3`
- Diese statt des rohen Integers als Features

**Ergebnis:** **68,39% Testgenauigkeit** — kein Unterschied.

**Erkenntnis:** Zone ist grundsätzlich kein hilfreicher Feature, unabhängig vom Encoding. Wurde wieder entfernt.

---

## Fazit & offene Ideen

| Experiment | Testgenauigkeit | Δ vs. Baseline |
|---|---|---|
| Baseline (Corner-Phases, 1 Seg/Phase) | **73,33%** | — |
| + Sliding Windows | 68,57% | -4,76 pp |
| + Sliding Windows + Zone (int) | 68,39% | -4,94 pp |
| + Sliding Windows + Zone (one-hot) | 68,39% | -4,94 pp |

**Was aktuell gut funktioniert:**
- Corner-Phase-Segmentierung ist semantisch sinnvoll
- 5/5 Fahrer korrekt per Majority-Vote auf allen Runden
- `--test-only` Flag isoliert Runden 6 & 7 (nie im Training gesehen)

**Mögliche nächste Schritte (nicht implementiert):**
- Mehr Fahrerdaten sammeln (mehr Runden)
- Feature Selection: Top-10 statt alle 29 Features → weniger Rauschen
- Andere Modelle: XGBoost, SVM (schon vorbereitet in der Pipeline)
- Augmentierung ohne Überlappung: leichtes Gaussian-Noise statt Stride-Windows
- Driver-spezifische Streckenabschnitte (Lieblingslinien) als Feature
