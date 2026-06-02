# Wichtige Fragen für die Professoren

## Datenqualität & Umfang

### 1. Datenmengen

**Frage:** Wie viele Fahrer oder Daten sollen wir nutzen?

- **Hintergrund:** Durchschnittlich ~116.000 samples/Fahrer, aber ungleiche Verteilung (66k bis 273k)
- **Kontext:** Vergleichbare Studien in der Literatur?

### 2. Daten-Imbalance

**Frage:** Sollte ich die ungleiche Datenverteilung zwischen Fahrern (4:1 Verhältnis zwischen größtem und kleinstem Datensatz) durch Undersampling/Oversampling adressieren?

- **Aktuell:** MAAKZ19001 hat 273k samples, MAMCZ06001 nur 67k
- **Alternativen:** SMOTE, ADASYN, oder stratified sampling?

### 3. Datenquellen-Kombination

**Frage:** Ist die Kombination von zwei unterschiedlichen Telemetrie-Formaten (HTF und LD) methodisch sauber, oder sollten separate Modelle trainiert werden?

HTF daten sind auch mit hochgeladen von uns.
Prüfen ob die HTF Daten das kürzel enthalten

- **Risiko:** Unterschiedliche Sampling-Raten und Sensoren
- **Vorteil:** Mehr Daten und generalisierbareres Modell

---

## Methodik & Feature Engineering

### 4. Segmentierungsstrategie

**Frage:** Ist die gewählte Segment-Länge von 10 Sekunden (500 Samples @ 50Hz) optimal für Fahrstil-Erkennung, oder sollte ich adaptive Segmentierung basierend auf Streckenabschnitten testen?

- **Aktuell:** Fixed-window mit 50% Overlap
- **Alternative:** Event-basierte Segmentierung (Kurven, Geraden, Brems-Zonen)

### 5. Temporal Dependencies

---

## Modell-Evaluation

### 6. Cross-Validation Strategie

**Frage:** Ist simple train/test split (80/20) ausreichend, oder sollte ich nested cross-validation oder leave-one-session-out verwenden?

- **Risiko:** Data leakage zwischen Sessions desselben Fahrers
- **Alternative:** Stratified K-Fold mit Session-Grouping

---

## Verbesserungsstrategien

### 7. Deep Learning Investment

**Frage:** Bei welcher Datenmenge wäre der Wechsel zu Deep Learning (LSTM, 1D-CNN) gerechtfertigt?

- **Aktuell:** 1.3M samples, 11 Fahrer
- **Threshold:** Literatur empfiehlt 10x mehr Daten für DL

### 8. Data Augmentation

**Frage:** Welche Data Augmentation Techniken sind bei Telemetrie-Daten sinnvoll? Time-warping, Gaussian noise, oder Adversarial examples?

- **Risiko:** Unrealistische synthetische Samples
- **Vorteil:** Bessere Generalisierung

---

## Validierung & Metriken

### 10. Evaluation Metrics

**Frage:** Sind Accuracy, Precision, Recall ausreichend, oder sollte ich auch F1-Score, ROC-AUC (one-vs-rest), Cohen's Kappa berichten?

- **Multi-Class Problem:** Welche Metriken sind Standard?

---

## Dokumentation & Präsentation

### 11. Visualisierungen

**Frage:** Welche Visualisierungen sind am wichtigsten für die Präsentation/Paper?

- Confusion Matrix ✓
- Feature Importance ✓
- t-SNE/UMAP Projektion der Features?
- Learning Curves?
- Error Analysis (false positives/negatives)?

### 12. Reproduzierbarkeit

**Frage:** Was sollte im Appendix/Repository stehen, damit jemand das Projekt nachbauen kann?

- Requirements.txt ✓
- README mit Pipeline ✓
- Hyperparameter configs?
- Seed values für Reproduzierbarkeit?
- Docker container?

---

Viel Erfolg! 🚀
