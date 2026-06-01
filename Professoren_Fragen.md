# Wichtige Fragen für die Professoren

## 📊 Datenqualität & Umfang

### 1. Datenmengen

**Frage:** Ist die aktuelle Datenmenge von 1,3 Millionen Samples (~7 Stunden Fahrtzeit) über 11 Fahrer ausreichend für eine wissenschaftlich valide Studie?

- **Hintergrund:** Durchschnittlich ~116.000 samples/Fahrer, aber ungleiche Verteilung (66k bis 273k)
- **Kontext:** Vergleichbare Studien in der Literatur?

### 2. Daten-Imbalance

**Frage:** Sollte ich die ungleiche Datenverteilung zwischen Fahrern (4:1 Verhältnis zwischen größtem und kleinstem Datensatz) durch Undersampling/Oversampling adressieren?

- **Aktuell:** MAAKZ19001 hat 273k samples, MAMCZ06001 nur 67k
- **Alternativen:** SMOTE, ADASYN, oder stratified sampling?

### 3. Datenquellen-Kombination

**Frage:** Ist die Kombination von zwei unterschiedlichen Telemetrie-Formaten (HTF und LD) methodisch sauber, oder sollten separate Modelle trainiert werden?

- **Risiko:** Unterschiedliche Sampling-Raten und Sensoren
- **Vorteil:** Mehr Daten und generalisierbareres Modell

---

## 🔬 Methodik & Feature Engineering

### 4. Segmentierungsstrategie

**Frage:** Ist die gewählte Segment-Länge von 10 Sekunden (500 Samples @ 50Hz) optimal für Fahrstil-Erkennung, oder sollte ich adaptive Segmentierung basierend auf Streckenabschnitten testen?

- **Aktuell:** Fixed-window mit 50% Overlap
- **Alternative:** Event-basierte Segmentierung (Kurven, Geraden, Brems-Zonen)

### 5. Feature-Auswahl

**Frage:** Mit 171 Features besteht die Gefahr von Overfitting bei nur 2.552 Segmenten. Sollte ich Feature-Selection (z.B. RFE, LASSO) durchführen oder ist das durch Cross-Validation bereits abgedeckt?

- **Feature-Kategorien:** Statistisch (mean, std, skew), Behavioral (jerk, steering rate), Frequency (FFT)
- **Alternativen:** PCA, Autoencoder für dimensionality reduction

### 6. Temporal Dependencies

**Frage:** Ignoriere ich durch die segment-basierte Klassifikation wichtige zeitliche Zusammenhänge? Wären LSTM/Transformer sinnvoller?

- **Aktuell:** Jedes Segment unabhängig
- **Risiko:** Strategie-Muster über mehrere Runden gehen verloren

---

## 🤖 Modell-Evaluation

### 7. Baseline-Vergleich

**Frage:** Wie bewerte ich die Modellperformance objektiv? Was wäre ein "gutes" Ergebnis bei 11-Klassen-Klassifikation?

- **Aktuell:** Random Forest 87.36%, XGBoost 70.76%
- **Baseline:** Random Guessing wäre 9.09%, aber was ist State-of-the-Art?

### 8. Cross-Validation Strategie

**Frage:** Ist simple train/test split (80/20) ausreichend, oder sollte ich nested cross-validation oder leave-one-session-out verwenden?

- **Risiko:** Data leakage zwischen Sessions desselben Fahrers
- **Alternative:** Stratified K-Fold mit Session-Grouping

### 9. Confusion Matrix Interpretation

**Frage:** Wie interpretiere ich systematische Verwechslungen zwischen bestimmten Fahrern? Könnte das auf ähnliche Fahrstile oder unzureichende Features hinweisen?

- **Beobachtung:** Gibt es "Fahrer-Cluster" mit ähnlichen Charakteristiken?

---

## 🚀 Praktische Anwendung

### 10. Real-Time Prediction

**Frage:** Für eine Live-Anwendung während eines Rennens: Wie lang muss das Minimum-Segment sein, um verlässliche Predictions zu bekommen?

- **Trade-off:** Schnellere Erkennung vs. höhere Genauigkeit
- **Aktuell:** 10 Sekunden - zu lang für Echtzeit?

### 11. Generalisierung auf neue Strecken

**Frage:** Wurde das Modell nur auf Nürburgring trainiert. Wie gut würde es auf anderen Strecken generalisieren?

- **Validierung:** Sollte ich Multi-Track Daten sammeln?
- **Alternative:** Track-agnostic features entwickeln

### 12. Transfer Learning

**Frage:** Wäre es sinnvoll, ein Pre-trained Model auf generischen Fahrdaten zu nutzen und dann fine-tuning für spezifische Fahrer durchzuführen?

- **Vorteil:** Weniger Daten pro neuem Fahrer nötig
- **Herausforderung:** Wo finde ich Pre-trained Models für Sim-Racing?

---

## 📈 Verbesserungsstrategien

### 13. Modell-Ensemble

**Frage:** Lohnt sich der zusätzliche Aufwand eines Ensemble-Modells (RF + XGBoost + SVM voting) oder ist Random Forest allein ausreichend?

- **Erwarteter Gain:** +2-5% laut Literatur
- **Komplexität:** Höherer Deployment-Aufwand

### 14. Deep Learning Investment

**Frage:** Bei welcher Datenmenge wäre der Wechsel zu Deep Learning (LSTM, 1D-CNN) gerechtfertigt?

- **Aktuell:** 1.3M samples, 11 Fahrer
- **Threshold:** Literatur empfiehlt 10x mehr Daten für DL

### 15. Data Augmentation

**Frage:** Welche Data Augmentation Techniken sind bei Telemetrie-Daten sinnvoll? Time-warping, Gaussian noise, oder Adversarial examples?

- **Risiko:** Unrealistische synthetische Samples
- **Vorteil:** Bessere Generalisierung

---

## 🎯 Projektausrichtung

### 16. Wissenschaftlicher Beitrag

**Frage:** Was ist der Haupt-Research Contribution dieses Projekts?

- a) Methodische Innovation (neues Feature Engineering)?
- b) Praktischer Use-Case (Cheat-Detection im E-Sports)?
- c) Datenset-Contribution (erster öffentlicher Sim-Racing Telemetrie-Datensatz)?

### 17. Reproduzierbarkeit

**Frage:** Sollte ich den Datensatz und Code öffentlich machen (z.B. auf GitHub/Zenodo) für Reproduzierbarkeit?

- **Pro:** Wissenschaftliche Best Practice
- **Contra:** Privacy-Concerns der Fahrer?

### 18. Zielgruppe

**Frage:** Für wen ist dieses System in erster Linie gedacht?

- a) E-Sports Organisatoren (Anti-Cheat)
- b) Fahrer-Training (Fahrstil-Analyse)
- c) Spieleentwickler (KI-Gegner-Training)
- d) Forensics (wer fuhr in einem bestimmten Rennen)

---

## 💡 Limitationen & Future Work

### 19. Bekannte Limitationen

**Frage:** Welche Limitationen sollte ich in der Arbeit explizit diskutieren?

- Nur ein Fahrzeug (Porsche 911 GT3 RS)
- Nur eine Strecke (Nürburgring)
- Nur Sim-Racing (nicht Real-World übertragbar)
- Supervised Learning (braucht gelabelte Daten)

### 20. Future Work Prioritäten

**Frage:** Was wäre die wichtigste nächste Forschungsfrage:

- a) Multi-Vehicle Support (verschiedene Auto-Typen)
- b) Unsupervised Clustering (Fahrstil-Archetypen)
- c) Anomaly Detection (Cheat-/Bot-Erkennung)
- d) Real-World Transfer (Sim-to-Real)

---

## ⚙️ Technische Details

### 21. Hyperparameter Tuning

**Frage:** Habe ich genug Hyperparameter-Tuning gemacht, oder sollte ich formale Optimierung (Bayesian Opt, Grid Search) durchführen?

- **Aktuell:** Scikit-learn Defaults mit minimal tuning
- **Risiko:** Suboptimale Performance

### 22. Computational Efficiency

**Frage:** Bei Skalierung auf 50+ Fahrer: Wird das aktuelle Feature-Engineering (171 features/segment) ein Bottleneck?

- **Alternative:** Online-Learning, Feature-Hashing?

### 23. Interpretability vs. Performance

**Frage:** Sollte ich Modell-Interpretability (SHAP, LIME) priorisieren, auch wenn das Performance kostet?

- **Use-Case:** Bei Anti-Cheat muss man Entscheidungen erklären können
- **Trade-off:** Random Forest ist interpretierbarer als Deep Learning

---

## 🔍 Validierung & Metriken

### 24. Evaluation Metrics

**Frage:** Sind Accuracy, Precision, Recall ausreichend, oder sollte ich auch F1-Score, ROC-AUC (one-vs-rest), Cohen's Kappa berichten?

- **Multi-Class Problem:** Welche Metriken sind Standard?

### 25. Statistical Significance

**Frage:** Sollte ich statistische Tests (McNemar, Wilcoxon) durchführen, um zu zeigen, dass RF signifikant besser ist als XGBoost/SVM?

- **Scientific Rigor:** Ist der Unterschied (87% vs 71%) statistisch signifikant?

---

## 📋 Dokumentation & Präsentation

### 26. Visualisierungen

**Frage:** Welche Visualisierungen sind am wichtigsten für die Präsentation/Paper?

- Confusion Matrix ✓
- Feature Importance ✓
- t-SNE/UMAP Projektion der Features?
- Learning Curves?
- Error Analysis (false positives/negatives)?

### 27. Reproduzierbarkeit

**Frage:** Was sollte im Appendix/Repository stehen, damit jemand das Projekt nachbauen kann?

- Requirements.txt ✓
- README mit Pipeline ✓
- Hyperparameter configs?
- Seed values für Reproduzierbarkeit?
- Docker container?

---

## 🎓 Literatur & Related Work

### 28. State-of-the-Art Vergleich

**Frage:** Welche Papers sollte ich als Vergleich heranziehen?

- Driver identification in real vehicles?
- Behavioral biometrics?
- E-Sports cheating detection?
- Time-series classification generell?

### 29. Novelty

**Frage:** Ist das Projekt eine inkrementelle Verbesserung oder echter novelty?

- **Was ist neu:** Sim-Racing spezifisch, HTF+LD Kombination
- **Was ist bekannt:** Telemetrie-basierte Classification existiert

### 30. Ethical Considerations

**Frage:** Gibt es ethische Bedenken bei biometrischer Identifikation von Gamern?

- Privacy: Können Fahrer getrackt werden?
- Consent: Wurde informed consent eingeholt?
- Misuse: Könnte das System missbraucht werden?

---

## ✅ Zusammenfassung der wichtigsten 5 Fragen

Wenn Zeit knapp ist, sollten Sie mindestens diese Fragen klären:

1. **Datenqualität (Frage 1):** Ist 1.3M samples / 11 Fahrer ausreichend?
2. **Methodische Validierung (Frage 8):** Ist meine Train/Test Split Strategie robust?
3. **Performance Interpretation (Frage 7):** Ist 87.36% "gut" für 11-Klassen-Problem?
4. **Nächste Schritte (Frage 20):** Was ist die wichtigste Verbesserung?
5. **Wissenschaftlicher Beitrag (Frage 16):** Was ist der Kern-Contribution?

---

## 📞 Vorbereitung für die Diskussion

**Bevor Sie ins Meeting gehen:**

- Übersicht über aktuelle Daten ausdrucken (results/00_data_overview.txt)
- Confusion Matrix und Feature Importance Plots bereithalten
- README.md durchlesen für vollständige Pipeline
- Überlegen: Was ist IHRE Hauptfrage?

**Notizen während des Meetings:**

- Welche Fragen fand der Prof am wichtigsten?
- Welche neuen Perspektiven wurden genannt?
- Gibt es neue Literatur-Empfehlungen?
- Zeitplan für nächste Schritte?

Viel Erfolg! 🚀
