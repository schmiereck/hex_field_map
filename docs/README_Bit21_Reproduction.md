# Bit-Test 21 — Falsifikation des q-abhängigen Paket-Richtungswechsels

Reproduzierbarer, endlicher Falsifikationsversuch zur exakt reversiblen D4-Bit-8-Dynamik: Verdopplung der verborgenen Periode, separate Vergrößerung der sichtbaren y/z-Richtungen und ein zusätzlicher Test der k-Abhängigkeit. **Physikalische Wellenzahlen sind in allen Geometriekontrollen gleich.**

- Wissenschaftlicher Bericht: `docs/RESULTS_Bit21_QFiniteSizeFalsification_de.md`
- Quellcode: `src/phase_bit_s4_open40_falsify_q.py`, `_analyze.py`, `_validate.py`
- `data/bit21_summary.json`: volle zusammenfassende Ergebnisse und Konfidenzintervalle
- `data/bit21_validation.json`: vier Vorwärts-/Rückwärtsticks auf drei Geometrien und lokale Regelvergleiche
- Weitere `data/bit21_*.json`: pro-Seed-Ergebnisse; `data/bit21_*.npz`: diagnostische Antworttrajektorien
- `figures/*.svg`: geometrische Robustheit und zusätzliche k-Abhängigkeit

Das ZIP enthält auch die komplette Vorgängersicherung von Bit 20. Voraussetzung: Python, NumPy, Numba, Matplotlib und die Vorgängerquellen in `src` bzw. in `PYTHONPATH`.

**Grenze:** Keine Ableitung von Masseneigenschaften, keine isolierte kollektive Eigenmode; 12 Ticks und frühe Schwerpunktfits über Ticks 2–8. Die Probe ist ein rekonstruierter Bit-6/Bit-19-Präparationstyp, nicht die verlorene originale Bit-6-Implementierung.
