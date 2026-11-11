# Bit-Test 20: q-abhängige, phasenkodierte 4D-Paketantwort

**Status:** alle neun Fälle `(q_mode in 1,2,3) × (k_mode in 0,+3,-3)` mit **8 unabhängigen Seeds je Fall** durchgerechnet, 24 Ticks auf `(48,4,4,16)` mit 6.144 D4-Knoten. Plus exaktem kollisionsfreien Kontrolllauf für jeden Seed und Fall. Bit-8-Kollision ist gegenüber Bit 19 unverändert; das neue Numba-Modul wurde referenzgeprüft.

Die schrittweise Rechnung wurde gegen Prozessabbruch mit getrennten q-Ausgaben gesichert. Rohdaten in `data/bit20_raw.npz` sowie `data/bit20_q{1,2,3}.npz`, Postprocessing und Validierung in JSON. Komplexe Zahlen nur im Analysecode, **nicht** im lokalen Update.

## Reproduktion

Benötigt Python mit NumPy, Numba und Matplotlib. Im Verzeichnis dieser Datei:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python src/phase_bit_s4_open39_dispersion.py --qs 1 --out data/bit20_q1.npz --metadata data/bit20_q1_metadata.json
python src/phase_bit_s4_open39_dispersion.py --qs 2 --out data/bit20_q2.npz --metadata data/bit20_q2_metadata.json
python src/phase_bit_s4_open39_dispersion.py --qs 3 --out data/bit20_q3.npz --metadata data/bit20_q3_metadata.json
python src/phase_bit_s4_open39_merge.py
python src/phase_bit_s4_open39_analyze.py
python src/phase_bit_s4_open39_plot.py
python src/phase_bit_s4_open39_validate.py
```

Vollständiger, differenzierter Befund: `docs/RESULTS_Bit20_PhaseQDispersion_de.md`.

**Wichtig:** In der frühen Intensitätsschwerpunktbewegung zeigt `q=π/4` ein anderes Vorzeichen als `q=π/8` und `q=3π/8`. Das belegt **noch nicht** eine Eigenmode oder negative Gruppengeschwindigkeit; es könnte eine Interferenz-/Präparationswirkung sein. Keine langlebige Bindung im untersuchten Zeitfenster.
