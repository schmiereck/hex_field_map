# Bit-Test 6: gesicherte Teilrekonstruktion (2026-11-09)

Ausgangspunkt: `schmiereck/hex_field_map`, `main` bei `e2b1e02995e5ad0a18798c223d721311eaaedd91`.

## Im ZIP

- `docs/RESULTS_Bit6_LocalPackets_de.md`: **originaler** Bericht des unterbrochenen Work-Laufs (unverändert).
- `figures/phase_bit_s4_open25.png`: **originale** Grafik des unterbrochenen Work-Laufs (unverändert).
- `src/phase_bit_s4_open25_symmetry_recovery.py`: **neu** rekonstruierter, eigenständiger, deterministischer Test der Kollisionssymmetrien.
- `data/phase_bit_s4_open25_symmetry_recovery.json`: **neu** aus dem Test erzeugtes Prüfprotokoll.

## Start

```bash
python src/phase_bit_s4_open25_symmetry_recovery.py --out data/phase_bit_s4_open25_symmetry_recovery.json
```

Nur Python-Standardbibliothek nötig. Der Test überprüft alle 2325 Zustände mit 0–3 Besetzungen, alle acht Geometrieabbildungen, Erhaltungsgrößen, Invertierbarkeit und das berichtete Gegenbeispiel `[0,22]`.

## Grenzen

Dies ist **nicht** der ursprüngliche Commit `283283b`, und **keine** vollständige Wiederherstellung von `phase_bit_s4_open25.py`, `phase_bit_s4_open25_campaign.py`, `phase_bit_s4_open25_figs.py` oder der numerischen JSON/NPZ-Dateien. Die publizierten Bit-5-Implementierungen dienten als Vorlage für eine unabhängige lokale Rekonstruktion. Die originalen Messzahlen wurden nicht neu gerechnet. Die Grafik und der Bericht stammen aus der Nutzereinreichung; die neuen Tests sind davon getrennt.

## Veröffentlichung

Zuerst ZIP lokal erstellt und geprüft; danach ggf. schrittweise GitHub-Publikation auf `recovery/bit6-local-packets-20261109`.