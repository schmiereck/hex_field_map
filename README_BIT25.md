# Bit-Test 25 — Kohärente ausgehende Moden und exakter lokaler quadratischer Vertex

Fortsetzung von Bit 24, 11. November 2026.

## Zusammenfassung

- **Exakter Mikroprozess:** 24-Bit-Reversibilität und alle fünf Ladungen bleiben erhalten; Parallelisierung verändert die Bit-8-Kollision nicht.
- **Vier vollständige Paketbedingungen:** insgesamt 30 Hintergrund-Seed-Fälle × 4 relative Phasen × 4 genaue Bittrajektorien = **480** Trajektorien, jeweils 12 Ticks auf **12.288 D4-Knoten**.
- **Nichtlinearität:** Der Vierzustands-XOR-Rest wird ab Tick 4 ungleich null; im Referenzfall sind an Tick 12 im Mittel 4.644 Knoten betroffen.
- **Kein gesicherter kohärenter Ausgangszweig:** Die komplexe, phasenzyklische Fourierantwort bei `(2k_x,2q)` und in allen weiteren untersuchten Ausgängen bleibt nach Mehrfachtest-Korrektur unauffällig.
- **Neuer exakter positiver Befund:** Durch **vollständige** Enumeration der 16.777.216 möglichen 24-Bit-Kollisionsinputs wurde für die gezielten x/x- und x/w-Mole eine **nichtverschwindende lokale quadratische Antwort** bestätigt. Ihre 24-Komponenten-Quellvektoren sind kinetisch und ladungsfrei, können nach Streaming aber nichtverschwindende räumliche Dichte-/Strommoden bilden.
- **Unterscheidung:** Der lokale quadratische Vertex um eine unabhängige Halbfüllung ist **kein** Beweis für eine langlebige ausgehende Welle oder einen physikalischen Dimensionsübergang.

[Wissenschaftlicher Bericht](docs/RESULTS_Bit25_CoherentOutgoingModes_de.md) · [Reproduktion](docs/README_Bit25_Reproduction.md)

Ausführbarer Code `src/phase_bit_s4_open44_*.py`, eingefrorene Bit24-Matrix `data/bit24_collision_matrix.json`, vollständige Daten `data/bit25_*.npz`, JSON-Analysen, Abbildungen in `figures/`.

Das ZIP enthält das vollständige Bit24-Archiv und die historische Feldgrafik. Inhalte sind per `manifest.json` SHA-256 geprüft. Neu auf `main` werden ausschließlich Bit25-Dateien geschrieben.
