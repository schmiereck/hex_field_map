# Bit-Test 22 — D4 Strukturfaktor / Richtungsstrom

Experiment: **stationärer dynamischer Strukturfaktor** $S(k_x,q,\omega)$ für die unveränderte, reversible Bit-8-D4-Mikrodynamik. Gemessene Größen: Dichte, $x$-Strom und verborgener $w$-Strom; Fourierdiagnostik nur extern.

- **Hauptergebnis:** Bei $q=\pi/4$, $k_x=\pi/8$ sind Peaks nahe $\omega=+\pi/8$ (überwiegend x-Strom) und $\omega=-\pi/4$ (überwiegend w-Strom) zu unterscheiden. Beide existieren bereits in freier D4-Bewegung. Keine neu entstandene einzelne stabile Eigenmode nachgewiesen.
- **Daten:** 80 Ticks, acht unabhängige Halbfüllungs-Seeds auf 12.288 Knoten; vier weitere auf 49.152 Knoten; jeweils kollisionsfreier Kontrolllauf mit denselben Initial-Bits. Vollständig reversibel beim ersten Seed jeder Geometrie. Analytische Streaming-Kontrolle für alle 21 Spatial-Kanäle mit Float32-Fehler <3×10^-7.
- **Statistik:** PSD über unabhängige Seeds, getrennte Seed-Hälften, gepaarter Bootstrap (1200 Resamples).
- **Einschränkung:** Der frühere q-abhängige Paket-Schwerpunktwechsel ist kein Beweis für eine Gruppengeschwindigkeit aus einem isolierten Eigenzweig.

Bericht: `docs/RESULTS_Bit22_DynamicStructureFactor_de.md`. Reproduktionsbefehle dort. Neue Dateien: `src/phase_bit_s4_open41_*.py`. Vollständige Rohzeitreihen: `data/bit22_equilibrium_*.npz`; Bit20-Vorgänger unter `support/`. Das historische Bit-6-Bild liegt unangetastet im bisherigen Repository und zusätzlich im eingebetteten Bit-21-Backup.
