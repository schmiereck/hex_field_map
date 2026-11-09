# Bit-Test 10: Vektorisierte Symmetrie-Orbits für den 4D-Kern

Stand: 9. November 2026. Ausgangspunkt: Bit-Test 8 (Kollision) und Bit-Test 9 (gepackter 4D-Kern).

## Status

**Numerisch positiv:** Die mikroskopische Kollisionsregel aus Bit-Test 8 ist unverändert. Eine neue Auswertung berechnet die B4-Symmetrie-Orbits mit 384 SIMD-artigen NumPy-Tabellenzugriffen (`uint32`) und verwendet vorberechnete kanonische Klassen für zulässige Austauschmasken. Die Auswahl einschließlich Mehrdeutigkeitsverwerfung bleibt exakt erhalten.

**Grenze:** Diese Optimierung beschleunigt die kalte Kollisionsentscheidung; für den langen 4D-Paketversuch und die physikalische Dynamik ist damit noch nichts bewiesen. Wiederholte Messungen und reale Speedups hängen von Architektur, NumPy, Zustandsdichte und Cache-Trefferrate ab.

## Konstruktionsprinzip

1. `TABS[384,3,256]` enthält die für Bit-Test 7/8 definierten B4-Permutationen. Ein 24-Bit-Zustand wird mit drei indizierten `uint32`-Tabellenoperationen für alle 384 Elemente gleichzeitig transformiert.
2. Für alle zulässigen Zweiteilchen-Austausche ist die kanonische Klasse der Vier-Bit-XOR-Maske einmalig berechnet. `MASK_CLASS[s ^ t]` ist exakt gleich `canonical(s ^ t)` aus Bit-Test 8.
3. Bei gleichwertigen Masken entscheidet weiterhin die lexikographisch kleinste kanonische ungerichtete Kante `(min(g.s,g.t), max(g.s,g.t))`, über **alle** `g` aus B4. Bei Mehrdeutigkeit erfolgt keine Bewegung.
4. Eine Kollision `s → t` erfolgt nur bei gegenseitiger Partnerwahl. Damit ist sie wieder eine Involution. Es wurde keine neue mikroskopische Regel eingeführt.

## Prüfungen

`python src/phase_bit_s4_open29_benchmark.py --out data/phase_bit_s4_open29_benchmark.json`

- 300 zufällige unterschiedliche Zustände mit exakt 12 Bits: jeder Ausgabewert identisch mit der bisherigen Bit-8-Auswertung.
- 86 Zustände einschließlich Nullzustand, Vollzustand und allen 24 Einbitzuständen: Involution, Erhaltung von Zahl und vier Impulskomponenten sowie Äquivarianz unter acht Generatorsymmetrien gegen Bit 8 geprüft.
- 128 Knoten, 3 Vorwärts- und 3 Rückwärtsticks sowie 512 Knoten, je 2 Vorwärts- und Rückwärtsticks: bitgenaue Gleichheit zu Bit 8 auf jedem geprüften Schritt.
- Gesonderter 512-Knoten-Test des gepackten `VectorOrbitD4` auf kalten halbgefüllten Zuständen: 560,768 ms (Bit 9) gegen 54,136 ms (Bit 10), Faktor 10,36. Hin- und Rücklauf bitgleich.
- 300 kalte lokale Vergleiche: 426,071 ms gegen 54,454 ms, Faktor 7,82. Diese Messung und der vollständige Tick wurden separat durchgeführt.

## Dateien

- `src/phase_bit_s4_open29_fast_collision.py`: identische Regel, schnellere Orbit-Auswertung.
- `src/phase_bit_s4_open29_engine.py`: 4D-Engine mit der optimierten Kollisionsentscheidung.
- `src/phase_bit_s4_open29_benchmark.py`: Referenzvergleich, Symmetrie- und Reversibilitätsprüfung.
- `data/phase_bit_s4_open29_benchmark.json`: reproduzierbarer Prüfbericht.
- `data/phase_bit_s4_open29_engine_benchmark.json`: gesonderte Cold-Tick-Messung.

## Nächster Test

Die vektorisierte Kollisionsentscheidung an vielen Tausend neuen 24-Bit-Zuständen bzw. einer größeren 4D-Box untersuchen. Dabei **kalten** und **warmen** Cache strikt trennen, Speicherbedarf messen, Laufzeitgrenzen einbauen, atomare Zwischenstände schreiben und erst nach bestandener 4D-Reversibilitätsprüfung zur Paketpräparation übergehen.