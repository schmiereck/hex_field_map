# Bit-Test 18 – Langzeitentwicklung der aktiven 4D-Region

**Stand: 10. November 2026.** Reproduzierbarer Nachfolger von Bit-Test 17. Die mikroskopische Bit-8-Kollisionsregel ist unverändert.

## Fragestellung

Bleibt eine lokale Störung vor einem bekannten, homogenen 24-Bit-Hintergrund als dünne, sich ausdehnende Kugelschale erhalten, oder füllt sie sukzessive das innere 4D-Volumen? Kann der exakte Sparse-Event-Solver eine solche Störung über 50–100 Ticks verwalten?

## Methodik und Präparation

`src/phase_bit_s4_open37_longfront.py` benutzt den exakten `SparseD4Events` aus Bit 17, den unveränderten `collide_fast` aus Bit 10 und das D4-Streaming aus Bit 9. Die Anfangsstörung ersetzt **genau einen** Gitterknoten durch `0xffffff` (24 Bits); der Hintergrund ist entweder überall `0x8cd1ae` (kollisionsaktiver homogener Bitzustand) oder `0` (Vakuumkontrolle). Der Ausgabewert ist die XOR-Abweichung zum *gleichzeitig* entwickelten Hintergrund.

Ein D4-Knoten liegt nur auf einer Hälfte der ganzzahligen 4D-Gitterplätze (gerade Koordinatensumme). Die Gitter sind **periodisch**; dies ist keine unendlich ausgedehnte Raum-Zeit. Die Radialdistanz verwendet minimale periodische Koordinatenabstände. Sie ist eine Diagnose, keine Annahme isotroper Ausbreitung. Eine ungestörte Kugel in vier Raumdimensionen hätte eine 3-dimensionale Hyperoberfläche; die gemessene Besetzungsmenge muss damit nicht identisch sein.

Die innere Aktivierungsrate wird durch die Anzahl aktiver Gitterknoten innerhalb von `r <= r90/2` geteilt durch die Anzahl sämtlicher D4-Gitterknoten darin definiert, wobei `r90` das 90%-Quantil der radialen Distanz *aktiver* Knoten ist. Diese Kenngröße unterscheidet grob eine dünne äußere Front von einer innen zunehmend ausgefüllten Region. Sie ist nicht unabhängig vom jeweiligen Gitter bzw. seiner periodischen Geometrie.

## Messungen

| Perioden (4D) | Physische Knoten | Ticks | Exakte vollständige Rückkehr | Befund |
|---|---:|---:|---|---|
| 8×8×8×8 | 2.048 | 100 | ja, 100 vorwärts + 100 rückwärts | ab Tick 30 vollständig aktiv |
| 16×16×16×16 | 32.768 | 25 | ja, 25 + 25 | 32.230/32.768 aktive Knoten bei Tick 25 |
| 16×16×16×16 | 32.768 | 50 | *nicht für diesen Lauf separat durchgeführt* | 32.765/32.768 aktiv bei Tick 50 |
| 24×24×24×24 | 165.888 | 16 | ja, 16 + 16 | 25.767/165.888 aktiv bei Tick 16 |
| 16×16×16×16, Vakuum | 32.768 | 50 | ja, 50 + 50 | höchstens 24 aktive Knoten, keine neue Kollisionsquelle |

Für jeden Fall wurde mindestens der Beginn des Laufs gegen den unabhängigen dichten Kernel verglichen (standardmäßig zwei Ticks). Exakte lokale Involution und Streaming-Permutation begründen die Fortsetzung, und bei vier der fünf Läufe wurde zusätzlich der vollständige Zustand durch den sparse Reverse-Algorithmus exakt rekonstruiert. Die volle Dichtesimulation über 100 Ticks wurde **nicht** parallel durchgeführt.

Ausgewählte Werte bei aktivem Hintergrund:

| Tick | 2.048 Knoten | 32.768 Knoten | 165.888 Knoten |
|---:|---:|---:|---:|
| 1 | 16 | 16 | 16 |
| 5 | 662 | 797 | 797 |
| 10 | 1.635 | 5.036 | 5.540 |
| 15 | 2.001 | 14.974 | 20.817 |
| 20 | 2.026 | 27.758 | — |
| 25 | 2.044 | 32.230 | — |
| 30 | 2.048 | 32.504 | — |
| 50 | 2.047 | 32.765 | — |
| 100 | 2.048 | — | — |

Für 165.888 Knoten ist bei Tick 8 die Aktivierung im definierten **inneren Vergleichsbereich 58,0 %**, bei Tick 10 **58,5 %** und bei Tick 15 **71,8 %**. Bereits in der frühen Entwicklungsphase enthält die Störung daher viele aktive **innere** Knoten: Sie ist keine dünne, rein äußere Wellenfront.

Im Vakuum bleiben aus der initialen 24-Kanal-Anregung 24 gerichtete frei laufende Bits erhalten, die sich teilweise periodisch auf weniger räumliche Knoten überlagern. Dort entstehen während 50 Ticks keine nichtlinearen Kollisionsquellen. Das bedeutet nicht, dass die beobachteten 24 Positionen eine kontinuierliche Kugelschale darstellen.

## Interpretation und Grenzen

1. **Falsifikation der einfachen dünnen Schale für diese Präparation und Regel**: Ein kollisionsaktiver homogener Hintergrund erzeugt stark innen besetzte Differenzzustände. Die Verdichtung ist bereits vor der vollständigen periodischen Sättigung beobachtbar.
2. **Periodische Randbedingungen**: Früheste potenzielle Umschläge eines gerichteten Bits können ab etwa halber minimaler Periodenlänge auftreten (4/8/12 Ticks für die drei symmetrischen Gitter). Daten danach sind ausdrücklich Torusdynamik, keine freie Ausbreitung im unbegrenzten Raum. Auch vor dieser Zeit können weit ausgreifende D4-Strukturen entstehen; die geometrische Interpretation bleibt begrenzt.
3. **Keine allgemeine Aussage über alle Wellenpakete**: Die Präparation `FULL` auf einem Knoten und die uniforme Hintergrundkonfiguration sind konkrete Sonderfälle. Die frühere langlebige Bit-6-Präparation wurde nicht rekonstruiert.
4. **Kompression**: Das sparse Dictionary ist bei Gitterfüllung nahezu 100 % ineffizient. Für dieses Modell ist nach der Sättigung ein Wechsel zu einem dichten `uint32`-Array sinnvoll. Das ist ein Implementierungsvorschlag, nicht hier benchmarked.
5. **D3/D4/D8**: Eine dimensionsübergreifende Interpretation der Oberflächen bleibt eine *offene Hypothese*. Diese Messung testet nur D4.

## Reproduzierbarkeit

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open37_longfront.py --shape 8 8 8 8 --ticks 100 --roundtrip --out data/bit18_uniform_8cube_100.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open37_longfront.py --shape 16 16 16 16 --ticks 25 --roundtrip --out data/bit18_uniform_16cube_25.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open37_longfront.py --shape 16 16 16 16 --ticks 50 --out data/bit18_uniform_16cube_50.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open37_longfront.py --shape 24 24 24 24 --ticks 16 --roundtrip --out data/bit18_uniform_24cube_16.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open37_longfront.py --shape 16 16 16 16 --ticks 50 --kind vacuum --roundtrip --out data/bit18_vacuum_16cube_50.json
```

## Nächster sinnvoller Test

Ein **hybrider Sparse/Dense-Solver** mit exakt gleicher Mikrodynamik und einem klaren Schwellwert für den Darstellungswechsel. Für die Physik: eine echte winkel- und phasenstrukturierte langlebige Paketpräparation statt nur eines einzelnen Vollbelegungs-Pulses; erneuter Test der radialen Dichte, der inneren Störung und des nichtlinearen Quellterms mit ausreichend langen Gitterperioden.