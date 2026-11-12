# Bit-Test 25 — Ergänzende unabhängige Kohärenzkontrolle

**Datum:** 12. November 2026. Dies ist **eine ergänzende, unabhängig berechnete Stichprobe**, nicht der Ersatz für den bereits auf `main` vorhandenen umfassenderen Bericht [`RESULTS_Bit25_CoherentOutgoingModes_de.md`](RESULTS_Bit25_CoherentOutgoingModes_de.md).

## Abgleich mit der Hauptuntersuchung

Die bereits abgeschlossene Hauptuntersuchung enthält eine **vollständige Auszählung aller 16.777.216 lokalen Bitwörter** der Bit-8-Kollision. Ihre lokalen quadratischen Antwortvektoren für x/x und x/w sind **streng ungleich null** (Normen ca. 0,01787 und 0,01736). Sie hat außerdem einen kontrollierten **Vierphasenzyklus** über 480 komplette Bit-Trajektorien ausgewertet und **keinen nach Mehrfachtestkorrektur signifikanten kohärenten Ausgangszweig** nachgewiesen.

Diese ergänzende Kontrolle nutzt unabhängig neue Ausgangs-Seeds, weniger Phasen, aber ein **breiteres Raster von 25 räumlichen Fourierkanälen** für die vorzeichenbehaftete Vierzustandsdifferenz. Es bestehen daher **unterschiedliche statistische Tests**, insbesondere **kein** vollständiger Phase-Cycling-Selektor. Ein unbereinigter p-Wert dieser Ergänzung darf den negativen, umfangreicheren Haupttest nicht überstimmen.

## Verfahren

Gitter `(48,4,4,32)` mit 12.288 physischen D4-Knoten, Eingangs-`k=π/8`, `q=π/4`. Modenvektoren stammen von der eingefrorenen Bit-24-Präparationsmatrix. Vier Anfangszustände mit gemeinsamen ganzzahligen 16-Bit-Schwellen und disjunkten Trägern werden mit dem **unveränderten reversiblen Bit-8-Kern** über zwölf Ticks simuliert. Die signierte verbundene Antwort ist `R=n(X11)-n(X10)-n(X01)+n(X00)`, unabhängig vom XOR-Indikator. Projektion auf Dichte, `Jx`, `Jw` sowie `k_out∈{-2,-1,0,+1,+2}π/8` und `q_out∈{-4,-2,0,+2,+4}π/8`. Je Eingangs-Kombination werden die komplexen Antwortkoeffizienten zuerst über Seeds gemittelt und erst dann quadriert.

Produktionsumfang: **28 Seed-Bedingungs-Läufe** (acht gleiche Seed-Hintergründe über verschiedene Bedingungen gepaart), also **112 separate 24-Bit-Evolutionen**. Im x/x-Referenzfall und im x/w-Kontrollfall je acht Seeds; weitere x/x-Gegenphasen `π/2`, `π` und Amplitude `0,10` mit je vier Seeds. Der erste Seed jedes Laufs wurde bitgenau rückwärts validiert; kollisionsfreies Streaming ergibt die verbundene Antwort **exakt null**, ebenso die Anfangskonfiguration. Alle fünf Ladungen werden in den vier Einzelläufen erhalten, der gemischte Gesamtwert bleibt null.

Die Frequenzauflösung aus nur zwölf Ticks ist zu gering, um eine präzise spektrale Resonanz oder einen stabilen ausgehenden Frequenzzweig zu beanspruchen. `q_out=0` ist die Summe über die w-Richtung, **nicht** das Signal einer einzelnen 3D-Schicht.

## Ergebnisse

Bei x-rechts + x-links setzt die verbundene Antwort in allen acht Seeds erstmals bei Tick 4 ein. Bei Tick 12 sind im Mittel **4.642,4** Knoten betroffen. Für x-rechts + w-Zweig tritt die Antwort in sieben von acht Seeds bei Tick 4, in einem Seed bei Tick 5 auf; bei Tick 12 sind es **4.443,1** Knoten. Das ist ein großer nichtlinearer Bereich, kein kompaktes ausgehendes Paket.

Über jeweils fünf sichtbare `k`-Werte, drei Observablen und das vorher definierte Zeitfenster gemittelte explorative **Seed-Vorzeichenpermutationstests**:

| Eingangsmoden | Zeitfenster | `q_out=0` | `q_out=π/4` | `q_out=π/2` |
|---|---|---:|---:|---:|
| x rechts + x links, 8 Seeds | Tick 4–7 | 0,062 | 0,523 | 0,070 |
| x rechts + x links, 8 Seeds | Tick 8–12 | 0,508 | 0,336 | 0,469 |
| x rechts + w, 8 Seeds | Tick 4–7 | 0,258 | 0,414 | **0,016** |
| x rechts + w, 8 Seeds | Tick 8–12 | 0,227 | 0,828 | 0,641 |

Der frühe `q_out=π/2`-Kanal beim x/w-Eingang ist damit **exploratorisch auffällig**, allerdings **unkorrigiert für mehrere Eingangsbedingungen, Zeitfenster und Ausgangskanäle** und mit einer anderen Nullmethode als der vollständige Haupttest. Seine rohe kohärente Leistung beträgt **36,4 %** der mittleren Seedleistung im Frühfenster, sinkt im Spätfenster auf **11,8 %**, nahe der endlichen Stichprobenrauschskala `1/8=12,5 %`.

Eine einfache Phasenrelation der Form `A_out(φ+π)≈−A_out(φ)` ist in den vier gepaarten Hintergrund-Seeds nicht belegt. Quadratische Amplitudenskalierung ist mit vier Seeds nicht entscheidbar.

## Fazit

Die exakte lokale Kopplungsstruktur der Hauptuntersuchung und diese Ergänzung sprechen beide für **eine reale lokale nichtlineare Kopplung**, aber **nicht** für eine gesicherte langlebige, phasenkohärente ausgehende Mode. Der kurzzeitige `q_out=π/2`-Befund ist als mögliche **vorab festzulegende** Hypothese für Bit-Test 26 nützlich, nicht als Bestätigung der D3/D4/D8-Dimensionsvermittlungs-Hypothese.

Die lokalen Ströme `Jx`, `Jw` transportieren Besetzungsimpulse; im Modell ist noch **keine separate mikroskopische Energie** definiert. Eine Aussage über Energieübertragung zwischen Dimensionen wäre daher unbegründet.

**Reproduktion:** Aus dem Bit-25-Ergänzungs-ZIP mit den vollständigen `.npz`-Zeitreihen: `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open44_significance_independent.py`; zusätzlich `phase_bit_s4_open44_validate_independent.py`. Für eine Neuberechnung siehe `phase_bit_s4_open44_mode_transfer.py`; die vollständigen Vorgängerdateien befinden sich im eingebetteten Bit-24-ZIP. Die Aggregationsdatei `bit25_significance_independent.json` enthält alle Teststatistiken und Paarkorrelationen.
