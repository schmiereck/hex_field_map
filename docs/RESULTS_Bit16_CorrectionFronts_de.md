# Bit-Test 16 — Räumlich zusammenhängende Korrekturfronten der 4D-Bitdynamik

Stand: 10. November 2026. Fortsetzung von Bit-Test 15. Unveränderte Bit-8-Kollision, Bit-10-Optimierung und Bit-12-Engine ohne Engine-Cache.

## Hypothese

Die Abweichungen einer ausgedehnten, frei bewegten Richtungsfront von der tatsächlichen kollidierenden Mikrodynamik könnten sich als wenige geometrisch zusammenhängende Korrekturobjekte darstellen lassen. Wenn solche Objekte kompakt bleiben, wäre langfristig ein Fronten-Solver vorstellbar. Geprüft wird die **verlustfreie Darstellung aller 24 Richtungsbits** der Differenz zwischen perturbiertem Feld und gleichzeitig entwickeltem Referenzfeld. Der Hintergrund selbst ist nicht mitkodiert.

## Exakte Rechnung und Einordnung

Für Baseline B, gestörten Zustand P, XOR-Differenz D = B XOR P, exakte lokale involutive Kollision C und invertierbares Richtungs-Streaming S:

    Q = C(P) XOR C(B) XOR D
    D_next = S(D XOR Q)

`Q` ist die neu erzeugte **nichtlineare Kollisionsquelle**, die freie Vorhersage `S(D)` wird durch sie korrigiert. Zusätzlich wird das kumulative Residuum `R_t = D_t XOR S^t(D_0)` gemessen. Sämtliche drei Felder sind vollständige 24-Bit-Felder, keine Dichte-Näherungen. Die Quelle wird im Test **aus der exakten Kollision von B und P** ermittelt; daraus ergibt sich **noch keine autonome oder schnellere obere Dynamik**.

Die aktiven Knoten eines Feldes werden in Komponenten des periodischen 4D-D4-Wurzelgitters gruppiert. Zwei Knoten sind benachbart, wenn sie durch einen der 24 D4-Root-Vektoren verbunden sind. Die Komponenten umfassen alle Knoten mit mindestens einem gesetzten Korrekturbit. Nichtnull-Wörter werden in einem ausdrücklich reversiblen Binärformat (Gruppe, Index, 24-Bit-Wort) kodiert und dekodiert; jedem Test liegt ein exakter Rundlauf zugrunde. Verglichen werden direkt komprimierte Dichte-Wortfolgen, eine einfache unstrukturierte Sparsliste der Korrekturen und die komponentengruppierte Sparsliste, jeweils mit zlib. Die anfängliche Ray-Liste wird nur einmal pro Zeitreihe bezahlt.

## Reproduzierbare Stichproben

Die Anfangspräparationen folgen Bit 14/15: Lokalisierter Zusatzpuls oder verteilte Störung, jeweils im Vakuum oder auf zufällig halbgefülltem Hintergrund. Periodische 4D-Gitter. Die ersten zwei Fälle untersuchen 8 Vorwärtsticks und 8 Rückwärtsticks auf 512 physikalischen D4-Knoten, die zwei großen Fälle 6+6 Ticks auf 2.048 Knoten. Die Rückwärtssimulation stellt jeweils die exakten Anfangswörter wieder her.

| Präparation | Knoten | Ticks | Direktes D (zlib) | Rays + ungruppiertes R (zlib) | Rays + Komponenten-R (zlib) |
|---|---:|---:|---:|---:|---:|
| Vakuumpuls | 512 | 0–8 | 520 B | **212 B** | **212 B** |
| Zufallsstörung im Vakuum | 512 | 0–8 | 681 B | **244 B** | **244 B** |
| Puls / Halbfüllung | 512 | 0–8 | **2.869 B** | 4.841 B | 4.864 B |
| Verteilte Störung / Halbfüllung | 512 | 0–8 | **4.166 B** | 7.239 B | 7.287 B |
| Vakuumpuls | 2.048 | 0–6 | 561 B | **178 B** | **178 B** |
| Puls / Halbfüllung | 2.048 | 0–6 | **937 B** | 978 B | 1.057 B |
| Verteilte Störung / Halbfüllung | 2.048 | 0–6 | **3.006 B** | 4.888 B | 5.028 B |

Keine der getesteten Komponentenvarianten verbessert die ungruppierte Spars-Kodierung. Das liegt auch am zusätzlichen Komponenten-Metadatenaufwand. Damit ist **nicht** belegt, dass jede geometrische Frontkodierung scheitert: Komponente plus explizite Indizes ist nur ein einfacher Kontrollencoder.

## Wachstum und Zusammenwachsen

Lokaler Puls auf 512 Knoten, kumulative Residualknoten `R_t`, geordnet (Tick, Anzahl Knoten, Komponenten, größte Komponente):

    (0,0,0,0), (1,0,0,0), (2,17,4,9), (3,33,3,31),
    (4,75,5,61), (5,167,1,167), (6,240,1,240),
    (7,358,1,358), (8,464,1,464)

Auf 2.048 Knoten wächst derselbe Präparationstyp langsamer:

    (0,0,0,0), (1,0,0,0), (2,4,4,1), (3,8,6,2),
    (4,31,6,17), (5,58,11,30), (6,135,13,104)

Die Quelle Q_t ist vorübergehend deutlich kleiner als das kumulierte Residuum. Auf 512 Knoten wächst sie beim Puls in 8 Quell-Ticks von 0 auf 176 Knoten und vereinigt sich schließlich zu einer Komponente. Auf 2.048 Knoten umfasst die Quelle bei t=5 nur 31 Knoten in 16 Komponenten. Das ist ein möglicher Ansatzpunkt für ein Ereignis-/Kollisionsquellen-Modell, **aber** das kumulative R muss dennoch korrekt fortgeschrieben werden.

Im Vakuum sind sämtliche Quellen und Residuen über die gesamte beobachtete Testdauer exakt null: Die anfänglichen Richtungsbits sind freie Strahlen, unabhängig von der Frontform.

## Interpretation und Grenzen

1. Physische Zusammenhängigkeit allein ist keine Informationskompression. Sobald viele unregelmäßig belegte Knoten in einer Komponente liegen, verschwindet der Vorteil einer einzigen Objekt-ID.
2. Die kleinen periodischen Dimensionen (besonders die beiden Achsen mit Länge 4) führen zu frühem Wrap-around und können die Clustervereinigung beschleunigen. Der Test belegt **kein** asymptotisches exponentielles Wachstum und keine universelle Perkolationsschwelle.
3. Der Zufallshintergrund ist ein harter Belastungstest, aber nicht identisch mit einer lang lebenden, kohärenten Paketanregung aus Bit 6. Andere Präparationen könnten zu stabileren Fronten führen.
4. Speicherzahlen sind relativ zur parallel simulierten Referenz und enthalten nicht deren Speicher-/Rechenkosten, keine Skript-/Softwarekosten und keine Metadaten für Gitterkonventionen.
5. `Q` und `R` sind XOR-Differenzen, keine separat physisch erhaltenen Teilchen. Die fünf Mikrodynamik-Invarianten wurden für die gesamten individuellen Trajektorien geprüft, nicht für die Differenzgrößen selbst.
6. Komponenten kodieren **keine** glatte Oberflächenform, keine Krümmung, keine Richtungskorrelation der 24 Bits und keine Dimensionskopplung D3/D4/D8. Diese Erweiterungen bleiben offen.

## Nächster Falsifikationstest

Eine **lokale Ereignisfront** statt kumuliertem Residuum testen: kleine Quell-Cluster Q separat beschreiben, nach freier Translation aggregieren und nur bei tatsächlicher Wechselwirkung eine exakte lokale Aktualisierung auslösen. Vergleich mit der vollen Bitdynamik, inklusive aller Baseline-Zugriffe und des Rechenaufwands. Zusätzlich weniger kurzperiodische Achsen (z.B. 8×8×8×8) und kohärent präparierte Hintergrundstrukturen prüfen. Nur eine autonome, exakt rekonstruierbare und über viele Ticks unterlinear wachsende Darstellung wäre eine wirkliche Beschleunigung.

Die Idee, Kugel-/Schalenoberflächen könnten Daten zwischen D3, D4 und D8 vermitteln, bleibt eine separate, in Bit 15 notierte geometrische Forschungshypothese. Bit 16 leitet daraus keinen dynamischen Dimensionsübergang ab.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open35_correction_fronts.py --kind background_pulse --ticks 8 --out data/bit16_background512.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open35_correction_fronts.py --shape 16 4 4 16 --kind background_pulse --ticks 6 --out data/bit16_background2048.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open35_correction_fronts.py --kind vacuum_pulse --ticks 8 --out data/bit16_vacuum512.json
```