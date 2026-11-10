# Bit-Test 15: Diskrete Oberflächenzustände auf 4D-Schalen

Stand: 10. November 2026. Ausgangspunkt: Bit-Test 14 (radiale Dichteschalen), unveränderte symmetrische, reversible Bit-8-Kollision in der optimierten Bit-10/12-Engine.

## Hypothese

Eine räumlich expandierende 4D-Front könnte als wenige holistische Objekte mit richtungsabhängigen Bits beschrieben werden, statt als unabhängiger 24-Bit-Zustand je Gitterknoten. Ziel ist eine **verlustfreie Beschreibung der vollständigen 24-Bit-Differenz** zwischen einer angeregten und einer gleichzeitig entwickelten Referenzsimulation. Die Referenz-/Hintergrundsimulation selbst wird nicht mitkodiert. Kompression der Differenz ist daher ausdrücklich **keine Kompression des vollständigen Universums**.

## Verglichene Modelle

1. **Radiale Bit-Schalen**: Eine der höchstens K gewählten Gruppen gleicher periodischer 4D-Radiusquadrate trägt ein konstantes 24-Bit-Muster. Der Rest wird als Liste von (Knotenindex, 24-Bit-XOR-Residuum) verlustfrei gespeichert.
2. **Radiale Schalen mit Oberflächensektoren**: Zusätzlich wird jeder Punkt nach dominierender Koordinatenachse und Vorzeichen in acht feste Sektoren eingeteilt; pro ausgewählter Radius-Sektor-Gruppe ist ein 24-Bit-Muster erlaubt. Alle Abweichungen werden exakt korrigiert.
3. **Gerichtete Front-Bits (Kontrolle)**: Die anfängliche Differenz besteht aus endlich vielen (Knoten, Richtung)-Bits. Sie werden unter reinem 4D-Streaming gemeinsam fortgeschrieben. Die Abweichung zur **tatsächlichen** nichtlinearen Kollision wird pro Tick als vollständiges XOR-Residuum mitgeführt.

Bei allen Verfahren erfolgt ein echter Encoder/Decoder-Roundtrip für **jedes** gespeicherte Bit; alle Vergleiche sind mit der exakten Mikrodynamik berechnet. Modellselektion ist offline und keine autonome Dynamik. Speicherzahlen enthalten einfach formatierte Binärdaten und zlib-Kompression, aber keine Software-, Metadaten- oder Hintergrundkosten.

## Ergebnis A: Konstante diskrete Bitmuster auf Schalen

Für das Gitter (8,4,4,8), 512 physische D4-Knoten, Seed 1509, 0..3 Ticks. Die Tabelle zeigt die jeweils beste zlib-Dateigröße unter K=0,1,4,8,16 beider Schalenvarianten und die zlib-Größe der unveränderten 24-Bit-Differenz.

| Tick | Vakuumpuls: Roh/zlib | Beste Schale/zlib | Halbgefüllter Hintergrund: Roh/zlib | Beste Schale/zlib |
|---:|---:|---:|---:|---:|
| 0 | 22 B | 23 B | 22 B | 23 B |
| 1 | 74 B | 115 B | 52 B | 70 B |
| 2 | 67 B | 76 B | 72 B | 95 B |
| 3 | 81 B | 114 B | 99 B | 142 B |

Die beste Wahl ist fast immer **K=0**: Das heißt, die geprüften Gruppenmuster helfen nicht; die expliziten Korrekturen sind teurer als die direkte zlib-Speicherung. Insbesondere tragen benachbarte Richtungsbits unterschiedliche Anteile auf derselben Kugelschale. Die sektorielle Aufteilung behebt das nicht. Bei (16,4,4,16) / 2.048 Knoten und halbgefülltem Hintergrund liefert der analoge Test kein robustes anderes Bild (Ticks 0..2: Roh 34/69/87 B, Schalen 23/68/83 B); winzige Vorteile entstehen bei K=0 durch unterschiedliche Kodierung von Nullbereichen, **nicht** durch Schalenobjekte.

## Ergebnis B: Gerichtete, frei propagierende Front-Bits

Beim Vergleich über mehrere Zeitpunkte wird die Ray-Startliste einmal bezahlt; das XOR-Residuum wird für jeden gespeicherten Tick bezahlt, auch wenn es leer ist.

| Experiment | Zeitpunkte | Komprimierte rohe Differenzen | Ray-Liste plus komprimierte Residuen | Residualstellen am letzten Tick |
|---|---:|---:|---:|---:|
| Vakuumpuls, 512 Knoten | 4 | 244 B | **127 B** | **0** |
| Zufällig verteilte Störung im Vakuum, 512 Knoten | 4 | 313 B | **159 B** | **0** |
| Puls vor halbgefülltem Hintergrund, 512 Knoten | 4 | **245 B** | 307 B | 29 |
| Zufallsstörung vor halbgefülltem Hintergrund, 512 Knoten | 4 | **548 B** | 794 B | 117 |
| Puls vor halbgefülltem Hintergrund, 2.048 Knoten | 3 | 190 B | **181 B** | 14 |
| Zufallsstörung vor halbgefülltem Hintergrund, 2.048 Knoten | 3 | **311 B** | 330 B | 43 |

**Positiver begrenzter Befund:** Im Vakuum sind die Front-Bits exakt freie gerichtete Bahnen. Die Front-Repräsentation bleibt konstant klein, weil die lokalen Kollisionen dort nicht aktiv werden. **Negativer Befund unter Kollisionen:** Aus der einfachen Ray-Vorhersage entwickeln sich zunehmend Residuen. Bei den getesteten endlichen Fenstern überwiegt im 512-Knoten-Fall der Korrekturaufwand. Dies belegt weder asymptotisches exponentielles Wachstum noch die Unmöglichkeit einer besseren holistischen Darstellung.

## Validierung und Grenzen

- Echter binärer Encoder und Decoder für sämtliche 24 Richtungsbits der Differenz; XOR-Korrekturen liefern bitgenau den jeweiligen Mikrozustand zurück, **gegeben den gleichzeitig entwickelten Hintergrund**.
- Alle Trajektorien sind exakt umkehrbar; Teilchenzahl und vier Impulskomponenten bleiben in der getrennten Mikrosimulation erhalten.
- Die drei Schalenkodierungen enthalten keinen Kompressionsbeweis für beliebige Felder: K und Oberflächensektoren wurden explorativ getestet und der gleiche Datensatz für die Modellwahl verwendet.
- Die Ray-Kontrolle ist keine Kugeloberfläche im strengen geometrischen Sinn, sondern eine diskrete gerichtete Front; sie trennt Transportsymmetrie von nichtlinearer Kollision.
- Die freie Vakuumdynamik ist kein Nachweis einer emergenten Quantenwelle oder stabilen kollidierenden Schale.
- Die getesteten maximal drei bzw. vier Ticks reichen nicht, um das Langzeitverhalten abzuleiten.
- Für echte Beschleunigung müsste die obere Darstellung **autonom** mit beschränktem Aufwand und ohne Kenntnis des exakten Gitters fortschreitbar sein.

## Nächster Test

Die Schale sollte nicht aus einem konstanten Muster je Radius bestehen, sondern aus einem **lokalen gerichteten Oberflächenfeld**, dessen freie Propagation analytisch erfolgt und dessen nichtlineare Wechselwirkung als kleine, lokalisierte Korrekturfront geprüft wird. Wir sollten zunächst über viele kleine Ticks ermitteln, ob die Zahl der aktiven, räumlich zusammenhängenden Kollisions-Korrekturcluster sublinear zur Gittergröße wächst. Wenn nicht, gibt es für die exakte Vollsimulation keinen belegten Kompressionsvorteil.

## Separate Forschungsnotiz D3/D4/D8

Eine 4D-Kugeloberfläche schneidet eine sichtbare 3D-Hyperebene geometrisch in einer 2D-Kugeloberfläche (sofern der Schnitt nicht leer oder tangential ist). Die D3-, D4- und D8-**Wurzelgitter** liefern jedoch nicht automatisch eine kanonische dynamische Projektions-/Kopplungsabbildung oder Erhaltungsgesetze über Dimensionswechsel. Die Hypothese, dass Wellenfronten und ihre Oberflächeninformation eine solche Dimensionskopplung tragen, bleibt als mögliche Forschungsrichtung notiert; Bit-Test 15 testet sie **nicht**.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open34_shell_bits.py --kind background_pulse --out data/bit15_background_pulse_512.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open34_front_rays.py --kind background_pulse --out data/bit15_ray_background_pulse_512.json
```

Beide Skripte erwarten die vorhandenen Bit-8/10/12/14-Module im selben `src`-Verzeichnis. Die Messwerte sind Ergebnisse endlicher kontrollierter Tests, keine theoretisch allgemeingültigen Aussagen.