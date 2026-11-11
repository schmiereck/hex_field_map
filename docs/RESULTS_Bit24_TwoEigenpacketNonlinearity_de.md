# Bit-Test 24 — Nichtlineare Wechselwirkung zweier 4D-Eigenmodenpakete

**Datum:** 11. November 2026  
**Repository:** `schmiereck/hex_field_map`, Fortsetzung Bit 20–23  
**Status:** Experiment und unabhängige technische Kontrollen abgeschlossen. **Nachweis einer kausal verzögerten nichtlinearen Bitantwort; kein Nachweis stabiler gebundener Teilchen oder einer asymptotischen Streumatrix.**

## 1. Frage und Ergebnis

Können zwei zunächst räumlich getrennte, gezielt aus der 24-Kanal-Linearisierung präparierte 4D-Wellenpakete über die bloße Überlagerung ihrer einzeln entwickelten Zustände hinaus eine messbare Wechselwirkung erzeugen? Und bildet sich nach der Begegnung ein kleiner, langlebiger ausgehender Modensatz?

**Ja zur ersten Frage. Nein zur zweiten im untersuchten Fenster.** Der rein nichtlineare Anteil setzt exakt erst dann ein, wenn die kausalen Einflussbereiche der beiden Präparationen aufeinander treffen. Er wächst aber auf einem aktiven halbgefüllten Hintergrund schnell zu einer großen Region an: Nach 14 Ticks sind im Referenzfall durchschnittlich **6.721,4 von 12.288 physischen D4-Knoten (54,7 %)** betroffen. Vergleichbare Wechselwirkung entsteht auch für gleichgerichtete und x/w-polarisierte Pakete.

## 2. Die vier Zustände und das streng exakte Wechselwirkungsmaß

Ein identischer halbgefüllter, räumlich zufälliger Bit-Hintergrund `X00` wird mit zwei anfangs **disjunkten** lokalen Bit-Flippatterns `A`, `B` kombiniert. Dies erzeugt `X10=X00 XOR A`, `X01=X00 XOR B`, `X11=X00 XOR A XOR B`. Alle vier Zustände folgen der **unveränderten** lokalen Bit-8-Kollision und Streamingregel `F_t=(S∘C)^t`.

Die exakte Viererdifferenz lautet:

\[
I_t=F_t(X_{00})\oplus F_t(X_{10})\oplus F_t(X_{01})\oplus F_t(X_{11}).
\]

`I_0=0` gilt bitgenau. Falls nur frei gestreamt wird, gilt `I_t=S^t(I_0)=0` **für jeden Tick**, weil Streaming eine lineare Bitpermutation ist. Ein positiver Wert unter der Bit-8-Regel ist somit eine zusätzliche, durch die **nichtlineare Kollision** ausgelöste Antwort; nicht nur die Überlagerung von bereits vorhandenen Störungen. Gezählt werden Gitterknoten, an denen mindestens ein Bit von `I_t` gesetzt ist.

Als physikalisch nähere, **vorzeichenbehaftete** Antwort prüfen wir ergänzend für jede der 24 Richtungen `R_i(t)=n_i(F_t(X11))-n_i(F_t(X10))-n_i(F_t(X01))+n_i(F_t(X00))`. Ihre Gesamtzahl und alle vier Gesamtimpulse sind null (anfängliche Disjunktheit und separate Erhaltung der fünf Ladungen). Dichte, x-Strom und verborgener w-Strom werden außerdem in räumlichen Fourierkanälen `q=0,π/4,π/2` ausgewertet. Wichtig: Die gespeicherten Spektralwerte sind **Mittelwerte der pro Seed quadrierten Amplituden**, nicht die Leistung eines kohärent über Seeds gemittelten Signals. Sie identifizieren deshalb noch keine eindeutig ausgehende Eigenmode.

## 3. Präparation und unveränderte Dynamik

- Je D4-Knoten genau **24 Boolesche Richtungsbits**; die Entwicklung enthält keine reellen oder stochastischen Übergangsparameter. Die bitidentische Numba-Beschleunigung des ursprünglichen Bit-8-Kollisionskerns stammt aus Bit 20.
- Die statistische Eigenvektorpräparation verwendet eine eingefrorene 24×24-Matrix aus der **ergänzenden Bit-23-Serie mit 400.000 lokalen Zustandsstichproben** (`data/bit24_collision_matrix.json`). Das ist eine einmalige **externe** Rechenhilfe, keine Änderung der physischen Regeln. Auf GitHub existiert zusätzlich eine andere Bit-23-Matrix aus 800.000 Stichproben; die hier verwendete Matrix ist im Archiv explizit fixiert und wird nicht mit ihr gleichgesetzt.
- Die Initialwahrscheinlichkeit lautet `p_i=0,5+ε g(x-c) Re[v_i exp(i(k_x x+q w+φ))]`, quantisiert durch **16-Bit-Ganzzahl-Zufallsschwellen**. Komplexzahlen und Gaußfunktionen kommen ausschließlich in Initialisierung/Analyse vor. Das mikroskopische Gitter rechnet ausschließlich mit endlichen Bitzuständen.
- Verwendete Träger `k_x=π/8`, verborgenes `q=π/4`; x-rechts/x-links sind statistisch linearisierte Zweige mit Gruppengeschwindigkeiten ungefähr `+1,0048` und `−1,0030`. Der w-polarisierte Kontrollzweig liegt bei ungefähr `−0,0015`.
- Beide räumlich anfangs disjunkten x-Profile haben `σ=3,5` und endliche Stützbereiche `|x-c|≤5` (keine anfangs überlappenden Ausläufer). `y,z,w` tragen die verborgenen Richtungsphasen; es sind **keine** vollständig in vier Raumrichtungen lokalisierten Kugeln.
- Hauptgeometrie `(48,4,4,32)`, **12.288 physische Knoten**. Quergitterkontrolle `(48,8,8,32)`, **49.152 Knoten**. Insgesamt **11 Bedingungen, 46 Seeds, 184 Bit-Evolutionen** mit 14 Ticks (Quergitter 12 Ticks), über vergleichbare Bedingungen mit denselben Seed-Indizes gepaart.

## 4. Kausalität und Anfangsabstand

| Anfangsabstand der Paketzentren | Erste nichtlineare Antwort | Mittlere aktive Knoten bei Tick 8 | Bei Tick 14 |
|---:|---:|---:|---:|
| 12 | **Tick 2** | 3.435,0 | 6.738,0 |
| 16 | **Tick 4** | 2.363,1 | 6.721,4 |
| 20 | **Tick 6** | 1.162,0 | 6.924,2 |

Mit Stützradius fünf ergeben sich minimale Abstände 2/6/10 zwischen den ursprünglichen Störungsgebieten. Die beobachteten ersten Ticks 2/4/6 entsprechen der maximalen x-Ausbreitungsgeschwindigkeit eines D4-Streamingbits (eine Einheit pro Tick) und der Kollisions-/Streaming-Reihenfolge. **Jeder Seed innerhalb einer Bedingung hat denselben ersten nichtlinearen Tick.** Kein nichtlokaler Fernwirkungseffekt ist zu erkennen.

Im Basisszenario wächst die betroffene Knotenzahl von `0` (Ticks 0–3) über `36,5` (Tick 4), `2.363,1` (Tick 8) auf `6.721,4` (Tick 14). Die große späte Region passt nicht zu einem einzelnen dauerhaft eng gebundenen Wechselwirkungsobjekt.

## 5. Amplitude: starker Früh-, schwacher Späteffekt

Bei Abstand 16, Phase null:

| Präparationsstärke ε | Aktive Knoten bei Tick 4 | Bei Tick 8 | Bei Tick 14 |
|---:|---:|---:|---:|
| 0,10 | 4,0 | 1.965,5 | 6.046,0 |
| 0,16 | 21,0 | 2.218,0 | 6.444,8 |
| 0,22 | 36,5 | 2.363,1 | 6.721,4 |
| 0,30 | 53,2 | 2.470,0 | 6.877,8 |

Vier Seeds pro ε, bei ε=0,22 acht. Die gestörte anfängliche Bitanzahl variiert mit ε. Die XOR-Knotenzahl ist weder eine lineare Wellenamplitude noch ein Wirkungsquerschnitt; insbesondere folgt aus den Werten **keine** gesicherte Kopplungskonstante oder Quadratgesetz. Nach 14 Ticks sind die Unterschiede relativ gering.

## 6. Relative verborgene Phase und Gegenproben

Bei gleichartigen x-rechts/x-links-Paketen, gleichem Bit-Hintergrund pro Seed, ε=0,22 und Abstand 16:

| Relative Phase φ | Aktive Knoten bei Tick 4 | Bei Tick 8 | Bei Tick 14 |
|---:|---:|---:|---:|
| 0 (erste vier Seed-Paare) | 39,5 | 2.356,5 | 6.682,3 |
| π/2 | 25,3 | 2.347,5 | 6.676,0 |
| π | 29,5 | 2.371,8 | 6.654,0 |

Der erste sichtbare Wechselwirkungseinsatz reagiert auf die relative Phase. Bei Tick 14 ist der Effekt klein. Vier Seeds und die Änderung der tatsächlich präparierten lokalen Flips reichen **nicht** zum Nachweis einer reproduzierbaren phasenabhängigen Streumatrix.

Ein weiterer wichtiger **negativer** Kontrollbefund (gleiche vier Seeds, Tick 8):

| Kombination | Mittlere aktive Knoten |
|---|---:|
| x-rechts + x-links | 2.356,5 |
| x-rechts + nahezu ortsfester w-Zweig | 2.145,3 |
| x-rechts + x-rechts | 2.201,8 |

Wechselwirkung entsteht somit **nicht allein** durch eine frontale Begegnung zweier gegenläufiger Pakete: Auch die kausalen Einflussbereiche gleichgerichteter oder unterschiedlich polarisierter Moden können sich unter der reversiblen lokalen Kollision gegenseitig beeinflussen. Ein charakteristischer Zweiteilchen-Streuprozess ist damit bislang nicht isoliert.

## 7. Kontrolle der Gittergeometrie

Auf `(48,8,8,32)` erfasst der nichtlineare Anteil bei Tick 12 im Mittel **18.902,5/49.152 = 38,5 %** aller Knoten, gegenüber **4.640,9/12.288 = 37,8 %** auf dem Hauptgitter. Die massive räumliche Verteilung ist demnach nicht ausschließlich durch die kurzen y/z-Perioden von vier Einheiten erzeugt. Beide Simulationen bleiben allerdings periodische endliche 4D-Tori; bei späteren Ticks sind zusätzliche Umrundungen und erneute Begegnungen möglich.

Die vorzeichenbehaftete Mischdifferenz besitzt sowohl im x- als auch w-Strom sowie in den projizierten `q=0,π/4,π/2`-Kanälen endliche (pro Seed quadrierte) Leistung. Das ist eine **Kanaldiagnose**, kein Nachweis einer neuen stabilen oder kohärenten w-Mode. Eine komplex gemittelte verbundene Antwort und Fehleranalyse fehlen noch.

## 8. Unabhängige Validierung und Reversibilität

- Der Numba-Kern stimmt für **157 unabhängige lokale Testzustände** bitgenau mit der vorherigen Referenz überein, ist involutiv und erhält die fünf Ladungen.
- Auf der kleinen D4-Geometrie `(32,4,4,16)` werden alle vier vollständigen Zustände nach sechs Vorwärts- und sechs Rückwärtsschritten **exakt** zum Ursprung rekonstruiert.
- Die fünf gemischten additiven Gesamtladungen verschwinden in der laufenden Auswertung; die vier einzelnen Mikrotrajektorien erhalten jeweils die fünf globalen Ladungen.
- `I_0=0` und die anfängliche disjunkte Präparation sind für alle 11 gespeicherten Bedingungen geprüft. Bei einem seedgleichen kollisionsfreien Streaming-Test gilt `I_t=0` für **jeden** geprüften Tick.
- Die acht Seeds des Hauptfalls zeigen sämtlich den ersten Wechselwirkungsterm bei Tick 4. Die beiden Eingangsmoden, Vorwärts-/Rückwärtsregeln und die separate Vorbereitungsmatrix sind im Archiv reproduzierbar.

## 9. Wissenschaftliche Konsequenz

Der Test zeigt auf einem endlichen D4-Gitter eine **lokal-konsistente, exakt isolierte nichtlineare Antwort auf zwei zuvor getrennte Eigenzweig-Anregungen**. Das ist ein belastbarer Fortschritt über die reine lineare Interferenz aus Bit 23 hinaus.

**Nicht belegt** sind: (a) die Existenz asymptotisch stabiler ein- und auslaufender Pakete; (b) ein teilchenähnlicher elastischer Streuquerschnitt; (c) ein eindeutiger neuer ausgehender kollektiver Eigenzweig; (d) ein dynamischer D3/D4/D8-Dimensionsübergang oder das Entstehen stabiler Kugeloberflächen. Vor allem die ebenfalls starke Wechselwirkung bei gleichlaufenden Paketen mahnt zur Vorsicht.

**Nächster sinnvoller Bit-Test 25:** Die **komplexe, ensemblegemittelte verbundene** Antwort für ausgehende `(k_out,q_out)`-Moden mit Fehlerbudget bestimmen, insbesondere die erwarteten Mischkanäle `(0,0)` und `(2k,2q)` bei schwachen Paaranregungen. Parallel die kurzzeitige kohärente x-/w-Stromantwort von dem stark anwachsenden inkohärenten XOR-Träger separieren. Nur dann kann man aus der Bit-Nichtlinearität eine physikalisch belastbare Modenkopplungstheorie gewinnen.

## 10. Reproduktion und Daten

Der ausführbare Test `src/phase_bit_s4_open43_pair_collision.py` verwendet `src/phase_bit_s4_open43_mode_prep.py` (den separat gespeicherten statistischen 24×24-Operator) und den im Repository bereits vorhandenen unveränderten `phase_bit_s4_open39_numba_collision.py`. Benötigt sind NumPy mit `bitwise_count`, Numba, SciPy und optional Matplotlib.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_validate.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_pair_collision.py \
  --matrix data/bit24_collision_matrix.json --seeds 1 --ticks 6 \
  --shape 32 4 4 16 --out data/test_reproduction.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_analyze.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_plot.py
```

Alle 11 Vollserien lassen sich mit `phase_bit_s4_open43_batch.py` und `phase_bit_s4_open43_branch_controls.py` reproduzieren (existierende Datendateien werden dabei weiterverwendet; für Neuberechnung vorher entfernen). Die vollständigen Seedzeitreihen sind im ZIP und der ChatGPT-Dateibibliothek gespeichert; kompakte Aggregate, Python-Quellen, Prüfungen und Abbildungen auf GitHub `main`. Das Archiv konserviert auch das Bit-23-Eigenzweig-ZIP und die historische ursprüngliche Feldgrafik.
