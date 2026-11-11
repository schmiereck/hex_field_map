# Bit-Test 25 — Kohärente ausgehende Mischmoden und vollständiger lokaler quadratischer Vertex

**Datum:** 11. November 2026  
**Repository:** `schmiereck/hex_field_map`  
**Fortsetzung:** Bit 20–24 (4D-Wellenpakete, 24-Richtungs-Moden, gezielte Doppelanregung).  
**Status:** Alle geplanten vier Paketserien und 16.777.216 lokale Kollisionszustände ausgewertet. **Exakte nichtverschwindende lokale quadratische Kopplung; noch kein statistisch gesicherter kohärenter ausgehender Zweig in der vollständigen Paketsimulation.**

## 1. Fragestellung und das wichtigste Ergebnis

Bit 24 zeigte eine kausal verzögert einsetzende, räumlich schnell anwachsende nichtlineare Antwort zweier anfänglich disjunkter D4-Eigenmodenpakete. Seine quadrierten, **pro Seed** berechneten räumlichen Fourierwerte belegten noch nicht, dass eine über viele Seeds gemeinsame **komplexe Phase** überlebt. Bit 25 untersucht deshalb:

1. Überlebt im ausgehenden räumlichen Mischkanal `(2k_x,2q)=(π/4,π/2)` eine phasenkohärente Antwort in **Dichte**, **Jx** oder **Jw**?
2. Wie abhängig ist die Antwort von Amplitude und Paarung (x rechts/x links oder x rechts/w)?
3. Ist bereits die **lokale deterministische** Kollisionsregel quadratisch an diese Eigenvektoren gekoppelt – unabhängig von der riesigen inkohärenten Antwort nach vielen Ticks?

**Resultat:** Alle vier Paketserien erzeugen eine wachsende nichtlineare Bitantwort. Doch kein getesteter ausgehender komplexer Fourierkanal besteht die Korrektur für die Mehrfachtests. Unabhängig davon ist die lokale quadratische Kopplung **streng nicht null**: Für zwei ausgewählte Modenpaare wurden alle `2^24 = 16.777.216` lokalen Bitwörter exakt durch die unveränderte Kollisionsregel geschickt und der zugehörige quadratische Antwortvektor ausgewertet.

Die beiden Befunde widersprechen einander nicht. Ein **lokal vorhandener Kopplungsterm** garantiert weder eine von zufälligen Hintergründen unabhängige Phase noch eine langlebige freie Welle nach mehreren Kollisionen.

## 2. Exakte Mikrodynamik und vier verbundene Zustände

Pro physischem D4-Knoten existieren 24 Boolesche Richtungsbits. Das mikroskopische Update ist weiterhin `F = S∘C` mit der Bit-8-Kollision `C` (B4-symmetrisch, involutiv, teilchenzahl- und impulserhaltend) und der exakt invertierbaren D4-Richtungsverschiebung `S`. Das Bit-20-Numba-Modul wird lediglich mit unverändertem Kern auf unabhängige Gitterknoten parallelisiert und gegen seine serielle Referenz verglichen.

Auf einem **identischen, räumlich zufälligen halbgefüllten** uint16-Schwellenhintergrund werden zwei getrennte vorbereitete Bit-Flippatterns `A` und `B` verwendet:

- `X00 = X0`, `X10 = X0 XOR A`, `X01 = X0 XOR B`, `X11 = X0 XOR A XOR B`.
- Die beiden Anfangsstützbereiche sind disjunkt. Beide präparierten Eingangsmoden stammen aus der in Bit 24 eingefrorenen 24×24-Matrix, die **nur zur Initialisierung und Analyse** dient.
- Die **vorzeichenbehaftete** kombinierte Antwort jedes Richtungsbits lautet `R_i(t)=n_i(F^t(X11))-n_i(F^t(X10))-n_i(F^t(X01))+n_i(F^t(X00))`.
- Der alternative bitweise XOR-Rest `I_t=F^t(X00) XOR F^t(X10) XOR F^t(X01) XOR F^t(X11)` ist ein **anderes Maß**, nämlich die Anzahl mikroskopischer Nichtlinearitätsstellen. XOR und vorzeichenbehaftete Fourieramplituden werden **nicht** gleichgesetzt.

`R_i(0)=0` ist exakt und für reines Streaming gilt `R_i(t)=0` für alle Ticks. Dasselbe gilt für den XOR-Rest. Für die fünf globalen additiven Ladungen gilt zu jedem Zeitpunkt `Σ_r Σ_i R_i=0` und `Σ_r Σ_i v_{i,α}R_i=0` für α=1,…,4. Darum sind die globalen Dichte- und Stromkanäle `(k,q)=(0,0)` **exakt null**. Bei der Float32-Fourier-Auswertung können numerische Rundungsreste von rund `10^-9` auftreten; sie sind keine physikalischen Signale und werden für diesen Kanal in der Analyse auf null gesetzt.

## 3. Phasen- und Richtungsauflösung

Die verborgene Eingangs-Wellenzahl lautet `q=π/4`, die sichtbare `k_x=π/8`. Die Anfangsprofile in x sind abgeschnittene Gaußfenster (`σ=3,5`, Stützradius fünf), 16 Gittereinheiten getrennt. Die volle Hauptgeometrie lautet `(48,4,4,32)` mit **12.288 physikalischen D4-Knoten**. Die zwei Moden sind in der vierten Richtung **phasenkodiert, aber nicht dort lokalisiert**. Es handelt sich nicht um vierdimensionale Kugeln.

Pro Hintergrundseed werden vier relative Phasen `φ = 0, π/2, π, 3π/2` des zweiten Pakets mit demselben uint16-Hintergrund gerechnet. Die komplexe externe Diagnose eines räumlichen Kanals ist

\[
A^{(s,φ)}_{k,q,o}(t)=\frac{1}{N}\sum_{r,i}\ell_{o,i}\,R^{(s,φ)}_i(r,t)\,e^{-i(kx+qw)}
\]

mit `o ∈ {Dichte,Jx,Jw}` und Richtungsgwichten `ℓ={1,v_x,v_w}`. Der **an die Phase der zweiten Anregung gebundene** Koeffizient lautet

\[
H^{(s)}_{k,q,o}(t)=\tfrac14\sum_{φ}e^{-iφ}\,A^{(s,φ)}_{k,q,o}(t).
\]

Die Kohärenzprüfung ist für das **vorab bestimmte Zeitfenster Tick 5–10** definiert (frühester möglicher Wechselwirkungstermin beim Hauptabstand ist Tick 4):

\[
C=\frac{\sum_{t=5}^{10}|\langle H_s(t)\rangle_s|^2}{\sum_{t=5}^{10}\langle |H_s(t)|^2\rangle_s}.
\]

`C` ist der Anteil des phasenzyklischen Leistungsmaßes, der **über die Seeds phasenstabil** ist; er hat unter der Nullhypothese ein endliches Stichprobenrauschen ungefähr der Größenordnung `1/S` und ist keine Messung eines Energieanteils. Als Teststatistik dient der Zähler. Für den Nullvergleich werden pro Seed unabhängige vierte Einheitswurzeln `{1,i,-1,-i}` multipliziert, über die ganze Zeitachse konstant. Damit bleiben innerseedliche Zeitkorrelationen und Amplitudenverläufe erhalten. Je Fall wurden **6.000 Zufallsrotationen** geprüft. Die 27 Nichtnull- räumlichen Kanal-/Observablenkombinationen je Fall werden zusätzlich konservativ mehrfach getestet; über vier Bedingungen betrachtet sind es 108 explorative Tests.

## 4. Vollständige Paketserien

Vier Bedingungen mit gemeinsamem Seed-Plan, jeweils 12 Ticks, 4 Phasen, 4 getrennte reversible Bit-Trajektorien je Seed und Phase:

| Fall | Stärke ε | Moden | Seeds | verbundene XOR-Knoten Tick 4 | Tick 8 | Tick 12 |
|---|---:|---|---:|---:|---:|---:|
| Schwach | 0,11 | x rechts + x links | 6 | 4,7 | 2.017,3 | 4.386,3 |
| Referenz | 0,22 | x rechts + x links | 12 | 30,0 | 2.373,7 | 4.644,5 |
| Stark | 0,30 | x rechts + x links | 6 | 47,6 | 2.472,7 | 4.715,8 |
| x/w | 0,22 | x rechts + w-Zweig | 6 | 5,9 | 2.148,1 | 4.461,1 |

Insgesamt sind dies **30 Seed/Fall-Kombinationen**, **120 phasenkontrollierte Vierzustands-Präparationen** und **480 tatsächlich simulierte Bit-Trajektorien** über jeweils 12 Ticks. Der Anfang ist lokal getrennt, die erste zusätzliche Antwort beginnt beim Referenzabstand 16 im **vierten Tick**; sie wächst im betrachteten Fenster trotz unterschiedlicher Amplituden zu einer großen Region. Dies entspricht dem negativen räumlichen Befund von Bit 24.

Für den **vorab ausgewählten Mischkanal `(2k_x,2q)`** im Referenzfall ergeben sich die folgenden Resultate:

| Observable | `C` (Tick 5–10) | Nullrotations-p-Wert |
|---|---:|---:|
| Dichte ρ | 0,0642 | 0,6194 |
| Jx | 0,0698 | 0,5501 |
| Jw | 0,0300 | 0,9310 |

Zum Vergleich: Bei zwölf Seeds beträgt eine einfache Nullrausch-Skala für `C` ungefähr `1/12 ≈ 0,083`. Keine dieser drei Messungen trennt sich sichtbar vom Nullmodell. Auch die übrigen untersuchten räumlichen Kanäle und die drei Kontrollfälle ergeben **keinen familiär-korrigierten Signifikanznachweis**. Der kleinste *unkorrigierte* p-Wert aller 108 Explorationswerte beträgt etwa **0,0090** (`stark`, `q_flip`, `Jw`), klar oberhalb des konservativen Gesamt-Schwellwerts `0,05/108 ≈ 0,00046`. Solche Einzelwerte sind daher **nicht** als Modennachweis zu bezeichnen.

Dieses negative Resultat **beweist keine verschwindende kohärente Kopplung**: Sechs bis zwölf unabhängige Hintergrund-Seeds setzen der Auflösung kleiner komplexer Amplituden enge Grenzen; die Zeitreihe ist nur zwölf Ticks lang und die wachsende ungerichtete Störungsregion erzeugt Rauschen. Die Fensterlänge nach dem Kontakt reicht zudem **nicht**, um beispielsweise zwei nahe Frequenzen `π/8` und `π/4` sicher als getrennte Spektralpole aufzulösen. Ein stabiler ausgehender Frequenz-/Streuzweig wurde hier **nicht** nachgewiesen.

## 5. Der vollständige lokale quadratische Vertex (neues positives Ergebnis)

Um zu trennen, ob die **Kollision selbst** überhaupt eine quadratische Modenantwort erlaubt, betrachten wir einen einzelnen Knoten mit **unabhängigen** Eingangsbits `X_j ~ Bernoulli(1/2)` und definiertem Ausgang `Y_i=C(X)_i`. Mit `s_j=2X_j-1`, `t_i=2Y_i-1` gilt für `j ≠ k` bei Halbfüllung

\[
\frac{\partial^2\mathbb E[Y_i]}{\partial p_j\partial p_k}\bigg|_{p_j=1/2}
=2\,\mathbb E[t_i s_j s_k],
\qquad \frac{\partial^2\mathbb E[Y_i]}{\partial p_j^2}=0.
\]

Da die identische Eingangsabbildung keine quadratische Antwort hat, darf zur isolierten **kollisionsbedingten** Kopplung `t_i-s_i` verwendet werden. Für zwei Eingangsvektoren `a,b` folgt das kontrahierte exakte Vertex

\[
V_i(a,b)=2\,2^{-24}\sum_{X\in\{0,1\}^{24}}
(t_i-s_i)\left[\left(\sum_j s_j a_j\right)\left(\sum_k s_k b_k\right)-\sum_j a_jb_j\right].
\]

**Die Summe wurde über alle 16.777.216 Bitwörter berechnet.** Der Bitkollisionskern ist unverändert und alle Ausgaben werden integergenau bestimmt. Nur die Multiplikation mit den aus Bit 24 numerisch geschätzten komplexen Eigenvektoren verwendet Gleitkommazahlen. Es ist also eine vollständige Auszählung des lokalen Zustandsraums für **diese festgelegten Modenvektoren**, keine symbolische geschlossene Lösung und auch keine Aussage über den korrelierten Vielknotenzustand.

| Eingangsmoden `a,b` | `||V||₂` | Betrag von ρ nach einer Streaming-Phase bei `(2k,2q)` | Betrag von Jx | Betrag von Jw |
|---|---:|---:|---:|---:|
| x rechts + x links | **0,0178685** | **0,0333443** | 0,0067982 | **0,0227252** |
| x rechts + w | **0,0173569** | **0,0154139** | 0,0055142 | **0,0166920** |

`V` ist ein lokaler **kinetischer** Antwortvektor: `Σ_i V_i = 0` und `Σ_i v_{i,α}V_i=0` für alle vier Richtungen (maximaler numerischer Rest unter `10^-15`). Die anschließende richtungsabhängige Streaming-Phase `e^{-i(2k v_x+2q v_w)}` kann dennoch eine nichtverschwindende **räumliche** Dichte- und Stromamplitude bei `k,q ≠ 0` ergeben. Dies verletzt keine globale Ladungserhaltung.

Die dargestellten Vertex-Beträge sind dimensionslose zweite Ableitungen beziehungsweise ihre Fourierprojektionen. Sie dürfen **nicht** direkt als Energie, Energiefluss oder als gemessene Paketamplitude interpretiert werden. Die ersten vier Zeitperioden einer echten Paketbegegnung bauen bereits räumliche Hintergrundkorrelationen auf; die iid-Halbfüllungs-Hesse-Matrix allein beschreibt diesen Verlauf daher **nicht quantitativ**.

Der unabhängige vorläufige Monte-Carlo-Test mit 480.000 lokalen Zufallszuständen liefert z.B. für den Dichte-Ausgang des x/x-Paares ebenfalls etwa `−0,03335` (real), in Übereinstimmung mit dem exakten Ergebnis `−0,0333433`. Dieser Test wurde **nicht** zur Auswahl oder Anpassung der vollständigen Auszählung benutzt.

## 6. Technische Kontrolle und reproduzierbarer Umfang

- Parallelisierter Integer-Kollisionskernel gegen bitidentische serielle Numba-/Bit-10-Kollision auf zwei kleinen D4-Geometrien und mehrere Vorwärts-/Rückwärtsschritte verglichen.
- Zusätzlich **385** lokale Bit-10/Bit-8-Vergleiche, Involution und fünf Ladungen erhalten.
- Vollständiger Vierzustands-Rücklauf im produktiven **12-Tick-Referenzfall** (erster Seed, alle vier Phasen-Zustände der ersten Phase) exakt zum Ausgangszustand.
- Separater Sechs-Tick-Rücklauf auf `(32,4,4,16)` mit vorzeichenbehafteter Ladungsauswertung und ungestörtem Streaming-Nulltest.
- Alle vier Reihen mit identischem Seed-Plan für gepaarte Phasenkontrollen. Einzeltrajektorien, phasenaufgelöste komplexe Fourierwerte und XOR-Zähler sind als `.npz` archiviert.
- Die lokale Enumeration berücksichtigt **exakt alle `2^24`** möglichen Bitwörter; der kinetische Vertex erfüllt die fünf Erhaltungsidentitäten bis auf numerische Rundung.

## 7. Physikalische Interpretation und Grenzen

**Bestätigt:** Die konkrete diskrete reversible Mikrodynamik besitzt eine **nichtverschwindende lokale quadratische Kopplung** der gezielt präparierten x/x- und x/w-Moden. Sie erzeugt unter einer Paketbegegnung viele zusätzliche mikroskopische Zustandsunterschiede. Auf der Ebene eines einzelnen lokalen iid-Streuschritts können daraus richtungsabhängige Fourierquellen entstehen.

**Nicht bestätigt:** Ein dauerhaft phasenkohärenter ausgehender Wellenzweig in der vollständigen endlichen 4D-Simulation, eine asymptotische S-Matrix, ein stabiler gebundener Teilchenzustand, eine Erhaltung beziehungsweise Umverteilung einer separat identifizierten mikroskopischen Energie, eine physikalische Vermittlung zwischen D3/D4 oder ein dynamischer D8-Beitrag. `Jx/Jw` sind hier **Teilchenzahl-Richtungsströme**, keine experimentell bestimmten Energieflüsse.

Damit ist die ursprüngliche Dimensionsidee präziser prüfbar: Der **verborgene w-Richtungsfreiheitsgrad** ist an einer lokalen Mischantwort beteiligt, deren räumliche Projektion sichtbare Dichtestrukturen beeinflussen kann. Eine *eigene Wechselwirkung zwischen verschiedenen Dimensionen* folgt daraus jedoch noch nicht: Wir simulieren derzeit **ein einziges 4D-Gitter**.

**Sinnvoller Bit-Test 26:** Ein **separat hergeleitetes, deterministisches quadratisches Antwortmodell** mit dem lokal exakt ausgezählten Vertex und dem Bit-23-Linearkern aufbauen. Seine vorhergesagte räumlich-zeitliche Zweitordnungsantwort gegen sehr kleine, mit **antithetischer/phase-cycling Varianzreduktion** präparierte Bit-Trajektorien testen, bevor die Antwort aufgrund des zufälligen Hintergrunds saturiert. Erst wenn dieses Modell die verbundenen komplexen Amplituden präzise vorhersagt, wäre ein belastbarer Weg zu einer reduzierten Modenkopplungstheorie erkennbar.

## 8. Reproduktion und Sicherung

Im entpackten Archiv vom Projektverzeichnis aus:

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4 BIT25_PARALLEL=1
python src/phase_bit_s4_open44_validate.py --root .
python src/phase_bit_s4_open44_outgoing.py --matrix data/bit24_collision_matrix.json --name baseline --seeds 12 --ticks 12 --reverse-first
python src/phase_bit_s4_open44_outgoing.py --name weak   --amplitude .11 --seeds 6 --ticks 12
python src/phase_bit_s4_open44_outgoing.py --name strong --amplitude .30 --seeds 6 --ticks 12
python src/phase_bit_s4_open44_outgoing.py --name xw     --amplitude .22 --branch w --seeds 6 --ticks 12
python src/phase_bit_s4_open44_analyze.py --names baseline weak strong xw --permutations 6000
python src/phase_bit_s4_open44_exact_vertex.py --root . --chunk 32768
python src/phase_bit_s4_open44_summary.py
python src/phase_bit_s4_open44_plot.py
```

Der Bit-8-Numba-Kern wird für den Vergleich aus Bit 20 mitgeliefert, die genaue 24×24-Mode-Präparationsmatrix aus Bit 24 ist **im Archiv eingefroren**. Das komplette Bit-24-Backup sowie die historische Feldgrafik sind ebenfalls gesichert. GitHub `main` enthält den neuen Quellcode, Report, **kompakte JSON-Einzelauswertungen** und die Diagramme; die vollständigen binären Bit25-Zeitreihen liegen im **vor dem Push erstellten ZIP** sowie separat in der ChatGPT-Dateibibliothek. Die ZIP-Inhaltsdateien wurden mit SHA-256 geprüft.
