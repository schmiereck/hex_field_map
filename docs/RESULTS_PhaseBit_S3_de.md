# Experiment S3: Spin ½ in 3D aus einem endlichen Phasen-Bit-Register

*Paper 3, Hauptlinie. Code: `src/phase_bit_fcc_2O.py` und `src/phase_bit_fcc_2O_figs.py`.
Daten: `data/phase_bit_fcc_2O.npz`. Abbildung: `figures/phase_bit_fcc_2O.png`.
Vorgänger: [`RESULTS_PhaseBit_S1_de.md`](RESULTS_PhaseBit_S1_de.md), [`RESULTS_PhaseBit_S2_de.md`](RESULTS_PhaseBit_S2_de.md).*

## Frage

In 2D reicht ein Z₄-Phasen-Bit pro Walker für Spin ½ (S1, S2). In 3D transportiert
der FCC-Walker (`quantum_fcc_3d.py`) den Spinor mit der minimalen SU(2)-Drehung.
Diese Drehungen erzeugen eine unendliche Gruppe, das Register wäre also
kontinuierlich. **Geht Spin ½ in 3D auch mit einem endlichen Register?**

## Konstruktion

Ersetzt wird die minimale Drehung durch ein Element der **binären Oktaedergruppe 2O**
(48 Elemente; das ist bis auf Phasen die Clifford-Gruppe eines Qubits). Für jedes
Paar benachbarter Richtungen (60°) gibt es in O genau zwei Drehungen, die d auf d′
abbilden: eine 90°- und eine 120°-Drehung. Die 90°-Drehung liegt immer näher an der
minimalen Drehung (Treue 0.8165 gegenüber 0.5774). Die Wahl ist also **eindeutig**:

> Jede 60°-Kurve der Bewegungsrichtung dreht den Spinor um eine
> **Vierteldrehung um eine Würfelachse**: Q = (1 + i·n·σ)/√2.

Prüfungen: Q(d′→d) = Q(d→d′)† gilt exakt (0.0). Die Inversionssymmetrie
Q(−d→−d′) = Q(d→d′) gilt exakt. √2·Q hat Einträge in Z[i] (ganze Gaußsche Zahlen).
Die 48 Q erzeugen genau die Gruppe 2O (Ordnung 48).

## Ergebnisse

### 1. Das Register ist endlich, und es ist wieder ein Z₄-Phasen-Bit

Der Walker trägt Richtung d und das aufgesammelte Gruppenelement g ∈ 2O. Von den
12 × 48 = 576 denkbaren Kombinationen werden **nur 48 erreicht, 4 pro Richtung**.
An der Startrichtung sind das {+1, +i·n·σ, −i·n·σ, −1}, also eine Z₄-Gruppe, erzeugt
von der 180°-Drehung um die eigene Richtung. Das ist genau die Struktur aus 2D
(S1: 12 von 24, ein Z₄-Bit pro Richtung). Der Unterschied: Beim Wechsel der Richtung
transformiert sich das Bit nichtabelsch.

Die Ganzzahl-Tabelle der Pfadzahlen N[x, y, z, d, Bit, T] reproduziert, ausgelesen
mit den irreduziblen Darstellungen von 2O, den Gleitkomma-Walk exakt (größte
relative Abweichung **1.8·10⁻¹⁵**, t = 8, ε ∈ {0.2, 0.6}, sechs Darstellungen). Aus
einer einzigen Tabelle entstehen damit:

| Auslesung | Dimension | Typ |
|-----------|-----------|-----|
| E₁/₂ (Spin ½) | 2 | Fermion (−1 ↦ −1) |
| E₅/₂ | 2 | Fermion |
| G₃/₂ (Spin 3/2) | 4 | Fermion |
| A₁ (Spin 0) | 1 | Boson |
| A₂ | 1 | Boson |
| T₁ (Vektor) | 3 | Boson |

### 2. Die Holonomie ist quantisiert: Dreieck π, Quadrat 2π

Die Spin-Holonomie um die Flächen des Kuboktaeders im Richtungsraum, angegeben als
Drehwinkel / 2π:

| Schleife | 2O-Register | minimale SU(2) |
|----------|-------------|----------------|
| Richtungs-Dreieck | **0.500** (π) | 0.088 (0.551 rad = Raumwinkel) |
| Richtungs-Quadrat | **1.000** (2π, also −1) | 0.216 (1.360 rad) |
| geschlossener Weg L = 6 | 1.000 (−1) | 1.000 (−1) |

Die minimale SU(2) verteilt die Krümmung der Kugel gleichmäßig nach Raumwinkel
(8·0.551 + 6·1.360 = 4π, Gauß–Bonnet). 2O **bündelt** sie in Vielfache von π. Das
Integral bleibt modulo 4π dasselbe. Alle geschlossenen Raumwege bis L = 8 (nur
60°-Kurven) und bis L = 7 (mit Geradeaus-Schritten) geben in **beiden** Modellen die
Holonomie −1 (4 Wege bei L = 6, 24 bei L = 8). Der Belt-Trick (−1 für die kürzeste
Schleife) bleibt also erhalten.

### 3. Spin ½ funktioniert: Kramers, Unitarität, Kausalität

Mit dem unitären Coin expm(iεG), G = Σ |d′⟩⟨d| ⊗ ρ(Q), ε = 0.1:

| Größe | 2O, Spin ½ | minimale SU(2), Spin ½ |
|-------|-----------|------------------------|
| Unitarität | 8.9·10⁻¹⁶ | 6.7·10⁻¹⁶ |
| Kramers-Aufspaltung, 40 zufällige k | **5.8·10⁻¹⁵** | 4.9·10⁻¹⁵ |
| max \|v_g\| | 1.687 < c = 1.732 | 1.693 |
| Ruheniveaus E/ε (Vielfachheit) | −4√2 (4), −2√2 (8), +2√2 (8), +4√2 (4) | −6.928 (4), −1.900 (8), +2.309 (4), +4.210 (8) |

Die Kramers-Verdopplung gilt bei jedem k, auch für E₅/₂ und G₃/₂. Die Ruheniveaus
des 2O-Modells sind exakte Vielfache von √2, die Zahlen des Clifford-Rings.

### 4. Neu: Teilchen-Antiteilchen-Symmetrie

Das kontinuierliche FCC-Modell hat **keine** Symmetrie E ↔ −E. Das war als
grundsätzliche Grenze vermerkt: Das Kuboktaeder enthält Dreiecke, ist also nicht
bipartit. Mit dem 2O-Register ist das Spektrum dagegen **bei jedem k exakt
symmetrisch**: max |E(k) − (−E)(k)| = **3.6·10⁻¹⁵** über 30 zufällige k, gegenüber
0.217 bei minimaler SU(2). Abbildung (c) und (d) zeigen das.

Mechanismus (plausibel, nicht bewiesen): Jedes Richtungs-Dreieck trägt die Holonomie
π, deren SU(2)-Element spurlos ist (±i·n·σ). Ungerade Zyklen tragen dadurch nicht
mehr zu tr(G^ungerade) bei, und der Graph verhält sich spektral wie ein bipartiter.
Die Grenze lag also nicht an der Geometrie, sondern an der kontinuierlichen
Verteilung der Krümmung.

### 5. Preis: mehr Anisotropie

Das obere Band ist richtungsabhängiger. Gemessen wurde die Spreizung von E − m₀
über [100], [110] und [111]:

| k | 0.30 | 0.60 | 1.00 |
|---|------|------|------|
| 2O | 16.8 % | 24.2 % | 29.5 % |
| minimale SU(2) | 7.9 % | 13.4 % | 20.5 % |

Bei k = 0 ist das obere Niveau vierfach entartet, die Zuordnung „oberes Band“ ist
dort also nur grob. Die Spreizung bei kleinem k kommt aus der Krümmung, nicht aus
dem Kegel selbst: Die vierfachen Niveaus (oben und unten) sind in **beiden**
Modellen masselose Dirac-Kegel mit der isotropen Steigung **c/3** (2O: ±0.5774 in
[100], [111] und [123] bei k = 10⁻⁴). Ein massives Spin-½-Band gibt es in keiner
der beiden Varianten (siehe `RESULTS_FCC_3D_de.md`, Abschnitt 6).

## Antwort

**Ja.** Spin ½ geht in 3D mit einem endlichen Register. Pro Richtung genügt ein
Z₄-Phasen-Bit, das sich beim Abbiegen um eine Vierteldrehung um eine Würfelachse
weiterdreht. Das Register ist die binäre Oktaedergruppe, alle Amplituden liegen im
Ring Z[i, 1/√2]. Erhalten bleiben Kramers-Verdopplung bei jedem k, Belt-Trick,
Unitarität und Kausalität. Dazu kommt die Teilchen-Antiteilchen-Symmetrie, die das
kontinuierliche Modell nicht hatte. Bezahlt wird mit stärkerer Anisotropie.

Damit ist die Idee aus dem Thread („Phasen-Bits auf dem Kuboktaeder“) in korrigierter
Form tragfähig. Das Bit sitzt nicht auf den Knoten eines 4er-Rings, sondern im
Richtungsraum, als Gruppenelement des Transports.

## Nebenbefund: Vorzeichenkonvention in `quantum_fcc_3d.py` (inzwischen behoben)

`su2_transport(u, v)` liefert cos(θ/2) + i·sin(θ/2)·n·σ. Unter U σ·w U† dreht dieses
Element um **−θ**, also von v weg (geprüft: |R(Q)u − v| = 1.414 statt 0). Die
Abbildung R ↦ Q ist damit ein Anti-Homomorphismus. Der Spinor folgt der Richtung
dann nicht konsistent. In ebenen Schleifen fällt das nicht auf (dort vertauschen
alle Drehungen, beide Konventionen geben −1). Das Spektrum ändert sich aber:

| Konvention | Ruheniveaus E/ε (Vielfachheit), Spin ½ |
|------------|----------------------------------------|
| konsistent, exp(−iθ n·σ/2) | −6.928 (4), −1.900 (8), +2.309 (4), +4.210 (8) |
| `quantum_fcc_3d.py` | −6.928 (2), −5.318 (4), −2.309 (2), 0 (6), +3.009 (4), +4.619 (6) |

Kramers-Verdopplung und Kausalität gelten in beiden Fällen (Zeitumkehr ist
unabhängig davon). Der Belt-Trick in `belt_trick()` nutzt Quaternionen in der
konsistenten Konvention und ist **nicht** betroffen. Betroffen sind die
spektralen Aussagen in `RESULTS_FCC_3D_de.md` (Massen, Bänder) und die darauf
aufbauenden FCC-Teile von `RESULTS_TwoParticle_de.md` und `RESULTS_Helicity_de.md`
(Skripte `quantum_fcc_3d.py`, `quantum_fcc_3d_figs.py`, `quantum_two_particle_figs.py`,
`quantum_helicity_test.py`).

**Behoben (2. Nov. 2026):** `su2_transport` liefert jetzt exp(−iθ n·σ/2). Alle
betroffenen Ergebnisse sind neu gerechnet und die Dokumente korrigiert. Die
wichtigste Änderung: Der Spin-½-Zweig des FCC-Walkers ist ein **masseloser
Dirac-Kegel** (Steigung c/3), kein massives Band. Gegenläufige Dublett-Zustände
haben deshalb exakt den Überlapp 0.

## Offene Punkte

- Beweis der E ↔ −E-Symmetrie: Gibt es einen expliziten chiralen Operator Γ mit
  ΓT(k)Γ† = T(k)*-artig, der aus der π-Holonomie der Dreiecke folgt?
- Gibt es eine Wahl in 2O, die die Anisotropie senkt? Mit der Bedingung
  „Transport bildet d auf d′ ab“ gibt es pro Kante nur zwei Kandidaten. Eine
  gemischte Wahl würde die Inversionssymmetrie brechen.
- Masse: Ein Term, der den Dirac-Punkt öffnet (in 3D auch T-erhaltend möglich), fehlt
  noch, sowohl mit 2O als auch mit minimaler SU(2). In 2D liefert die chirale
  Umkehr Masse (S2), dort aber nur unter Bruch der Zeitumkehr.
