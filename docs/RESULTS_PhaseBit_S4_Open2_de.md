# S4, neue offene Punkte: 2O ohne Untergrund, θ, Domänenwände, Zitterbewegung

*Paper 3, Hauptlinie. Code: `src/phase_bit_s4_open2.py` und `src/phase_bit_s4_open2_figs.py`.
Daten: `data/phase_bit_s4_open2.npz`. Abbildung: `figures/phase_bit_s4_open2.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open_de.md`](RESULTS_PhaseBit_S4_Open_de.md).*

Speicher: Kein Lauf brauchte mehr als etwa 0.3 GB. Die größten Matrizen sind die
Schichtoperatoren der Domänenwände mit 2880 × 2880 Einträgen.

## 1. 2O-Register: E ↔ −E und kein Untergrund

**Welche Terme kommen in Frage?** Gesucht waren alle Kopplungen, deren Spinfaktor
eine Drehung aus O ist, die d auf d′ abbildet. Damit bleibt das Register endlich
(2O), und die Helizität H bleibt erhalten. Nur ein Term erhält zusätzlich P, T, PT
und E ↔ −E: **W₉₀**, die Kopplung der 90°-Richtungspaare mit einer Vierteldrehung um
die Würfelachse. Er vertauscht mit G₂O ([G₂O, W₉₀] = 0) und verschiebt nur dessen
vier Niveaus. Die anderen Kandidaten brechen E ↔ −E (60°/120° mit 120°-Drehung) oder
P und PT (die 180°-Drehungen).

**Mechanismus.** Der Untergrund des Dirac-Quartetts ist Niveauabstoßung in zweiter
Ordnung, ausgelöst von den beiden achtfachen Niveaus. Deren Geschwindigkeitsgewichte
sind 2 (für G = −√2) und 6 (für G = +√2). Zum gegenüberliegenden Quartett besteht
keine Kopplung. W₉₀ schiebt die Niveaus gegeneinander. In zweiter Ordnung
verschwindet die Summe 2/(G_D − G_A) + 6/(G_D − G_B) bei **λ = −5 exakt**.

**Ergebnis** (Abbildung a):

- Der volle Floquet-Walker hat die Nullstelle bei **λ* = −4.979** (ε → 0).
- Danach bleibt ein Floquet-Rest der Ordnung ε²: α·ε = 5·10⁻⁴ / 2.3·10⁻³ / 3·10⁻²
  bei ε = 0.05 / 0.1 / 0.3. Ohne Zusatzterm ist α·ε = 0.196.
- Pro ε lässt sich λ*(ε) exakt nachstellen: −4.556 (ε = 0.1), −3.204 (ε = 0.3).
- Es gibt **keine Verdopplung**: Die beiden Dirac-Quartette liegen getrennt bei
  G = ±4.21, und E ↔ −E bildet das eine auf das andere ab.

Eigenschaften bei λ*(ε):

| ε, μ/ε | Lücke/μ | E ↔ −E | Kramers | d²E/dk² unten / oben (Dirac) | oberes Band gegen Dirac bei k = 0.2 / 0.4 | max \|v_g\| |
|---|---|---|---|---|---|---|
| 0.1, 0.1 | 4.00000 | 5·10⁻¹⁵ | 4·10⁻¹⁵ | −16.64 / +16.64 (±16.67) | −2.1 % / −7.5 % | 1.26 |
| 0.1, 0.5 | 4.00000 | 3·10⁻¹⁵ | 5·10⁻¹⁵ | −3.21 / +3.17 (±3.33) | −2.7 % / −8.8 % | 1.25 |
| **0.3, 0.1** | 4.00000 | 4·10⁻¹⁵ | 4·10⁻¹⁵ | −5.54 / +5.54 (±5.56) | **−0.3 % / −1.0 %** | 0.87 |
| **0.3, 0.5** | 4.00000 | 4·10⁻¹⁵ | 4·10⁻¹⁵ | −1.02 / +1.02 (±1.11) | **−0.6 % / −2.0 %** | 0.80 |

Bei ε = 0.3 ist die Abweichung in [100], [111] und [123] gleich groß, das Band ist also
isotrop. Das ist das sauberste massive Spin-½-Teilchen der ganzen Serie: Es hat ein
endliches Register, E ↔ −E und Kramers exakt, eine Lücke von exakt 4μ und ist bis
k = 0.4 auf 2 % Dirac-artig, **auch bei μ/ε = 0.5** (Abbildung b). Zum Vergleich:
Das in S4+ kompensierte SU(2)-Modell weicht bei k = 0.4 schon um etwa −35 % ab.

**Ein Fehlschluss, den ich unterwegs gemacht und korrigiert habe.** Bei λ = −2 treffen sich
das obere und das untere Quartett bei E = 0 zu einem achtfachen Punkt. Der Mittelwert
aller acht Bänder verschwindet dann wegen E ↔ −E identisch (10⁻¹⁶). Jeder der beiden
„Flavours“ hat aber weiterhin seinen eigenen Untergrund, mit entgegengesetztem
Vorzeichen: Bei k = 0.4 liegt das Zentrum des oberen Flavours bei +0.021. Dieser Punkt
ist also **keine** Lösung.

## 2. θ wird beobachtbar, sobald die Helizität gebrochen ist

S4+ hat gezeigt: Die Masse μ(cos θ·Π + sin θ·Π iH) hat ein θ-unabhängiges
Spektrum, weil H eine exakte chirale Symmetrie ist. Kommt aber eine zweite Masse dazu,
die H bricht, dann zählt der relative Winkel. Das gilt für jede Kopplung, die den Spin
beim Richtungswechsel *nicht* mitdreht (60° oder 90° mit Spin-Identität, [H, W] ≠ 0).
Am Dirac-Punkt projiziert eine solche Kopplung exakt auf Π. Für ν·W mit gleichem
Gewicht wie μΠ gilt:

> Lücke = 4μ·|1 + e^{iθ}|, auf 10⁻¹⁴ genau (beide Modelle, beide Kopplungen; Abbildung c)

Bei θ = π heben sich die beiden Massen auf, und die Lücke schließt sich. Das ist die
Gitterversion der „twisted mass“ der Gitter-QCD: Der chirale Winkel einer Masse ist
erst relativ zu einem chiralitätsbrechenden Term physikalisch.

Eine echte chirale *Anomalie* (Kopplung von θ an ein Eichfeld) ist das nicht. Dazu
bräuchte man ein Eichfeld in 3D, das hier nicht gebaut wurde.

## 3. Massen-Domänenwände: der 2D-Spin-Wirbel als Wandzustand

**Aufbau:** Eine Schicht aus 120 FCC-Lagen in z-Richtung, periodisch, mit Bloch-Impuls
k_∥ in x und y. Gerechnet wird exakt, als Floquet-Operator der Dimension 2880. Es
gibt zwei Wände im Abstand von 60 Lagen. Modell: das kompensierte SU(2)-Modell aus S4+,
ε = 0.3, μ = 0.05 (m = 0.1). Gezählt werden Zustände mit |E − E_c| < 0.9m, die zu
mehr als 60 % an den Wänden sitzen.

| Wandprofil | k_∥ = 0 | 0.02 | 0.04 | 0.06 |
|---|---|---|---|---|
| **Vorzeichenwechsel** μΠ → −μΠ (T überall erhalten) | **−0.0006** (×4) | ±0.011 | ±0.022 | ±0.033 |
| **chirale Drehung** θ: 0 → π (T in der Wand gebrochen) | −0.033, +0.038 | −0.034, +0.039 | −0.050, +0.044 | −0.054, +0.051 |

(Werte relativ zur Lückenmitte. Jeder Eintrag ist ein Kramers-Paar pro Wand.)

- **Vorzeichenwechsel:** Jede Wand trägt einen **masselosen 2D-Dirac-Kegel**. Seine
  Steigung ist 0.55 bis 0.57, also c/3 (Abbildung d). Bei k = 0 ist er ein
  Kramers-Paar, also genau das Objekt aus S4+ Abschnitt 2: Der 2D-Spin-Wirbel ist
  durch T geschützt.
- **Chirale Drehung:** Hier ist |m| überall gleich, nur die chirale Phase dreht
  sich, und T ist in der Wand gebrochen. Die Wandzustände sind **gegapt** (Lücke etwa
  0.07 bei einer Volumenlücke von 0.2).
- Im verdoppelten 2O-Modell (λ = −2) zeigt sich dasselbe Bild: acht fast
  entartete Nullmoden beim Vorzeichenwechsel (±0.002), keine bei der chiralen Drehung.

**Deutung.** Das ist die Physik eines topologischen Isolators. μΠ und −μΠ sind
zwei T-symmetrische Massen mit verschiedenem θ (0 und π, dem Axion-Winkel). An ihrer
Grenze muss ein T-geschützter Dirac-Kegel liegen. Damit ist θ ∈ {0, π} auch ohne
zusätzliche Kopplung **beobachtbar**: nicht im Volumen, aber an jeder Grenze, an der es
springt. Und die Verbindung zu 2D ist direkt: **Der masselose 2D-Spin-Wirbel ist der
Randzustand des massiven 3D-Spin-Wirbels.**

**Einschränkung.** Bei einem Floquet-Walker überlappen Bänder bei großen Impulsen
(|k| > 1, Gitterskala) energetisch mit der Massenlücke, in etwa 17 % der
Brillouin-Zone. Eine globale Energielücke gibt es also nicht. Beim 2O-Modell mit
λ = −2 sind es nur etwa 1 %. Wandzustände und Dirac-Teilchen sind trotzdem sauber, weil
der Impulsunterschied zu diesen Zuständen groß ist. Strenge topologische Invarianten
(Z₂-Index) wurden deshalb nicht berechnet.

## 4. Amplitude der Zitterbewegung

In S4+ war der Anfangsspinor innerhalb der entarteten Dubletts beliebig, die
Amplitude also basisabhängig. Jetzt ist er eindeutig gewählt: der Zustand mit
Helizität +1 und größtem S_z im Dirac-Raum bei k = 0. Die Vorhersage für den
oszillierenden Teil des Schwerpunkts ist eichinvariant formuliert, mit
Bandprojektoren:

y_osc(t) = 2 Re Σ_k g(k)² · i⟨u₀|P_unten(k) ∂_y P_oben(k)|u₀⟩ · e^{−iω(k)t} / Σ g²

Damit lassen sich ganze Kurven vergleichen (ε = 0.3, μ = 0.03, σ = 40, k₀ = 0.03;
Abbildung e, f):

| Modell | Amplitude bei t = 0: Simulation / Bandtheorie / Dirac 4×4 | rms(Sim − Band) | rms(Sim − Dirac) | Signal-rms |
|---|---|---|---|---|
| ohne Kompensation | 1.136 / 1.146 / 1.144 | 0.003 | 0.010 | 0.556 |
| kompensiert | 1.139 / 1.145 / 1.144 | 0.004 | 0.016 | 0.558 |

Die Zitterbewegung stimmt in Amplitude, Frequenz und Abklingen mit der Bandtheorie auf
0.6 % und mit der effektiven Dirac-Theorie auf 2 bis 3 % überein. Das Abklingen kommt
allein von der Dephasierung über die k-Breite des Pakets. Die einfache
Einzel-k-Formel c′²m/(2E²) = 2.56 überschätzt die Amplitude um mehr als das Doppelte.
Sie gilt nicht für Pakete.

## Antwort

1. **2O mit E ↔ −E und α = 0:** Das geht, mit einem einzigen zusätzlichen 2O-Term:
   λ*·ε·W₉₀ mit λ* ≈ −5. Es gibt keine Verdopplung und keinen Symmetriebruch. Bei
   ε = 0.3 entsteht das bisher sauberste massive Spin-½-Teilchen (2 % Dirac bis
   k = 0.4 bei μ/ε = 0.5). Preis: ein großer Zusatzterm und eine Restabhängigkeit der
   Ordnung ε², die man durch λ*(ε) ausgleicht.
2. **θ beobachtbar:** Im Volumen erst neben einer helizitätsbrechenden Masse, dann gilt
   die Lücke 4μ|1 + e^{iθ}|. Ein Sprung θ = 0 → π (Vorzeichen der Masse) ist dagegen
   immer sichtbar: Er trägt einen masselosen, T-geschützten 2D-Dirac-Kegel.
3. **Zitterbewegung:** Amplitude und Form folgen der Bandtheorie auf 0.6 % und der
   Dirac-Theorie auf 2 bis 3 %.

## Offene Punkte

Beantwortet in [`RESULTS_PhaseBit_S4_Open3_de.md`](RESULTS_PhaseBit_S4_Open3_de.md):
- **Gravitation:** Mit Vakuum-Gegenterm gilt R′ = 0.997 für Teilchen und Antiteilchen.
  Der Uhren-Lapse ist Nordström, der volle Lapse ist Einstein 1911.
- **Z₂ praktisch:** Ein Kramers-Paar an der Wand ist robust, zwei Paare bekommen eine Lücke.
- **Eichfeld:** [H, W_B] = 0 exakt, θ bleibt unbeobachtbar. Die Anomalie betrifft die
  Knoten-Chiralität.
