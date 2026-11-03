# S4, vierte Runde: lokaler Lapse, räumliche Metrik, Vakuum-Gegenterm, innerer Knoten

*Paper 3. Code: `src/phase_bit_s4_open4.py` (Teile `A`, `B`, `C`) und
`src/phase_bit_s4_open4_figs.py`. Daten: `data/phase_bit_s4_open4.npz`.
Abbildung: `figures/phase_bit_s4_open4.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open3_de.md`](RESULTS_PhaseBit_S4_Open3_de.md).*

Speicher: Alle Regeln sind lokal, deshalb wird der Zustand nur Schritt für Schritt
weitergerechnet (dünn besetzte Matrizen, wenige MB). Das Größte war die
Diagonalisierung für die Landau-Anfangszustände: Kette mit 260 Lagen, voller
Schrittoperator 6240² vor der Projektion auf einen Helizitätssektor (3120). Der
Spitzenbedarf lag bei etwa 1.5 GB, frei waren 2.8 GB.

Modell für die Gravitation: **2O\*** aus Runde 2 (ε = 0.3, λ\* = −3.2036) mit
Vakuum-Gegenterm.

## 1. Ein streng lokaler voller Lapse: der Warte-Puffer

In Runde 3 wurde der volle Lapse über den globalen Floquet-Hamiltonoperator gebaut.
Er war also nicht lokal. Gesucht war eine lokale, unitäre Regel für die ursprüngliche
Idee: „Jeder Knoten hat nur begrenzte Rechenzeit.“

**Regel:** Jeder Zustand (x, d, s) bekommt einen **Puffer-Partner** am selben Knoten.
Ein Takt lautet

```
U = R(a(x)) · [ S·C  ⊕  (−1) ]        R(a) = cos a + i sin a · σ_x  (aktiv ↔ Puffer)
```

Das aktive Register macht einen normalen Walker-Schritt (Coin, dann Verschiebung).
Der Puffer wartet und nimmt nur die Phase −1 auf. Danach mischt R(a) beide am selben
Knoten. Die Regel ist also streng lokal und exakt unitär (1·10⁻¹⁵).

**Exakte Lösung:** R(a) wirkt auf den inneren Index wie die Identität. Deshalb wird
jeder Eigenzustand des Walkers mit Eigenphase θ zu einem 2×2-Problem, und der Takt hat
die Eigenphase

```
f_a(θ) = θ/2 + arcsin(cos a · sin(θ/2))   ≈   N·θ  mit  N = (1 + cos a)/2
```

Geprüft gegen das exakte 48×48-Spektrum (20 zufällige k, drei Werte von a) stimmt das
auf **4·10⁻¹⁵**. Kramers bleibt erhalten (5.7·10⁻¹⁵), und weil f ungerade ist, auch
E ↔ −E. Das Dirac-Quartett bei ±0.060 wird bei N = 0.75 genau zu ±0.045 (Verhältnis
0.7500).

**Deutung:** In der Nähe von E = 0 wird die **ganze** Quasienergie mit N skaliert, also
Coin *und* Ausbreitung. Das ist der volle Lapse, jetzt aus einer lokalen Regel. Der
Puffer trägt genau den Anteil **1 − N** der Amplitude (gemessen: 0.2500 bei N = 0.75).
Ein Knoten mit Lapse N verbringt also den Bruchteil 1 − N mit Warten. Das ist die
Idee der begrenzten Rechenzeit, wörtlich umgesetzt. Mit einem Puffer reicht der
Bereich bis N ≥ 1/2. Für kleinere N bräuchte man mehrere Puffer.

### Warum die Pufferphase π sein muss

Mit einer anderen Pufferphase φ liegt der Dirac-Punkt nach dem Puffern bei
E_vak(N) = (−φ/2 + arccos(cos a · cos(φ/2)))/Δt. Bei N = 1/2 sind das π − φ (mit
Δt = 1/2). Nur für **φ = π** bleibt das Vakuum für jedes N bei E = 0 (Gitter:
5·10⁻¹⁶). Bei φ = 0.95π oder 0.9π hängt seine Energie dagegen mit bis zu 0.16 bzw.
0.31 vom Lapse ab und würde gravitieren (Abbildung b). Der wartende Teil muss also
genau gegenphasig sein.

### Querbeschleunigung mit dem Puffer-Lapse

Aufbau wie in Runde 3: x-Kette aus 160 FCC-Lagen, Bloch-Impuls in y und z,
N(x) = N₀(1 + g·x) mit N₀ = 0.75 und g = 2·10⁻³, 120 Takte. Der Anfangszustand liegt
auf dem physikalischen Zweig: Die Pufferamplitude kommt aus dem lokalen
2×2-Eigenvektor, und ein kurzer Spektralfilter entfernt den Rest des Zweigs bei π.
Bezugswert ist −g·N₀²·c′².

| Zustand | Puffer-Lapse: Simulation (Halbklassik) | /(−g N₀² c′²) |
|---|---|---|
| massiv, ruhend (μ = 0.15) | −3.24 (−3.32)·10⁻⁴ | 0.86 |
| massiv, k_y = 0.3 | −3.19 (−3.26)·10⁻⁴ | 0.85 |
| massiv, k_y = 0.6 | −2.96 (−3.00)·10⁻⁴ | 0.79 |
| masselos, k_y = 0.2 | −2.68 (−3.21)·10⁻⁴ | 0.72 |
| masselos, k_y = 0.4 | −3.31 (−3.45)·10⁻⁴ | 0.88 |

Das sind dieselben Verhältnisse wie beim globalen vollen Lapse ½{N, H} in Runde 3
(0.86 / 0.85 / 0.79 / 0.73 / 0.88). **Der lokale Puffer reproduziert Einstein 1911**:
Alles fällt gleich, Licht eingeschlossen. Die Halbklassik mit der exakten
Dispersion f_{a(x)}(θ(k)) stimmt auf 1 bis 4 % überein. Beim schmalen masselosen
Strahl mit k = 0.2 sind es wie in Runde 3 etwa 15 %.

## 2. Exp. 1c: räumliche Metrik und die volle ART-Lichtablenkung

In isotropen Koordinaten lautet die schwache Metrik ds² = −N²dt² + A²dx² mit N = 1 + Φ und
A = 1 − Φ. Für ein Teilchen heißt das H = N·√(k²c′²/A² + m²), also

```
H = (N/A)·K + N·M        (K: masseloser Anteil, M: Masse)
```

**Lokale Umsetzung:**

- Der Puffer-Lapse bekommt den Wert **N/A** statt N.
- Die Masse wird auf jedem Knoten mit **A** skaliert: μ(x) = μ₀·A(x).

Dann gilt (N/A)·(K + A·M) = (N/A)·K + N·M. Anschaulich ist A die physikalische Länge
einer Gitterkante: Der dimensionslose Massenparameter pro Schritt ist m·(Kantenlänge).
Beide Zutaten sind lokale Einstellungen auf dem Knoten.

| Zustand | 1c: Simulation (Halbklassik) | /(−g N₀² c′²) | Verhältnis 1c / Puffer: Sim. (Halbkl.) | ART: 1 + (p c′/E)² |
|---|---|---|---|---|
| massiv, ruhend | −3.15 (−3.41)·10⁻⁴ | 0.84 | 0.97 (1.03) | 1 |
| massiv, k_y = 0.3 | −3.96 (−4.14)·10⁻⁴ | 1.06 | **1.24** (1.27) | 1.25 |
| massiv, k_y = 0.6 | −4.69 (−4.80)·10⁻⁴ | 1.25 | **1.58** (1.60) | 1.57 |
| masselos, k_y = 0.2 | −5.37 (−6.43)·10⁻⁴ | 1.43 | **1.999** (2.000) | 2 |
| masselos, k_y = 0.4 | −6.58 (−6.90)·10⁻⁴ | 1.75 | **1.989** (2.000) | 2 |

- **Licht fällt doppelt so stark wie mit dem Lapse allein:** Das Verhältnis ist 1.999
  bzw. 1.989. Das ist der volle ART-Faktor 2 der Lichtablenkung.
- **Ruhende Masse fällt gleich** (0.97, Halbklassik 1.03): Die räumliche Metrik wirkt
  nicht auf ruhende Teilchen, wie in der ART.
- **Bewegte Masse** folgt dem ART-Faktor 1 + v²/c′², wobei v = p c′/E die
  Dirac-Geschwindigkeit ist (m = 2μ = 0.3). Gemessen: 1.24 gegen 1.25 und 1.58 gegen 1.57.
  Mit der Gruppengeschwindigkeit des Gitterbands (0.45 c′ und 0.66 c′) passt es
  schlechter (1.20 und 1.43). Das Gitterband ist bei μ/ε = 0.5 langsamer als Dirac. Die
  Kopplung an A misst aber den Anteil des kinetischen Terms an der Energie, nicht die
  Gruppengeschwindigkeit.

**Damit ist die Stufenfolge aus Experiment 1 komplett:**

| Stufe | Regel | Lichtablenkung |
|---|---|---|
| Uhren-Lapse | nur der Coin wird langsamer | Nordström: keine |
| Puffer-Lapse | lokaler Warte-Puffer | Einstein 1911: halbe |
| Puffer-Lapse + räumliche Metrik | Puffer-Lapse N/A, Masse μ·A | ART: volle |

Jede Stufe besteht aus streng lokalen, unitären Gitterregeln.

## 3. Woher kommt der Vakuum-Gegenterm?

Der Gegenterm wurde in Runde 3 von Hand gesetzt. Hier zeigt sich: Er ist **nicht
frei wählbar**, sondern doppelt erzwungen.

**(a) Äquivalenzprinzip für Antiteilchen.** Liegt der Dirac-Punkt bei einer
Restenergie δ, dann fallen Teilchen und Antiteilchen mit R′ = 1 ± δ/m. Gemessen mit
dem Puffer-Lapse (μ = 0.03, m = 0.06):

| δ | −0.06 | −0.03 | −0.01 | 0 | +0.01 | +0.03 | +0.06 |
|---|---|---|---|---|---|---|---|
| R′ Teilchen | 0.000 | 0.498 | 0.831 | **0.997** | 1.163 | 1.495 | 1.993 |
| R′ Antiteilchen | 1.993 | 1.495 | 1.163 | **0.997** | 0.831 | 0.498 | 0.000 |

Gleiches Fallen von Materie und Antimaterie verlangt also δ = 0 exakt. Der volle
Floquet-Offset (E_c = 1.02 ≫ m) wäre ein krasser Fall davon. Mit dem Puffer-Lapse würde
das obere Dublett dann 4.4-mal zu stark gezogen (∂E′/∂N = 1.315 statt 0.300). Das
untere Dublett hätte das falsche Vorzeichen (+0.720 statt −0.300) und fiele nach oben.

**(b) Der lokale Lapse verlangt es selbst.** Der Puffer-Takt ist nur um seinen Fixpunkt
E = 0 eine reine Zeitdilatation. Ein Zustand bei E = 0 „rechnet nicht“: Er wird vom
Puffer nicht verändert, egal wie groß N ist. Pufferphase π und Gegenterm sind
dieselbe Forderung: **Das Vakuum ist der Fixpunkt der Warte-Regel.**

**Was nicht geht:** Eine Gittersymmetrie, die ein *einzelnes* Quartett auf E = 0 legt,
gibt es in 2O\* nicht. E ↔ −E bildet das Quartett bei +E_c auf das zweite bei −E_c ab.
Nur am verdoppelten Punkt λ = −2 liegen beide von selbst bei 0, und dort gibt es zwei
Flavours mit je eigenem Untergrund (Runde 2). Der Gegenterm ist also der Preis für
einen einzelnen Dirac-Punkt. Er ist aber eindeutig bestimmt und lokal: eine Phase pro
Takt, gleich auf allen Knoten. Unter dem Lapse skaliert er wie alles andere.

## 4. Der chirale Zweig am inneren Knoten, gepumpt in der Zeit

Aufbau: minimales SU(2), μ = 0, Landau-Eichung, offene x-Kette, Helizitätssektor +1.
Als Anfangszustand dient der unterste Landau-Zustand (LLL) am jeweiligen Knoten. Er
ist der zentrierte Eigenzustand nahe dem Knoten mit der größten z-Geschwindigkeit
v_z = ⟨m_z⟩/Δt. Dann wird ein elektrisches Feld E ∥ B simuliert: k_z läuft linear in der
Zeit (2·10⁻⁴ pro Takt, ±0.25). Gemessen wird ⟨E⟩(t) = −arg⟨ψ|U|ψ⟩/Δt.

Zuerst ein Befund aus dem Volumenband: **Beide Knoten sind isotrope Weyl-Kegel mit
v = c′** (Steigungen ±1.000 c′ in allen Richtungen). Ihre Chiralitäten sind
entgegengesetzt.

| B | Knoten | LLL bei E − E_K | v_z des Zustands | gepumpt dE/dk_z |
|---|---|---|---|---|
| 0.05 | unten, E = −2.078 (χ = +1) | −0.031 | −0.989 c′ | **−0.988 c′** |
| 0.05 | innen, E = +0.693 (χ = −1) | −0.032 | +0.699 c′ | **+0.698 c′** |
| 0.035 | unten | −0.022 | −0.993 c′ | **−0.992 c′** |
| 0.035 | innen | −0.016 | +0.815 c′ | **+0.851 c′** |

- **Der innere Knoten hat einen chiralen Zweig mit entgegengesetzter Steigung.** Beim
  Pumpen in der Zeit steigt seine Energie mit k_z, die des unteren Knotens fällt
  (Abbildung f). Der Zustand bleibt dabei kohärent (|⟨U⟩| ≥ 0.998). Damit ist die
  Bilanz „netto null pro Helizitätssektor“ jetzt an beiden Knoten direkt gezeigt.
- Die Geschwindigkeit am inneren Knoten liegt unter c′ (0.70 bzw. 0.82 c′) und nähert
  sich c′ mit kleinerem B. Der innere Knoten ist nur 0.57 von den nächsten vierfachen
  Niveaus entfernt (bei −0.57 und +1.26 im Sektor), sein Weyl-Bereich ist also klein.
  Die Landau-Skala √(2B)·c′ = 0.18 bzw. 0.15 reicht schon an die Krümmung heran.
- **Physikalisch:** Paralleles E und B pumpen in jedem Helizitätssektor Zustände am
  unteren Knoten nach unten und am inneren Knoten nach oben. Die Anomalie der
  Knoten-Chiralität verteilt sich so auf zwei Quasienergien.
- **Grenze:** Längere Sweeps (|Δk_z| > 1) verlassen die Knotenumgebung. Das Band hat
  dort offene Bahnen in der Ebene senkrecht zu B, der Zustand delokalisiert und läuft
  aus der offenen Kette. Den globalen Verlauf des Zweigs über die ganze Zone
  (verbindet er die Knoten direkt oder über den Zonenrand?) zeigt diese Rechnung nicht.

## Antwort

1. **Lokaler voller Lapse:** Ein Warte-Puffer pro Knoten mit Phase π und lokaler
   Mischung R(a) ist exakt unitär und skaliert das Spektrum um E = 0 mit
   N = (1 + cos a)/2. Der Puffer hält den Anteil 1 − N der Amplitude. Der volle Lapse
   (Einstein 1911) folgt damit aus der Idee der begrenzten Rechenzeit, ohne globalen
   Operator.
2. **Exp. 1c:** Puffer-Lapse N/A plus Masse μ·A ergibt die schwache isotrope Metrik.
   Licht wird doppelt so stark abgelenkt (1.999), ruhende Masse fällt unverändert, und
   bewegte Masse folgt 1 + v²/c′² (1.24 / 1.58 gegen 1.25 / 1.57). Das ist die volle
   ART im schwachen Feld.
3. **Gegenterm:** Er ist erzwungen, durch gleiches Fallen von Teilchen und Antiteilchen
   (R′ = 1 ± δ/m) und durch die lokale Lapse-Regel, deren Fixpunkt das Vakuum sein muss.
   Eine Gittersymmetrie, die ihn ersetzt, gibt es nur um den Preis der Verdopplung.
4. **Innerer Knoten:** Pumpen mit E ∥ B in der Zeit zeigt seinen chiralen Zweig mit
   +0.70 c′ (B = 0.05) bzw. +0.85 c′ (B = 0.035), entgegengesetzt zum unteren Knoten
   (−0.99 c′).

## Offene Punkte

- **Exp. 1d:** Der Lapse kommt bisher von außen. Nächster Schritt wäre ein Lapse aus der
  lokalen Last, also N(x) aus der Teilchendichte (Rückwirkung, Poisson-Gleichung?).
  Mit dem Puffer ist das jetzt eine lokale Regel: Der Mischwinkel a(x) hängt von der
  Dichte am Knoten ab.
- Mehrere Puffer für starke Felder (N < 1/2) sowie zeitabhängige Lapse-Felder
  (Gravitationswellen?).
- Die räumliche Metrik wird über die Masse und den Puffer eingestellt, nicht über
  echte Kantenlängen. Ob ein unregelmäßiges Gitter (variable Kantenlänge) dasselbe
  leistet, ist offen.
- Der globale Verlauf der chiralen Landau-Zweige über die ganze Zone (Floquet-Anomalie
  über den Zonenrand oder direkte Verbindung der Knoten). Dafür bräuchte man eine
  periodische Kette mit kommensurablem Fluss.
