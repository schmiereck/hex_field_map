# Experiment 3: Isotropie der Kuboktaeder-Schalen und der Lichtkegel-Front

*Paper 3. Code: `src/phase_bit_exp3_isotropy.py` (Teile `A`–`D`) und
`src/phase_bit_exp3_isotropy_figs.py`. Daten: `data/phase_bit_exp3_isotropy.npz`.
Abbildung: `figures/phase_bit_exp3_isotropy.png`.
Roadmap: Experiment 3 in [`ROADMAP_Paper3_de.md`](ROADMAP_Paper3_de.md).*

## Frage

Die Ausgangsidee des Gemini-Threads: Die Knoten der Kuboktaeder-Schale n des
FCC-Gitters (Graphabstand n, 10n² + 2 Knoten) bilden eine diskrete Bloch-Kugel. Wird
sie mit wachsendem n isotrop? Und wie isotrop ist das, was der Walker tatsächlich
tut?

**Maß:** Anisotropie-Leistung einer gewichteten Richtungsmenge

```
C_l = 4π/(2l+1) · Σ_m |a_lm|²,      a_lm = Σ_i w_i Y_lm(r̂_i) / Σ_i w_i
```

C_l = 0 bedeutet isotrop, C_l = 1 einen einzelnen Punkt. Jede würfelsymmetrische Menge
ist ein sphärisches 3-Design (C₁ = C₂ = C₃ = 0). Der erste echte Test ist deshalb
l = 4, danach l = 6 und 8.

Modell für den Walker: **2O\*** (ε = 0.3, λ\*, Vakuum-Gegenterm), ohne Gravitation.

## A. Die Schalen selbst: nie isotrop

| n | 1 | 2 | 4 | 8 | 16 | 64 | 200 | Kontinuumsgrenze |
|---|---|---|---|---|---|---|---|---|
| Knoten | 12 | 42 | 162 | 642 | 2562 | 40962 | 400002 | |
| C₄ | 3.6·10⁻² | 3.0·10⁻³ | 3.39·10⁻³ | 3.27·10⁻³ | 3.23·10⁻³ | 3.220·10⁻³ | 3.219·10⁻³ | 3.23·10⁻³ |
| C₆ | 0.33 | 9.6·10⁻⁵ | 1.14·10⁻³ | 1.04·10⁻³ | 1.02·10⁻³ | 1.006·10⁻³ | 1.006·10⁻³ | 1.01·10⁻³ |
| C₈ | 0.16 | 2.9·10⁻³ | 1.35·10⁻⁴ | 1.23·10⁻⁴ | 1.20·10⁻⁴ | 1.19·10⁻⁴ | 1.19·10⁻⁴ | 1.24·10⁻⁴ |

- **C_l sättigt ab n ≈ 8.** Die Schale n ist ein maßstäblich vergrößertes
  Kuboktaeder, und ein Polyeder wird durch Vergrößern nicht rund. Der Grenzwert ist die
  Kuboktaeder-Oberfläche, gewichtet mit der Knotendichte der Flächen ({111}: 1/√3,
  {100}: 1/2), radial auf die Kugel projiziert. Er stimmt auf 0.3 % (C₄, C₆) bzw.
  4 % (C₈) mit n = 200.
- Die verbleibende Anisotropie hat die Amplitude √C₄ ≈ 5.7 %.
- Zufällig verteilte Punkte gleicher Anzahl erreichen C ≈ 1/N und sind ab n ≈ 10
  isotroper als die Schale.
- **n = 2 ist ein Sonderfall:** C₆ = 10⁻⁴, C₄ = 3·10⁻³. Für die Auflösung einer
  Bloch-Kugel bringt eine größere Schale aber keinen Gewinn mehr.

**Antwort auf die ursprüngliche Frage:** Als Punktmenge ist die Kuboktaeder-Schale
keine zunehmend bessere Bloch-Kugel. Sie bleibt ein 3-Design mit einem festen Rest bei
l = 4, 6, 8.

## B. Der Walker: das Teilchen wird isotrop, die Schale ist dafür nicht der Ort

**Langzeitgrenze einer Punktquelle** (alle 24 inneren Zustände inkohärent): Die
Richtungsverteilung ist die der Gruppengeschwindigkeiten aller Bänder über die
Brillouin-Zone (Hellmann–Feynman, exakt; 120 000 zufällige k-Punkte).

| μ | alle 24 Moden: C₄ / C₆ / C₈ | nur Dirac-Quartett |
|---|---|---|
| 0 | 8.6·10⁻⁴ / 1.4·10⁻⁴ / 8.1·10⁻⁴ | 5.0·10⁻⁴ / 1.1·10⁻⁴ / 3.5·10⁻⁴ |
| 0.03 | 9.6·10⁻⁴ / 1.3·10⁻⁴ / 8.4·10⁻⁴ | 3.9·10⁻⁴ / 1.1·10⁻⁴ / 3.4·10⁻⁴ |

Eine Punktquelle enthält alle k bis zum Zonenrand. Ihre Richtungsverteilung behält eine
feste kubische Form, die nicht mit der Zeit verschwindet.

**Glatte Quelle der Breite σ** (|ψ(k)|² ∝ exp(−σ²k²); deterministische Quadratur aus
3000 Fibonacci-Richtungen × 16 Radien, Boden 2·10⁻¹⁰):

| σ | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Dirac-Quartett C₄ | 1.5·10⁻⁴ | 1.06·10⁻⁴ | 7.5·10⁻⁶ | 4.7·10⁻⁷ | 3.0·10⁻⁸ | 2.1·10⁻⁹ |
| Dirac-Quartett C₆ | 3.5·10⁻⁴ | 7.8·10⁻⁷ | 6.8·10⁻⁹ | Boden | Boden | Boden |
| alle 24 Moden C₄ | 3.2·10⁻⁵ | 1.05·10⁻³ | 1.96·10⁻³ | 2.23·10⁻³ | 2.30·10⁻³ | 2.32·10⁻³ |

(μ = 0.03; μ = 0 gleich bis auf wenige Prozent.)

- **Das Teilchen (Dirac-Quartett) wird beliebig isotrop:** C₄ ∝ σ⁻⁴, also Faktor 16
  pro Verdopplung. Die Anisotropie-Amplitude fällt wie (a/σ)². Der masselose Kegel hat
  bei k = 0.05 / 0.1 / 0.2 / 0.4 eine Richtungsabhängigkeit der Geschwindigkeit von
  1.1·10⁻⁴ / 3.6·10⁻⁴ / 8.7·10⁻⁴ / 1.5·10⁻³.
- **Die übrigen 20 Moden bleiben verzerrt:** C₄ → 2.3·10⁻³ auch bei langen Wellen. Sie
  sitzen in entarteten Multipletts, deren Aufspaltung bei kleinem k kubisch verzerrt
  ist (wie das „warping“ der Lochbänder im Halbleiter). Das Teilchen des Modells ist
  aber das Quartett.

**Frontgeschwindigkeit** (max_k v·n̂ über alle Bänder): 0.94 / 0.87 / 0.94 entlang
[100] / [110] / [111], für das Quartett 0.94 / 0.86 / 0.84. Zum Vergleich: c′ = 0.577
(Dirac-Kegel), c = 1.732 (Gitter-Lichtkegel = Schale n).

## C. 3D-Simulation einer Punktquelle

Echte Zeitentwicklung bis n = 64 Schritte:
- Box 131³, complex64, 0.43 GB pro Zustand, 108 s.
- Gerechnet wird eine Startrichtung mit beiden Spins. Die Symmetrisierung über die 24
  Würfeldrehungen ergibt dann die inkohärente Summe über alle 24 inneren Zustände.
- Die Norm bleibt 1.000000.

| n | 8 | 16 | 24 | 32 | 48 | 64 | Langzeitgrenze |
|---|---|---|---|---|---|---|---|
| C₄ | 3.9·10⁻⁴ | 6.0·10⁻⁴ | 7.1·10⁻⁴ | 7.5·10⁻⁴ | 8.7·10⁻⁴ | 8.9·10⁻⁴ | 9.6·10⁻⁴ |
| C₈ | 1.5·10⁻³ | 2.0·10⁻⁴ | 1.6·10⁻⁴ | 3.9·10⁻⁴ | 5.3·10⁻⁴ | 6.4·10⁻⁴ | 8.4·10⁻⁴ |
| Gewicht auf der Schale (Graphabstand ≥ 0.8n) | 0.021 | 0.004 | 0.001 | < 10⁻³ | < 10⁻³ | < 10⁻³ | 0 |

- Die Richtungsverteilung läuft gegen die k-Raum-Grenze. C₆ wechselt bei n ≈ 16 das
  Vorzeichen seiner Amplitude und ist deshalb dort fast null.
- **Auf der geometrischen Schale liegt praktisch nichts.** Die Schale n ist der strenge
  Gitter-Lichtkegel r = c·t. Die Wahrscheinlichkeit endet aber bei der maximalen
  Gruppengeschwindigkeit. Simuliert (Teil D, 99 % in einem 12°-Kegel) liegt die Front bei
  r/t = 0.89 / 0.86 / 0.92 entlang [100] / [110] / [111], also bei etwa 0.5 c. Das
  stimmt mit dem k-Raum-Maximum überein. Das Maximum der Dichte liegt nahe c′.

## Antwort

1. **Die Schalen werden nie isotrop:** C₄ = 3.22·10⁻³ ab n ≈ 8, der
   Kuboktaeder-Grenzwert. Als „Bloch-Kugel“ mit wachsender Auflösung taugt die Schale
   nicht. Jede O_h-Schale ist nur ein 3-Design.
2. **Isotropie entsteht dynamisch, nicht geometrisch.** Das Teilchen (Dirac-Quartett)
   bewegt sich umso isotroper, je glatter es ist: C₄ ∝ (a/σ)⁴. Wie bei jedem guten
   Gitter-Dirac-Operator ist die Rotationssymmetrie eine Langwellen-Eigenschaft, keine
   Eigenschaft einer Schale.
3. **Die Schale ist nicht die Front.** Der Walker bleibt weit innerhalb des
   geometrischen Lichtkegels (Front bei 0.5 c, Dichtemaximum bei c′ = c/3). Die
   Front selbst ist um etwa 7 % anisotrop ([110] am langsamsten).
4. Für eine 3D-Gravitation im Modell heißt das: Felder und Teilchen müssen glatt auf
   der Gitterskala sein. Dann sind die Abweichungen von der Isotropie von der Ordnung
   (a/λ)².

**Was das nicht zeigt:**
- Geprüft ist das freie 2O\*-Modell. Mit Bleibe-Gatter und λ(N) (H2) ändert sich
  die Dispersion bei großem k, die Langwellen-Isotropie des Quartetts sollte bleiben.
  Nicht geprüft.
- Die 20 Nicht-Quartett-Moden sind im Modell nicht das Teilchen. Ob sie physikalisch
  stören (etwa bei harter Streuung), ist offen.

## Offene Punkte

*Weiter mit Gravitation in 3D: [`RESULTS_Gravity3D_de.md`](RESULTS_Gravity3D_de.md).*

- 3D-Feldgleichungen für Φ und b mit 1/r (Speicher: ein Feld ist billig, das
  Testteilchen braucht den k-Patch-Ansatz aus den S4-Runden).
- Lense–Thirring in 3D: rotierende Quelle, Präzession eines Spins.
- Ob eine andere Schalen-Gewichtung (etwa nach Knotendichte normiert) für die
  ursprüngliche Bloch-Kugel-Idee eine bessere Kugel ergibt. Bei n = 2 sind C₆ und C₈
  schon klein.
