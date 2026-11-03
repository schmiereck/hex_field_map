# Gravitation in 3D: 1/r-Feld und Lichtablenkung an einer Punktmasse

*Paper 3. Code: `src/phase_bit_s4_open12.py` (Teile `A`, `B`) und
`src/phase_bit_s4_open12_figs.py`. Daten: `data/phase_bit_s4_open12.npz`.
Abbildung: `figures/phase_bit_s4_open12.png`.
Vorgänger: [`RESULTS_Exp3_Isotropy_de.md`](RESULTS_Exp3_Isotropy_de.md),
[`RESULTS_G0i_Causal_de.md`](RESULTS_G0i_Causal_de.md).*

Modell: **2O\*** (ε = 0.3, λ\*, Vakuum-Gegenterm). In B läuft der volle 3D-Walker:
- Warte-Gatter (W) bzw. Warte- plus Bleibe-Gatter (H2) mit kompensierter Kopplung
  λ(N), die Münzen in 48 Klassen nach N.
- Box 112 × 88 × 80 (788 480 Knoten), complex64, 0.15 GB pro Register, 4 Register.
- Etwa 3.5 min pro Lauf mit W, 8 min mit H2. Frei waren 4.5–5.6 GB.

## A. Die Feldgleichung in 3D

**Statisch:** ∇²Φ = κs auf dem Würfelgitter (FFT, periodische Box mit 96³ Knoten),
Quelle gaußförmig mit σ = 2 und Gesamtlast 1.

| r | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| Φ / (−κ/4πr) | 1.000 | 1.005 | 1.015 | 1.022 | 1.048 | 1.090 |

Das ist das 1/r-Potential. Die wachsende Abweichung bei großem r kommt von den
Spiegelquellen der periodischen Box. Der Gradient ∂Φ/∂r liegt bei r = 6.1 / 9.8 auf
2.3 % / 1.6 % bei κ/(4πr²).

**Kausal:** Φ_tt = c_g²(∇²Φ − κs) mit c_g = c′, die Quelle wird bei t = 0
eingeschaltet, Φ = 0 am Anfang (Abb. b).
- Der Feldgradient setzt bei t ≈ r/c_g ein. Als die Front durchläuft, gibt es einen Puls
  (bis zum 3.7-Fachen des statischen Werts). Danach bleibt der statische Wert stehen
  (0.95 / 1.05 / 1.01 bei r = 6 / 10 / 16).
- **Kein Anwachsen:** Das Potential im Zentrum ist bei t = 21.7 und bei t = 43.2
  gleich (−0.03190, Freiraumwert −0.03175). In 1D wuchs das retardierte Feld wie c_g t,
  und Φ musste an jedem Teilchen referenziert werden (Runde 6). In 3D ist das nicht
  nötig.
- Der Puls ist die Front des Sprungs, den eine plötzlich „erschaffene“ Masse im
  Potential erzeugt. Eine echte Quelle entsteht nicht aus dem Nichts, deshalb gehört
  der Puls zum Testaufbau, nicht zur Physik.

## B. Lichtablenkung an einer Punktmasse

Aufbau:
- Statisches Feld U = GM/√(r² + r_c²) mit GM = 0.25, r_c = 2, und N = exp(−U)
  (Produktregel, β = 1). N_min = 0.88.
- Ein masseloses Paket (oberes Weyl-Dublett, k₀ = 0.6 entlang x, Breite σ = 5)
  fliegt an der Masse vorbei, mit Stoßparameter b = 7.35 bzw. 12.25. Es läuft von
  x = −18 bis +14, Masse bei x = 0.
- Gemessen wird Δ⟨k_y⟩, gemittelt über beide Kramers-Partner. Beide geben auf 10⁻⁵
  dasselbe; in 3D gab es hier keinen spinabhängigen Anteil.

Vorhersagen:
- **Lokale Kraft:** dk_y/dt = n_s N^{n_s} c′⟨|k|⟩ ∂U/∂y, gemittelt über die simulierte
  Dichte zu jeder Zeit. Dabei ist n_s = 1 für W (Lichtgeschwindigkeit c′N) und
  n_s = 2 für H2 (c′N²).
- **Dünner gerader Strahl:** n_s k₀ ∫∂U/∂y dx über die durchlaufene Strecke. Für den
  ganzen Weg ergibt das die bekannten Werte 2GM/b (W, Einstein 1911) und 4GM/b (H2,
  ART).

| b | Schema | Δk_y Simulation | lokale Kraft (Verhältnis) | gerader Strahl (Verhältnis) | Winkel Δk_y/k₀ |
|---|---|---|---|---|---|
| 7.35 | W | −0.02174 | −0.02136 (1.018) | −0.03418 (0.64) | 0.036 rad |
| 7.35 | H2 | −0.04503 | −0.04384 (1.027) | −0.06790 (0.66) | 0.075 rad |
| 12.25 | W | −0.01907 | −0.01907 (1.000) | −0.01876 (1.017) | 0.032 rad |
| 12.25 | H2 | −0.03925 | −0.03877 (1.012) | −0.03716 (1.056) | 0.065 rad |

**Verhältnis H2 / W = 2.071 (b = 7.35) und 2.058 (b = 12.25).**

- **Die räumliche Metrik der H2-Regel verdoppelt die Lichtablenkung auch in 3D.** Das
  ist die volle ART-Ablenkung 4GM/b gegen die halbe von Einstein 1911.
- **Lokale Kraft:** Sie trifft die Simulation auf 0–2.7 %. Mit ⟨k_x⟩ statt ⟨|k|⟩ wären
  es 5.5 % weniger gewesen; die Impulsbreite 1/σ = 0.2 neben k₀ = 0.6 macht ⟨|k|⟩
  größer.
- **Gerader Strahl:** Bei b = 7.35 ist das Paket so breit wie der Abstand zur Masse,
  ein dünner Strahl ist dann die falsche Näherung (0.64). Bei b = 12.25 passt er auf
  2–6 %.
- **Warum 2.06 und nicht 2.00:** Beide Läufe dauern gleich lang, aber Licht ist in H2
  im Feld langsamer (c′N² statt c′N). Es verbringt mehr Zeit nahe der Masse und kommt
  etwas weniger weit (x = +13.1 statt +13.9). Pro durchlaufener Strecke sagt der
  gerade Strahl 1.99 / 1.98 voraus. Die lokale Kraft gibt den Zeiteffekt wieder.

## Sieht ein Beobachter im Gitter Kugeln?

Die Frage war: Muss die Verteilung für uns isotrop aussehen? Ein Beobachter, der im
Gitter lebt, „sieht“ doch von innen Flächen, die von jedem Punkt aus gleich weit
entfernt sind.

**Ja, in einem genauen Sinn.** Der Beobachter hat keinen Zugang zu „Knoten zählen“.
Er misst Abstand mit seinen eigenen Signalen, etwa mit der Radarzeit von Licht oder
mit Maßstäben aus Materie. Seine Kugel ist die Menge der Punkte gleicher Licht-Laufzeit.
Die ist für ihn per Definition eine Kugel, gleich wie sie für uns von außen auf dem
Gitter aussieht.

Daraus folgen drei Punkte:

1. **Die Kuboktaeder-Schalen sind für ihn unsichtbar.** Kein Signal folgt dem
   Graphabstand, auf der Schale liegt praktisch keine Wahrscheinlichkeit
   (Experiment 3). Dass die Schalen nie rund werden, kann er nicht feststellen.
2. **Was er prüfen kann, ist Übereinstimmung.** Eine Anisotropie wird erst messbar,
   wenn zwei verschiedene Sonden verschiedene Kugeln liefern: Licht hoher gegen
   niedriger Energie, Licht gegen massive Teilchen, oder ein gedrehter Maßstab gegen
   eine Lichtlaufzeit (Michelson–Morley). Isotropie heißt im Modell also: Alle
   Signale niedriger Energie haben denselben Lichtkegel. Wie es von außen aussieht,
   spielt keine Rolle.
3. **Genau das misst unsere Größe C_l(σ).** Für das Teilchen des Modells hängt die
   Kegelgeschwindigkeit nur wie (a/λ)² von der Richtung ab: 1.1·10⁻⁴ bei k = 0.05,
   3.6·10⁻⁴ bei 0.1, 1.5·10⁻³ bei 0.4. Ein Radar mit langwelligem Licht ergibt also
   Kugeln, die auf diesen Bruchteil mit denen eines anderen langwelligen Signals
   übereinstimmen. Erst mit Signalen nahe der Gitterskala würde der Beobachter eine
   Vorzugsrichtung finden: eine energie- und richtungsabhängige Lichtgeschwindigkeit,
   wie sie in der Quantengravitations-Phänomenologie gesucht wird.

Die 20 übrigen Moden aus Experiment 3 bleiben auch bei langen Wellen kubisch
verzerrt. Wären sie physikalische Teilchen, könnte ein Beobachter die Gitterrichtung
mit ihnen gegen Licht vergleichen. Im Modell ist das Teilchen aber das Quartett. Ob
die anderen Moden bei niedriger Energie angeregt werden können, ist eine offene Frage.

## Antwort

1. **1/r in 3D:** Die lokale Feldgleichung gibt das Newton-Potential. Ihre kausale
   Version erreicht den statischen Wert hinter der Front und wächst nicht weiter. Das
   1D-Artefakt (lineares Anwachsen) ist verschwunden.
2. **Lichtablenkung an einer Punktmasse:** H2 lenkt doppelt so stark ab wie Warten
   allein (2.07 / 2.06 bei gleicher Laufzeit, 1.98–1.99 pro Strecke), also mit der
   vollen ART-Ablenkung. Die lokale Kraftformel trifft die volle 3D-Walker-Simulation
   auf ≤ 2.7 %.
3. **Beobachter im Gitter:** Er sieht Kugeln, weil er sie mit seinem Licht definiert.
   Messbar ist nur, ob verschiedene Signale dieselben Kugeln geben. Für langwellige
   Signale gilt das im Modell auf (a/λ)².

**Was das nicht zeigt:**
- Die Masse ist ein vorgegebenes statisches Feld U = GM/r, nicht die Last eines
  simulierten schweren Teilchens. Das Kopplungsgesetz aus Teil A ist aber dasselbe.
- Gemessen ist die Ablenkung auf einer endlichen Strecke (x von −18 bis +14), nicht der
  volle Winkel von −∞ bis +∞.
- Massive Testteilchen (Newton 1/r² und Perihel) und Shapiro-Verzögerung in 3D sind
  noch nicht gerechnet.

## Offene Punkte

- Selbstkonsistent: Feld aus der Last eines 3D-Teilchens (Feld billig, Teilchen teuer).
- Bahn eines massiven Teilchens um die Masse: 1/r²-Gesetz, mit genug Laufzeit auch die
  Periheldrehung (2 + 2γ − β)/3 = 1. Braucht lange Läufe.
- Lense–Thirring: Shift b aus einer rotierenden Quelle, Präzession eines Spins.
- Ein „Michelson–Morley“-Test im Gitter: Radar mit zwei Lichtenergien in verschiedenen
  Richtungen, als direkte Messung der Isotropie aus Sicht des Beobachters.
