# H2: Eine Budget-Regel pro Knoten ergibt Lapse und räumliche Metrik (γ_PPN = 1)

*Paper 3. Code: `src/phase_bit_s4_open8.py` (Teile `A`, `B`) und
`src/phase_bit_s4_open8_figs.py`. Daten: `data/phase_bit_s4_open8.npz`.
Abbildung: `figures/phase_bit_s4_open8.png`.
Vorgänger: [`RESULTS_H1_Antimatter_ProperTime_de.md`](RESULTS_H1_Antimatter_ProperTime_de.md).*

Modell wie in Runde 4–6: **2O\*** (ε = 0.3) mit Vakuum-Gegenterm. Bandebene mit
Blochmatrizen bis 96×96, x-Kette mit 300 Lagen. Speicher: wenige 100 MB, frei waren
3.9 GB.

## Frage

In Exp. 1c (Runde 4) wurde die räumliche Metrik von Hand eingesetzt: Puffer-Lapse N/A
und Masse μ·A mit A = 1/N. Damit kam die volle ART-Lichtablenkung heraus (γ = 1).
Hypothese H2: Es gibt **eine** lokale Regel pro Knoten, die Lapse N und räumliche
Metrik A zugleich erzeugt, und sie ergibt γ_PPN = 1.

Hinweis aus der Eigenzeit-Runde: Die Rechenlast teilt sich in E = m·dτ/dt + p·v. Die
Uhr-Rechnung sollte nur N sehen, die Bewegungs-Rechnung N und A.

## 1. No-Go: Warten allein ergibt nie eine räumliche Metrik

Ein Warte-Puffer (Runde 4) ist ein Register, in dem der Zustand nichts tut (Phase −1).
Er skaliert die Quasienergie um N = (1 + cos a)/2. Naheliegend wäre, dem Hüpfen und der
Uhr (Münze) **getrennte** Warte-Puffer zu geben. Das hilft nicht:

| Schema | N₁, N₂ | Lücke / m | v / c′ | Vorhersage |
|---|---|---|---|---|
| ein Puffer | 0.75 | 0.7500 | 0.7500 | 0.75 |
| getrennte Puffer für Schritt und Münze | 0.75, 0.75 | 0.6000 | 0.6000 | 1/(1/N₁ + 1/N₂ − 1) = 0.600 |
| | 0.50, 0.60 | 0.3750 | 0.3750 | 0.375 |
| | 0.40, 0.40 | 0.2500 | 0.2500 | 0.250 |

Beide Puffer greifen auf dieselbe aktive Amplitude zu. Die Wartezeiten addieren sich,
und nach Hellmann–Feynman sehen Schritt und Münze denselben Lapse, nämlich den aktiven
Anteil des gemeinsamen Vakuum-Fixvektors (die allgemeine Regel aus Runde 5). **Jede**
Kombination von Warte-Puffern ist eine reine Zeitumskalierung: Hüpfen und Uhr werden
gleich verlangsamt, die Lichtgeschwindigkeit in Koordinaten ist c′N wie die Uhr, also
γ = 0 (Einstein 1911, halbe Ablenkung).

## 2. Die Regel: jeder Prozess braucht jeden beteiligten Knoten frei

Die Uhr (Münze, Massendrehung) betrifft einen Knoten. Ein Sprung betrifft zwei,
Abfahrt und Ankunft. Die Regel lautet: Ein Knoten ist mit dem Gewicht N frei. Jeder
Prozess braucht alle seine Knoten frei. Dafür reichen zwei lokale Gatter:

```
Warten   W(a) = R(a)[ V  ⊕ (−1) ]       Knoten belegt: nichts wird gerechnet
Bleiben  H(b) = R(b)[ S C ⊕ (−C) ]      Sprung abgelehnt: der Zustand bleibt, die Uhr läuft weiter
H2-Takt  U = W(a) [ H(b) (S C ⊕ −C) ⊕ (−1) ],    N_w = N_h = N(x)
```

Der Unterschied liegt im Register: Im Warte-Register passiert nichts. Im
Bleibe-Register wird die Münze angewandt, aber nicht verschoben. Das Vorzeichen −1
(Pufferphase π) hält den Dirac-Punkt bei E = 0 fest (gemessen ≤ 7·10⁻¹⁶).

**Bandebene** (Lücke bei k = 0 mit μ = 0.03, masseloser Kegel bei |k| = 0.02):

| Schema | N_w | N_h | Lücke / m | v / c′ |
|---|---|---|---|---|
| nur Bleiben | 1 | 0.75 | **1.0000** | 0.7484 |
| H2 | 0.75 | 0.75 | **0.7500** | 0.5613 (N_w N_h = 0.5625) |
| H2 | 0.50 | 0.60 | 0.5000 | 0.2990 (0.300) |
| H2 | 0.40 | 0.40 | 0.4000 | 0.1592 (0.160) |

Das Bleibe-Gatter verlangsamt nur das Hüpfen, die Uhr bleibt unberührt. Mit N_w = N_h = N
gilt also

```
H = N² K + N M        (Exp. 1c mit A = 1/N)
```

Die effektive räumliche Metrik A = (Lücke/m)/(v/c′) (Steigung auf k → 0 extrapoliert):

| N | 0.95 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| A | 1.0526 | 1.1111 | 1.2500 | 1.4285 | 1.6666 | 2.0000 | 2.5000 | 3.3333 |
| γ = −ln A / ln N | 0.9997 | 0.9998 | 0.9999 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**γ_PPN = 1, nicht nur im schwachen Feld:** A = 1/N gilt bis N = 0.3, also
g₀₀·g_rr = −1. Allgemein ergäbe die Regel γ = (n_Sprung − n_Uhr)/n_Uhr, wenn ein Prozess
n Knoten braucht. Für Uhr = 1 und Sprung = 2 ist das genau 1. In 3D braucht jeder
Sprung in jeder Richtung zwei Knoten, die Metrik ist also isotrop.

**Bild auf der Bloch-Kugel:** Jedes Gatter ist eine Drehung R(a) = cos a + i sin a σ_x
eines Register-Qubits (aktiv/wartend bzw. springend/bleibend). R(a)·diag(1, −1) =
cos a σ_z + sin a σ_y. Der Vakuum-Fixvektor ist der Bloch-Vektor mit Polarwinkel a, und
der Lapse ist sein aktiver Anteil N = cos²(a/2) = (1 + z)/2. Ein belasteter Knoten
kippt diese beiden Bloch-Vektoren von der Polkappe weg.

## 3. Preis: ein Untergrund −α k² der Ordnung Δt

Das Bleibe-Gatter hat einen Effekt zweiter Ordnung. Das Verhältnis aus Bleibe- und
Sprung-Amplitude im Fixvektor hängt von der Schrittphase der Richtung ab, und die Münze
mischt die Richtungen. Das gibt eine zusätzliche Krümmung des Bandes (Teil A3, gegen
Dirac mit Geschwindigkeit N_h c′):

| N_h | 0.99 | 0.9 | 0.75 | 0.5 | 0.3 |
|---|---|---|---|---|---|
| α_extra, μ = 0 | 0.0024 | 0.0221 | 0.0460 | 0.0614 | 0.0516 |
| α_extra / (N_h(1 − N_h)) | 0.245 | 0.245 | 0.245 | 0.246 | 0.246 |
| dasselbe, μ = 0.03 | 0.256 | 0.255 | 0.254 | 0.248 | 0.236 |
| dasselbe, μ = 0.15 | 0.339 | 0.335 | 0.324 | 0.292 | 0.228 |

- α_extra ≈ 0.245·N_h(1 − N_h) ist fast massenunabhängig und auch für Licht vorhanden.
- Der Term ist ein Gitterartefakt der Ordnung Δt. Relativ zur kinetischen Energie
  p²/2m ist er ∝ m·Δt, relativ zu c′k ∝ k·Δt. Er verschwindet also im Kontinuumslimes.
- Er ist erster Ordnung im Feld (1 − N) und bricht in belasteten Gebieten die lokale
  Lorentz-Invarianz bei endlichem k. Bei μ = 0.15 (μ/ε = 0.5, weit außerhalb des
  Dirac-Bereichs) dominiert er die Dynamik schwerer Pakete.

## 4. x-Kette: Ablenkung im Gradienten von N

Aufbau wie Exp. 1b/1c: N(x) = N₀(1 + g x) mit N₀ = 0.75, g = 2·10⁻³, Pakete quer
laufend (Bloch-k_y), Breite σ = 16, 160 Takte. Verglichen wird H2 mit Warten allein
(1b) bei gleichem N(x). Nach der ART gilt für die Ablenkung
ẍ = −(N/A²)·∂N·(1 + γ v²). Die Kennzahl ist (a_H2/N₀²)/a_1b. Die ART-Vorhersage ist
halbklassisch mit derselben Impulsverteilung gerechnet, weil die breiten Pakete
schon in „Ruhe“ eine Impulsbreite der Größenordnung m/c′ haben.

| Paket | Simulation | Halbklassik (Gitterband) | ART γ = 1, ideal | ART γ = 1 + Untergrund α | γ angepasst (mit α) |
|---|---|---|---|---|---|
| μ = 0.03, Ruhe | 1.078 | 1.085 | 1.126 | 1.084 | — (unempfindlich) |
| μ = 0.03, k_y = 0.1 | 1.282 | 1.350 | 1.404 | 1.349 | 0.82 |
| μ = 0.03, k_y = 0.2 | 1.583 | 1.613 | 1.693 | 1.611 | 0.96 |
| μ = 0.15, Ruhe | 0.717 | 0.718 | 1.007 | 0.718 | — |
| μ = 0.15, k_y = 0.3 | 0.823 | 0.824 | 1.163 | 0.813 | 1.09 |
| Licht, k_y = 0.1 | **1.986** | 1.958 | 2.000 | 1.958 | **1.03** |
| Licht, k_y = 0.2 | **1.922** | 1.928 | 2.000 | 1.930 | **0.99** |
| Licht, k_y = 0.4 | **1.860** | 1.866 | 2.000 | 1.866 | **0.99** |

- **Licht wird doppelt so stark abgelenkt wie mit Warten allein**, also mit der vollen
  ART-Ablenkung. Ohne Untergrund-Korrektur ist γ = 0.99 / 0.92 / 0.86 bei
  k = 0.1 / 0.2 / 0.4, mit Korrektur 1.03 / 0.99 / 0.99.
- **„ART mit γ = 1 plus gemessener Untergrund α“ trifft die Gitter-Halbklassik in allen
  acht Fällen auf ≤ 0.011.** Die Abweichungen von der idealen ART sind also vollständig
  das Artefakt aus Abschnitt 3, nicht eine andere Metrik.
- Simulation und Halbklassik stimmen auf 0.1–2 % überein, außer bei μ = 0.03, k_y = 0.1
  (5 %). Dort ist auch das angepasste γ = 0.82 am schlechtesten, weil das Paket
  mit v ≈ 0.56 wenig auf γ reagiert. Die absoluten Beschleunigungen liegen in beiden
  Schemata gleich um 0.85–0.99 unter der Halbklassik (endliche Paketdauer), das kürzt
  sich im Verhältnis heraus.
- Rotverschiebung und Eigenzeit bleiben richtig: Die Ruhe-Lücke skaliert mit N_w = N,
  die Uhr eines ruhenden Teilchens geht also mit dτ/dt = N wie in der H1-Runde.

## Antwort

1. **No-Go:** Jede Kombination von Warte-Puffern (auch getrennte für Sprung und Münze)
   skaliert Hüpfen und Uhr gleich, ist also reiner Lapse mit γ = 0.
2. **H2 bestätigt:** Die Regel „jeder Prozess braucht jeden beteiligten Knoten frei“
   (Uhr: ein Knoten, Sprung: zwei) ergibt mit einem Warte- und einem Bleibe-Gatter
   H = N²K + N M, also A = 1/N und γ_PPN = 1. Auf Bandebene gilt das auf 3·10⁻⁴ bis
   N = 0.3, auf der Kette wird Licht doppelt abgelenkt (γ = 0.99–1.03 nach Abzug des
   Artefakts).
3. **Preis:** Das Bleibe-Gatter erzeugt einen Untergrund −0.245·N(1 − N)·k² (Ordnung Δt).
   Er verschwindet im Kontinuumslimes, ist aber im Dirac-Bereich bei μ = 0.03 mit
   3–5 % sichtbar und dominiert bei μ = 0.15.

**Was das nicht zeigt:** γ = 1 folgt hier aus der Kinematik der Regel, nicht aus einer
Feldgleichung. Die Quelle von N bleibt s = 2e − e_M (Runde 6). Der zweite PPN-Parameter
β (nichtlineare Überlagerung) und die Rückwirkung von A auf die Quelle sind nicht
geprüft. Dass Abfahrt und Ankunft gleich gewichtet werden (N_w = N_h), ist Teil der
Hypothese.

## Offene Punkte

- **Untergrund kompensieren:** Wie beim Hintergrund des Dirac-Quartetts (λ*·W₉₀,
  Runde 2) könnte eine N-abhängige Nachstimmung von λ den Term −α k² aufheben. Oder ein
  Gatter, dessen Bleibe-Anteil nicht von der Schrittphase abhängt.
- **β_PPN:** nichtlineare Überlagerung zweier Quellen; prüfen, ob N(x) mit der
  Budget-Regel exponentiell (N = e^Φ) oder linear überlagert.
- **Selbstkonsistent:** Teil B mit Φ aus der Rechenlast-Quelle (Runde 6) statt einem
  vorgegebenen Gradienten, und Shapiro-Verzögerung eines Lichtpulses durch ein
  belastetes Gebiet.
- **Experiment 3** der Roadmap (Isotropie der Kuboktaeder-Schalen, die ursprüngliche
  Bloch-Kugel-Frage) ist weiterhin offen.
