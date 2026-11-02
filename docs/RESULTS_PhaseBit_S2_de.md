# Experiment S2: Dynamik der Spin-Wirbel (Phasen-Bit-Walker, exakt unitär)

*Paper 3, Hauptlinie. Code: `src/phase_bit_vortex_dynamics.py` und `src/phase_bit_vortex_dynamics_figs.py`.
Daten: `data/phase_bit_vortex_dynamics.npz`. Abbildung: `figures/phase_bit_vortex_dynamics.png`.
Vorgänger: [`RESULTS_PhaseBit_S1_de.md`](RESULTS_PhaseBit_S1_de.md).*

## Fragen

1. Bleibt die Halbwindung des Spin-Wirbels in der Dynamik erhalten?
2. Kann ein Spin-Wirbel ruhen, und bewegt er sich mit v < c?
3. Sieht man eine Zitterbewegung im Richtungsraum?

## Modell

Der Walker lebt direkt auf der Doppelüberlagerung. Sein Label ist N ∈ Z₁₂
(Netto-Drehung mod 12), die Richtung d = N mod 6, das Phasen-Bit p = N mod 4. Die
Phasen stecken also im Label und nicht in den Amplituden. Die Dynamik ist exakt
unitär:

```
C  = expm(i·(ε·G0 + μ·G3))
G0 = S + S⁻¹              ±60°-Drehung (ganzzahliger Generator)
G3 = i·(S³ − S⁻³)          180°-Umkehr, die immer als LINKS-Drehung zählt (Phasen-Bit +3)
S : N → N + 1
```

Danach springt jedes Label entlang seiner Richtung d = N mod 6. Gerechnet wird mit
ε = 0.3.

## Ergebnisse

### 1. Die Halbwindung kann nicht zerfallen (Superauswahl)

Die 360°-Drehung R₆ : N → N + 6 vertauscht mit jedem Schritt: Der Sprung hängt nur
von N mod 6 ab, und G0, G3 sind Funktionen von S. Spin-Sektor (R₆ = −1,
Halbquanten-Wirbel) und Boson-Sektor (R₆ = +1) mischen daher **nie**.

Getestet wurde mit einem zufälligen Anfangszustand (Spin-Gewicht 0.4988), einem
**zufälligen ortsabhängigen ε(r)** (±50 %), μ = 0.05 und periodischen Rändern über
60 Schritte. Die größte Drift des Spin-Gewichts beträgt **8.3·10⁻¹⁶**. In allen
Paketläufen bleibt das Spin-Gewicht 1.000000000000.

Die Halbwindung ist also nicht dynamisch stabil, sondern **symmetriegeschützt**.
Keine Störung, die nur auf die Richtung wirkt (Felder, Lapse, Unordnung), kann aus
einem Spin-Wirbel einen gewöhnlichen Wirbel machen.

Gegenprobe: Bei μ = 0 und Spin-Auslesung (Faktor i^p) ist der Z₁₂-Walker identisch
mit dem 6-Richtungs-Turning-Modell bei α = 3/2 ≡ ½. Die größte relative Abweichung
nach 60 Schritten beträgt 7.0·10⁻¹⁵.

### 2. Ohne Umkehr ist der Spin-Wirbel masselos und kann nicht ruhen

Bei μ = 0 ist das obere Kramers-Dublett bei k = 0 entartet (Wirbel j = 5 und j = 7)
und spaltet **linear** auf, mit Kegelsteigung c/2 (Abbildung a). Ein Paket aus dem
reinen Wirbel j = 7 bei k = 0 bleibt deshalb nicht am Ort. Es wird zu einem
**Ring, der mit c/2 wächst** (Abbildung b): Der rms-Radius wächst mit 0.824
(c/2 = 0.866), der Schwerpunkt bleibt bei 10⁻¹⁴.

### 3. Masse durch die chirale Umkehr

Zählt die 180°-Umkehr immer als Linksdrehung (G3), spaltet das Kramers-Paar bei
k = 0 auf. Die Aufspaltung ist **exakt 8μ**:

| μ | 0 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|------|------|------|-----|
| Aufspaltung | 0 | 0.08000 | 0.16000 | 0.40000 | 0.80000 |

Oben liegt der reine Wirbel j = 7, unten j = 5. Es öffnet sich eine Lücke, und der
obere Zweig wird ein massives Band mit Minimum bei k = 0 (Abbildung a).

**Preis:** Die Masse bricht die Kramers-Entartung, denn die Umkehr unterscheidet
links von rechts, das bricht die Zeitumkehr. In 2D ist das die bekannte Situation:
Ein zweikomponentiges massives Dirac-Teilchen in 2+1D bricht T (Paritätsanomalie).
In 3D können Masse und Kramers-Paare nebeneinander bestehen. Das ist ein Grund mehr,
S3 anzugehen.

### 4. Der massive Spin-Wirbel ruht und bewegt sich mit v < c

- **Ruhend** (bandprojiziert, k = 0): Der Schwerpunkt bleibt am Ort, das Paket
  zerfließt nur (rms-Wachstum 0.612 gegenüber 0.824 masselos, Abbildung c, d). Der
  Wirbel-Anteil j = 7 beträgt am Ende 0.908, der Rest kommt von der k-Breite des
  Pakets.
- **Bewegt** (bandprojiziert, k entlang x):

| k | v Schwerpunkt (Gitter) | v_g des Bandes |
|---|-----------------------|----------------|
| 0.1 | 0.389 | 0.443 |
| 0.2 | 0.701 | 0.763 |
| 0.4 | 1.064 | 1.097 |
| 0.6 | 1.237 | 1.255 |

  Alle Geschwindigkeiten liegen unter c = √3. Die Werte des Schwerpunkts liegen
  etwas unter v_g, weil das Paket über das nichtlineare Band mittelt. Für größere k
  wird c/2 überschritten. Die Kegelsteigung c/2 gilt nur in der Nähe des Dubletts,
  das Band ist keine Dirac-Hyperbel mit c/2.

### 5. Zitterbewegung im Richtungsraum

Ein **reiner** j = 7-Wirbel mit Impuls k₀ = 0.3, also nicht bandprojiziert, überlappt
beide Zweige des aufgespaltenen Dubletts (0.770 und 0.189). Der Wirbel pendelt
dann zwischen j = 7 und j = 5 hin und her (Abbildung f), und gleichzeitig zittert
der Schwerpunkt quer zur Bewegungsrichtung:

| Größe | ω (Fit, t ≤ 60) | Dämpfung |
|-------|----------------|----------|
| Anteil Wirbel j = 7 | 0.5981 | 0.0358 |
| Schwerpunkt y | 0.5972 | 0.0372 |
| Bandabstand E₊(k₀) − E₋(k₀) | **0.5949** | — |

Die Frequenz stimmt auf 0.5 % mit dem Bandabstand überein, das ist die
Zitterbewegung. Die Dämpfung ist ebenfalls verstanden: Die beiden Zweige laufen
mit v_g = 0.965 und 0.005 auseinander. Bei σ = 16 trennen sie sich nach
σ/Δv = 16.7 Zeiteinheiten, erwartet ist eine Dämpfung von etwa 1/(1.6·16.7) ≈ 0.037.
Die Amplitude der Querbewegung beträgt 1.4 Längeneinheiten.

**Bild:** Der Spin-Wirbel eines bewegten massiven Teilchens ist kein starres Muster.
Er wechselt mit der Zitterbewegungsfrequenz seinen Drehsinn im Richtungsraum, j = 7
gegen j = 5, während die Halbwindung (Spin-Sektor) exakt erhalten bleibt.

## Antworten

1. **Ja**, die Halbwindung bleibt erhalten, und zwar exakt durch Symmetrie
   (Superauswahl R₆ = −1), nicht nur näherungsweise.
2. **Ohne Masse nein:** Der Phasen-Bit-Wirbel ist masselos (Kramers-Kegel, c/2).
   **Mit chiraler Umkehr ja:** Er ruht und bewegt sich mit v < c. Die Masse kostet
   in 2D die Kramers-Entartung.
3. **Ja:** Die Zitterbewegung ist sichtbar als Pendeln zwischen zwei Wirbelzuständen,
   mit der Frequenz des Bandabstands.

## Offene Punkte

- Lässt sich eine Masse finden, die T erhält? In 2D wäre das nur mit einer
  Verdopplung möglich (zwei Chiralitäten). Zu prüfen ist, ob die gerade/zeitartige
  Bewegung aus Paper 1 (Ruhe-Kanal) das leistet.
- Die niedrigste Masse-Skala ist 8μ, die Kegelsteigung c/2. Warum gerade c/2? Das
  ist auch im Turning-Modell noch nicht hergeleitet.
- **S3:** 3D mit endlichem Spin-Register. Dort können Masse und Kramers-Paare
  nebeneinander bestehen.
