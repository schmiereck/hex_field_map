# Experiment 1a: Gravitation als Rechenzeit, Lapse auf der inneren Uhr

*Paper 3, erster Schritt. Code: `src/quantum_gravity_lapse.py` und `src/quantum_gravity_lapse_figs.py`.
Daten: `data/gravity_lapse.npz`. Abbildung: `figures/gravity_lapse.png`.
Einordnung: [`ROADMAP_Paper3_de.md`](ROADMAP_Paper3_de.md).*

## Idee

Die Ausgangsidee: In der Nähe einer großen Masse rechnen die Knoten langsamer.
Die einfachste Umsetzung ist streng lokal und exakt unitär. Sie verlangsamt die
**innere Uhr** jedes Knotens, also die Abbiegerate im Turning-Phase-Modell:

```
eps(r) = eps0 · N(r),        C(r) = expm(i · eps(r) · G_alpha)
```

Der Shift, also der Lichtkegel-Sprung, bleibt unverändert. Die Masse ist in diesem
Modell die Abbiegerate (diskrete Zitterbewegung), daher bedeutet das:
„nahe der Masse wird pro Takt seltener abgebogen“. N(r) heißt Lapse, in Anlehnung
an die ART, wo N = √g₀₀ die lokale Uhrrate ist.

Gemessen wird in einem linearen Lapse `N = 1 + g·x` (homogenes Feld, g = 5·10⁻⁴)
mit bandprojizierten Wellenpaketen (oberes Band, σ = 12). Die Gitterrechnung wird
mit einem halbklassischen Ensemble verglichen, das das **exakte** Gitterband
verwendet, ohne Parabelnäherung. Dabei gilt `dk/dt = −g·eps0·∂E/∂eps`, und
`∂E/∂eps = −⟨u|G|u⟩/dt` ist exakt (Hellmann–Feynman).

## Ergebnisse

### 1. Freier Fall ist universell, aber doppelt so stark wie in der ART

Bei k = 0 ist die Ruheenergie exakt linear in eps (E = 4·eps bei α = 0). Die Kraft
ist daher `F = −m₀·∇N`, und die Trägheit ist die Bandkrümmung 1/m*. Daraus folgt
`a = −(m₀/m*)·∇N =: −R·c²·∇N`. Im Newtonschen Grenzfall der ART gilt R = 1.

| eps0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 |
|------|------|------|-----|-----|-----|-----|-----|
| R (α = 0)   | 1.99968 | 1.99933 | 1.99808 | 1.99308 | 1.98473 | 1.95792 | 1.91742 |
| R (α = 1/4) | 2.54829 | 2.54796 | 2.54680 | 2.54213 | 2.53433 | 2.50923 | 2.47119 |

- **Äquivalenzprinzip:** R hängt nicht von der Masse ab. Für eps0 → 0 geht R
  exakt gegen 2 (α = 0). Die Abweichung ist O(eps²): 0.1 % bei eps0 = 0.1 und
  4 % bei eps0 = 0.7. Ein Gitter-Eötvös-Effekt existiert also, verschwindet aber
  im Kontinuumslimes.
- **Stärke:** R = 2 und nicht 1. Das liegt nicht am Lapse, sondern an der
  Dispersion. Das massive Band des Turning-Modells hat `m* = m₀/(2c²)`, verhält
  sich bei kleinem k also nicht wie eine Dirac-Hyperbel mit c = √3. Bei großem k
  läuft es trotzdem gegen c (v_g = 1.72 bei k = 3). Bei α = 1/4 ist R = 2.55, die
  Fallstärke hängt also vom internen Parameter α ab. Das ist ein zweiter,
  α-abhängiger Bruch des starken Äquivalenzprinzips.
- **Hinweis:** Die Ruheenergie m₀ ist nicht dieselbe Größe wie die träge Masse
  m*·c². Für eine echte Gravitationskopplung braucht man ein Band mit
  m* = m₀/c², sonst stimmt die Kalibrierung zwischen Rotverschiebung (Uhrrate)
  und Fallbeschleunigung nicht. Das ist ein konkreter Prüfstein für jedes
  Gittermodell.

### 2. Gitter und Halbklassik stimmen auf 10⁻³ überein

| eps0 | a (Gitter) | a (Halbklassik, exaktes Band) | a (parabolisch, R·c²·g) | a (ART, c²·g) | Norm am Ende |
|------|-----------|-----------------------------|------------------------|--------------|------------|
| 0.2 | −2.4801e-3 | −2.4785e-3 | −2.9896e-3 | −1.5000e-3 | 0.998753 |
| 0.3 | −2.6605e-3 | −2.6611e-3 | −2.9771e-3 | −1.5000e-3 | 0.999992 |
| 0.5 | −2.7491e-3 | −2.7498e-3 | −2.9369e-3 | −1.5000e-3 | 1.000000 |
| 0.7 | −2.7311e-3 | −2.7318e-3 | −2.8761e-3 | −1.5000e-3 | 1.000000 |

Das Paket fällt im Mittel etwas schwächer als R·c²·g, weil seine k-Breite
(1/σ) das nichtparabolische Band abtastet. Bei leichten Teilchen (eps0 = 0.2) ist
das stärker. Der Effekt hängt vom Zustand ab, nicht von der Masse als solcher, und
die Halbklassik mit exaktem Band erfasst ihn vollständig. Der Normverlust bei
eps0 = 0.2 kommt vom Gitterrand und nicht von der Dynamik, denn der Lapse-Schritt
ist exakt unitär.

### 3. Bewegte Teilchen entkoppeln: Nordström statt Einstein

Der Uhren-Lapse koppelt an `κ(k) = eps·∂E/∂eps / E`, also an den Anteil der
Energie, der „Abbiege-Energie“ ist. Für Dirac wäre das m²/E² = 1/γ². Ein
**voller** Lapse, bei dem die gesamte Rechnung verlangsamt wird, koppelt an E,
also κ = 1.

| k | 0 | 0.3 | 0.6 | 0.9 | 1.5 | 3.0 |
|---|---|-----|-----|-----|-----|-----|
| κ (eps0 = 0.1) | 1 | 0.405 | 0.188 | 0.103 | 0.042 | 0.0099 |
| (m₀/E)² (eps0 = 0.1) | 1 | 0.314 | 0.118 | 0.059 | 0.023 | 0.0059 |
| κ (eps0 = 0.3) | 1 | 0.779 | 0.550 | 0.402 | 0.231 | 0.075 |

κ fällt wie 1/γ², mit einem Vorfaktor von etwa 1.5 bis 1.7. Schnelle Teilchen
spüren den Uhren-Lapse kaum noch. Die direkte Messung der Querbeschleunigung
(Feld entlang x, Bewegung entlang y, eps0 = 0.3) bestätigt das:

| k | a_x Gitter | a_x Halbklassik | a_x voller Lapse (Schätzung) | Gitter / voll |
|---|-----------|----------------|-----------------------------|--------------|
| 0.0 | −2.688e-3 | −2.689e-3 | −2.977e-3 | 0.90 (Paket vs. parabolisch) |
| 0.4 | −1.464e-3 | −1.464e-3 | −2.099e-3 | 0.70 |
| 0.8 | −8.66e-4  | −8.66e-4  | −1.657e-3 | 0.52 |
| 1.2 | −6.59e-4  | −6.59e-4  | −1.480e-3 | 0.45 |

(„Voller Lapse“ ist hier die halbklassische Schätzung `−g·E·v/k`, nicht
simuliert. Bei k = 0 ist es der parabolische Wert und daher nicht direkt mit dem
Paket vergleichbar.)

**Interpretation:** Eine Verlangsamung nur der inneren Uhr ist eine
**skalare Gravitation vom Nordström-Typ** (1913). Ruhende Massen fallen universell,
aber Licht, also der Grenzfall κ → 0, wird **nicht** abgelenkt. Genau diese
Theorie wurde 1919 durch die Lichtablenkung widerlegt.

## Folgerungen für Paper 3

1. „Rechenzeit“ muss **auch die Ausbreitung** verlangsamen, nicht nur das Abbiegen.
   Das ist Experiment 1b mit vollem Lapse: lokal reduzierte Lichtgeschwindigkeit
   c(r) = N(r)·c. Erwartet wird Einsteins Wert von 1911, also die halbe
   ART-Ablenkung.
2. Für die volle ART-Ablenkung muss zusätzlich die räumliche Metrik mitgehen
   (Experiment 1c).
3. Unabhängig davon braucht das Modell ein massives Band mit m* = m₀/c² (R = 1).
   Das Turning-Modell bei α = 0 hat R = 2. Zu prüfen ist, ob das hexagonale Modell
   aus Paper 1 oder der FCC-Walker R = 1 liefern.

## Offene Punkte

- Ist R = 2 exakt analytisch herleitbar? Der Grenzwert ist numerisch auf 3·10⁻⁴
  gesichert. Vermutlich folgt er aus der Struktur von G_α und dem
  Verschiebungsoperator bei k → 0.
- Der Vorfaktor von κ gegenüber 1/γ² (etwa 1.5 bis 1.7) ist noch nicht verstanden.
- Masselose Moden (α = 1/2): Der Kegel hat eine Steigung unabhängig von eps, aber
  einen eps-abhängigen Energie-Offset. Der Uhren-Lapse lenkt sie daher abhängig von
  der Frequenz ab, eine Art „Regenbogen-Gravitation“. Das ist noch nicht gemessen.
