# H2 fortgesetzt: Untergrund kompensiert, und β_PPN aus der Budget-Regel

*Paper 3. Code: `src/phase_bit_s4_open9.py` (Teile `A`, `B`, `C`) und
`src/phase_bit_s4_open9_figs.py`. Daten: `data/phase_bit_s4_open9.npz`.
Abbildung: `figures/phase_bit_s4_open9.png`.
Vorgänger: [`RESULTS_H2_BudgetRule_de.md`](RESULTS_H2_BudgetRule_de.md).*

Modell wie in H2: **2O\*** (ε = 0.3) mit Vakuum-Gegenterm, Warte-Gatter W und
Bleibe-Gatter H mit N_w = N_h = N. Bandebene mit Blochmatrizen, x-Kette mit 300 Lagen
(dünn besetzt, Münze pro Lage). Speicher: wenige 100 MB.

## 1. Kompensation des Untergrunds

In H2 erzeugte das Bleibe-Gatter einen Untergrund −α k² mit α ≈ 0.245·N_h(1 − N_h).
**Ansatz:** Die W₉₀-Kopplung λ (Runde 2) ändert nur den Untergrund. Bei λ = λ* ± 0.3
bleiben der Dirac-Punkt (≤ 2·10⁻¹⁵), die Lücke (1.00000) und die Geschwindigkeit
(bis auf 2·10⁻⁴) gleich. Ein belasteter Knoten stimmt deshalb λ mit N_h nach, sodass
α(μ = 0) wieder den Vakuumwert hat.

| N_h | 0.99 | 0.95 | 0.9 | 0.8 | 0.75 | 0.6 | 0.5 | 0.4 |
|---|---|---|---|---|---|---|---|---|
| λ(N_h) − λ* | +0.037 | +0.161 | +0.286 | +0.488 | +0.576 | +0.816 | +0.967 | +1.120 |
| α(μ = 0.03) ohne → mit | 0.0115 → 0.0091 | 0.0211 → 0.0094 | 0.0320 → 0.0095 | 0.0496 → 0.0091 | 0.0565 → 0.0088 | 0.0691 → 0.0074 | 0.0709 → 0.0063 | 0.0673 → 0.0051 |
| α(μ = 0.15) ohne → mit | 0.051 → 0.049 | 0.064 → 0.051 | 0.078 → 0.051 | 0.100 → 0.048 | 0.109 → 0.046 | 0.122 → 0.038 | 0.121 → 0.031 | 0.112 → 0.025 |

Vakuumwerte: α = 0.0004 (μ = 0), 0.0090 (μ = 0.03), 0.048 (μ = 0.15).

- Im schwachen Feld gilt Δλ ≈ 3.3·(1 − N_h). Für die Kette wird ein Polynom 4. Grades
  verwendet (Rest 3·10⁻³).
- Im Dirac-Bereich (μ = 0.03) liegt der Untergrund für N_h ≥ 0.7 innerhalb von 7 % des
  Vakuumwerts (vorher bis zum 7-Fachen). Bei stärkerem Feld fällt er unter den
  Vakuumwert (0.0051 bei N_h = 0.4).
- **Grenze:** Unter N_h ≈ 0.4 gibt es keine Lösung mehr. Dort müsste λ über −2 hinaus,
  wo sich die beiden Quartette treffen (Runde 2).
- **Einordnung:** Das ist ein lokaler Gegenterm wie λ* selbst: Der Knoten kennt sein N
  und stellt danach eine Kopplung ein. Ein zweites Feld kommt nicht dazu, die Regel
  wird aber um eine Funktion λ(N) länger.

## 2. x-Kette mit Kompensation

Aufbau wie H2, Teil B: N(x) = N₀(1 + g x), N₀ = 0.75, g = 2·10⁻³, σ = 16, 160 Takte. Die
Münze jeder Lage hat jetzt λ(N(x)). Verglichen wird mit Warten allein (1b), die
ART-Vorhersage ist halbklassisch für dieselbe Impulsverteilung gerechnet.

| Paket | ohne Kompensation (H2) | **mit Kompensation** | Halbklassik (Band) | ART γ = 1 | γ angepasst (Sim. / Band) |
|---|---|---|---|---|---|
| μ = 0.03, Ruhe | 1.078 | **1.116** | 1.123 | 1.126 | unempfindlich |
| μ = 0.03, k_y = 0.1 | 1.282 | **1.414** | 1.402 | 1.404 | 1.03 / 0.99 |
| μ = 0.03, k_y = 0.2 | 1.583 | **1.659** | 1.692 | 1.693 | 0.95 / 1.00 |
| μ = 0.15, Ruhe | 0.717 | 0.938 | 0.940 | 1.007 | unempfindlich |
| μ = 0.15, k_y = 0.3 | 0.823 | 1.087 | 1.089 | 1.163 | 0.53 / 0.54 |
| Licht, k_y = 0.1 | 1.986 | **2.005** | 2.001 | 2.000 | **1.005** / 1.001 |
| Licht, k_y = 0.2 | 1.922 | **1.998** | 2.003 | 2.000 | **0.998** / 1.003 |
| Licht, k_y = 0.4 | 1.860 | **2.007** | 2.013 | 2.000 | **1.007** / 1.013 |

- **Licht:** γ = 1 auf 0.7 % bei allen k, die k-Abhängigkeit aus H2 (0.86 bei k = 0.4)
  ist verschwunden.
- **Dirac-Bereich (μ = 0.03):** Das Gitterband liegt auf ≤ 0.3 % bei der ART
  (γ = 0.99 / 1.00). Die Simulation weicht davon um bis zu 2 % ab (Paketdauer).
- **μ = 0.15 (μ/ε = 0.5):** Hier bleibt eine Abweichung. Teil B2 klärt sie: Der Walker
  hat in diesem Bereich einen eigenen, massenabhängigen Untergrund (α = 0.048), und die
  Kompensation bei μ = 0 skaliert ihn nicht mit N_h² wie die Kinetik. „ART + α_c(N)“
  ergibt 0.940 / 1.077 (Gitter 0.940 / 1.089). Würde der eigene Untergrund mit N²
  skalieren, käme 1.007 / 1.163 heraus, also genau die ART. Der Rest ist damit die
  bekannte Nicht-Dirac-Abweichung des Walkers bei großer Masse, keine andere Metrik.

## 3. β_PPN: wie sich zwei Lasten an einem Knoten verbinden

Im isotropen PPN-Eichmaß gilt g₀₀ = −N² = −1 + 2U − 2βU² + …, also
ln N = −U + (β − 1)U². Für zwei Quellen mit U_i = −ln N_i folgt

```
β − 1 = [ln N₁₂ − ln N₁ − ln N₂] / (2 ln N₁ ln N₂)
```

Ist das Feldgesetz für U linear (die Quelle s = 2e − e_M aus Runde 6 geht linear ein),
entscheidet allein das **Kompositionsgesetz** zweier Lasten an einem Knoten über β.
Gemessen ist die Uhrrate N = Lücke/m auf Bandebene:

| N₁, N₂ | ein Gatter, Φ linear überlagert | parallele Warte-Register | verschachtelte Gatter |
|---|---|---|---|
| 0.99, 0.99 | β = 0.495 | 1.495 | **1.0001** |
| 0.98, 0.95 | 0.482 | 1.483 | **1.0001** |
| 0.9, 0.9 | 0.441 | 1.453 | **1.0001** |
| 0.8, 0.9 | 0.401 | 1.430 | **1.0000** |
| 0.7, 0.7 | 0.203 | 1.371 | **1.0000** |
| Gesetz | N = 1 − ΣU_i | 1/N = 1 + Σ(1/N_i − 1) | N = Π N_i |

- **Linear überlagerter Lapse** (so war Φ in Runde 5/6 eingekoppelt, N = N₀(1 + Φ)):
  β = 1/2.
- **Parallele Warte-Register** (jede Last hat ihr eigenes Register am aktiven
  Zustand): Die Wartezeiten addieren sich wie in einer Warteschlange, β = 3/2.
- **Verschachtelte Gatter** (das zweite Gatter wirkt auch auf das Register des ersten,
  der Zustand muss für beide Lasten frei sein): N = N₁N₂, also ln N additiv und **β = 1
  exakt, auch im starken Feld** (N₁₂ = 0.49).

Das Produkt ist dieselbe Logik wie die H2-Regel selbst: Ein Sprung braucht zwei freie
Knoten (N·N), ein Zustand braucht seinen Knoten frei für jede Last (N₁·N₂). Mit beiden
Lasten verschachtelt (Bleibe- und Warte-Gatter) gilt auch die räumliche Metrik weiter:
Uhr 0.85499 gegen N₁N₂ = 0.85500, Hüpfen 0.73104 gegen (N₁N₂)² = 0.73102, also
A = 1/(N₁N₂) auf 10⁻⁴.

**Folgerung für die Kopplung:** Das Feldgesetz muss linear in ln N sein:
N = N₀·exp(Φ) statt N₀(1 + Φ). In erster Ordnung (alle bisherigen Tests) ist das
gleich, in zweiter Ordnung legt es β fest.

### Quellen im Potential (Φ₂-Term) und bewegte Quellen (Φ₁-Term)

Teil C2 misst die Last einer ruhenden Quelle bei gleichförmigem N im H2-Schema:
Q/m = Q_M/m = Quelle/m = N auf 2·10⁻⁵ (N = 0.999 … 0.5). Eine Quelle, die selbst im
Potential einer anderen sitzt, zieht also mit N·m. Mit der Teilchenzahl pro
Koordinatenzelle n* (auf dem Gitter erhalten) und U* = Feld von m·n* folgt aus der
Produktregel, analytisch:

```
ln N = −U* − (3/2) Φ₁* + Φ₂*          (Φ₁*: Quelle E(1 + v²) aus Runde 6, Φ₂*: Rotverschiebung der Quelle)
g₀₀  = −1 + 2U* − 2U*² + 3Φ₁* − 2Φ₂*
```

Das ist die 1PN-Metrik der ART, geschrieben mit der erhaltenen Dichte
ρ* = ρ(1 + ½v² + 3U). In der Standardform mit ρ (g₀₀ ⊃ −2βU² + 2(γ + 1)Φ₁ + 2(3γ − 2β + 1)Φ₂)
entspricht das γ = 1 und β = 1. Die drei gemessenen Zutaten passen also zusammen:
Quelle T₀₀ + T_ii (Runde 6), Last ∝ N (H1, C2) und Produktregel (C1). Mit dem
linearen Gesetz gäbe der U²-Term β = 1/2, der Φ₂-Term aber β = 1, also keine
konsistente Metriktheorie.

## Antwort

1. **Kompensation:** Ein Knoten, der seine W₉₀-Kopplung mit N nachstimmt
   (Δλ ≈ 3.3·(1 − N)), hebt den Untergrund des Bleibe-Gatters auf. Licht wird dann bei
   allen k mit γ = 1.00 ± 0.007 abgelenkt, massive Pakete im Dirac-Bereich mit
   γ = 0.99–1.00 (Gitterband). Das funktioniert für N ≥ 0.4. Bei μ/ε = 0.5 bleibt der
   eigene Untergrund des Walkers als Rest.
2. **β_PPN = 1**, wenn sich Lasten an einem Knoten wie unabhängige Belegungen verbinden
   (N = ΠN_i, verschachtelte Gatter). Das ist dieselbe Regel „frei für alle Beteiligten“
   wie bei H2 und gilt auch im starken Feld. Warteschlangen (Wartezeiten addieren sich)
   gäben β = 3/2, linear überlagerter Lapse β = 1/2. Beide sind durch die
   Perihelverschiebung und die Mondbahn (|β − 1| ≲ 10⁻⁴) ausgeschlossen.
3. Zusammen mit der Quelle aus Runde 6 und der Last ∝ N ergibt die Produktregel die
   statische und geschwindigkeitsabhängige 1PN-Metrik g₀₀ der ART.

**Was das nicht zeigt:**
- β ist auf Bandebene als Kompositionsgesetz gemessen, nicht dynamisch (Perihel
  verlangt Bahnen in 2D/3D).
- Die Φ₁/Φ₂-Konsistenz ist eine analytische Folgerung aus gemessenen Zutaten.
- Innere Energie und Druck (Φ₃, Φ₄) sind nicht geprüft.
- **g₀ᵢ fehlt:** Das Modell hat nur die Skalare N und A. Die gravitomagnetischen
  Terme (Lense–Thirring, mitgeführte Bezugssysteme) brauchen ein Vektorfeld. Ohne sie
  ist das Gitter ein bevorzugtes Bezugssystem der Gravitation (PPN α₁ ≠ 0 zu
  erwarten).

## Offene Punkte

- **g₀ᵢ:** ein Verschiebungsfeld (Shift) aus dem Impulsstrom der Rechenlast. Lokal
  etwa als richtungsabhängiges Bleiben in Bewegungsrichtung der Quelle.
- Massenabhängigen Untergrund bei μ/ε = 0.5 mit N² skalieren, etwa mit λ(N, μ) oder
  einem zweiten Gegenterm.
- Dynamischer β-Test: Bahn in 2D mit Perihelverschiebung (Speicher prüfen).
- Selbstkonsistente Rechnung mit N = N₀ exp(Φ) und Φ aus der Rechenlast, dazu die
  Shapiro-Verzögerung.
- Experiment 3 der Roadmap (Isotropie der Schalen) ist weiterhin offen.
