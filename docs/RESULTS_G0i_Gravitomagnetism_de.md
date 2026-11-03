# g₀ᵢ: Gravitomagnetismus aus dem Impulsstrom der Rechenlast

*Paper 3. Code: `src/phase_bit_s4_open10.py` (Teile `A`, `B`) und
`src/phase_bit_s4_open10_figs.py`. Daten: `data/phase_bit_s4_open10.npz`.
Abbildung: `figures/phase_bit_s4_open10.png`.
Vorgänger: [`RESULTS_H2b_Compensation_Beta_de.md`](RESULTS_H2b_Compensation_Beta_de.md).*

Modell wie in H2b: **2O\*** (ε = 0.3) mit Vakuum-Gegenterm, Warte- und Bleibe-Gatter
(N_w = N_h = N), kompensierte W₉₀-Kopplung λ(N), Kopplung N = N₀·exp(Φ). x-Kette
mit 640 Lagen, Bloch-Impuls k_y, k_z. Speicher: wenige 100 MB, frei waren 3.9 GB.

## Frage

Bisher hatte das Modell nur die Skalare N (Lapse) und A (räumliche Metrik). In der
linearisierten ART erzeugt eine bewegte Quelle zusätzlich einen Shift g₀ᵢ, sourced
von der Impulsdichte T₀ᵢ. Für ein Testteilchen heißt das H = N·E(p/A) + b·p, ein
Drift entlang der Bewegung der Quelle (Mitführung). Für zwei Körper, die sich beide
entlang y bewegen, trägt die Querkraft dann den Faktor

```
(1 + v_s²)(1 + v_t²) − C·v_s·v_t        mit  C = 4  (ART)
```

Daraus folgen drei klare Vorhersagen:

- Parallele Lichtstrahlen ziehen sich **nicht** an.
- Antiparallele Lichtstrahlen ziehen sich doppelt so stark an wie ohne g₀ᵢ.
- Mitbewegte Körper merken nichts von ihrer Bewegung gegen das Gitter. Für Schichten
  gilt a(mitbewegt) = a(Ruhe)/γ³, und das erzwingt C = 4, denn
  (1 + v²)² − C·v² = (1 − v²)² nur für C = 4. Ohne g₀ᵢ wäre das Gitter ein
  bevorzugtes Bezugssystem der Gravitation.

## 1. Bandebene: wie b·p lokal entsteht

**Erster Versuch, verworfen:** ein richtungsabhängiges Bleibe-Gatter
N_h(d) = N_h(1 + u_d·w), bei dem Sprünge in Richtung w leichter angenommen werden. Das
gibt zwar einen linearen Term, aber:

- **Kramers-Paare spalten:** Bei massiven Zuständen mit k ⊥ w trennen sich die beiden
  Partner um 1.4·10⁻⁴ (bei w = 0.02, k = 0.02). Das ist eine unphysikalische
  Spin-Kopplung.
- **Das Vakuum verschiebt sich** in zweiter Ordnung (−3·10⁻⁴ ≈ −0.74 w² am
  Dirac-Punkt), dann gravitiert es.

Der Grund: Im Walker ist die innere Richtung zugleich die Bewegungsrichtung. Eine
Gewichtung nach Richtung trifft deshalb auch den Spin-Anteil des Zustands.

**Lösung: richtungsunabhängiger Mitnahme-Sprung.** b·p ist der Erzeuger einer
Verschiebung. Lokal bedeutet das: Jeder Knoten gibt einen kleinen Anteil **jedes**
Zustands, ganz gleich, wohin dieser zeigt, an den Nachbarn in Richtung des
Impulsstroms weiter. Über das Nachbarpaar (0, ℓ, ±ℓ) mit ℓ = 0.612 ist der Erzeuger
b·sin(k_y ℓ)cos(k_z ℓ)/ℓ ≈ b·k_y. Er wirkt auf alle inneren Komponenten und Register
gleich.

| Prüfung | Ergebnis |
|---|---|
| Verschiebung aller vier Niveaus, masselos und μ = 0.03, sechs k-Vektoren | b·sin(k_y ℓ)cos(k_z ℓ)/ℓ auf ≤ 5·10⁻¹⁵ |
| Dirac-Punkt (k = 0) | unverändert (0) |
| Kramers-Paare | bleiben entartet |

## 2. Kopplungsgesetz

Das Feld kommt aus der Impulsdichte der Quelle, mit derselben Green-Funktion wie Φ:

```
Φ(x) = (κ/2) ∫|x − x′| s(x′) dx′,           s = 2e − e_M  (Runde 6)
b(x) = −4 c′² N₀⁴ (κ/2) ∫|x − x′| k_s ρ_s(x′) dx′
```

- k_s·ρ_s ist die Impulsdichte, also die Noether-Ladung der Verschiebung und eine
  lokale Größe.
- Der Faktor 4 ist der ART-Wert, den das Relativitätsprinzip verlangt (siehe oben).
- N₀⁴ ist die Umrechnung zwischen Gitter- und physikalischen Koordinaten um den
  Hintergrund N₀ mit A₀ = 1/N₀. Im Vakuum (N₀ = 1) fällt er weg.

## 3. x-Kette: zwei bewegte Schichten

Aufbau:
- Quelle und Testteilchen sind Schichten (Bloch-k_y) bei ±67, mit Breite σ = 32.
- Grundlapse N₀ = 0.6, Feldgradient am Testort festgelegt (d ln N/dx = 3·10⁻³).
- Gemessen wird die Kraft F = d⟨k_x⟩/dt über 320 Takte, gemittelt über beide
  Kramers-Partner. Die Zitterbewegung mit ω = 2E wird mitgefittet.
- Die ART-Vorhersage ist für genau diese Pakete gerechnet: halbklassisch mit idealer
  Dirac-Dispersion in N, A = 1/N und b, gemittelt über die Orts- und
  Impulsverteilung. Zusätzlich steht die Formel für ideale Strahlen da.

**Lapse-Kraft (ohne g₀ᵢ):** Simulation/ART = 0.996–1.002 in allen elf Fällen.

| Quelle → Test | v_s, v_t | Mitnahme-Kraft ΔF Sim. / ART (10⁻³κ) | Anziehung mit / ohne g₀ᵢ | ART, diese Pakete | ART, ideale Strahlen |
|---|---|---|---|---|---|
| μ = 0.03, k = 0.1 → +0.1 | 0.50, 0.50 | +0.867 / +0.863 | **0.378** | 0.381 | 0.366 |
| μ = 0.03, k = 0.2 → +0.2 | 0.75, 0.75 | +3.454 / +3.447 | **0.088** | 0.093 | 0.079 |
| μ = 0.03, k = 0.1 → −0.1 | 0.50, −0.50 | −0.872 / −0.863 | **1.626** | 1.619 | 1.634 |
| μ = 0.03, k = 0.2 → −0.2 | 0.75, −0.75 | −3.488 / −3.447 | **1.921** | 1.907 | 1.921 |
| Licht, k = 0.2 → +0.2 | 1, 1 | +3.449 / +3.447 | **0.019** | 0.022 | 0 |
| Licht, k = 0.2 → −0.2 | 1, −1 | −3.487 / −3.447 | **1.992** | 1.978 | 2 |
| Licht, k = 0.1 → +0.1 | 1, 1 | +0.864 / +0.863 | **0.053** | 0.055 | 0 |
| Licht, k = 0.1 → −0.1 | 1, −1 | −0.873 / −0.863 | **1.957** | 1.945 | 2 |
| μ = 0.03, ruhend → k = 0.2, und umgekehrt | | 0 | 1.000 | 1 | 1 |

- **Parallele Lichtstrahlen ziehen sich praktisch nicht mehr an** (0.019 bzw. 0.053
  statt 1). Der Rest ist die Impulsbreite der Pakete: Die Last zählt |k|, die
  Mitnahme nur k_y. Die Paket-ART sagt 0.022 und 0.055 voraus.
- **Antiparallele Lichtstrahlen ziehen sich doppelt so stark an** (1.99 bzw. 1.96).
- Die gravitomagnetische Kraft selbst stimmt auf 0.1–1.2 % mit der ART. Antiparallel
  liegt sie etwa 1 % über parallel, ein kleiner Gitterrest.
- Ruhen Quelle oder Test, gibt es keinen Effekt, wie verlangt.

**Relativitätsprinzip (mitbewegte, gleiche Schichten):**

| v | mit g₀ᵢ | ART, diese Pakete | ideale Schichten 1/γ³ | ohne g₀ᵢ | nur Lapse + A: γ(1 + v²)² |
|---|---|---|---|---|---|
| 0.50 | **0.669** | 0.675 | 0.655 | 1.769 | 1.788 |
| 0.75 | **0.319** | 0.339 | 0.292 | 3.630 | 3.672 |

Ohne g₀ᵢ ziehen sich zwei mitbewegte Schichten um den Faktor 1.8 bzw. 3.6 stärker an
als in Ruhe. Ein mitbewegter Beobachter könnte so seine Geschwindigkeit gegen das
Gitter messen. Mit g₀ᵢ fällt die Anziehung wie die Zeitdilatation (1/γ³). Damit ist
die Gravitation des Modells in dieser Hinsicht frei von einem bevorzugten
Bezugssystem.

**Messfallen, die hier aufgetreten sind:**
- Ein einzelner Kramers-Partner hat bei Querbewegung im Gradienten einen
  spinabhängigen Kraftanteil (±8 % bei k = 0.2). Das Mittel über beide Partner
  entfernt ihn.
- Der Ortsfit ist durch die Zitterbewegung bei 2E verfälscht, deren Periode etwa so
  lang ist wie der Lauf. Die Kraft aus d⟨k_x⟩/dt ist robuster.
- Ein Filter (U − e^{iEΔt}) gegen das Gegenband gewichtet das Paket um und verfälscht
  die Kraft. Verworfen.
- Pakete näher als 4σ am offenen Kettenende geben bis zu 23 % zu viel Kraft. Daher 640
  statt 440 Lagen.

## Antwort

1. **g₀ᵢ lässt sich lokal bauen:** als richtungsunabhängiger Mitnahme-Sprung, dessen
   Stärke aus der Impulsdichte der Rechenlast kommt. Er ist auf Bandebene exakt b·p
   für alle Zustände und lässt Vakuum und Kramers-Paare unberührt. Ein
   richtungsabhängiges Gatter geht nicht: Spin-Kopplung und Vakuum-Verschiebung.
2. **Die Dynamik ist die der ART:** Parallele Lichtstrahlen ziehen sich nicht an,
   antiparallele doppelt. Die gravitomagnetische Kraft stimmt auf ≤ 1.2 %.
3. **Relativitätsprinzip:** Mitbewegte Schichten verhalten sich wie in Ruhe, bis auf die
   Zeitdilatation (0.669 / 0.319 gegen 0.675 / 0.339). Der Koeffizient 4 ist genau der,
   der das sicherstellt.

**Was das nicht zeigt:**
- Der Koeffizient 4 ist eingesetzt: Er ist der Wert, den das Relativitätsprinzip
  verlangt, nicht aus einer Rechenregel abgeleitet. Was gezeigt ist: Mit ihm passen
  Lapse, räumliche Metrik und Shift auf dem Gitter zusammen.
- b ist hier instantan (Green-Funktion). Die kausale Version bräuchte eine
  Wellengleichung für b wie in Runde 6 für Φ.
- Geprüft ist nur der Fall, dass Impuls und Shift entlang derselben Achse liegen (y)
  und das Feld in x variiert. Die Rotation von b (Lense–Thirring, Präzession) braucht
  2D/3D.
- Der Gegenterm λ(N) und die Kopplung N₀⁴ sind Gitter-Buchhaltung des Hintergrunds
  N₀ < 1.

## Offene Punkte

- **Kausales g₀ᵢ:** Wellengleichung für b mit der Impulsdichte als Quelle. Dann ist
  auch eine Gravitationswelle (Quadrupol aus T_ij) in Reichweite, zusammen mit den
  anisotropen Coin-Gewichten aus Runde 5.
- **Lense–Thirring:** rotierende Quelle in 2D, Präzession eines Spins oder einer Bahn.
- **Woher kommt die 4?** Gesucht ist eine Budget-Regel, die den Faktor aus der
  Aufteilung der Rechenlast liefert (Lapse-Quelle E(1 + v²) gegen Impulsstrom E·v),
  statt ihn aus dem Relativitätsprinzip zu übernehmen.
- Experiment 3 der Roadmap (Isotropie der Schalen) ist weiterhin offen.
