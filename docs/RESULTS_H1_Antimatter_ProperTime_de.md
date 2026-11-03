# H1: aktive schwere Masse der Antimaterie, und die Eigenzeit als Massenanteil der Rechenlast

*Paper 3. Code: `src/phase_bit_s4_open7.py` (Teile `A`, `B`) und
`src/phase_bit_s4_open7_figs.py`. Daten: `data/phase_bit_s4_open7.npz`.
Abbildung: `figures/phase_bit_s4_open7.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open6_de.md`](RESULTS_PhaseBit_S4_Open6_de.md).*

Modell wie in Runde 4–6: **2O\*** (ε = 0.3) mit Vakuum-Gegenterm, Warte-Puffer
(N₀ = 0.75), x-Kette mit 400 bzw. 1000 Lagen, 1D-Green-Funktion. Speicher: wenige
100 MB.

Diese Runde prüft zwei Fragen, deren Antwort nicht vorgegeben ist:

- **H1:** Gravitiert Antimaterie aktiv wie Materie?
- **Eigenzeit:** Frisst Bewegung Rechenzeit, sodass die Eigenzeit eines bewegten
  Teilchens als lokale Größe im Modell auftaucht?

## 1. H1: aktive schwere Masse der Antimaterie

**Ausgangslage:** Die Quelle aus Runde 6 ist s = 2e − e_M. Beide Lasten sind
Hellmann–Feynman-Größen der Quasienergie, also für das untere Band negativ:

| Zustand | E | Q = Σe | Q_M = Σe_M | Quelle 2Q − Q_M |
|---|---|---|---|---|
| Teilchen, k_y = 0 | +0.060 | +0.0453 | +0.0447 | +0.0459 |
| Teilchen, k_y = 0.3 | +0.182 | +0.1366 | +0.0144 | +0.2588 |
| Antiteilchen, k_y = 0 | −0.060 | −0.0453 | −0.0447 | −0.0459 |
| Antiteilchen, k_y = 0.3 | −0.182 | −0.1365 | −0.0144 | −0.2586 |

Das passive Fallen des Antiteilchens ist normal: R′ = 0.997 aus Runde 4. Wörtlich im
Einteilchen-Bild genommen wäre Antimaterie also ein Körper, der normal fällt, aber
aktiv abstößt.

**Test: Paar aus Teilchen und Antiteilchen** (beide μ = 0.03, Abstand 245, 200 Takte):

| Bild | a_Teilchen | a_Antiteilchen | Schwerpunkt | Relativbeschleunigung (aufeinander zu > 0) |
|---|---|---|---|---|
| Einteilchen (Quelle des Antiteilchens = seine Last) | −1.849·10⁻⁴ | −1.858·10⁻⁴ | **−1.85·10⁻⁴** | +8.9·10⁻⁷ |
| Loch im Dirac-See (Quelle = −Last des fehlenden Zustands) | +1.858·10⁻⁴ | −1.859·10⁻⁴ | −1.8·10⁻⁸ | **+3.72·10⁻⁴** |
| Referenz: zwei Teilchen | +1.858·10⁻⁴ | −1.859·10⁻⁴ | — | +3.72·10⁻⁴ |

- **Einteilchen-Bild:** Beide laufen in dieselbe Richtung und beschleunigen gemeinsam
  (das „runaway“-Paar von Bondi). Der Impuls ist nicht erhalten, das dritte Gesetz
  gilt nicht. Das Modell schließt diese Lesart also selbst aus.
- **Loch-Bild:** Das Antiteilchen ist ein fehlender Zustand negativer Energie im
  gefüllten See. Seine Last relativ zum Vakuum ist −(Last des fehlenden Zustands) =
  +|E|(1 + v²). Teilchen und Antiteilchen ziehen sich dann an, **genau wie zwei
  Teilchen** (Beschleunigungen auf 10⁻³ gleich), und der Schwerpunkt ruht. Passiv fällt
  das Loch wie der fehlende Zustand, also ebenfalls normal.

**Ergebnis H1:** Im Modell gravitiert Antimaterie aktiv und passiv wie Materie,
vorausgesetzt die Last wird relativ zum gefüllten See gemessen. Das ist die einzige
Lesart mit Impulserhaltung. Die Idee „Rechenlast kann nicht negativ sein“ fällt damit
mit der Löcher-Interpretation zusammen. Der Vakuum-Fixpunkt aus Runde 4 muss demnach
der Vielteilchen-Grundzustand sein, nicht das leere Gitter.

**Preis (offen):** Der gefüllte See hat selbst eine riesige Last Σ_{E<0} E. Dass sie
nicht gravitiert, muss man fordern (Normalordnung), wie beim Gegenterm für das
einzelne Quartett. Das ist die Gitterform des Problems der kosmologischen Konstante.
Hier wird es nicht gelöst, sondern sichtbar gemacht.

**Ausgeschlossene Alternative:** Eine lokal nicht-negative Last wie (1 − cos EΔt)/Δt
wäre quadratisch in E. Die aktive Masse wüchse dann wie E², die passive wie E, und das
dritte Gesetz verletzte sich für ungleiche Massen um den Faktor m₁/m₂. Das folgt
analytisch und wurde nicht simuliert.

## 2. Eigenzeit: Bewegung frisst Rechenzeit

**Idee:** Die Energie (Rechenlast) eines Teilchens teilt sich in zwei Teile:

```
E  =  m · dτ/dt   +   p · v
      Uhr-Anteil      Bewegungs-Anteil
```

Der Uhr-Anteil ist der Massenanteil μ·∂E/∂μ. Für Dirac gilt m²/E = m·√(1 − v²) =
m·dτ/dt. Je schneller das Teilchen, desto mehr Rechenzeit geht in die Bewegung und desto
weniger tickt die innere Uhr. Die lokale Massen-Last e_M(x) aus Runde 6 misst also die
**Eigenzeitrate pro Ort**, und ihr Raum-Zeit-Integral ist

```
∫∫ e_M dx dt  =  m · τ  =  −S        (S: Wirkung des freien Teilchens)
```

Mit Lapse N und räumlicher Metrik A (Exp. 1c, physikalischer Impuls k/A) sollte
dτ/dt = N·A·m/√(k²c′² + A²m²) = N·√(1 − v_phys²) herauskommen: Gravitations- und
Geschwindigkeits-Zeitdilatation zugleich.

**Test:** Pakete bewegen sich quer (k_y = 0, 0.15, 0.3, 0.6), mit gleichförmigem Lapse
N = 0.4, 0.75 und 0.999 und mit räumlicher Metrik A = 0.9 und 1.1, für μ = 0.03 und 0.15
(40 Fälle). Gemessen wird Q_M/m, gemittelt über 60 Takte.

- **Gegen das Gitterband** (μ∂E/∂μ der Banddispersion mit N und A): alle bewegten Fälle
  auf 0.9 %. Die ruhenden Fälle liegen bei μ = 0.03 um 2–3 % darunter, weil das Paket eine
  Impulsbreite hat und ⟨m/E⟩ < m/E(0) ist. Bei μ = 0.15 sind es 0.2 %.
- **Gegen Dirac N√(1 − v_phys²):** 0.5 % bei k = 0.15, 2.1 % bei k = 0.3, 9 % bei
  k = 0.6. Die Abweichung ist die bekannte Abweichung des Gitterbands von Dirac bei
  großem k.
- **Aufteilung der Last** (μ = 0.15, N ≈ 1): Uhr-Anteil Q_M/Q = 1.00 / 0.92 / 0.75 / 0.41
  bei k = 0 / 0.15 / 0.3 / 0.6, Dirac 1/γ² = 1 / 0.92 / 0.75 / 0.40 (Abbildung d).
- **Lapse und Metrik:** Bei k = 0 ist dτ/dt = N unabhängig von A, bei Bewegung wirkt A
  über den physikalischen Impuls (k_y = 0.6, μ = 0.15: 0.420 bei A = 0.9, 0.471 bei
  A = 1.1; Band 0.421 / 0.472).

**Bezug zu Paper 1:** Dort war die quantenmechanische Eigenzeit
τ_quantum = T·m·⟨1/E(k)⟩ (1+1D, Zeitdilatation 36 % bei v = 0.9c). Das ist genau
∫Q_M dt/m für N = 1: Die Eigenzeit von Paper 1 ist die Massen-Last, jetzt lokal
gemessen und mit Gravitation (N) und räumlicher Metrik (A) kombiniert.

**Was das ist und was nicht:** Die Identität Massenanteil = m·dτ/dt ist für Dirac exakt
(Hellmann–Feynman plus Kinematik). Neu ist dreierlei:

- Sie ist auf dem Gitter eine **lokale** Größe, gemessen aus zwei Nachbar-Takten.
- Sie enthält Lapse und räumliche Metrik richtig, ohne Zusatzannahme.
- Dieselbe Größe ist die Massen-Last in der ART-Quelle s = 2e − e_M.

Die Rechenlast-Bilanz E = m·dτ/dt + p·v verbindet also Eigenzeit, Wirkung und
gravitative Quelle.

## Antwort

1. **H1:** Antimaterie gravitiert im Modell aktiv und passiv wie Materie, aber nur im
   Loch-Bild (Last relativ zum gefüllten See). Das Einteilchen-Bild gäbe negative aktive
   Masse und ein davonlaufendes Paar ohne Impulserhaltung (Schwerpunkt −1.85·10⁻⁴). Der
   Preis ist die Normalordnung der See-Last.
2. **Eigenzeit:** Die lokale Massen-Last ist m·dτ/dt, mit dτ/dt = N√(1 − v_phys²):
   Gravitation, räumliche Metrik und Bewegung in einer Größe (Gitterband auf 0.9 %).
   Bewegung frisst Rechenzeit: Der Uhr-Anteil der Last fällt wie 1/γ². Das
   Raum-Zeit-Integral der Massen-Last ist m·τ = −S.

## Offene Punkte

*Weiter mit der Hypothese H2: [`RESULTS_H2_BudgetRule_de.md`](RESULTS_H2_BudgetRule_de.md).*

- **See-Last:** Kann der gefüllte See selbst ein Fixpunkt der Warte-Regel sein
  (Gesamtphase pro Takt null), sodass die Normalordnung nicht gefordert, sondern
  erzwungen ist? Das wäre ein Gitterargument zur kosmologischen Konstante.
- **H2:** Eine einzige Budget-Regel pro Knoten, die N und A gemeinsam erzeugt; prüfen,
  ob γ_PPN = 1 herauskommt. Die Aufteilung E = m·dτ/dt + p·v legt eine Form nahe: Die
  Uhr-Rechnung sieht nur N, die Bewegungs-Rechnung N und A.
- Zwillingsparadoxon auf einer Bahn mit Umkehr: ∫Q_M dt entlang der Bahn gegen die
  Eigenzeit der Bahn, auch im Gravitationsfeld.
