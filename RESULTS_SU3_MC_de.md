# SU(3)-Gittereichtheorie mit Wilson-Wirkung auf dem Dreiecksgitter

*Dateien:* `quantum_hex_su3_mc.py` (Monte Carlo), `quantum_hex_su3_mc_figs.py` (Report + Abbildung)
*Schritt 1 aus:* `ROADMAP_QCD_3D_de.md`

---

## 1. Warum dieser Schritt

`quantum_hex_su3.py` hatte gezeigt: ein **statischer** SU(3)-Zufallshintergrund
gehorcht einem Umfanggesetz `⟨W⟩ = c^Umfang` und confined damit **nicht**.
Confinement verlangt, dass die Links durch die Wilson-Wirkung korreliert werden:

```
Gewicht  ~  exp( beta * sum_Plaketten (1/N) Re tr U_P )
```

Das Eichfeld bekommt hier also zum ersten Mal eine eigene Dynamik.

---

## 2. Gitterstruktur — ein glücklicher Umstand

Drei positive Linkrichtungen `e0 = (1,0)`, `e1 = (0,1)`, `e2 = (-1,1)` mit
`e0 + e2 = e1`. Zwei Elementardreiecke pro Platz:

```
T_A(x):  x -(+0)-> x+e0 -(+2)-> x+e1 -(-1)-> x     W_A = U1(x)^+ U2(x+e0) U0(x)
T_B(x):  x -(+2)-> x+e2 -(+0)-> x+e1 -(-1)-> x     W_B = U1(x)^+ U0(x+e2) U2(x)
```

**Jedes Dreieck enthält genau einen Link jeder Richtung.** Links derselben
Richtung teilen sich also nie eine Plakette — das ist eine perfekte
3-Färbung. Alle Links einer Richtung lassen sich gleichzeitig aktualisieren,
der komplette Metropolis-Sweep ist über das Gitter vektorisiert. Auf dem
Quadratgitter braucht man dafür eine Schachbrettzerlegung; hier fällt es
geschenkt an.

Abzählung (periodisch, L×L): `3L²` Links, `2L²` Dreiecke,
`Links − Plätze + 1 = 2L²+1` eichfixierte Freiheitsgrade. Die Plaketten sind
also bis auf eine globale Nebenbedingung unabhängig — das ist die Grundlage
der exakten 2D-Faktorisierung weiter unten.

### Staples

Aus `Re tr(U[x,μ] · Σ_μ(x))` ergibt sich

```
Sigma_0(x) = U1(x)^+ U2(x+e0)  +  U2(x-e2) U1(x-e2)^+
Sigma_1(x) = U0(x)^+ U2(x+e0)^+  +  U2(x)^+ U0(x+e2)^+
Sigma_2(x) = U1(x)^+ U0(x+e2)  +  U0(x-e0) U1(x-e0)^+
```

**Verifikation:** die aus dem Staple berechnete Wirkungsänderung stimmt mit der
vollständig neu berechneten Wirkung überein — für alle drei Richtungen auf
`1.3e-15`. Das ist der kritische Korrektheitstest des Verfahrens.

---

## 3. Validierung gegen ein exaktes Ergebnis

In **zwei** Dimensionen faktorisiert reine Eichtheorie: die Plaketten sind
unabhängige Variablen, und die Wilson-Schleife gehorcht einem **exakten**
Flächengesetz

```
<W(C)>  =  w1 ^ (Anzahl eingeschlossener Dreiecke),     w1 = <(1/N) Re tr U_P>
```

wobei `w1` aus einem **Ein-Matrix-Integral** folgt. Das wird hier als
Validierung benutzt, nicht als Entdeckung: Confinement in 2D ist kinematisch.
Der Sinn von Schritt 1 ist ein korrektes, geprüftes dynamisches Eichfeld, das
sich nach 3+1D mitnehmen lässt — dort ist das Flächengesetz nicht mehr trivial.

Drei unabhängige Referenzen wurden verglichen:

1. **Ein-Plaketten-Metropolis** (exakt für alle β),
2. **Haar-Reweighting** (unabhängig, unverzerrt, nur für kleines β effizient),
3. **Starkkopplungsentwicklung** `w1 ≈ β/(2N²) = β/18`.

Zwei scharfe Vorabtests:

| Test | Ergebnis |
|---|---|
| `⟨plaq⟩` bei β = 0 muss exakt 0 sein | +0.0009 ± 0.0006 (L=16) ✓ |
| vektorisierte vs. explizit ausgeführte Wilson-Schleife | 1.1e-16 ✓ |

### Ergebnis der Validierung

L = 24, sieben Kopplungen:

| β | ε | Akzeptanz | Gitter ⟨plaq⟩ | exaktes 1-Plaketten-Integral | Haar-Reweighting | Pull |
|---|---|---|---|---|---|---|
| 1.0 | 4.00 | 0.777 | 0.05989(29) | 0.05973(65) | 0.06096 | +0.23 |
| 2.0 | 4.00 | 0.556 | 0.12811(35) | 0.12756(156) | 0.12984 | +0.35 |
| 3.0 | 2.22 | 0.519 | 0.20299(33) | 0.20402(146) | 0.20450 | −0.69 |
| 4.0 | 1.51 | 0.529 | 0.28037(40) | 0.28130(180) | – | −0.50 |
| 6.0 | 0.98 | 0.526 | 0.42242(36) | 0.42174(147) | – | +0.44 |
| 8.0 | 0.85 | 0.473 | 0.53601(36) | 0.53727(138) | – | −0.89 |
| 12.0 | 0.54 | 0.525 | 0.67846(25) | 0.67991(146) | – | −0.98 |

**Alle sieben Kopplungen stimmen innerhalb von 0.98σ überein.** Der Monte Carlo
ist damit gegen ein exaktes Ergebnis validiert.

---

## 4. Was beim ersten Anlauf schiefging

Der erste Produktionslauf maß Wilson-Schleifen bis zu 32 eingeschlossenen
Dreiecken und bildete daraus Creutz-Verhältnisse. Das war **kein brauchbarer
Beleg**: ab einer bestimmten Fläche fällt ⟨W⟩ unter den statistischen
Rauschboden (~4·10⁻⁴), die Messwerte werden mit dem Rauschen vergleichbar und
sogar negativ. Die Creutz-Verhältnisse ergaben dann Werte wie −40 oder +20
statt der vorhergesagten ~0.8.

Das ist das gewöhnliche **Signal-Rausch-Problem** von Wilson-Schleifen — ⟨W⟩
fällt exponentiell mit der Fläche, das Rauschen bleibt konstant —, kein Defekt
des Ensembles. Aber es kann keine Schlussfolgerung tragen. Die Auswertung
wurde deshalb ersetzt.

---

## 5. Der entscheidende Test

Statt tief ins Rauschen zu messen: **Schleifen mit gleichem Umfang, aber
verschiedener Fläche** — und umgekehrt —, gemessen in zwei Ensembles, die auf
**dieselbe Plakette** eingestellt sind.

* Wilson-Wirkung, dynamisches Eichfeld.
* Statisch: unabhängig gezogene Links `U = exp(i·g·H)`. Deren Plakette ist
  `c³` mit `c = ⟨U⟩/N` (numerisch bestätigt), also lässt sich g durch
  Bisektion so wählen, dass beide Ensembles dieselbe Plakette haben.

Ein Umfanggesetz sagt innerhalb einer Gruppe gleichen Umfangs gleiche Werte
voraus, ein Flächengesetz einen Abfall um `w1^Δ`.

### β = 12 (w₁ = 0.67771, statisch auf g = 0.8705 eingestellt, ⟨plaq⟩ = 0.67583)

| (a,b) | Umfang | Fläche | Wilson-Wirkung ⟨W⟩ | w₁^Fläche | statisch ⟨W⟩ | c^Umfang |
|---|---|---|---|---|---|---|
| (1,1) | 4 | 2 | 0.459012 ± 0.000047 | 0.459291 | 0.592762 ± 0.000057 | 0.593254 |
| (2,1) | 6 | 4 | 0.210794 ± 0.000056 | 0.210948 | 0.457438 ± 0.000056 | 0.456942 |
| (3,1) | 8 | 6 | 0.096983 ± 0.000045 | 0.096887 | 0.353038 ± 0.000078 | 0.351951 |
| (2,2) | 8 | 8 | 0.044716 ± 0.000042 | 0.044499 | 0.352629 ± 0.000115 | 0.351951 |
| (4,1) | 10 | 8 | 0.044699 ± 0.000039 | 0.044499 | 0.272401 ± 0.000119 | 0.271083 |
| (3,2) | 10 | 12 | 0.009601 ± 0.000031 | 0.009387 | 0.273623 ± 0.000152 | 0.271083 |
| (5,1) | 12 | 10 | 0.020579 ± 0.000046 | 0.020438 | 0.210565 ± 0.000160 | 0.208796 |
| (4,2) | 12 | 16 | 0.001811 ± 0.000032 | 0.001980 | 0.209977 ± 0.000183 | 0.208796 |
| (3,3) | 12 | 18 | 0.000822 ± 0.000030 | 0.000909 | 0.209672 ± 0.000129 | 0.208796 |

### Das schärfste Paar

`(2,2)` und `(4,1)` haben **dieselbe Fläche 8**, aber **verschiedene Umfänge
8 und 10**:

| | (2,2), U=8 | (4,1), U=10 | Verhältnis |
|---|---|---|---|
| **Wilson-Wirkung** | 0.044716 ± 0.000042 | 0.044699 ± 0.000039 | **1.0004** — identisch |
| **statisch** | 0.352629 ± 0.000115 | 0.272401 ± 0.000119 | **1.294** — verschieden |

Und umgekehrt, `(3,1)` und `(2,2)` mit **demselben Umfang 8**, Flächen 6 und 8:

| | (3,1), A=6 | (2,2), A=8 | Verhältnis | Vorhersage |
|---|---|---|---|---|
| **Wilson-Wirkung** | 0.096983 | 0.044716 | **2.17** | 2.18 (Flächengesetz) |
| **statisch** | 0.353038 | 0.352629 | **1.001** | 1 (Umfanggesetz) |

Damit ist es sauber: **das dynamische Eichfeld hängt nur von der Fläche ab,
der statische Hintergrund nur vom Umfang** — bei identischer Plakette.

### Weitere Gruppen bei β = 12

| Umfang | Flächen | Wilson-Verhältnis | Flächengesetz sagt | statisch | Umfanggesetz sagt |
|---|---|---|---|---|---|
| 8 | 6, 8 | 2.17 | 2.18 | 1.00 | 1 |
| 10 | 8, 12 | 4.66 | 4.74 | 1.00 | 1 |
| 12 | 10, 16, 18 | 25.02 | 22.47 | 1.00 | 1 |

Bei β = 8 gilt dasselbe, solange das Signal reicht (Umfang 8: 3.51 gegen
vorhergesagte 3.48). Bei **β = 4 ist der Schleifenvergleich vollständig
rauschdominiert** — die Werte liegen bei ~10⁻⁴ mit Fehlern von 3·10⁻⁴ — und
trägt dort nichts bei. Das ist in der Abbildung durch den eingezeichneten
3σ-Rauschboden sichtbar gemacht.

---

## 6. Stringspannung

`σ = −ln w₁` pro Elementardreieck:

| β | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| σ | 2.815 | 2.055 | 1.594 | 1.272 | 0.862 | 0.624 | 0.389 |

In 2D ist σ bei **jeder** Kopplung endlich — reine Eichtheorie confined dort
immer. Es gibt keinen Deconfinement-Übergang und keine asymptotische Freiheit.

---

## 7. Einordnung — was dieser Schritt zeigt und was nicht

**Was er zeigt:** ein korrektes, gegen ein exaktes Ergebnis validiertes
dynamisches SU(3)-Eichfeld auf diesem Gitter, und den kontrollierten Nachweis,
dass die Wilson-Wirkung ein Flächengesetz erzeugt, wo unabhängige Links bei
gleicher Plakette nur ein Umfanggesetz geben.

**Was er nicht zeigt:** Confinement als nichttriviales Phänomen. In zwei
Dimensionen faktorisiert reine Eichtheorie, das Flächengesetz ist exakt und
kinematisch, und es gilt für jedes β. Der eigentliche Gehalt von Confinement —
dass es ein Ergebnis der Gluondynamik ist und nicht der Abzählung — wird erst
in 3+1D sichtbar. Auch asymptotische Freiheit gibt es erst dort.

Der Wert von Schritt 1 ist deshalb die geprüfte Maschinerie, nicht das
Resultat: Staples, vektorisierter Sweep, Ensemble-Vergleich und
Fehlerbehandlung lassen sich unverändert nach 3+1D mitnehmen, wo dieselben
Messungen dann etwas Nichttriviales aussagen.

---

## 8. Abbildung

`su3_mc_confinement.png`

| Tafel | Inhalt |
|---|---|
| 1 | Plakette: Monte Carlo gegen exaktes Ein-Matrix-Integral, alle 7 Kopplungen |
| 2 | ⟨W⟩ gegen Fläche bei β=12, beide Ensembles bei gleicher Plakette, mit 3σ-Rauschboden |
| 3 | Bei festem Umfang: statische Kurven flach, dynamische fallen mit der Fläche |
| 4 | Stringspannung σ(β) |

---

## 9. Offene Punkte

* **Signal-Rausch-Problem.** Größere Schleifen brauchen entweder exponentiell
  mehr Statistik oder Varianzreduktion (Multi-Level-Algorithmen, Smearing).
  Für 3+1D ist das keine Nebensache, sondern die zentrale technische Hürde.
* **Kein Skalenverhalten.** In 2D gibt es kein Kontinuumslimes-Programm; die
  Umrechnung von σ in physikalische Einheiten wartet auf 3+1D.
* **Quenched.** Das Eichfeld hat jetzt Dynamik, aber die Materie wirkt nicht
  zurück. Die Fermiondeterminante fehlt.
* **Nächster Schritt** laut `ROADMAP_QCD_3D_de.md`: der 3+1D-FCC-Walker mit
  SU(2)-Münze — der Schritt, der Spin von einer Wahl zu einer Notwendigkeit
  macht.
