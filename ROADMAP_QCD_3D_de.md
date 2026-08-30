# Roadmap: 3D-Spin und QCD

*Dateien:* `quantum_fcc_holonomy.py`, `quantum_hex_su3.py`, `quantum_qcd_3d_report.py`
*Abbildung:* `qcd_3d_overview.png`

Zwei Fragen: (A) Braucht echter Spin 3D? (B) Was fehlt für QCD?
Beide sind hier gerechnet, nicht nur diskutiert.

---

## A. Spin: die Vermutung "erst in 3D" ist richtig — und der Grund ist topologisch

### Was in 2D passiert

Der Richtungsraum ist ein **Kreis**. Für jeden geschlossenen Weg, der auf dieselbe
gerichtete Kante zurückkommt, ist die Winkelsumme exakt `360°·w` mit ganzzahliger
Windungszahl `w`. Wegen `π₁(S¹) = ℤ` kann durch diesen Kreis ein **Fluss** gefädelt
werden: die Schleifenphase `exp(i·2πα·w)` ist für **jedes reelle α** wohldefiniert.

Das ist unser α. Es ist ein *kontinuierlicher* Parameter — die Anyon-Situation.
Spin ist in 2D ein frei drehbarer Knopf, nicht eine erzwungene Größe.

### Was in 3D passiert (FCC, 12 Nachbarn, gerechnet)

Der Richtungsraum ist eine **Kugel**. Die Richtung wird mit der minimalen Drehung
weitertransportiert, akkumuliert als Quaternion (also in SU(2)).

| | 2D Dreiecksgitter | 3D FCC |
|---|---|---|
| geschlossene Wege (bis L=6) | 10 | 1584 |
| Drehachse = Anfangsrichtung | 1.7e-16 | 8.0e-16 |
| **verschiedene Werte von φ/2π** | **1** | **23** |
| max. Abweichung von einer ganzen Zahl | **0.000000** | **0.500000** |
| Holonomien um verschiedene Achsen | vertauschen | **‖QaQb − QbQa‖ = 0.333** |

Die Holonomiewinkel sind algebraisch, aber irrational: `cos(φ/2)` nimmt die Werte
`1/3`, `1/√3`, `√(2/3)`, `2√2/3` an — `arccos(1/3)/π` ist irrational.

**Es gibt in 3D keine Windungszahl.** Der 2D-Umlaufsatz überlebt nicht.

### Warum daraus Spin-Starrheit folgt

* `π₁(S²) = 0` — auf der Kugel gibt es keine Schleife, durch die man einen Fluss
  fädeln könnte. Das kontinuierliche α aus 2D hat schlicht kein 3D-Gegenstück.
* Die Rahmen leben in `SO(3)`, überlagert von `SU(2)`. `SU(2)` ist **einfach**:
  der einzige nichttriviale Normalteiler ist das Zentrum `ℤ₂`. Also gibt es
  **keinen** stetigen Homomorphismus `SU(2) → U(1)` außer dem trivialen.
* Übrig bleiben genau die beiden Charaktere von `ℤ₂`: **+1 und −1**.

```
2D:  pi_1(S^1) = Z    -> alpha in R/Z   (Kontinuum, Anyonen)
3D:  pi_1(S^2) = 0,  SO(3) <- SU(2) doppelt ueberlagert
                      -> alpha in {0, 1/2}   (Boson oder Fermion)
```

**Spin wird quantisiert, weil die 3D-Drehgruppe doppelt — nicht unendlich —
zusammenhängend ist.** Genau das war die Vermutung.

### Rückbezug auf die bisherigen Ergebnisse

Der Punkt α = ½ des 2D-Modells (Kramers-Verdopplung, `(−1)^w`, Berry-Versatz
γ = 0 statt ½) ist damit **kein Zufall und keine Wahl**: er ist der Schatten
dessen, was in 3D die einzige nichttriviale Möglichkeit ist. In 2D mussten wir
α = ½ von Hand einstellen; in 3D wäre es erzwungen. Das ist ein starkes Argument,
den 3+1D-FCC-Schritt tatsächlich zu gehen.

### Was für ein echtes 3+1D-Modell noch fehlt

1. **Münzoperator auf 12 Richtungen** statt 6, erzeugt aus `SU(2)`-Generatoren
   statt aus der `U(1)`-Drehung `R`. `C = exp(i·ε·G)` mit `G` aus den
   Transport-Quaternionen — die Bausteine liegen in `quantum_fcc_holonomy.py`.
2. **Zwei Spinorkomponenten pro Richtung** (der Rahmen hat in 3D eine
   Zusatzfreiheit, die es in 2D nicht gibt): `amp[x,y,z,d,s]` mit s = ±½.
3. **Lichtgeschwindigkeit und Masse neu vermessen** — c ist auf FCC nicht √3.
4. Dispersion, Isotropie, Kegelsteigung: alles wie in 2D, aber mit 12 Richtungen.

Rechenaufwand: das 2D-Modell hatte `(Nx·Ny·6)`; 3D hat `(Nx·Ny·Nz·12·2)`. Bei
Nx=Ny=Nz=200 sind das 1.9e8 komplexe Zahlen ≈ 3 GB pro Zustandsarray. Also
entweder kleineres Gitter (~120³) oder float32/Streaming.

---

## B. QCD: was die Maschinerie schon kann und was fehlt

### Schon vorhanden

Das Magnetfeld-Modul ist bereits eine **Gittereichtheorie**, nur mit der
Eichgruppe U(1):

| Begriff der Eichtheorie | wo es im Projekt schon steht |
|---|---|
| Eichzusammenhang auf Links | Peierls-Phase pro Schritt |
| Wilson-Schleife | `loop_flux` = exp(i·B·Fläche) |
| Plakettenwirkung | Fluss pro Elementardreieck |
| Eichinvarianz | verifiziert auf 1.7e-15 (symmetrisch vs. Landau) |
| Materie im Feld | der bandprojizierte Walker |

### Was der Übergang zu SU(3) ändert (gerechnet, `quantum_hex_su3.py`)

Die Linkphase wird eine Matrix `U ∈ SU(3)`, der Zustand bekommt einen Farbindex
`amp[x,y,d,c]`.

| Prüfung | Ergebnis |
|---|---|
| Links sind in SU(3) | `‖UU† − 1‖ = 4e-16`, `|det U − 1| = 5e-16` |
| `tr W` eichinvariant, Dreieck | **8.9e-16** |
| `tr W` eichinvariant, 2×3-Raute | **1.8e-15** |
| **U(1): Flüsse addieren sich** | **exakt 0.00e+00** |
| **SU(3) mit gemeinsamem Basispunkt** | **9.7e-17** (korrekte Regel) |
| SU(3) ohne Transporter | 0.0646 (falsch) |
| SU(3): Plaketten vertauschen nicht | 0.82 |

Der entscheidende Unterschied in einem Satz: **in U(1) heben sich die Transporter
weg und Flüsse addieren sich; in SU(3) muss man zum gemeinsamen Basispunkt
konjugieren, das Produkt ist wegordnungsabhängig, und es gibt keinen additiven
Fluss.** Das ist der ganze Unterschied zwischen Elektrodynamik und QCD.

### Ein sauberes Negativergebnis: statischer Hintergrund confined nicht

Mit unabhängig gezogenen Links (statischer, quenched Hintergrund) gilt exakt ein
**Umfanggesetz**, kein Flächengesetz:

| g | c = ⟨U⟩/N | Schleife | Umfang | Fläche | ⟨W⟩ gemessen | c^Umfang |
|---|---|---|---|---|---|---|
| 0.6 | 0.94089 | (1,1) | 4 | 2 | 0.77985 | 0.78372 |
| | | (2,1) | 6 | 4 | 0.69623 | 0.69382 |
| | | (2,2) | 8 | 8 | 0.61864 | 0.61422 |
| | | (3,2) | 10 | 12 | 0.54128 | 0.54376 |
| | | (3,3) | 12 | 18 | 0.47427 | 0.48138 |
| 1.0 | 0.84016 | (1,1) | 4 | 2 | 0.49016 | 0.49824 |
| | | (3,3) | 12 | 18 | 0.10864 | 0.12369 |

`⟨W⟩ = c^Umfang` trifft auf drei Stellen. Der Grund ist elementar: bei
unabhängigen Links faktorisiert der Erwartungswert über die Perimeterlinks,
`⟨U⟩ = c·1`. **Confinement kann so nicht entstehen** — dafür müssen die Links
durch die Wilson-Wirkung korreliert werden.

### Was für echte QCD fehlt — ehrliche Liste

1. **Dynamisches Eichfeld.** Links müssen aus `exp(β Σ_Plaketten (1/3)Re tr U_P)`
   gezogen werden (Metropolis/Wärmebad). Das ist Standard und machbar: auf dem
   Dreiecksgitter bordert jeder Link genau 2 Dreiecke, die Staple-Summe hat also
   nur 2 Terme. Erst damit gibt es ein Flächengesetz und eine Stringspannung.
   **Das ist der nächste konkrete Schritt.**
2. **3+1 Dimensionen.** In 2 räumlichen Dimensionen ist reine Eichtheorie
   (euklidisch 2D) exakt lösbar und trivial confining; asymptotische Freiheit und
   echte Gluondynamik gibt es erst in 3+1D. Der 3D-Schritt aus Teil A wird also
   **zweimal unabhängig** gebraucht: für starren Spin und für echte QCD.
3. **Rückwirkung der Materie** (unquenched): unser Walker spürt das Feld, aber
   das Feld spürt den Walker nicht. Volle QCD braucht die Fermiondeterminante.
4. **Reelle Zeit.** Unsere Zeitentwicklung ist unitär in reeller Zeit; die
   etablierten QCD-Methoden sind euklidische Monte-Carlo-Verfahren. Reelle Zeit
   hat das Vorzeichenproblem. Das ist die **härteste** Barriere, und sie ist
   nicht durch Fleiß zu umgehen. Realistisch: euklidisch für Spektren und
   Stringspannung, reelle Zeit nur für Propagation im festen Hintergrund.
5. **Zwei-Teilchen-Raum.** Ein Farbsingulett `q q̄` braucht einen echten
   Zweiteilchen-Hilbertraum — dieselbe offene Baustelle wie bei der
   Ausschließlichkeit in `RESULTS_Turning_2D_de.md`.

### Ein Haken, den es zu prüfen lohnt

Das Dreiecksgitter trägt eine natürliche **ℤ₃**-Struktur (drei Untergitter, das
kleinste geschlossene Element ist das Dreieck aus 3×120°). Das Zentrum von SU(3)
ist ebenfalls ℤ₃, und die Zentrumssymmetrie ist genau das, was Confinement
klassifiziert (Polyakov-Schleife). Ob das mehr als eine Koinzidenz ist, ist
offen — es wäre aber der erste Ort, an dem ich nachsehen würde.

---

## Reihenfolge, die ich vorschlagen würde

1. **SU(3) mit Wilson-Wirkung** auf dem bestehenden 2D-Gitter (Monte Carlo,
   Stringspannung, Flächengesetz). Klein, abgeschlossen, liefert die erste echte
   Confinement-Messung. Baut nur auf `quantum_hex_su3.py` auf.
2. **3+1D FCC-Walker** mit `SU(2)`-Münze (Teil A, Punkte 1–4). Groß, aber es ist
   der Schritt, der den Spin von einer Wahl zu einer Notwendigkeit macht.
3. **Zwei-Teilchen-Raum** — löst Ausschließlichkeit *und* Farbsingulett in einem.
4. Erst danach: unquenched, Vorzeichenproblem, echte QCD-Observablen.
