# 3+1D-Walker auf dem FCC-Gitter mit SU(2)-Münze

*Dateien:* `quantum_fcc_3d.py` (Modell), `quantum_fcc_3d_figs.py` (Report + Abbildung)
*Schritt 2 aus:* `ROADMAP_QCD_3D_de.md`
*Abbildung:* `fcc_3d_spin.png`

> **Korrektur (2. Nov. 2026).** `su2_transport` hatte das falsche Vorzeichen:
> cos(θ/2) + i·sin(θ/2)·n·σ dreht den Spinor um −θ, also von der neuen Richtung
> weg (gefunden in Paper 3, Experiment S3). Jetzt gilt U(R) = exp(−iθ n·σ/2). Alle
> Zahlen unten sind mit dem korrigierten Code neu gerechnet. Unverändert bleiben
> Unitarität, Kramers bei jedem k, Gürteltrick (Quaternionen, war schon korrekt),
> die Spin-0-Werte und die Kausalität. **Grundlegend geändert hat sich der
> Spin-½-Zweig:** Er ist kein massives Band, sondern ein **masseloser Dirac-Kegel**
> (vierfach entartet bei k = 0, Steigung exakt c/3). Die Aussage
> „m = 4√3·ε, Dispersion relativistisch auf 0.8–4.7 %“ war ein Artefakt der
> falschen Konvention.

---

## 1. Worum es geht

Im 2D-Modell lebt die Richtung auf einem **Kreis**. Deshalb ist eine
Schleifenphase `exp(i·2πα·w)` für *jedes* reelle α zulässig — Spin ist ein
freier Knopf, den man von Hand einstellen muss. α = ½ wurde gewählt, weil es
Kramers-Verdopplung und die Dirac-Berry-Phase liefert.

In 3D lebt die Richtung auf einer **Kugel**. `quantum_fcc_holonomy.py` hatte
gezeigt: es gibt dort keine Windungszahl und die Holonomien vertauschen nicht,
also existiert kein kontinuierliches α. Was an seine Stelle tritt, ist der
Paralleltransport selbst: der Walker trägt einen Spinor, und eine
Richtungsänderung dreht diesen Spinor mit dem SU(2)-Element der minimalen
Drehung.

**α verschwindet. Übrig bleibt die Wahl der Darstellung: Spin 0 oder Spin ½ —
Boson oder Fermion.** Genau die beiden Möglichkeiten, die die Topologie zulässt.

---

## 2. Geometrie — und ein Ärgernis weniger

12 FCC-Richtungen. Die Winkel zwischen ihnen sind 60°, 90°, 120°, 180°, mit
je 4, 2, 4, 1 Nachbarn.

Die Münze verbindet **nur die 60°-Paare**. Das ist der Kuboktaeder-Graph
(12 Ecken, 24 Kanten, Grad 4, zusammenhängend). Damit gilt:

> **Die 180°-Umkehr, deren Drehachse undefiniert ist und die im 2D-Modell eine
> Konvention von Hand erzwang, kommt hier gar nicht vor.** Sie ist im
> 60°-Graphen nicht enthalten.

Skalierung wie in 2D (Raumzeit-Kantenlänge 1): `|Schritt| = √3/2`, `Δt = ½`,
also **c = √3** — dieselbe Lichtgeschwindigkeit wie im 2D-Modell, per
Konvention. Gitterkonstante pro ganzzahliger Einheit: `a = √6/4`.

*(Korrektur zur Roadmap: dort stand, c sei auf FCC nicht √3. Mit derselben
Kantenlängen-Konvention ist es √3.)*

---

## 3. Der Münzoperator

```
G = sum_{d' ~ d}  |d'><d| (x) Q(d -> d')        Q = SU(2)-Transport
C = expm( i * eps * G )
```

`G` ist hermitesch per Konstruktion (das Rückwärtspaar trägt `Q†`), also ist
`C` **exakt unitär**. Zu erster Ordnung in ε ist das „geradeaus = 1,
60°-Drehung = i·ε·(Spinordrehung)" — dieselbe Amplitudenregel wie bisher, nur
mit der Dreh*phase* ersetzt durch die Dreh*ung*.

| | Dimension | Unitaritätsfehler | G hermitesch |
|---|---|---|---|
| Spin 0 | 12 | 5.6·10⁻¹⁶ | 0.0 |
| Spin ½ | 24 | 6.7·10⁻¹⁶ | 0.0 |

---

## 4. Kramers-Verdopplung ist automatisch

Ruhespektrum bei ε = 0.1 (Vielfachheiten in Klammern):

| Spin 0 (Boson) | Spin ½ (Fermion) |
|---|---|
| +0.4000 (×5) | +0.4210 (×8) |
| 0.0000 (×3) | +0.2309 (×4) |
| −0.4000 (×3) | −0.1900 (×8) |
| −0.8000 (×1) | −0.6928 (×4) |

**Bei Spin ½ sind alle Vielfachheiten gerade. Bei Spin 0 nicht.** Das ist die
Kramers-Entartung — und sie musste hier *nicht eingestellt* werden.

### Und sie gilt bei jedem k, nicht nur bei k = 0

| |k| | 0.0 | 0.1 | 0.3 | 0.6 |
|---|---|---|---|---|
Über den ganzen gescannten Bereich |k| ≤ 1.2 bleibt die Aufspaltung des
untersten Paares (Spin ½) auf Maschinengenauigkeit (max 5.1·10⁻¹⁵). Bei Spin 0 ist der entsprechende Abstand von Ordnung 1.

Das ist der scharfe Fermion-Boson-Unterschied: bei Spin ½ ist **jeder** Zustand
ein Kramers-Dublett. Im 2D-Modell gab es die Verdopplung nur bei k = 0 und nur
am Punkt α = ½.

---

## 5. Der Gürteltrick auf dem Gitter

Geschlossene Wege, die auf dieselbe gerichtete Kante zurückkommen (nur
60°-Drehungen), klassifiziert nach ihrer SU(2)-Holonomie:

| Länge L | SU(2) = +1 | SU(2) = −1 |
|---|---|---|
| 6 | 0 | 4 |
| 8 | 0 | 24 |
| 10 | 0 | 100 |
| 11 | 0 | 176 |
| 12 | 148 | 1296 |

**Die kürzeste geschlossene Richtungsschleife hat L = 6 und bringt den Spinor
mit −1 zurück. Erst bei L = 12 — zwei solche Schleifen — tritt +1 auf.**

Das ist der Gürteltrick, direkt gemessen: 2π gibt −1, 4π gibt +1. Beide
ℤ₂-Klassen sind besetzt, also ist π₁(SO(3)) = ℤ₂ auf dem Gitter realisiert.

Die Holonomiewinkel selbst nehmen 19 verschiedene Werte von φ/2π an, darunter
±0.216347 und ±0.391827 — **nicht quantisiert**. Es gibt keine Windungszahl,
genau wie in `ROADMAP_QCD_3D_de.md` gezeigt. Nur die ℤ₂-Klasse überlebt.

---

## 6. Spin ½ ist masselos: ein Dirac-Kegel mit c/3

### Ruheniveaus linear in ε

| ε | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| unterstes Niveau / ε, Spin 0 | −8.000 | −8.000 | −8.000 | −8.000 | −8.000 |
| unterstes Niveau / ε, Spin ½ | −6.928 | −6.928 | −6.928 | −6.928 | −6.928 |

Für Spin 0 ist das ein massives Band mit Minimum bei k = 0 (m = 8ε). Für Spin ½
ist das unterste Niveau dagegen **vierfach entartet** und spaltet **linear** auf:

```
E(k) = −4√3·ε ± (c/3)·|k| + O(k²)        (zwei Kramers-Dubletts)
```

Gemessen bei k = 10⁻⁴ in den Richtungen [100], [110], [111] und [123]:
Steigungen −0.57748/+0.57722 (ε = 0.1). Der Mittelwert ist c/3 = 0.57735, für
ε = 0.05 und 0.3 ebenso. Der Kegel ist also **isotrop und unabhängig von ε**.
Das ist ein masseloses vierkomponentiges Dirac-Teilchen bei der versetzten
Energie −4√3·ε, das genaue 3D-Gegenstück zum Kramers-Kegel des 2D-Modells bei
α = ½ (Steigung c/2, siehe `RESULTS_PhaseBit_S2_de.md`).

Vergleich mit √(c²k² + m²) und m = 0.69282 (ε = 0.1): Die Abweichung beträgt
+3.8 % bei k = 0.05, +6.6 % bei 0.10, +9.2 % bei 0.20, +6.3 % bei 0.40 und −4.6 %
bei 0.80. Das ist kein relativistisches massives Band, sondern ein Kegel.

**Folgerung:** In 3D liefert die SU(2)-Münze automatisch Spin ½, aber **keine
Masse**. Für ein massives Fermion braucht man einen zusätzlichen Term, der den
Dirac-Punkt öffnet. In 3D ist das, anders als in 2D, auch unter Erhaltung der
Zeitumkehr möglich. Das ist offen.

### Kausalität und Isotropie

`v_g = ⟨Δr⟩/Δt` ist exakt (Hellmann-Feynman für die unitäre Münze), und wegen
`|⟨Δr⟩| ≤ |Schritt|` ist `|v_g| ≤ c` **strukturell garantiert**. Gemessen über
300 zufällige k und alle 24 Bänder:

```
max |v_g| = 1.72239   <   c = 1.73205
```

Isotropie von |v_g| über zufällige 3D-Richtungen:

| Spin | \|k\|=0.2 | \|k\|=0.5 |
|---|---|---|
| 0 | 8.6 % | 19.7 % |
| ½ | 9.0 % | 20.8 % |

Das ist **schlechter als in 2D** (dort 3.5 % bei |k|=0.3). Die kubische
Anisotropie des FCC-Gitters ist stärker als die sechszählige Welligkeit des
Dreiecksgitters. Für kleine k verschwindet sie wie erwartet.

---

## 7. Ausbreitung in beliebige 3D-Richtungen

Gaußsches Paket auf dem FCC-Untergitter, Spin ½, unterster Zweig (Kegel), ε = 0.1,
|k| = 0.5, 24 Schritte:

| k ∥ | σ | \|v_g(k₀)\| | \|v\| gemessen | Winkelfehler | Norm |
|---|---|---|---|---|---|
| (1,0,0) | 4 | 1.0696 | 0.9522 | 0.00° | 0.99995 |
| (1,1,1) | 4 | 1.2599 | 1.0544 | 0.00° | 0.99995 |
| (1,2,3) | 6 | 1.2702 | 1.1511 | 2.19° | 0.99974 |

Die Norm bleibt erhalten. Der Betragsunterschied ist die k-Mittelung des
Pakets. Entlang der Symmetrieachsen ist der Winkelfehler jetzt exakt 0 (vorher
1.65° und 0.71°, eine Folge des inkonsistenten Transports).

Die Winkelfehler sind größer als die 0.07° des 2D-Modells — Folge der
stärkeren FCC-Anisotropie und der kürzeren Läufe (24 statt 120 Schritte, weil
ein 3D-Gitter mit 91³ Plätzen × 24 Komponenten schon 200 MB belegt).

---

## 8. Eine gefundene Einschränkung

**Das Spektrum ist nicht teilchen-antiteilchen-symmetrisch.** Es gibt keine
±E-Paare: bei Spin ½ ist der unterste Wert −0.6928, aber es gibt kein +0.6928.

Der Grund ist strukturell: das Spektrum einer Adjazenzmatrix ist genau dann
symmetrisch, wenn der Graph **bipartit** ist. Der Richtungsgraph des
2D-Modells war ein 6-Zyklus — bipartit, daher `E_m = −4ε cos(π(α−m)/3)`
symmetrisch. Der Kuboktaeder enthält **Dreiecke** und ist damit nicht
bipartit.

Das gilt auch nach der Korrektur (max |E(k) − (−E)(k)| = 0.217).

**Gelöst in Paper 3, Experiment S3** (`RESULTS_PhaseBit_S3_de.md`): Ersetzt man
den kontinuierlichen Transport durch Vierteldrehungen aus der binären
Oktaedergruppe 2O, trägt jedes Richtungs-Dreieck die Holonomie π (spurlos), und
das Spektrum wird bei jedem k exakt symmetrisch (3.6·10⁻¹⁵). Die Grenze lag also
nicht an der Geometrie, sondern an der kontinuierlichen Verteilung der Krümmung.

---

## 9. Was Schritt 2 zeigt

| Frage | 2D-Modell | 3+1D-FCC |
|---|---|---|
| Spin | freier Parameter α ∈ ℝ/ℤ | **nur 0 oder ½** |
| Kramers-Verdopplung | nur bei α=½ und nur bei k=0 | **automatisch, bei jedem k** |
| 180°-Umkehr | Konvention nötig | kommt nicht vor |
| Windungszahl | w ∈ ℤ | existiert nicht; nur ℤ₂ |
| 2π-Drehung | (−1)^w bei α=½ | −1, direkt gemessen (L=6) |
| Masse | 4ε·cos(πδ/3), masselos bei α=½ (Kegel c/2) | 8ε (Boson); Fermion **masselos** (Dirac-Kegel c/3) |
| Isotropie bei \|k\|≈0.2–0.3 | 3.5 % | 8.6–9.0 % |

**Der Kern: In 2D musste Spin ½ gewählt werden. In 3D bleibt nichts anderes
übrig.** Das war die Vermutung, und das Modell macht sie jetzt konstruktiv —
nicht nur als topologisches Argument, sondern als lauffähige Zeitentwicklung.

---

## 10. Offene Punkte

* **Masse für Spin ½** (Abschnitt 6): Ein Term, der den Dirac-Punkt öffnet und
  T erhält, fehlt noch.
* **Ladungskonjugation** (Abschnitt 8): gelöst mit dem 2O-Register (Paper 3, S3).
* **Speicher.** Ein 3D-Gitter mit 121³ Plätzen × 24 Komponenten wären 800 MB
  in complex64. Längere Läufe brauchen entweder Streaming, ein Ausnutzen der
  FCC-Parität (nur die Hälfte der Plätze ist besetzt) oder eine
  k-Raum-Formulierung.
* **Magnetfeld in 3D.** Der Peierls-Faktor überträgt sich direkt; interessant
  wäre, ob der Berry-Versatz γ = 0 gegen ½ auch hier Boson und Fermion trennt.
* **Schritt 3** laut Roadmap: der Zwei-Teilchen-Raum — löst Ausschließlichkeit
  und Farbsingulett gemeinsam.
