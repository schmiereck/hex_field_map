# Hebt echter Spin die Deckelung der Austauschinterferenz auf?

*Dateien:* `quantum_helicity_test.py`, `quantum_helicity_test_figs.py`
*Anschluss an:* `RESULTS_Scatter_2D_de.md`
*Abbildung:* `helicity_test.png`

---

## 1. Die Frage

In Schritt 4 war die Mott-Interferenz vorhanden, aber schwächer als im
Lehrbuch. Die Ursache: der Bandeigenvektor ist **an den Impuls gekoppelt** —
er ist helizitätsartig. Zwei gegenläufige Teilchen haben deshalb nur den
Überlapp `|⟨u(k)|u(−k)⟩| < 1`, und die Austauschinterferenz wird entsprechend
schwächer.

Naheliegende Vermutung: das liegt daran, dass der innere Index im
2D-Modell *nur* die Bewegungsrichtung ist. Ein Modell mit einem **vom Impuls
unabhängigen** inneren Freiheitsgrad — echtem Spin zusätzlich zum Heading, wie
im 3+1D-FCC-Modell — sollte die volle Mott-Nullstelle liefern.

**Ergebnis: nein. Spin hilft nicht. Aber der nichtrelativistische Grenzfall
schon.**

---

## 2. Eine Zwischenhypothese, die sich nicht bestätigt hat

Für einen kontinuierlichen Dirac-Spinor gilt exakt

```
u†(k,s) u(-k,s) = m/E = 1/gamma
```

Wenn das auch hier gälte, wäre die Deckelung schlicht Relativistik und keine
Modellschwäche. Geprüft (2D, α=0):

| ε | k | m/E | gemessener Überlapp | Verhältnis |
|---|---|---|---|---|
| 0.5 | 0.1 | 0.9928 | 0.9705 | 0.978 |
| 0.5 | 0.4 | 0.9085 | 0.6922 | 0.762 |
| 0.5 | 1.0 | 0.6961 | 0.2910 | 0.418 |
| 0.2 | 0.4 | 0.6924 | 0.3147 | 0.455 |
| 0.2 | 1.0 | 0.3963 | 0.0697 | 0.176 |

**Der Überlapp fällt deutlich schneller als 1/γ.** Das 6-komponentige Heading
ist kein Dirac-Spinor; zur relativistischen Unterdrückung kommt hinzu, dass
sich die Richtungsverteilung mit wachsendem k immer stärker auf die
Vorwärtsrichtung konzentriert.

Bestätigt wird nur der masselose Endpunkt: bei α = ½ (Dirac-Kegel, m = 0) ist
`m/E = 0`, und der gemessene Überlapp ist **exakt 0** (3·10⁻¹⁶ bei jedem k).

---

## 3. Der eigentliche Test: hilft Spin?

Im FCC-Modell ist der massive Zweig ein **Kramers-Dublett** bei jedem k. Die
richtige Größe sind deshalb die Singulärwerte der 2×2-Überlappmatrix zwischen
dem Dublett bei +k und bei −k. Zum Vergleich die **Heading-Fidelity**
`Σ_d √(p_d(+k)·p_d(−k))` — der klassische Überlapp der beiden
Richtungsverteilungen mit ausgespurtem Spinor.

(ε = 0.1)

| k | Dublett-Singulärwerte | Heading-Fidelity | FCC Spin 0 | 2D-Vergleich |
|---|---|---|---|---|
| 0.05 | 0.8897, 0.8897 | 0.9114 | 0.9699 | 0.9924 |
| 0.10 | 0.6927, 0.6927 | 0.7497 | — | 0.9705 |
| 0.20 | 0.4158, 0.4158 | 0.5078 | 0.6921 | 0.8938 |
| 0.40 | 0.1949, 0.1949 | 0.2834 | — | 0.6922 |
| 0.80 | 0.0723, 0.0723 | 0.1283 | 0.1675 | 0.3855 |

Drei Befunde:

1. **Beide Singulärwerte sind gleich** (Kramers-Entartung). Es gibt also
   keinen „guten" und keinen „schlechten" Kanal im Dublett — man kann sich
   keinen hochüberlappenden Spinzustand aussuchen.
2. **Die Singulärwerte liegen unter der Heading-Fidelity.** Der Flaschenhals
   ist die Richtungsverteilung, und der Spinor verschlechtert das Ergebnis
   zusätzlich, weil er beim Transport mitgedreht wird.
3. **Spin 0 hat durchweg höheren Überlapp als Spin ½** (0.692 gegen 0.416 bei
   k = 0.2). Der zusätzliche Freiheitsgrad macht es also **schlechter**, nicht
   besser.

Der Grund ist strukturell und nicht reparierbar: **der innere Index *ist* die
Bewegungsrichtung.** Kein zusätzliches Etikett ändert daran etwas.

---

## 4. Was die volle Mott-Nullstelle doch liefert

Alle Kurven laufen bei k → 0 gegen 1 — dort wird die Richtungsverteilung
isotrop. Getestet mit einem echten Streulauf bei ε = 1.5, k = 0.3, wo der
Überlapp 0.965 beträgt (gegen 0.794 in Schritt 4):

| Ringradius | 9–14 | 14–18 | 18–22 | 22–26 | 26–31 | 31–35 |
|---|---|---|---|---|---|---|
| Kontrast bei 90° | **+0.9926** | +0.9937 | +0.9932 | +0.9917 | +0.9896 | +0.9854 |

```
bester Kontrast 0.9937   bei Ueberlapp 0.9650   (Norm 0.997 erhalten)
Fermion-Querschnitt bei 90 Grad = 0.32 % des bosonischen
```

Der Kontrast ist über alle Radien stabil — anders als im Fall mit niedrigem
Überlapp, wo er mit dem Radius dephasierte. Zum Vergleich im selben Lauf, mit
sauberer Norm gemessen:

| | Überlapp | Kontrast | Verhältnis | Norm |
|---|---|---|---|---|
| ε = 1.5, k = 0.3 | 0.9650 | **0.9937** | 1.030 | 0.997 |
| ε = 0.5, k = 0.3 | 0.7943 | 0.7893 | 0.994 | 0.951 |

**Das ist die volle Mott-Nullstelle.** Sie wird nicht durch zusätzlichen Spin
erreicht, sondern durch den nichtrelativistischen Grenzfall.

Der Preis: bei kleinem k ist die Gruppengeschwindigkeit klein (v_g = 0.235),
das Paket muss groß sein (σ ≥ 1/k) und der Lauf lang — 260 Schritte auf einem
407×235-Gitter statt 91 auf 241×141.

---

## 5. Eine Korrektur an Schritt 4

In `RESULTS_Scatter_2D_de.md` steht, der Kontrast sei durch den Überlapp
**gedeckelt**. Das ist zu stark formuliert. Die gemessenen Verhältnisse
Kontrast/Überlapp sind:

```
0.95   0.88   0.78   1.12   0.99   1.03
```

Sie streuen um 1 und überschreiten sie mehrfach. Richtig ist: **der Kontrast
folgt dem Überlapp als Skalierungsgesetz**, nicht als strenge Schranke.

---

## 6. Ergebnis

| Frage | Antwort |
|---|---|
| Ist die Deckelung einfach Relativistik (1/γ)? | Nein — der Überlapp fällt schneller |
| Hebt ein impulsunabhängiger Spin sie auf? | **Nein** — Spin ½ ist schlechter als Spin 0 |
| Woran liegt es dann? | Der innere Index *ist* die Bewegungsrichtung |
| Gibt es die volle Mott-Nullstelle? | **Ja**, im nichtrelativistischen Grenzfall: Kontrast 0.9937, Fermion bei 90° = 0.32 % des Bosons |
| Bei α = ½? | Nie — Überlapp exakt 0 bei jedem k |

---

## 7. Nachtrag: Flavour hilft nicht — und zwar beweisbar nicht

Die in Abschnitt 6 offen gelassene Idee war, den inneren Raum künstlich zu
vergrößern (mehrere Kopien pro Richtung), sodass zwei Teilchen bei
entgegengesetztem Impuls denselben „Flavour" tragen können.

### Ein passives Etikett trägt Faktor 1 bei

Wird die Münze zu `C ⊗ 1_f`, faktorisiert der Bandeigenvektor: `u(k) ⊗ χ`.
Dann ist `⟨u(k)⊗χ | u(−k)⊗χ⟩ = ⟨u(k)|u(−k)⟩ · ⟨χ|χ⟩`. Gemessen für N_f = 2
und 4:

| k | ohne Flavour | mit N_f=2 | mit N_f=4 |
|---|---|---|---|
| 0.3 | 0.7943075987 | 0.7943075987 | 0.7943075987 |
| 0.6 | 0.5159907613 | 0.5159907613 | 0.5159907613 |

Identisch auf zehn Stellen. Das Etikett ändert nichts.

### Es geht allgemeiner: eine kinematische Schranke

Man muss gar nicht ausprobieren. Mit den Randverteilungen der Richtungen
`p_d`, `q_d`, `a_d = √p_d`, `b_d = √q_d` und `c_d = n_d·v̂ ∈ [−1,1]` gilt

```
|<u|v>|  <=  sum_d a_d b_d  =:  F           (Cauchy-Schwarz je Richtung)

2 beta = sum_d c_d (a_d - b_d)(a_d + b_d)
       <= max|c_d| * sqrt(sum (a-b)^2) * sqrt(sum (a+b)^2)
       <= sqrt(2-2F) * sqrt(2+2F) = 2 sqrt(1-F^2)

=>   F <= sqrt(1 - beta^2) = 1/gamma
```

**Der Austauschüberlapp zweier Zustände mit den Geschwindigkeiten ±v ist durch
1/γ beschränkt — für jedes Gitter und jeden inneren Raum.** Der Grund: die
Schranke hängt nur von den Richtungs-Randverteilungen ab, und die sind durch
die Geschwindigkeit festgenagelt. Kein zusätzlicher Index kann das umgehen.

Numerisch bestätigt: die optimierte Fidelity trifft √(1−β²) auf **9·10⁻¹³**
(2D-Dreiecksgitter entlang einer Achse). Gleichheit verlangt `max|c_d| = 1`,
also Gitterrichtungen exakt entlang ±v — auf FCC entlang x zeigt keine
Richtung dorthin, dort liegt das Optimum echt darunter (0.977 statt 0.989).

### Der Dirac-Spinor sättigt die Schranke

`u†(k,s)u(−k,s) = m/E = 1/γ` — genau der Wert der Schranke. **Dirac ist
optimal.** Meine frühere Feststellung „die Dirac-Hypothese ist widerlegt"
bekommt damit ihre richtige Deutung: der Dirac-Wert ist nicht der *erwartete*,
sondern der *maximal mögliche*, und die Gittermodelle bleiben darunter.

| Modell | β | Überlapp | 1/γ | Ausschöpfung |
|---|---|---|---|---|
| 2D, ε=1.5 | 0.046 | 0.9960 | 0.9989 | 0.997 |
| 2D, ε=0.5 | 0.166 | 0.9705 | 0.9862 | 0.984 |
| FCC Spin 0 | 0.255 | 0.8889 | 0.9669 | 0.919 |
| FCC Spin ½ | 0.381 | 0.6896 | 0.9247 | 0.746 |
| 2D, ε=0.5 | 0.731 | 0.2910 | 0.6824 | 0.426 |

Keine einzige Messung verletzt die Schranke.

### Was daraus folgt

Die Deckelung ist **kein Modelldefekt und nicht reparierbar**. Sie ist
kinematisch: zwei Teilchen, die sich mit ±v bewegen, *sind* durch ihre
Bewegung teilweise unterscheidbar, und zwar genau um den Faktor 1/γ. Die volle
Mott-Nullstelle existiert nur im Grenzfall v → 0 — was die Messung aus
Abschnitt 4 (Kontrast 0.9937 bei β = 0.136) genau bestätigt.

*Abbildung:* `overlap_bound.png`

---

## 8. Offen

* ~~Ein Modell ohne Heading-Kopplung~~ — erledigt, siehe Abschnitt 7: ein
  Etikett ist wirkungslos, und die Schranke 1/γ gilt für jeden inneren Raum.
  Was offen bleibt, ist nur noch, wie nah ein Gittermodell an die
  Dirac-Sättigung herankommen kann; die hier gebauten schöpfen zwischen 43 %
  und 99.7 % aus.
* **Bandumkehr.** Bei ε = 1.5 hat das Teilchenband bei k ≈ 0.87 ein Maximum:
  `v_g` kippt dort von +0.459 (k=0.80) über +0.056 (k=0.87) auf −0.278
  (k=0.90). Jenseits davon läuft der Zustand rückwärts und „gegenläufig"
  verliert seinen Sinn; die Überlappkurven sind in der Abbildung dort
  abgeschnitten (Dreieck-Marker). Das ist keine Bandkreuzung — der Abstand zum
  nächsten Band bleibt konstant bei 3.24.
* **Randverlust.** Ein früherer Referenzlauf verlor 60 % der Norm über den
  Gitterrand; die hier angegebenen Zahlen stammen aus Läufen mit 0.997 bzw.
  0.951 erhaltener Norm. Absorbierende Ränder wären der nächste technische
  Schritt.
