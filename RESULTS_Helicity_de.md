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

## 7. Offen

* **Ein Modell ohne Heading-Kopplung.** Man könnte den inneren Raum künstlich
  vergrößern (mehrere Kopien pro Richtung), sodass zwei Teilchen denselben
  „Flavour"-Zustand bei entgegengesetztem Impuls tragen können. Ob das eine
  konsistente unitäre Dynamik ergibt oder nur ein Etikett ohne Wirkung ist,
  wurde nicht geprüft.
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
