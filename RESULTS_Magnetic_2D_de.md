# Magnetfeld und Chiralitätsfamilien im Winkelsummen-Modell

*Dateien:* `quantum_hex_magnetic.py` (Modell), `quantum_hex_magnetic_figs.py` (Auswertung)
*Baut auf:* `RESULTS_Turning_2D_de.md`

---

## 1. Konstruktion

Zum Winkelsummen-Modell kommt ein ortsabhängiger Peierls-Faktor. Ein Schritt, der
bei `r` startet und um `dr` versetzt, bekommt zusätzlich

```
exp( i * Integral A.dl )  =  exp( i * (B/2) * (x*dy - y*dx) )     A = (B/2)(-y, x)
```

Die Mittelpunktsregel ist hier **exakt**, weil sich die `dx*dy/2`-Kreuzterme
wegheben. Die Summe dieser Phase über einen geschlossenen Weg ist genau
`B * (vorzeichenbehaftete eingeschlossene Fläche)` — die Gaußsche Trapezformel.

Das ist damit **dieselbe Größe** wie die Flächenbilanz der geschlossenen Wege aus
dem Winkelsummen-Modell: Windungszahl und eingeschlossene Fläche sind die beiden
Erhaltungsgrößen einer geschlossenen Bahn, und `alpha` bzw. `B` sind die beiden
Flüsse, die daran koppeln.

### Verifikation

| Weg | Fluss/B gemessen | exakt |
|---|---|---|
| kleinstes Dreieck | +0.324759526 | +3√3/16 = +0.324759526 |
| Sechseck | +1.948557159 | +1.948557159 |
| Dreieck im Gegensinn | −0.324759526 | −0.324759526 |
| Raute | +0.649519053 | +0.649519053 |

* **Unitarität:** exakt erhalten (Diagonalphase × Verschiebung × unitäre Münze);
  Norm `1.000000000000` über 8 Schritte mit Zufallszustand.
* **Eichinvarianz:** die Dichte im symmetrischen Feld `A=(B/2)(−y,x)` und im
  Landau-Feld `A=B(−y,0)` (verbunden durch `chi = −Bxy/2`) stimmen auf
  **1.7e-15** relativ überein. Das schließt Implementierungsfehler aus.

---

## 2. Zyklotronbahnen — die Chiralitätsfamilien trennen sich

| k | B | R gemessen | k/B | Verhältnis | Norm |
|---|---|---|---|---|---|
| 0.3 | 0.0150 | 19.02 | 20.00 | 0.951 | 1.000000 |
| 0.4 | 0.0160 | 23.58 | 25.00 | 0.943 | 1.000000 |
| 0.4 | 0.0114 | 33.16 | 35.00 | 0.947 | 0.999999 |
| 0.5 | 0.0167 | 28.11 | 30.00 | 0.937 | 1.000000 |
| 0.6 | 0.0150 | 37.70 | 40.00 | 0.942 | 0.999999 |

Der Radius folgt `R = k/B` mit einem systematischen Defizit von ~6 %: der
gemessene Schwerpunkt läuft auf einem etwas kleineren Kreis als die ideale Bahn,
weil das Paket während des Umlaufs zerfließt und der Schwerpunkt nach innen
gezogen wird. Der Umlaufsinn kippt exakt mit dem Vorzeichen von B — die beiden
Chiralitätsfamilien, die im feldfreien Modell nur kombinatorisch unterscheidbar
waren, sind jetzt **dynamisch getrennt**.

### Ordnungsparameter

```
L_z = sum_{r,d} |w_d(r)|^2 * (x*dy_d - y*dx_d) / dt
```

(zweimal die Rate, mit der die Wahrscheinlichkeit Fläche überstreicht).

| B | L_z(t=60), bandprojiziertes ruhendes Paket |
|---|---|
| 0.00 | −0.0000 |
| +0.03 | −2.36 |
| −0.03 | **+2.36** (exakt gespiegelt) |
| +0.06 | −3.01 |

### Ein wichtiger Nebenbefund

Eine **nackte angeregte Kante** — der ursprüngliche Startzustand der Idee —
besetzt alle sechs Bänder mit **exakt 1/6** Gewicht. Teilchen- und lochartige
Bänder umlaufen gegensinnig, ihre Beiträge löschen sich exakt aus:

```
L_z (nackte Kante, B = +0.05, t = 60)  =  -1.7e-14
```

**Die Chiralitätsfamilien trennen sich erst, wenn man ein Band auswählt**, also
ein Vorzeichen der Ladung festlegt. Das ist keine numerische Feinheit, sondern
die Aussage, dass Teilchen und Antiteilchen im Feld gegensinnig umlaufen.

### Grenze des semiklassischen Bildes

Bei `eps = 0.1` ist die effektive Masse winzig (`m* = 0.087`), also ist
`hbar*omega_c = B/m*` **größer** als die Bandlücke — das Paket spaltet sich in
einen gekrümmten und einen geradeaus laufenden Teil auf (magnetischer
Durchbruch). Die Rechnungen oben laufen deshalb bei `eps = 0.5`, wo
`omega_c/Lücke = 0.05` ist.

| eps | m_Ruhe | Lücke(k=0) | m*_eff | omega_c/Lücke bei R=25 |
|---|---|---|---|---|
| 0.1 | 0.400 | 0.200 | 0.087 | 0.91 |
| 0.2 | 0.800 | 0.400 | 0.144 | 0.28 |
| **0.5** | 2.000 | 1.000 | 0.345 | **0.05** |
| 0.8 | 3.200 | 1.600 | 0.566 | 0.02 |

---

## 3. Die stehende Welle auf der Umlaufbahn

Die Idee: eine zusätzliche Welle *entlang* der Bahn, die nur bei bestimmten
Bahngrößen konstruktiv beiträgt, gibt eine Energie, die die Bahngröße abbildet.

Das ist exakt die **Onsager-Quantisierung** (Bohr-Sommerfeld für Bahnen im
Magnetfeld). Gemessen wurde sie zeitaufgelöst: aus einem lokalisierten Startzustand
wird `c(t) = <psi_0|psi(t)>` aufgezeichnet und Fourier-transformiert — Spitzen
stehen genau dort, wo eine Bahn sich selbst verstärkt.

### Gemessene Niveaus (eps=0.5, B=0.05)

| alpha | Niveaus E_n |
|---|---|
| 0.0 | 2.0716, 2.1921, 2.2967, 2.3913, 2.4776, 2.5578 |
| 0.5 | 1.7727, 2.0436, 2.1866, 2.3017, 2.4014, 2.4914 |

Die Abstände sind **nicht** gleich (0.120, 0.105, 0.095, 0.086, 0.080): die
effektive Masse wächst mit der Energie, das Band ist kein Parabel-Band. Der
richtige, parameterfreie Test ist die eingeschlossene Fläche im k-Raum:

```
A_k(E_n) / (2*pi*B)  =  n + gamma
```

| alpha | A_k/(2πB) | Steigung (erwartet 1) | gamma |
|---|---|---|---|
| 0.0 | 0.517, 1.516, 2.513, 3.516, 4.511, 5.506 | **0.9980** | **0.518** |
| 0.5 | 0.021, 1.017, 2.016, 3.016, 4.013, 5.014 | **0.9985** | **0.020** |

**Die Quantisierung ist auf 0.2 % exakt.** Und der Versatz `gamma` springt bei
`alpha = 1/2` von 1/2 auf 0.

Das ist die Berry-Phase π des Dirac-Punktes — und sie ist **derselbe
Spinorfaktor** `(-1)^w` wie die Windungszahl-Phase aus dem Winkelsummen-Modell.
Die beiden Hälften des Modells treffen sich hier: was dort die Schleifenphase
`exp(i*2*pi*alpha*w)` war, ist hier der halbzahlige Versatz der
Bahnquantisierung.

### Warum das dem früheren Nulltest nicht widerspricht

Im Winkelsummen-Modell wurde gezeigt: eine konstante Zusatzphase `delta` pro
Schritt verschiebt nur `E -> E - delta/dt` und bewegt nichts. Das bleibt richtig.
Beides zusammen ergibt aber genau die hiesige Aussage: `E` ist der Regler, der
auswählt, **welche** geschlossene Bahn sich verstärkt. Die konstante Phase pro
Schritt bewegt kein Paket, aber sie stimmt die Bahnresonanz um.

---

## 4. Richtung aus Überlagerung (Fourier)

Die energiegefilterten Zustände

```
phi_n = sum_t w(t) * exp(i*E_n*t*dt) * psi(t)      (Hann-Fenster)
```

sind **stationäre Ringe** — stehende Wellen ohne Richtung:

| n | E_n | ⟨r⟩ | Dichtedrift über 40 Schritte |
|---|---|---|---|
| 0 | 2.072 | 7.11 | 0.3 % |
| 1 | 2.192 | 10.28 | 0.2 % |
| 2 | 2.297 | 12.75 | 0.6 % |
| 3 | 2.391 | 14.86 | 1.0 % |
| 4 | 2.478 | 16.70 | — |
| 5 | 2.558 | 18.31 | — |

Der Bahnradius wächst wie `sqrt(n + 1/2)` — die Energie bildet also tatsächlich
die Bahngröße ab.

Die **kohärente Summe** dieser sechs richtungslosen Ringe ist ein lokalisiertes
Paket, das die Bahn abläuft:

```
|<r>|  läuft von 0.71 bis 16.83 und kehrt bei t = 67.5 zum Start zurück
```

Die Bahn schließt sich zu einem Kreis (Abbildung `magnetic_standing_waves.png`,
unten rechts). **Aus der Überlagerung stehender Wellen verschiedener Wellenlänge
entsteht eine Richtung** — genau wie vermutet.

Die Bahn spiralt dabei leicht nach innen: weil die Niveauabstände nicht exakt
gleich sind (Abschnitt 3), ist die Überlagerung kein exakter kohärenter Zustand
und dephasiert über wenige Umläufe. In einem exakt parabolischen Band wäre der
Umlauf periodisch.

---

## 5. Abbildungen

| Datei | Inhalt |
|---|---|
| `magnetic_geometry.png` | Peierls-Konstruktion, Flussverifikation, Eichinvarianz, Links-/Rechts-Familien vs. B |
| `magnetic_orbits.png` | Zyklotronbahnen, Spiegelbahnen bei ±B, R vs k/B, Ordnungsparameter L_z |
| `magnetic_landau.png` | Zeitaufgelöstes Spektrum, Niveauabstände, Onsager-Gerade mit gamma = 1/2 vs 0 |
| `magnetic_standing_waves.png` | Stehende Ringwellen, ⟨r⟩ vs n, kohärente Summe läuft die Bahn |

---

## 6. Offene Punkte

* **Der 6 %-Defizit bei R = k/B** ist bisher nur als Zerfließeffekt plausibel
  gemacht, nicht quantitativ hergeleitet.
* **Landau-Entartung**: der Entartungsgrad pro Niveau (Flussquanten pro Fläche)
  wurde nicht gemessen; dafür bräuchte es eine Diagonalisierung auf endlichem
  Gebiet oder eine magnetische Einheitszelle bei rationalem Fluss
  (Hofstadter-Spektrum). Das wäre der natürliche nächste Schritt und würde auch
  den Übergang zum Hofstadter-Schmetterling auf diesem Gitter zeigen.
* **Kollisionen mit Ausschließlichkeit** bleiben offen (siehe
  `RESULTS_Turning_2D_de.md`, Abschnitt 10) — das Modell ist weiterhin linear.
