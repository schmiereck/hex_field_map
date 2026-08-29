# Winkelsummen-Modell auf dem 2D-Dreiecksgitter (Turning-Phase Model)

*Dateien:* `quantum_hex_turning.py` (Modell), `quantum_hex_turning_figs.py` (Auswertung)

---

## 1. Die Idee

Eine gerichtete Kante wird angeregt. Die Welle läuft über das Gitter und ändert
in jedem Schritt ihre Richtung um ein Vielfaches von 60°. Als Pfadgewicht wird
die **Summe der durchlaufenen Winkel** verwendet:

```
w_Pfad = prod_Schritte  a(|n|) · exp(i · alpha · 60° · n)
```

`n` = vorzeichenbehaftete Anzahl der 60°-Einheiten, um die in diesem Schritt
gedreht wurde. Die Schrittzahl `N` bleibt die Uhr der Welle.

### Kante oder Knoten?

Die Frage löst sich auf: **gerichtete Kante = (Knoten, Richtung)**. Beides ist
dieselbe Darstellung, und es ist genau das `amp[x,y,d]` des bestehenden Modells.
Ein reines Skalarfeld auf Knoten *kann* die Amplitudenregel nicht ausdrücken,
weil diese von der vorherigen Richtung abhängt — es liefert Klein-Gordon
(spinlos), nicht Dirac. Für ein Feld mit Spin ist die gerichtete Kante
zwingend.

---

## 2. Ergebnis 1: Die Winkelsumme ist quantisiert — und deshalb allein wertlos

Für **jeden** geschlossenen Weg, der auf dieselbe gerichtete Kante zurückkommt,
gilt exakt

```
Sigma(theta) = 360° · w  ,   w in Z   (Windungszahl)
```

Das ist kein Zufall, sondern der diskrete Whitney'sche Umlaufsatz: Endrichtung =
Anfangsrichtung, also ist die aufsummierte Richtungsänderung ein Vielfaches von
360°. Numerisch verifiziert für **alle** geschlossenen Wege bis Länge 9:

| Länge L | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| Wege gesamt | 2 | 4 | 10 | 74 | 280 | 1244 | 5840 |
| Verletzungen des Satzes | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Konsequenz:** Mit dem Gewicht `exp(i·Sigma(theta))` bekommt jede Schleife exakt
die Phase 1 — es gäbe überhaupt keine Interferenz, nur Pfadzählung. Der
Skalenfaktor `alpha` ist das, was die Konstruktion physikalisch macht.

### Windungszahl-Familien

Die Verteilung der Windungszahlen ist der eigentliche Inhalt:

| L | w=-2 | w=-1 | w=0 | w=+1 | w=+2 |
|---|---|---|---|---|---|
| 3 | – | 1 | – | 1 | – |
| 4 | – | 2 | – | 2 | – |
| 5 | – | 5 | – | 5 | – |
| 6 | 1 | 24 | **24** | 24 | 1 |
| 7 | 7 | 84 | 98 | 84 | 7 |
| 8 | 34 | 348 | 480 | 348 | 34 |
| 9 | 213 | 1563 | 2286 | 1563 | 213 |

Die vermuteten **Familien existieren wirklich**: bis L=5 gibt es *ausschließlich*
w=±1 (reine Links- bzw. Rechtsbahnen — das Dreieck aus 3×120° und seine
Verlängerungen). Erst ab L=6 treten w=0-Schleifen auf: das sind genau die
**Achten** (Dreieck links + Dreieck rechts). Ab L=9 kommt w=±3 dazu.

---

## 3. Ergebnis 2: alpha ist ein Aharonov-Bohm-Fluss im Richtungsraum

Die Schleifenphase ist

```
exp(i · alpha · Sigma(theta)) = exp(i · 2*pi * alpha * w)
```

* **alpha ist nur mod 1 physikalisch.** Für ganzzahliges alpha lässt sich die
  Phase durch `psi_d -> exp(i*alpha*60°*d) * psi_d` wegeichen. Diese Eichung ist
  aber nur global definiert, wenn `alpha` ganzzahlig ist (weil `d` mod 6 lebt).
  Genau darin liegt die Physik: `alpha` ist ein Fluss, die Windungszahl `w` ist
  die davon umschlossene Ladung. Numerisch bestätigt: Spektrum bei alpha und
  alpha+1 identisch.
* **alpha = 1/2 gibt (-1)^w**: eine 2pi-Drehung liefert -1. Das ist die
  Spinor-Doppelüberlagerung, also Spin 1/2.
* Symmetrie `alpha <-> -alpha` (mod 1): die einzigen selbstkonjugierten Punkte
  sind alpha = 0 und alpha = 1/2 — die beiden zeitumkehr-symmetrischen Flüsse
  (0 und pi).

---

## 4. Das Modell: ein exakt unitärer Münzoperator

Statt Gewichte von Hand zu setzen:

```
C = exp( i * eps * G_alpha ) ,    G_alpha = e^{i*pi*alpha/3} R + h.c.
```

mit `R` = Drehung des Headings um +60°. Eigenschaften:

* **exakt unitär** (Fehler 2.2e-16) — löst das bekannte `|lambda| > 1`-Problem
  des bisherigen 2+1D-Modells. Die Norm bleibt im Test über 120 Schritte auf
  `1.00000000` erhalten.
* zu erster Ordnung in `eps`: „geradeaus = 1, ±60°-Drehung = i·eps" — also die
  bekannte Amplitudenregel.
* Mehrfachdrehungen und die 180°-Umkehr entstehen konsistent aus der
  Exponentialfunktion; die Vorzeichen-Mehrdeutigkeit von ±180° verschwindet auf
  Operatorebene.

Zum Vergleich sind zwei nicht-unitäre Varianten implementiert: `mode="graded"`
(`w(n) = (i*eps)^|n| · exp(...)`) und `mode="flat"`. Letztere reproduziert bei
`alpha=0` exakt die Amplitudenregel von `quantum_hex_2d.py` (ohne die
Ruhe-Richtung): Münz-Eigenwerte `1+5i*eps` (einfach) und `1-i*eps` (5-fach) —
die bekannte 5-fache Entartung.

---

## 5. Ergebnis 3: Ruhespektrum — analytisch exakt

Bei k=0 ist die Transfermatrix zirkulant. Mit der Drehimpulsquantenzahl
`m = 0..5`:

```
E_m(k=0)  =  -4 * eps * cos( pi * (alpha - m) / 3 )
```

Numerik vs. Analytik: `max |E_num - E_ana| = 2.4e-15`.

`alpha` erscheint **ausschließlich** in der Kombination `(alpha - m)` — es
verschiebt also das Drehimpulsspektrum, genau wie ein Fluss es tut.

### Kramers-Verdopplung bei alpha = 1/2

Anzahl verschiedener Ruheniveaus:

| alpha | 0 | 1/4 | **1/2** |
|---|---|---|---|
| verschiedene Niveaus | 4 (2 Dubletts + 2 Singuletts) | 6 (alle einfach) | **3 (drei Dubletts)** |

Nur bei `alpha = 1/2` paaren sich **alle sechs** Niveaus. Das ist die
Kramers-Entartung — das Kennzeichen von Spin 1/2. Die Winkelsumme *ist* der
Spin, sofern man sie mit 1/2 skaliert.

---

## 6. Ergebnis 4: alpha ist ein Massenregler

Das war nicht erwartet und ist der interessanteste Befund.

| alpha | oberstes Band bei k -> 0 | Interpretation |
|---|---|---|
| 0 … 0.49 | `dE/dk -> 0`, quadratisch | **massiv**, gapped |
| **0.5** | `dE/dk -> 0.866020` | **masselos**, Dirac-Kegel |
| 0.51 … 1 | `dE/dk -> 0`, quadratisch | massiv (symmetrisch zu 1-alpha) |

Die Ruheenergie des obersten Bandes ist analytisch

```
m(alpha) = 4 * eps * cos( pi * delta / 3 ) ,   delta = Abstand von alpha zur nächsten ganzen Zahl
```

und variiert nur zwischen `4*eps` (alpha=0) und `2*sqrt(3)*eps ≈ 3.46*eps`
(alpha=1/2). Das **Band** ändert dagegen bei alpha=1/2 schlagartig seinen
Charakter: die dort erzwungene Kramers-Entartung spaltet in k **linear** statt
quadratisch auf und öffnet einen Kegel.

**Die Kegelsteigung ist exakt `sqrt(3)/2 = c/2 = 0.866025`, unabhängig von eps**
(gemessen 0.866020 bei eps=0.05, 0.1 und 0.2).

Physikalische Lesart: In Feynmans Schachbrett entsteht Masse durch Drehungen.
Gibt man den Drehungen eine Zusatzphase, stört diese die Massenerzeugung — und
bei `alpha = 1/2`, wo Links- und Rechtsschleifen mit entgegengesetztem Vorzeichen
eingehen, hebt sie sich exakt auf. Die Winkelsumme ist damit gleichzeitig
Spin *und* Massenparameter.

---

## 7. Ergebnis 5: Das bewegte Teilchen

### Vollständige Klassifikation der „Zusatzphase pro Schritt"

| Phase hängt ab von … | physikalisches Objekt | bewegt sich? |
|---|---|---|
| **nichts** (konstantes delta pro Schritt, auch nur bei jedem n-ten) | Energie-Offset | **nein** |
| **Drehwinkel** `n` | alpha = Fluss im Richtungsraum (Spin, Masse) | nein |
| **Schrittvektor** `dr` → `k·dr` | Impuls (Peierls-Phase) | **ja, in beliebige Richtung** |
| **Ort** `r`, mit Rotation | Magnetfeld B | ja, Kreisbahnen |

### Nulltest (bestätigt)

Ein richtungsunabhängiges `delta` pro Schritt multipliziert **jeden** Eigenwert
mit demselben Faktor: `E -> E - delta/dt`. Die Dichte `|psi|²` ist invariant.
Gemessen bei delta = 0, 0.5, 1.5:

| delta/Schritt | E | Bahn |
|---|---|---|
| 0.0 | 2.6034 | identisch |
| 0.5 | 1.6034 | identisch |
| 1.5 | -0.3966 | identisch |

Die Energie verschiebt sich exakt um `delta/dt = 2*delta`, die Trajektorien
liegen exakt aufeinander. **Ein konstanter Winkelzuschlag pro Schritt kann kein
Teilchen bewegen** — er ist eine reine Ruheenergie-Verschiebung.

### Bewegung in beliebige Richtungen (bestätigt)

Gaußsches Paket, `eps=0.5`, `alpha=0.5`, `|k|=0.8`, 120 Schritte, Spinor =
Bandeigenvektor bei k:

| Winkel | v vorhergesagt | v gemessen | Richtungsfehler | Norm |
|---|---|---|---|---|
| 0° | 1.2483 | 1.2231 | 0.000° | 1.00000000 |
| 15° | 1.2427 | 1.2181 | 0.070° | 1.00000000 |
| 30° | 1.2371 | 1.2131 | 0.000° | 1.00000000 |
| 45° | 1.2427 | 1.2181 | 0.070° | 1.00000000 |
| 60° | 1.2483 | 1.2231 | 0.000° | 1.00000000 |
| 90° | 1.2371 | 1.2131 | 0.000° | 1.00000000 |
| 135°, 180°, 225°, 270°, 315° | wie oben | wie oben | ≤ 0.070° | 1.00000000 |

**Maximaler Richtungsfehler 0.070° über alle 11 getesteten Winkel.** Das Teilchen
läuft in jede gewünschte Richtung, nicht nur entlang der 6 Gitterachsen.

Der kleine Betragsunterschied (1.223 gemessen vs. 1.248 vorhergesagt) ist die
bekannte k-Mittelung des Pakets: bei sigma=6 ist sigma_k = 1/6 nicht klein gegen
k=0.8, und `<v_g>` über das Paket liegt unter `v_g(k_0)`.

### Kausalität und Isotropie

Die Gruppengeschwindigkeit ist exakt `v_g = <dr>_u / dt` (Hellmann-Feynman für
den unitären Münzoperator). Da `|<dr>| <= sqrt(3)/2`, ist `|v_g| <= c = sqrt(3)`
**strukturell garantiert**, ohne jede Zusatzannahme.

Isotropie von `|v_g|` (6-zählige Gitterwelligkeit):

| |k| | Streuung von \|v_g\| |
|---|---|
| 0.3 | 3.5 % |
| 0.8 | 10.1 % |

---

## 8. Ergebnis 6: Zwei Teilchen ohne Ausschließlichkeit

Die Zeitentwicklung ist linear — zwei Pakete durchdringen einander
vollständig. Die einzige Spur der Begegnung ist ein Interferenzmuster.

**Neuer Befund:** Ob überhaupt interferiert wird, entscheidet der Überlapp der
inneren Spinoren, nicht der räumliche Überlapp:

| Öffnungswinkel zwischen k_A und k_B | \|<u_A\|u_B>\| | Interferenzterm bei vollem Überlapp |
|---|---|---|
| 60° | 0.794 | 76 % der Spitzendichte |
| 120° | 0.379 | – |
| **180° (frontal)** | **3.7e-16 (exakt 0)** | **3.0 %** (nur Restbreite in k) |

Zwei **frontal** aufeinander zulaufende Pakete gleicher Energie haben *exakt
orthogonale* innere Zustände und interferieren in der Dichte praktisch nicht,
obwohl sie räumlich vollständig überlappen — analog zu orthogonalen
Polarisationen. Kreuzende Pakete (±30°) zeigen dagegen kräftige Streifen mit
dem erwarteten Abstand `2*pi/|k_A - k_B|`.

---

## 9. Abbildungen

| Datei | Inhalt |
|---|---|
| `turning_geometry.png` | Richtungen, Drehtabelle `n`, Münzoperator `arg C` bei alpha=0 und 1/2 |
| `turning_loops.png` | Windungszahlen aller geschlossenen Wege bis L=9, Strukturfaktor A(alpha) |
| `turning_spectrum.png` | `E_m(alpha)` numerisch vs. analytisch, Kramers-Verdopplung, Massenregler |
| `turning_dispersion.png` | 6 Bänder bei alpha=0/0.25/0.5, Zoom auf den Kegel, Isotropie, `v_g(k)` |
| `turning_motion.png` | Schwerpunktbahnen in 11 Richtungen, Richtungsfehler, Nulltest für delta |
| `turning_two_packets.png` | Kreuzende Pakete mit Streifen, frontale Pakete ohne |

---

## 10. Offene Punkte / nächste Schritte

* **Kollisionen mit Ausschließlichkeit.** Das jetzige Modell ist linear, also
  ist *nur* Interferenz möglich. Echte Kollisionen brauchen entweder
  (a) einen Zwei-Teilchen-Hilbertraum mit antisymmetrisierter Amplitude auf
  Paaren gerichteter Kanten — exakt, aber teuer (`(N_edges)²`), oder
  (b) eine nichtlineare Selbstwechselwirkung (Gross-Pitaevskii-artiger Term
  `g|psi|²psi`) — billig, aber ad hoc, und sie bricht die Unitarität nicht,
  wohl aber die Superposition.
  Der Ausschluss (Fermi-Statistik) folgt in (a) automatisch aus der
  Antisymmetrisierung; interessant wäre, ob die Windungszahl-Phase `(-1)^w` bei
  alpha=1/2 dabei konsistent die Vertauschungsstatistik erzeugt.
* **Chiralitätsfamilien / Magnetfeld.** Links-, Rechts- und Achten-Familien sind
  kombinatorisch bereits nachgewiesen (Abschnitt 2). Um sie *dynamisch* zu
  trennen, braucht es einen ortsabhängigen Peierls-Faktor
  `exp(i*A(r)·dr)` mit `A = (B/2)(-y, x)` — dann erwartet man Zyklotronbahnen
  und Landau-artige Niveaus. Das ist eine kleine Erweiterung von `step()`.
* **Lokale Kantenanregung.** Statt eines Gaußpakets eine einzelne angeregte
  Kante als Quelle — das von der Idee vermutete Interferenzmuster in der Fläche.
  Mit dem unitären Münzoperator ist das jetzt über beliebig lange Zeiten stabil.
* **Verbindung zur Eigenzeit.** Die Schrittzahl `N` ist die Uhr; die Gesamtphase
  zerfällt in `alpha*Sigma(theta)` (Spin/Windung) und `N*E*dt` (Energie). Der
  Vergleich mit `quantum_proper_time.py` steht aus.
* **Warum genau c/2?** Die Kegelsteigung ist exakt `sqrt(3)/2`, unabhängig von
  eps. Eine geschlossene Herleitung aus der Struktur der Kramers-Partner steht
  noch aus.
