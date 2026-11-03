# S4, dritte Runde: Gravitation auf dem sauberen Teilchen, Z₂, Magnetfeld

*Paper 3. Code: `src/phase_bit_s4_open3.py` (Teile `main`, `B2`, `C2`) und
`src/phase_bit_s4_open3_figs.py`. Daten: `data/phase_bit_s4_open3.npz`.
Abbildung: `figures/phase_bit_s4_open3.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open2_de.md`](RESULTS_PhaseBit_S4_Open2_de.md),
[`RESULTS_Gravity_Lapse_de.md`](RESULTS_Gravity_Lapse_de.md).*

Speicher: Die größten Rechnungen sind die x-Kette des Lapse-Tests (Dimension 3840:
Schur-Zerlegung und eigh) und die z-Schicht (2880). Der Spitzenbedarf lag bei etwa
1.2 GB, frei waren 2 bis 3 GB.

Modell: **2O\*** = G₂O + λ\*(ε)·W₉₀ aus der zweiten Runde, ε = 0.3, λ\* = −3.2036.

## 1. Gravitation (Exp. 1b) auf dem sauberen Teilchen

### Zuerst ein Befund: Der Floquet-Offset gravitiert

Ein Lapse skaliert die Quasienergie, also auch den Abstand E_c des Dirac-Quartetts
von E = 0. Bei 2O\* liegt das Quartett bei E_c = +1.02, die Masse ist nur 0.06. Ohne
Korrektur wäre die schwere Masse gleich E_c + m, und das Teilchen fiele 18-mal zu
stark. Physikalisch bedeutet das: Gravitieren darf nur die Energie relativ zum Vakuum. Deshalb
kommt ein **Vakuum-Gegenterm** hinzu, ein lokaler Term −ε·lv·𝟙 im Generator, der
den Dirac-Punkt auf E = 0 legt. Alle folgenden Ergebnisse verwenden ihn.

### Äquivalenzprinzip

R′ = E₀ / (m\* c′²), gemessen mit der eigenen Grenzgeschwindigkeit c′ = c/3 des
Teilchens. R′ = 1 entspricht der ART. Relativ zur Gitter-Lichtgeschwindigkeit
c = √3 wäre R = R′/9. Dieser Faktor 9 stammt nur daher, dass der Fermion-Lichtkegel
(c/3) nicht der Gitterkegel ist.

| Modell | μ/ε | R′ Teilchen | R′ Antiteilchen |
|---|---|---|---|
| 2O ohne Kompensation | 0.05 / **0.1** / 0.5 | 0.877 / **0.734** / −2.77 | 1.107 / **1.201** / 1.66 |
| SU(2) kompensiert (S4+) | 0.05 / 0.1 / 0.5 | 0.995 / 0.979 / 0.311 | 0.995 / 0.979 / 0.311 |
| **2O\*** | 0.05 / **0.1** / 0.5 | 0.999 / **0.997** / 0.914 | 0.999 / **0.997** / 0.922 |

**Der Untergrund −αk² verletzt das Äquivalenzprinzip zwischen Teilchen und
Antiteilchen.** Ohne Kompensation fallen sie verschieden stark (0.73 gegenüber 1.20).
Mit 2O\* fallen beide mit R′ = 0.997 (μ/ε = 0.1). Die Kompensation aus Runde 2 hat also
eine physikalische Bedeutung: Sie stellt das Äquivalenzprinzip her.

### Uhren-Lapse gegen vollen Lapse

- **Uhren-Lapse** C(r) = expm(i N(r) X): Er ist exakt lokal und unitär, es wird nur der
  Coin verlangsamt.
- **Voller Lapse** H → ½(N H + H N): Hier wird auch die Ausbreitung verlangsamt.
  Gerechnet wird mit dem Floquet-Hamiltonoperator, der glatt auf die Dirac-Umgebung
  projiziert ist (|E| < 1.2), damit der Schnitt an der Zonengrenze nicht stört.

Aufbau: eine x-Kette aus 160 FCC-Lagen mit Bloch-Impuls in y und z, linearer Lapse
N = 1 + g·x mit g = 2·10⁻³. Die Pakete sind bandprojiziert (σ = 8). Gemessen wird die
Querbeschleunigung a_x, Bezugswert ist −g·c′² = −6.67·10⁻⁴ (Abbildung b).

| Zustand | Uhren-Lapse: Simulation (Halbklassik) | voller Lapse: Simulation (Halbklassik) |
|---|---|---|
| massiv, ruhend (μ = 0.15) | −5.94 (−5.75)·10⁻⁴ | −5.76 (−5.91)·10⁻⁴ |
| massiv, k_y = 0.3 | −4.31 (−4.19)·10⁻⁴ | −5.68 (−5.80)·10⁻⁴ |
| massiv, k_y = 0.6 | −2.15 (−2.12)·10⁻⁴ | −5.28 (−5.35)·10⁻⁴ |
| **masselos**, k_y = 0.2 | **+0.10**·10⁻⁴ (+0.16) | **−4.85**·10⁻⁴ (−5.71) |
| **masselos**, k_y = 0.4 | **+0.22**·10⁻⁴ (+0.24) | **−5.86**·10⁻⁴ (−6.14) |

- **Uhren-Lapse = Nordström:** Ruhende Massen fallen, schnelle immer schwächer, Licht
  gar nicht. Die Kopplung κ(k) = N ∂E/∂N / E folgt jetzt (m/E)² = 1/γ² ohne den
  rätselhaften Vorfaktor 1.5 bis 1.7 aus Exp. 1a (κ = 0.509 gegenüber 0.520 bei
  k = 0.1). Bei k ≳ ε fällt κ zusätzlich ab, weil die Kegelkrümmung von ε abhängt
  (Abbildung c).
- **Voller Lapse = Einstein 1911:** Alles fällt etwa mit −g·c′², unabhängig von der
  Geschwindigkeit, also **wird auch Licht abgelenkt**, und zwar mit derselben
  Beschleunigung wie langsame Materie. Das ist Einsteins Wert von 1911, also die
  Hälfte der ART-Ablenkung. Den Faktor 2 liefert erst die räumliche Metrik
  (Exp. 1c).
- Die Simulation stimmt mit der Halbklassik (exaktes Band) auf 2 bis 3 % überein, beim
  masselosen Strahl mit k = 0.2 auf 15 %. Dort ist der Strahl im Verhältnis zur
  Paketbreite (σ_k = 0.125) schmal, und seine Winkelverteilung ist breit.

**Antwort auf die Frage aus Runde 2:** Ja. Das saubere 2O\*-Teilchen erfüllt R′ = 1 auf
0.3 %, für Teilchen und Antiteilchen, wenn man c′ = c/3 als seine Lichtgeschwindigkeit
nimmt und den Vakuum-Offset abzieht. Mit vollem Lapse fällt Licht wie bei Einstein
1911.

## 2. Z₂ praktisch: Ist der Wandkegel robust?

Ein strenger Floquet-Z₂-Index braucht eine globale Lücke, und die gibt es nicht
(siehe Runde 2). Geprüft wurde deshalb direkt die Aussage, die ein Z₂-Index macht: Eine
zufällige, **T-gerade** Störung an einer Wand kann den Wandkegel nur dann öffnen, wenn
die Wand eine **gerade** Zahl von Kramers-Paaren trägt.

Aufbau: die z-Schicht aus Runde 2, Vorzeichenwechsel der Masse (μ = 0.05), dazu eine
zufällige T-gerade 24×24-Störung (T = Π·Θ, geprüft auf 0) in den sieben Lagen um
Wand 1, Stärke 1.2μ und 2.4μ.

- **Ein Quartett (2O\*):** Die Wand trägt ein Kramers-Paar. Die Störung verschiebt die
  Kegelspitze (auf −0.012), aber die beiden mittleren Zweige berühren sich bei k = 0
  exakt (Abstand 6·10⁻¹⁵ bzw. 2·10⁻¹⁵). Die Zweige laufen vom unteren ins obere
  Volumenband (Abbildung d), der Kegel bleibt **lückenlos**.
- **Zwei Quartette (2O, λ = −2):** Die Wand trägt zwei Kramers-Paare. Die mittleren Zweige
  treffen sich außerhalb von k = 0, wo nichts die Kreuzung schützt, und stoßen sich ab.
  Der minimale Abstand beträgt **4.3·10⁻³** (1.2μ) bzw. **4.9·10⁻³** (2.4μ) und ist
  nie null (Abbildung e). Die Wand ist **gegapt**.

Das ist die Z₂-Arithmetik: 1 Paar ist geschützt, 2 Paare sind trivial. Der Sprung
θ = 0 → π der Masse ist also genau dann topologisch, wenn er einen einzelnen
Dirac-Punkt betrifft.

## 3. Magnetfeld und Anomalie

**Exakter Befund:** Peierls-Phasen sind diagonal in Ort und Richtung, die Helizität
H = Σ_d |d⟩⟨d|⊗u_d·σ ist es auch. Daraus folgt [H, W_B] = 3.5·10⁻¹⁶. Das gilt für jedes
U(1)-Feld und ebenso für Farb-Links, die nur auf einen Farbindex wirken. Die
Gitter-Chiralität H bleibt also **auch mit Eichfeld exakt erhalten**. Es folgt:

- **θ bleibt mit Eichfeld unbeobachtbar.** Getestet in einer offenen x-Kette mit B = 0.05:
  Das Spektrum ändert sich über θ ∈ [0, π] um 3.4·10⁻¹⁴. Kommt eine helizitätsbrechende
  Masse hinzu, ändert es sich um 0.26. Eine Kopplung θ ↔ F F̃ über H gibt es nicht.
- **Warum es trotzdem keinen Widerspruch zur Anomalie gibt:** H ist nicht die axiale
  Ladung der Kontinuumstheorie. In jedem Helizitätssektor sitzen zwei Weyl-Knoten mit
  *entgegengesetzter* Chiralität, bei verschiedenen Quasienergien (Runde 1). Die
  anomale axiale Ladung ist die Knoten-Chiralität, nicht H.
  - **Landau-Niveaus** (Landau-Eichung A = (0, Bx, 0), B = 0.05, x-Kette aus 200 Lagen,
    getrennt nach Helizität): Am untersten Knoten (E = −2.078) gibt es in jedem Sektor genau
    **einen chiralen Zweig**. Seine Steigung ist −0.549 für Helizität +1 (Chiralität +1)
    und +0.590 für Helizität −1 (Chiralität −1), also |v| ≈ c/3 und Vorzeichen = −χ
    (Abbildung f).
  - Am inneren Knoten (E = +0.693, entgegengesetzte Chiralität) liegt der chirale Zweig
    zwischen höheren Landau-Niveaus des gekrümmten Bandes. Mit dieser Auswertung ließ
    er sich nicht sauber trennen.
  - Die Bilanz „netto null pro Sektor“ folgt aus den Weyl-Chiralitäten (Runde 1). Hier
    ist sie nur für den unteren Knoten direkt gezeigt.

Mit anderen Worten: Das Gitter verlagert die Anomalie. Paralleles E und B pumpen
Teilchen vom Knoten bei −2.08 zum Knoten bei +0.69 desselben Helizitätssektors, also
innerhalb der Quasienergie-Zone und nicht zwischen den Helizitäten.

## Antwort

1. **Gravitation:** Mit Vakuum-Gegenterm fällt das saubere 2O\*-Teilchen mit
   R′ = 0.997, für Teilchen und Antiteilchen gleich. Ohne die Kompensation des
   Untergrunds fallen beide verschieden stark. Der Uhren-Lapse ist Nordström-Gravitation
   (Licht unabgelenkt, Kopplung 1/γ²). Der volle Lapse ist Einstein 1911 (Licht fällt wie
   langsame Materie).
2. **Z₂:** Eine T-gerade Störung kann den Wandkegel eines einzelnen Dirac-Punkts nicht
   öffnen, den eines doppelten schon (vermiedene Kreuzung, Lücke 4 bis 5·10⁻³).
3. **Eichfeld:** Die Helizität bleibt mit jedem Eichfeld exakt erhalten, θ bleibt
   unbeobachtbar. Die echte Anomalie betrifft die Knoten-Chiralität, und deren
   Partner liegt bei einer anderen Quasienergie.

## Offene Punkte

*Alle vier Punkte sind in der vierten Runde bearbeitet, siehe [`RESULTS_PhaseBit_S4_Open4_de.md`](RESULTS_PhaseBit_S4_Open4_de.md).*

- **Exp. 1c:** räumliche Metrik, also auch die Schrittlänge oder Hüpfamplitude über N
  ändern, für den vollen ART-Faktor 2 der Lichtablenkung.
- **Lokaler voller Lapse:** Der volle Lapse wurde hier über den globalen
  Floquet-Hamiltonoperator gebaut. Eine streng lokale, unitäre Gitterregel für
  „langsamere Ausbreitung“ fehlt noch (die ursprüngliche Idee der begrenzten
  Rechenzeit pro Knoten).
- **Herkunft des Vakuum-Gegenterms:** Er wurde von Hand gesetzt. Ist er natürlich, etwa
  als Symmetrieforderung „das Vakuum rechnet nicht“?
- Den chiralen Zweig am inneren Knoten sauber trennen, zum Beispiel durch Pumpen mit
  E ∥ B in der Zeit statt über Spektren.
