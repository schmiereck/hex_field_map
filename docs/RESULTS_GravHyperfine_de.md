# Gravitative Hyperfeinstruktur: Lense-Thirring-Kopplung des Atoms an den Drehimpuls des Zentralkörpers

*Paper 3. Code: `src/phase_bit_s4_open16.py` (Teile `A`, `B <Spin> <±1> <rule|norule>`, `C`) und
`src/phase_bit_s4_open16_figs.py`. Daten: `data/phase_bit_s4_open16.npz` (Referenz und
Auswertung) und `data/phase_bit_s4_open16_B_<Spin>_<p|m>_<rule|norule>.npz` (Zeitsignale).
Die J = 0-Läufe stammen aus der vorigen Runde (`data/phase_bit_s4_open15_B_H2_<Spin>.npz`).
Abbildung: `figures/phase_bit_s4_open16.png`.
Vorgänger: [`RESULTS_GravSpinOrbit_de.md`](RESULTS_GravSpinOrbit_de.md).*

## Frage

Die vorige Runde hat die Feinstruktur des Gravitationsatoms gemessen, also die Kopplung des
Teilchenspins an die eigene Bahn. Dreht sich der Zentralkörper, kommt das gravitomagnetische
Feld dazu (Lense-Thirring). Es koppelt an den Bahndrehimpuls des Teilchens und an seinen Spin.
Das ist das gravitative Gegenstück zur Hyperfeinstruktur, bei der der Kernspin an das Elektron
koppelt. Die Fragen:

1. Welche Verschiebung der 2p-Niveaus sagt die Dirac-Gleichung in der Metrik mit
   gravitomagnetischem Feld voraus, und wie verteilt sie sich auf Bahn und Spin?
2. Liefert der Walker diese Verschiebung, mit dem lokalen Mitführ-Hop (`drag3d`) und der
   internen Drehregel aus Runde 13?

## Vorhersage: Bahn : Spin = 2 : −1

Ein Zentralkörper mit Drehimpuls J entlang z erzeugt (Runde 13, hier wie U weich gemacht)

```
b = B_J (ẑ × r) / (r² + r_c,b²)^{3/2},      B_J = 2 c′² (κ/4π) J
b_φ = B_J ρ / (ρ² + r_c,b²)^{3/2},          (rot b)_z = B_J (2 r_c,b² − ρ²) / (ρ² + r_c,b²)^{5/2}
```

Der Walker koppelt daran auf zwei Wegen:

- **Bahn:** Der Mitführ-Hop ergibt H₁ = b·p = (b_φ/ρ) L_z.
- **Spin:** Die Drehregel exp(−i dt ½ rot b·J_int) wirkt mit gyrogravitomagnetischem
  Faktor 1.

Weit außerhalb des Kerns gilt in der Ebene des Atoms

```
b·p = B_J L_z / ρ³,      ½ rot b · S = −(B_J/2) S_z / ρ³
```

Bahn : Spin = 2 : −1 ist das Verhältnis der Allgemeinen Relativitätstheorie:
2G/(c²r³) J·L für die Bahn, Ω_LT = −GJ/(c²r³) in der Äquatorebene für den Kreisel. Für die
große Komponente (m_l = 1) folgt in erster Ordnung

```
ΔE ∝ B_J ⟨1/ρ³⟩ (m_l − s_z/2):      2p3/2 : 2p1/2 = 3 : 5 = 0.6
```

Ohne Spinkopplung wäre das Verhältnis 1 : 1, mit dem elektromagnetischen g = 2 wäre es 1 : 3.
Das Verhältnis misst also den gyrogravitomagnetischen Faktor.

## A. Dirac-Referenz

Grundlage ist die radiale 2D-Dirac-Gleichung der vorigen Runde (H2, μ = 0.1, α_f = 0.3,
a₀ = 9.62, r_c = a₀/2, mit dem Wilson-Term des Gitterbands). Zu jeder Komponente kommt der
Diagonalterm

```
m_l · b_φ/ρ + s_z · (rot b)_z / 2
```

mit den Quantenzahlen der Komponenten:

- **2p3/2:** groß (m_l = 1, s_z = +½), klein (m_l = 2, s_z = −½).
- **2p1/2:** groß (m_l = 1, s_z = −½), klein (m_l = 0, s_z = +½).
- Die kleine Komponente trägt etwa 1 % des Gewichts.

Gerechnet wird exakt in B_J (Eigenwert mit den Zusatztermen) und in erster Ordnung, Bahn- und
Spinanteil getrennt. Bei ±B_J trennt sich der ungerade Anteil (E(+J) − E(−J))/2 vom geraden,
der quadratisch in B_J ist.

**Kernradius des Drehimpulses.** Mit r_c,b = r_c = a₀/2 fällt der Spinanteil fast weg:
Spin/Bahn ist dann −0.06 statt −¼. Der Grund ist, dass rot b im Kern das Vorzeichen wechselt
(Panel d der Abbildung) und die 2p-Dichte dort noch merklich ist. Der rotierende Körper wird
deshalb kompakter angesetzt, r_c,b = 0.25 a₀ = 2.41. Das Potential U behält r_c = a₀/2.

**Referenzwerte** (B_J = 0.15):

| | Bahn (1. Ord.) | Spin (1. Ord.) | Spin / Bahn | ungerade, exakt | gerade |
|---|---|---|---|---|---|
| 2p3/2 | 4.806e-5 | −6.588e-6 | −0.137 (fern: −¼) | 4.151e-5 | −1.16e-6 |
| 2p1/2 | 6.366e-5 | +1.034e-5 | +0.162 (fern: +¼) | 7.407e-5 | −2.39e-6 |

- **Verhältnis 2p3/2 : 2p1/2:** 0.560 mit Spin, 0.756 nur mit Bahn.
- **Woher die Abweichung von 0.6 und 1 kommt:**
  - Die 2p1/2-Hülle liegt im starken Feld etwas weiter innen, also ist ihr ⟨1/ρ³⟩ größer.
  - Die kleinen Komponenten tragen andere m_l.
  - Der Kern dämpft den Spinanteil stärker als den Bahnanteil.

## B. Gitter

**Aufbau:**
- Lauf: der H2-Lauf der vorigen Runde (Schicht 340 × 340 × 2 von `Walker3D`, 8000 Takte),
  mit demselben C4-projizierten Startzustand und Vorfilter (`o15.setup_B`).
- Hinzu kommen der Mitführ-Hop (Bahn) und wahlweise die Spinregel nach jedem Takt.
- Läufe:
  - nur Bahn, +J: 2 Läufe (2p3/2, 2p1/2);
  - Bahn + Spin, ±J: 4 Läufe.
- Laufzeit: je 4.9 h, zwei Prozesse parallel. Das Niveau kommt wie zuvor aus der
  Filterdiagonalisierung.

**Mitführ-Hop.** Der Hop ist die Ziegelmauer von Bindungsdrehungen aus Runde 13: Die Bindung
(x, x + s) wird um den Winkel dt (b·s)/(8 l²) gedreht, mit b in der Bindungsmitte. Die Schicht
erforderte zwei Änderungen:

- **Parität aus der Ebene.** Mit z-Periode 2 zerstört die volle Parität (x·s)//2 mod 2 den
  Wechsel entlang s. Die Parität wird daher nur aus dem Anteil in der Ebene gebildet,
  ((I₀s₀ + I₁s₁)//(s₀² + s₁²)) mod 2. Die Boxlänge muss ein Vielfaches von 4 sein.
- **Schnellere Umsetzung.** Mit `np.roll` brauchte ein Takt 3.9 s (n = 200). Jetzt werden die
  Bindungen über Indexlisten gesammelt und zurückgeschrieben, nur mit Starts auf dem geraden
  Untergitter. Das braucht 0.36 s pro Register bei n = 340. Jede Bindungslage ist exakt
  unitär (Normdrift 0).

**Hop nur alle 4 Takte (DRAGK = 4), mit 4-fachem Winkel.** Das ist in erster Ordnung in b
exakt. Der Fehler sind geschachtelte Kommutatoren mit dem Walker-Schritt U₀, und deren
Erwartungswert in einem Eigenzustand von U₀ ist null. Der gerade Anteil zweiter Ordnung fällt
in E(+J) − E(−J) heraus. Er wird hier aber auch selbst verglichen (unten).

**Spinregel.** J_z = V diag(j) V^H (aus `J_int`, Runde 13). Der Phasenfaktor ist
exp(−i dt ½ (rot b)_z(x) j), ortsabhängig. Die Zustände sind Zeilenvektoren (S, 24).
Daraus folgt

```
a → ((a · conj(V)) · Phase) · Vᵀ
```

**Falle:** Die erste Fassung hatte Konjugation und Transposition für Spaltenvektoren gesetzt.
Auf Zeilenvektoren angewandt dreht das mit der falschen Matrix. Die Folgen:

- **Symptom:** Der Bahnanteil (Läufe ohne Regel) stimmte auf 2.8 %, der Spinanteil hatte das
  falsche Vorzeichen: +1.11e-5 / −1.13e-5 gegen −0.66e-5 / +1.03e-5.
- **Diagnose:** Eine kleine Box (n = 200) und die Zerlegung des gefilterten Startzustands nach
  J_int-Eigenwerten:
  - Mit dem Fehler lagen 66 % des 2p3/2-Gewichts bei J_int = −3/2.
  - Korrigiert liegen 98.8 % bei +½.
  - Der korrigierte Erwartungswert ⟨½ rot b·J_int⟩ ist −7.22e-6 (Referenz −7.58e-6) für 2p3/2
    und +1.155e-5 (+1.145e-5) für 2p1/2.
  - Bei 2p1/2 trägt die kleine Komponente (1.3 % bei J_int = +½) allein 2.7e-6 bei.
- Die fehlerhaften Läufe wurden verworfen.

## C. Ergebnis

**Verschiebungen** durch J (Gitter gegen Referenz):

| | Größe | Gitter | Referenz | Gitter / Ref. |
|---|---|---|---|---|
| 2p3/2, nur Bahn | E(+J) − E(0) | 4.808e-5 | 4.678e-5 | 1.028 |
| 2p1/2, nur Bahn | E(+J) − E(0) | 6.357e-5 | 6.187e-5 | 1.027 |
| 2p3/2, Bahn + Spin | (E(+J) − E(−J))/2 | 4.295e-5 | 4.151e-5 | 1.035 |
| 2p1/2, Bahn + Spin | (E(+J) − E(−J))/2 | 7.618e-5 | 7.407e-5 | 1.028 |
| 2p3/2, gerader Anteil | (E(+J) + E(−J))/2 − E(0) | −1.23e-6 | −1.16e-6 | 1.06 |
| 2p1/2, gerader Anteil | (E(+J) + E(−J))/2 − E(0) | −2.49e-6 | −2.39e-6 | 1.04 |

Die Streuung über die Filtergitter liegt bei 10⁻¹⁰ bis 10⁻⁹, also 10⁴-mal unter den
Verschiebungen.

**Verhältnis 2p3/2 : 2p1/2:**

| | Gitter | Referenz | schwaches Feld |
|---|---|---|---|
| Bahn + Spin | 0.5638 | 0.5604 | 0.6 |
| nur Bahn | 0.7564 | 0.7560 | 1 |

**Spinanteil** (Bahn + Spin) − Bahn, beide als ungerader Anteil. Bei den Läufen ohne Regel
wird der gerade Anteil der Referenz abgezogen.

| | Gitter | Referenz (exakt) | Gitter / Ref. | Referenz 1. Ordnung |
|---|---|---|---|---|
| 2p3/2 | −6.46e-6 | −6.59e-6 | 0.980 | −6.59e-6 |
| 2p1/2 | +1.077e-5 | +1.036e-5 | 1.039 | +1.034e-5 |

- **Der Walker hat die gravitative Hyperfeinstruktur.** Ein rotierender Zentralkörper
  verschiebt die 2p-Niveaus um den Betrag, den die Dirac-Gleichung mit gravitomagnetischem
  Feld vorhersagt, auf 2.7–3.5 %. Das gilt für den Bahnanteil (Mitführ-Hop), den Spinanteil
  (Drehregel) und sogar für den geraden Anteil zweiter Ordnung (4–6 %).
- **Der Spin koppelt mit Faktor 1 und dem richtigen Vorzeichen.** Der Spinanteil folgt s_z:
  2p3/2 (s_z = +½) wird abgesenkt, 2p1/2 angehoben. Seine Größe trifft die Referenz auf 2 %
  bzw. 4 %. Das Verhältnis 2p3/2 : 2p1/2 (0.564) zeigt das ohne jede Referenz: Ohne Spin
  wäre es 0.756, mit g = 2 läge es weit darunter.
- **Panel (a) der Abbildung** zeigt die Messgröße direkt: Die Phase des Laufs mit J gegen den
  Lauf mit J = 0 wächst linear mit der Steigung der Referenz.

**Gemeinsamer Faktor 1.03.** Alle Bahnverschiebungen liegen um denselben Faktor 1.027–1.035
über der Referenz. Er hängt nicht vom Spin ab und fällt im Verhältnis heraus (0.7564 gegen
0.7560). Zwei Ursachen kommen in Frage; getrennt wurden sie nicht:

1. Das Gitter bindet 1.0–1.1 % stärker (vorige Runde). Eine kompaktere Hülle hat ein größeres
   ⟨1/ρ³⟩, das gibt einige Prozent mehr.
2. Der Mitführ-Hop selbst lag schon in Runde 13 um 1.5 % über der starren Rotation
   (0.00406 gegen 0.004).

## Antwort

1. **Vorhersage:** Bahn : Spin = 2 : −1 wie in der Allgemeinen Relativitätstheorie. Daraus
   folgt 2p3/2 : 2p1/2 = 3 : 5 weit außerhalb des Kerns. Mit Kern und starkem Feld wird
   daraus 0.560 (Referenz). Das Verhältnis ist ein direktes Maß für den
   gyrogravitomagnetischen Faktor.
2. **Gitter:** Der Walker reproduziert die Verschiebungen auf 2.7–3.5 % und das Verhältnis
   auf 0.6 % (0.5638 gegen 0.5604). Mitführ-Hop und Drehregel aus Runde 13 sind damit nicht
   nur an klassischen Bahnen und starrer Rotation geprüft, sondern an einem gebundenen
   Quantenzustand, Bahn und Spin getrennt.
3. **Spin im Gravitationsfeld:** Die vorige Runde hat gezeigt, dass der Spin an die eigene
   Bahn koppelt (de Sitter). Diese Runde zeigt, dass er auch an den Drehimpuls des
   Zentralkörpers koppelt (Lense-Thirring, Schiff-Präzession), mit Faktor 1.

**Was das nicht zeigt:**
- **Die Spinregel ist gesetzt, nicht hergeleitet.** exp(−i dt ½ rot b·J_int) ist die Regel aus
  Runde 13. Gemessen ist, dass sie im gebundenen Zustand den richtigen Spinanteil gibt, nicht,
  dass das Gitter sie erzwingt.
- **Feld vorgegeben.** b ist das Feld eines rotierenden Körpers mit weichem Kern, nicht aus
  einer bewegten Quelle im Gitter erzeugt (das wäre die kausale g₀ᵢ-Runde in 3D).
- **Nur 2D, nur 2p, starkes Feld** (α_f = 0.3), wie in der vorigen Runde.
- **Kernradius gewählt.** Mit r_c,b = 0.25 a₀ ist der Spinanteil groß genug zum Messen. Er
  hängt aber merklich von r_c,b ab (Spin/Bahn −0.137 hier, −0.062 bei 0.5 a₀).

**Messhinweise:**
- Der gerade Anteil (quadratisch in B_J) ist 3–4 % des ungeraden. Wer E(+J) − E(0) mit der
  ungeraden Referenz vergleicht, macht 3 % Fehler. Ohne −J-Lauf muss die volle Referenz
  E(+J) − E(0) zum Vergleich dienen.
- **Speicher:** Beide Male hat Claude Code die Bash-Hülle wegen Speichermangels beendet. Die
  Python-Prozesse und die wartenden Folgeläufe liefen trotzdem weiter. Vor einem Neustart
  deshalb die Prozessliste prüfen.

## Offene Punkte

- Den gemeinsamen Faktor 1.03 trennen: Ursache Hülle (aus den absoluten Niveaus) oder
  Mitführ-Hop (an einem freien Paket mit bekanntem L).
- Die Spinregel aus der Gitterdynamik herleiten, statt sie zu setzen. Zum Beispiel aus einem
  Hop, der die Bindung nicht nur verschiebt, sondern das 2O-Register mitdreht.
- b aus einer rotierenden Quelle im Gitter erzeugen (kausal, wie in der g₀ᵢ-Runde), statt es
  vorzugeben.
