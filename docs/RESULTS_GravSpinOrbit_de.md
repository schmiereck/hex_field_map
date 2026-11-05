# Gravitative Spin-Bahn-Kopplung: die Feinstruktur des Gravitationsatoms

*Paper 3. Code: `src/phase_bit_s4_open15.py` (Teile `A`, `B <Variante> <Spin>`, `C`) und
`src/phase_bit_s4_open15_figs.py`. Daten: `data/phase_bit_s4_open15.npz` (Referenz und
Auswertung) und `data/phase_bit_s4_open15_B_<Variante>_<Spin>.npz` (Zeitsignale).
Abbildung: `figures/phase_bit_s4_open15.png`.
Vorgänger: [`RESULTS_GravAtom_de.md`](RESULTS_GravAtom_de.md).*

## Frage

Das Teilchen des Modells hat Spin ½. In einem Gravitationsfeld koppelt der Spin an die
Bahn: Ein Kreisel auf einer Bahn präzediert (de-Sitter- bzw. geodätische Präzession), und im
Atom spaltet das die Niveaus mit gleichem l nach j = l ± ½ auf. Die Fragen:

1. Welche Feinstruktur sagt die Dirac-Gleichung in der Metrik der einzelnen Regeln
   (nur Warten, H2, linearer Lapse) voraus?
2. Liefert der Walker selbst diese Feinstruktur, also ohne effektiven Hamiltonian?

## Vorhersage: der de-Sitter-Faktor (2γ + 1)

Ein Dirac-Teilchen in der statischen Metrik ds² = −N² dt² + A² dx² hat (flaches Maß,
symmetrisch geordnet)

```
H = β m N + (c′/2) {F, α·p},     F = N / A
```

Im schwachen Feld (N = 1 − U, A = 1 + γU) enthält die Foldy-Wouthuysen-Form den Term

```
H_SO = (2γ + 1) · c′² / (4m) · (1/r)(dU/dr) · σ·L
```

- Pro Spin-Einheit ist das γ + ½. Für γ = 1 ist das die de-Sitter-Präzession 3/2.
- Für γ = 0 (beschleunigtes Bezugssystem ohne Raumkrümmung) bleibt ½, also der bekannte
  Hehl-Ni-Term.
- β tritt nicht auf.

Daraus folgt für die Feinstruktur E(2p3/2) − E(2p1/2):

| Regel | γ | Faktor 2γ + 1 | relativ zu W |
|---|---|---|---|
| A = N (keine Regel des Modells, nur Kontrolle) | −1 | −1 | −1 |
| nur Warten (W) | 0 | 1 | 1 |
| H2 | 1 | 3 | 3 |
| linearer Lapse | 1 | 3 | 3 |

Die l-Aufspaltung der vorigen Runde (die Quanten-Periheldrehung) geht hier nicht ein, weil
2p3/2 und 2p1/2 dasselbe Bahn-|m_l| haben.

## A. Dirac-Referenz

Die Radialgleichung wird in 2D gerechnet (k_z = 0). Die beiden Spin-Blöcke tragen κ = ±j:

- **2p3/2** ist κ = +3/2 (Spin parallel zum Bahndrehimpuls).
- **2p1/2** ist κ = −1/2 (Spin antiparallel).
- Beide haben in der großen Komponente das Bahn-m_l = 1.

Diskretisierung:
- Versetztes Gitter. Die Komponente, die bei r = 0 regulär ist (∼ r^½), liegt auf dem
  halben Gitter.
- Am äußeren Rand gilt P = 0, damit kein Randzustand entsteht.

**Schwaches Feld** (α_f = m GM/c′ = 0.05): Die Feinstruktur verhält sich wie
**−0.978 : 1 : 2.997 : 3.011** (A = N : W : H2 : linear). Das ist die Vorhersage
−1 : 1 : 3 : 3.

**Wachsende Kopplung** (Verhältnis zu W):

| α_f | A = N | H2 | linear |
|---|---|---|---|
| 0.05 | −0.978 | 2.997 | 3.011 |
| 0.10 | −0.983 | 3.061 | 3.117 |
| 0.20 | −0.951 | 3.275 | 3.548 |
| 0.30 | −0.894 | 3.665 | 4.530 |

Im starken Feld liegen H2 und linearer Lapse über 3. Der lineare Lapse steigt stärker, weil
dort β = ½ statt 1 ist und das in höherer Ordnung eingeht.

**Referenz für das Gitter.** Das Gitterband hat bei μ = 0.1 neben der Dirac-Struktur einen
spinunabhängigen Wilson-artigen Term:

- Die Lücke ist Δ(k) = m + (1 + r_b) c′²k²/(2m) mit r_b = −0.0370.
- Die lineare Dirac-Kopplung ist c′ auf 0.2 % (gemessen aus der Mischung der Eigenvektoren).

Der Term geht als β r_b c′²/(2m) N^½ Π² N^½ auf der großen Komponente in die Referenz ein.
Ohne ihn liegt die Referenz-Feinstruktur um 7–8 % tiefer (Tabelle unten).

## B. Das 2D-Gravitationsatom im vollen 3D-Walker

Ein 3D-Atom passt nicht in den Speicher (vorige Runde). Ein 2D-Atom in einer dünnen Schicht
des vollen 3D-Walkers passt. Für die Feinstruktur reicht das, denn Spin und
Bahndrehimpuls stehen beide senkrecht zur Schicht.

Aufbau:
- `Walker3D` (aus Runde 12) auf einer periodischen Schicht 340 × 340 × 2, Bloch-k_z = 0,
  gerades Untergitter.
- μ = 0.1 (m = 0.2), α_f = 0.3, also GM = 0.866 und Bohr-Radius a₀ = 9.62.
- U = GM/√(ρ² + r_c²) um die z-Achse, r_c = a₀/2, U(0) = 0.18. Die halbe Box ist
  104 = 10.8 a₀.
- Regeln: W, H2 und linearer Lapse, jeweils mit kompensiertem λ(N).

Startzustand und Messung:
- **Start:** Referenz-2p-Hülle × e^{iφ} × Ruhe-Spinor mit J_z = +½ (2p3/2) bzw. −½
  (2p1/2), J_z aus `J_int` (Runde 13).
- **Symmetrie:** Der Start wird exakt auf den C4-Sektor j mod 4 der Gitterdrehung
  projiziert (Platzdrehung × R90_int). Er vertauscht danach mit der Drehung auf
  Float32-Genauigkeit.
- **Vorfilter:** 4 Takte gegen die π-Zweige.
- **Lauf:** 8000 Takte pro Lauf (t = 4000). Die Läufe dauerten 1.2 h (W) bzw. 2.1–2.6 h
  (H2, linear), mit drei parallelen Prozessen zu je 4 Threads.
- **Niveau:** Filterdiagonalisierung allein aus der Autokorrelation c(t) = ⟨ψ₀|ψ(t)⟩.
  - Basis: Hann-gefensterte Fouriervektoren bei K Energien um das Ziel.
  - Matrixelemente ⟨Φ_j|U^p|Φ_k⟩ = Σ_d c(d + p) R_jk(d), mit R_jk per FFT.
  - Rayleigh-Ritz; das Niveau ist der Ritz-Wert mit dem größten Gewicht im Start
    (0.986–0.990).

Die FFT allein löst die Feinstruktur nicht auf: Die Auflösung 2π/t ≈ 1.6·10⁻³ ist 4- bis
15-mal gröber als die Aufspaltung. Die Filterdiagonalisierung tut es. Ihr Ergebnis hängt
vom Filtergitter kaum ab (K = 6/10/14, Abstand 0.5–0.75 Bins): Die Streuung liegt bei
10⁻⁷ (W), 2·10⁻⁸ (H2) bzw. 10⁻¹⁰ (linear), also mehr als 1000-mal unter der Feinstruktur.

## C. Ergebnis

**2p-Niveaus** (E − m, 10⁻³):

| Regel | Gitter 2p3/2 | Gitter 2p1/2 | Referenz 2p3/2 | Referenz 2p1/2 |
|---|---|---|---|---|
| nur Warten | −3.7830 | −3.8902 | −3.7723 | −3.8811 |
| H2 | −4.1446 | −4.5439 | −4.1017 | −4.4979 |
| linear | −4.3882 | −4.8841 | −4.3388 | −4.8289 |

**Feinstruktur** E(2p3/2) − E(2p1/2) (10⁻⁴):

| Regel | Gitter | Dirac + Wilson-Term | Gitter / Ref. | Dirac rein | Gitter / rein |
|---|---|---|---|---|---|
| nur Warten (γ = 0) | 1.0715 | 1.0881 | **0.985** | 1.0126 | 1.058 |
| H2 (γ = β = 1) | 3.9928 | 3.9621 | **1.008** | 3.6909 | 1.082 |
| linear (γ = 1, β = ½) | 4.9590 | 4.9009 | **1.012** | 4.5555 | 1.089 |

**Verhältnisse zu W:**

| | H2 / W | linear / W |
|---|---|---|
| Gitter | 3.726 | 4.628 |
| Referenz (α_f = 0.3) | 3.641 | 4.504 |
| schwaches Feld (FW) | 3 | 3 |

- **Der Walker hat die gravitative Spin-Bahn-Kopplung, und sie hat die richtige Größe.**
  Seine Feinstruktur stimmt mit der Dirac-Gleichung in der Metrik auf 1.5 % (W), 0.8 % (H2)
  und 1.2 % (linear) überein, sobald der spinunabhängige Wilson-Term des Gitterbands
  mitgenommen wird.
- **2p3/2 liegt höher als 2p1/2.** Spin parallel zum Bahndrehimpuls ist schwächer
  gebunden, wie bei der elektromagnetischen Feinstruktur.
- **Die Aufspaltung hängt von der Raumkrümmung ab.** H2 und linearer Lapse haben die
  3- bis 4.6-fache Feinstruktur von „nur Warten“. Gitter und Referenz stimmen darin auf
  2–3 % überein, ebenso in der Reihenfolge W < H2 < linear.
- Panel (a) der Abbildung zeigt die Messgröße direkt: Die Phasendifferenz der beiden
  Spin-Signale wächst linear mit der Steigung der Referenz.

**Absolute Niveaus:** Das Gitter bindet etwas stärker als die Referenz, um 0.3 % (W) bzw.
1.0–1.1 % (H2, linear). Die Abweichung ist für beide Spins fast gleich und fällt daher in
der Feinstruktur größtenteils heraus. Wahrscheinliche Ursachen sind Gitterterme höherer
Ordnung im Band, die in der Referenz fehlen (der gemeinsame Untergrund E_c und die
Anisotropie), sowie die Ordnung der Operatoren.

**Boxgröße:** Mit der Referenz in einer fast dreimal so großen Box (R = 289 statt 104)
ändert sich die Feinstruktur um 0.6 % (W), die Niveaus um etwa 4·10⁻⁶. Der Box-Effekt ist
also klein gegen die verbleibenden Abweichungen. Die Referenz rechnet ohnehin mit derselben
Box wie das Gitter.

**Messfallen:**
- Das Spektrum per FFT (auch mit Blackman-Harris) zeigt bei 8000 Takten nur eine breite
  Linie. Eine Aufspaltung daraus abzulesen wäre Unsinn.
- Für 2p1/2 (κ < 0) konvergierte die Referenz zuerst langsam und hatte einen falschen
  Zustand nahe 0. Die Ursache war die Lage der Komponenten auf dem versetzten Gitter:
  Die bei r = 0 reguläre Komponente gehört auf das halbe Gitter.
- In kleinen Boxen verdarb ein Randzustand am äußeren Rand das 2p1/2-Niveau. Abhilfe ist
  P = 0 am Rand.
- Beim linearen Lapse ist das Residuum √(1 − |λ|²) als 0 angegeben, weil |λ| dort im
  Rahmen der Rechengenauigkeit 1 ist (abgeschnitten). Das Niveau ist trotzdem stabil:
  Die Streuung über die Filtergitter liegt bei 10⁻¹⁰.
- Drei parallele Läufe brachten den Rechner an die Speichergrenze. Das Skript speichert
  das Signal deshalb jetzt alle 1000 Takte.

## Antwort

1. **Vorhersage:** Die Dirac-Gleichung in der Metrik gibt im schwachen Feld eine
   Feinstruktur proportional zu 2γ + 1, also W : H2 : linear = 1 : 3 : 3. Der Faktor ist
   die de-Sitter-Präzession pro Spin-Einheit (γ + ½); β geht nicht ein. Im starken Feld
   (α_f = 0.3) werden daraus 1 : 3.64 : 4.50.
2. **Gitter:** Der Walker selbst zeigt diese Feinstruktur, als 2D-Gravitationsatom in
   einer Schicht des vollen 3D-Walkers. Er trifft die Referenz auf 0.8–1.5 %. Damit ist
   auch der effektive Hamiltonian der vorigen Runde zum ersten Mal in mehr als einer
   Dimension direkt am Walker geprüft: Die absoluten 2p-Niveaus stimmen auf 0.3–1.1 %.
3. **Spin präzediert im Gravitationsfeld wie ein Kreisel.** Die spinabhängige Kraft auf
   einen einzelnen Kramers-Partner (±8 %, g₀ᵢ-Runde) war ein erstes Anzeichen dafür. Sie
   ist hier als Feinstruktur quantitativ bestätigt.

**Was das nicht zeigt:**
- **Nur 2D.** In 3D kommen Zustände mit Spin schräg zum Bahndrehimpuls hinzu. Die
  Kopplung σ·L ist aber dieselbe, und die Gitterisotropie wurde in Exp. 3 geprüft.
- **Nur 2p.** Andere Niveaus (3p, 3d) wären teurer (größere Box, mehr Takte).
- **Der Wilson-Term gehört dazu.** Ohne ihn liegt das Gitter 6–9 % über der reinen
  Dirac-Feinstruktur. Er ist eine Eigenschaft des Gitterbands bei endlichem μ und wird
  mit kleinerem μ kleiner (r_b = −0.082 bei μ = 0.15, −0.037 bei μ = 0.1). Er ist
  gemessen, nicht angepasst.
- **Starkes Feld.** Für den Vergleich mit 1 : 3 : 3 bräuchte das Gitter α_f ≲ 0.1. Dann
  wäre die Feinstruktur etwa 100-mal kleiner und bräuchte etwa 100-mal mehr Takte.

## Offene Punkte

- Die gemeinsame Verschiebung der absoluten Niveaus (0.3–1.1 %) aus dem Band ableiten
  (E_c-Untergrund, Anisotropie).
- Feinstruktur bei kleinerem α_f, um das Verhältnis 1 : 3 : 3 direkt im Gitter zu sehen.
  Das wäre mit harmonischer Inversion und kürzeren Signalen eventuell machbar.
- ~~Spin-Spin-Kopplung (gravitative Hyperfeinstruktur) über das gravitomagnetische Feld
  eines rotierenden Zentralkörpers, mit dem `drag3d` aus Runde 13.~~ **Erledigt**, siehe
  [`RESULTS_GravHyperfine_de.md`](RESULTS_GravHyperfine_de.md): Gitter / Referenz
  1.027–1.035, Verhältnis 2p3/2 : 2p1/2 = 0.564 gegen 0.560.
