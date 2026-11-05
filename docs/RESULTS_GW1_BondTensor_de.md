# Gravitationswellen 1: Trägt die Budgetregel einen Tensor?

Skript: `src/phase_bit_s4_open17.py` (Teile A, B; Laufzeit < 1 s), Daten: `data/phase_bit_s4_open17.npz`, Log `data/phase_bit_s4_open17.log`.

## Ausgangslage

Paper 3 endet mit der offenen Frage, ob das Modell überhaupt Gravitationswellen mit zwei transversal-spurfreien
Polarisationen haben kann. Bisher hat es zwei Skalare (Lapse N, Raummetrik A = 1/N) und einen Vektor (Mitnahme-Hop b),
aber keinen spurfreien räumlichen Tensor h_ij. Mit Φ und b allein strahlt ein Doppelstern nur Skalar- und
Vektorwellen ab (Atmungs- und Longitudinalmode). Das wäre durch den Doppelpulsar und die LIGO-Polarisationstests
ausgeschlossen.

## Idee: die Budgetregel pro Bindung

Die Regel lautet: „Jeder Prozess braucht jeden Knoten, den er beteiligt, frei.“ Ein Hop entlang der FCC-Bindung d
beteiligt die Bindung (x, x + d). Ist die Last eines Knotens nach Bindungsrichtung aufgelöst, wird das Stay-Gate zu
N_h,d. Es hängt nur von der Achse ab und ist gerade unter d → −d. Das ist der Unterschied zum verworfenen
richtungsabhängigen Gate aus g0i: Jenes war ungerade in u_d und hat Kramers gespalten.

## A. Bandebene

**A0 – Algebra (exakt).** Das FCC-Gitter hat genau 6 ungerichtete Bindungsachsen. Ihre Dyaden u_a u_aᵀ bilden eine
Basis der symmetrischen 3×3-Matrizen (Rang 6, Singulärwerte √2 und 5 × 1/√2, Kondition 2). Ein Skalar pro
Bindungsachse *ist* damit ein symmetrischer Tensor: die Spur ist das isotrope A, dazu kommen 5 spurfreie Komponenten.
Das kubische Gitter (3 Achsen) könnte das nicht.

**A1 – Antwort des Dirac-Kegels** (2O*, ε = 0,3, μ = 0, N₀ = 0,75, nur Stay-Gate). Gemessen wird der Kegeltensor M
aus E² = c′² kᵀMk:

| Störung S (N_h,d = N₀(1 + u_dᵀSu_d)) | Antwort dM/M₀ pro Einheit S | Dirac-Punkt | Kramers |
|---|---|---|---|
| keine | M = 0,562495 · 1 (N_h² = 0,5625), Nebendiagonale 1e-13 | 1,5e-16 | 3,5e-15 |
| Spur | +2,010 (M ∝ N_h²) | 8e-16 | 4e-15 |
| E_g (diag(1, −1, 0)) | +0,500 | 3,7e-5 (2. Ordnung) | 2e-12 |
| T_2g (xy) | +1,000 | −1,7e-4 (2. Ordnung) | 6e-8 |

Die Antwort ist immer ein Ellipsoid (Fit-Residuum ≤ 3e-6). Kramers bleibt erhalten, und der Dirac-Punkt verschiebt
sich erst in zweiter Ordnung.

**A2 – GR-Ziel M = N₀² g⁻¹ mit g = 1 + 2h** (h spurfrei, zufällig; S aus der gemessenen linearen Antwort):

| \|h\| | \|M − N₀² g⁻¹\| / N₀² | Dirac-Punkt | Kramers |
|---|---|---|---|
| 0,0005 | 2,1e-6 | 1,6e-6 | 1e-8 |
| 0,001 | 1,7e-5 | 8,9e-6 | 3e-8 |
| 0,002 | 7,7e-5 | 3,5e-5 | 1e-7 |
| 0,005 | 2,4e-4 | 2,7e-4 | 4e-7 |
| 0,01 | 7,4e-4 | 4,1e-4 | 9e-7 |
| 0,02 | 4,9e-3 | 4,5e-3 | 6e-6 |

Ein Kontrolllauf mit einem festen h ergibt einen Fehler von ≈ 6 h² und eine Vakuumverschiebung von ≈ 3,2 h². Das gilt
unabhängig von k (0,01 bzw. 0,004). In linearer Ordnung erzeugt das Bindungs-Gate also **jede** symmetrische inverse
Metrik exakt, und das ist die Ordnung, auf die es bei Gravitationswellen ankommt (h ~ 1e-21). Die Vakuumverschiebung
O(h²) braucht wie bisher einen Gegenterm. Bei Wellen ist sie ein Effekt zweiter Ordnung, also eine Energiedichte der
Welle.

**A3 – Bindungslokales Gesetz.** M = Σ_d N_d² u_d u_dᵀ / 4 (Summe über die 12 Richtungen) gibt die Spurantwort exakt
wieder, E_g auf 0,9 % und T_2g auf 4,5 %. Mit w = N_d statt N_d² liegt der Fehler bei 50 %. Die Hop-Rate einer
Bindung ist also das Produkt der Freianteile ihrer beiden Knoten – dieselbe Zählung, die isotrop γ = 1 ergab.
Die zwei kubischen Konstanten (E_g 0,5, T_2g 1,0) stellen kein Hindernis dar: Die Basis ist vollständig, und S wird
aus der gemessenen Antwort gesetzt.

## B. Quelle: die Last einer Bindung ist der Spannungstensor

Die Bindungslast wird nach Hellmann–Feynman definiert, L_a = ∂E/∂ln N_a. Das ist dasselbe Prinzip wie bei der
Massenlast e_M = μ ∂E/∂μ in Runde 6. Verglichen wird mit (N²/2) u_aᵀ T u_a, wobei T^ij = c′² p^i p^j / E ist
(λ(N) kompensiert, 4 Richtungen je |k|):

| μ | \|k\| = 0,02 | 0,05 | 0,1 | 0,2 |
|---|---|---|---|---|
| 0 (Licht) | 4,4e-3 | 1,1e-2 | 2,1e-2 | 3,9e-2 |
| 0,03 | 2,9e-2 | 3,0e-2 | 3,4e-2 | 4,6e-2 |

(Größte Abweichung relativ zum größten u T u.) Für Licht geht die Abweichung linear in k gegen 0. Das passt zur
N-Abhängigkeit des Gitter-Hintergrunds −αk² (relativ O(k); nicht separat geprüft). Mit Masse bleibt ein k-unabhängiger Rest von 2,9 %, derselbe Typ
Wilson-artiger Term wie bei der Feinstruktur (open15).

Damit ist der Kreis geschlossen wie bei Lapse und Masse: Wie das Teilchen auf das Gate einer Bindung reagiert, legt
zugleich fest, wie stark es diese Bindung belastet. Das Kontinuumsgegenstück ist T^ij = (2/√g) δS/δg_ij.

## Was das für Gravitationswellen bedeutet

In harmonischer Eichung der linearisierten ART gilt □h̄_μν = −16πG T_μν. Das Modell reproduziert davon bereits:

- h₀₀ = ½(h̄₀₀ + h̄_kk), Quelle T₀₀ + T_kk (Runde 6) ✓
- h₀ᵢ, Quelle T₀ᵢ (Mitnahme-Hop, C = 4) ✓

Für die Raummetrik verlangt die ART h_ij = h̄_ij + ½δ_ij(h̄₀₀ − h̄_kk). Das Modell setzt bisher h_ij = h₀₀ δ_ij
(A = 1/N). Die Differenz ist genau

  **H_ij = h̄_ij − δ_ij h̄_kk, Quelle T_ij − δ_ij T_kk.**

Für ruhende Materie ist sie O(v⁴). Alle PPN-Ergebnisse von Paper 3 (γ, β, Periheldrehung, Lense–Thirring,
Feinstruktur) bleiben deshalb unverändert. Das Bindungs-Gate kann H_ij tragen (A), und seine Quelle ist die
gemessene Bindungslast (B). **Eingeführt** wird dabei, wie schon bei Φ und b, die Feldgleichung (Wellengleichung pro
Bindungsachse mit derselben Geschwindigkeit c_g).

Mit allen drei Feldern ist das System auf linearer Ordnung die linearisierte ART in harmonischer Eichung. Die
Nicht-TT-Anteile der Fernfelder sind dann Eichanteile und für einen kovariant gekoppelten Detektor unsichtbar. Genau
hier kann das Gitter abweichen: Ein Eichanteil ist nur unsichtbar, wenn (i) die Quelle auf dem Gitter erhalten ist,
(ii) alle Felder dieselbe Ausbreitung haben und (iii) der Walker auf alle Komponenten kovariant reagiert. Jede
Verletzung macht aus Eichanteilen messbare Zusatzpolarisationen (Atmungsmode, Longitudinalmode).

## Fahrplan

1. **GW2 – Detektor.** Walker-Interferometer (wie Runde 5 C3, jetzt mit Bindungs-Gate) in einer reinen TT-Welle und
   in einer reinen Eichwelle h_μν = ∂_μξ_ν + ∂_νξ_μ, bei der N, b und das Bindungs-Gate zusammen gestört werden. Die
   ART verlangt Antwort auf h₊ und null Antwort auf die Eichwelle. Gemessen wird der Gitterrest als Funktion von k
   und Wellenlänge. Billig (Bloch, zeitabhängig).
2. **GW3 – Feld.** Die drei Wellengleichungen (Φ, b, Bindungsfeld) auf dem 3D-Gitter mit einer erhaltenen
   rotierenden Quadrupolquelle: Fernfeld 1/r, Winkelverteilung von h₊ und h× ((1 + cos²θ)/2, cos θ), harmonische
   Bedingung, Riemann-Projektion auf die Eardley-Klassen (nur Ψ₄ erlaubt). Vergleichswert: das veröffentlichte
   Modell ohne Bindungsfeld, das Ψ₂ ≠ 0 und Atmungsmode zeigen muss. Nur Felder, ~1 GB.
3. **GW4 – Rückwirkung.** Abgestrahlte Leistung aus der Rückwirkungskraft des retardierten Feldes auf die Quelle,
   verglichen mit der Quadrupolformel 32/5 G μ² a⁴ ω⁶.

Offen bleibt, ob die Feldgleichungen selbst aus einer lokalen Regel folgen. Das ist dieselbe Lücke wie bei Φ und b.
