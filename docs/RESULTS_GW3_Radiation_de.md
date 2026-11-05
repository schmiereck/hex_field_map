# Gravitationswellen 3: Was die Feldgleichungen abstrahlen

Skript: `src/phase_bit_s4_open19.py` (~10 min, ~1,3 GB), Daten: `data/phase_bit_s4_open19.npz`, Log
`data/phase_bit_s4_open19.log`. Die Lock-in-Felder (330 MB) liegen nicht im Repo. `GW3_SCRATCH=<datei>` speichert sie,
`python phase_bit_s4_open19.py reanalyze <datei>` wertet sie neu aus.
Vorgänger: [`RESULTS_GW1_BondTensor_de.md`](RESULTS_GW1_BondTensor_de.md), [`RESULTS_GW2_Detector_de.md`](RESULTS_GW2_Detector_de.md).

## Frage

GW2 hat gezeigt: Der Detektor im Gitter ist bei k → 0 kovariant und sieht genau das, was die Felder tragen. Die
Polarisationsfrage entscheidet sich deshalb hier: Strahlt das Gitter nur die zwei TT-Moden ab, oder auch Atmungs-,
Longitudinal- und Vektormoden?

## Aufbau

Alle Felder des Modells sind Linearkombinationen der spurumgekehrten Störung h̄_μν mit □h̄_μν = −16π T_μν:

| Feld | Metrik | Quelle |
|---|---|---|
| Lapse Φ (Runde 6) | h₀₀ = ½(h̄₀₀ + h̄_kk) | T₀₀ + T_kk |
| Mitnahme-Hop b (g0i) | h₀ᵢ = h̄₀ᵢ | T₀ᵢ |
| Raum, veröffentlicht (A = 1/N) | h_ij = h₀₀ δ_ij | – |
| Raum mit Bindungsfeld (GW1) | h_ij = h₀₀ δ_ij + H_ij, H_ij = h̄_ij − δ_ij h̄_kk | T_ij − δ_ij T_kk |

Die zehn Komponenten h̄_μν laufen mit derselben Gitter-Wellengleichung:

- 160³, h = 1, dt = 0,5, 7-Punkt-Laplace, Leapfrog, c = 1.
- Absorbierender Schwamm von 32 Zellen (2 Wellenlängen, kubisches Profil).

Die Quelle ist ein rotierendes m=2-Muster, kompakt und identisch erhalten über ein Superpotential:
Ψ^{0i0j} = A_ij(t) g(x) mit A = Re[a e^{−iΩt}], a = (e_x + i e_y)(e_x + i e_y)ᵀ und g gaußförmig (Breite 2). Daraus folgen
T⁰⁰ = A_ij ∂_i∂_j g, T⁰ʲ = −Ȧ_ij ∂_i g und T^{ij} = Ä_ij g. Die Frequenz ist Ω = 0,393 (Wellenlänge 16), die Erhaltung
gilt auf 3e-7.

Nach dem Einschwingen wird jede Komponente über zwei Perioden eingelockt (komplexe Amplitude). Ausgewertet wird der
eichinvariante elektrische Riemann-Tensor E_ij = R₀ᵢ₀ⱼ = −½(h_ij,00 + h₀₀,ij − h₀ᵢ,0j − h₀ⱼ,0i) im Rahmen (e_θ, e_φ, n),
zerlegt in:

- Tensor: plus, cross
- Atmung: transversale Spur
- longitudinal: nn
- Vektor: n e

Verwendet werden spektrale Ableitungen auf Gitterpunkten. Kontrollen: Finite-Differenzen-Auswertung mit Interpolation
und die gittereigenen Symbole (2/h) sin(kh/2) und (2/dt) sin(Ωdt/2), für die eine Gitter-Ebenenwelle exakt lichtartig
ist. Alle drei Auswertungen stimmen überein.

**Zwei Fehlversuche, festgehalten:**

1. Der Schwamm war mit 16 Zellen schmaler als die Wellenlänge (24). Die Folge waren stehende Wellen, |h×| ≈ 0,1|h₊| am
   Pol statt 1, und eine um 5–50 % verletzte harmonische Bedingung.
2. Die Quelle mit T⁰ⁱ = ∇χ und spektral gelöster Spannung hatte Ausläufer ~1/r³ bis in den Schwamm. Das Fernfeld war
   dann nicht 1/r.

## Ergebnis 1: Das TT-Feld ist korrekt

| Prüfung | Ergebnis |
|---|---|
| h̄₊, h̄× gegen die retardierte Fernfeldformel (4/r) e^{iΩr} T̂_ij(Ωn) | 0,93 … 1,08 (Rest: Reflexionen des Schwamms) |
| Winkelverteilung bei r = 30: \|plus\| gegen (1 + cos²θ)/2 | 1,000 / 0,876 / 0,628 / 0,500 (Soll 1 / 0,875 / 0,625 / 0,5) |
| \|cross\| gegen cos θ | 1,020 / 0,883 / 0,516 / 0,000 (Soll 1 / 0,866 / 0,5 / 0) |
| r·\|E₊\| bei r = 20 / 30 / 40 (θ = 60°) | 2,703 / 2,704 / 2,616 |
| harmonische Bedingung ∂^μ h̄_μν / \|Ω h̄₀₀\| | 0,2–2,5 % |

## Ergebnis 2: Polarisationsgehalt (relativ zum GR-TT-Signal am selben Ort)

Mittel über φ, spektrale Auswertung:

| Modell | Anteil | θ | r = 20 | r = 30 | r = 40 | Verhalten |
|---|---|---|---|---|---|---|
| mit Bindungsfeld | Vektor | 90° | 0,53 | 0,31 | 0,27 | ~ 1/(kr), Nahfeld (Ψ₃ ~ 1/r²) |
| | longitudinal | 90° | 0,20 | 0,12 | 0,05 | ~ 1/r², Nahfeld (Ψ₂ ~ 1/r³) |
| | Atmung | 90° | 0,10 | 0,06 | 0,03 | = ½ longitudinal (spurfrei im Vakuum) |
| | alle Nicht-TT | 0° | ≤ 3e-5 | ≤ 6e-5 | ≤ 8e-5 | |
| veröffentlicht (A = 1/N) | TT | alle | 0,26 | 0,17 | 0,13 | fällt ab, Nahfeld: keine TT-Strahlung |
| | Atmung | 90° | 1,12 | 1,07 | 1,05 | **konstant: Strahlung** |
| | Atmung | 60° / 30° | 0,52 / 0,11 | 0,50 / 0,11 | 0,49 / 0,11 | konstant |
| | longitudinal | 90° | 2,24 | 2,14 | 2,11 | konstant: Strahlung |
| | Vektor | 90° | 1,92 | 2,00 | 2,05 | konstant: Strahlung |

- **Mit Bindungsfeld:** Alles, was bei 1/r ankommt, ist TT. Die Nicht-TT-Anteile fallen wie das Nahfeld der ART ab
  (Peeling: Ψ₃ ~ 1/r², Ψ₂ ~ 1/r³; r·vec ≈ 10,6 / 9,3 / 10,8 ≈ konstant). Auf der Achse sind sie 1e-7 … 1e-5.
- **Veröffentlichtes Modell:** Es gibt keine TT-Strahlung. Der Rest fällt wie 1/r ab, ist also Nahfeld. Abgestrahlt
  werden stattdessen eine Atmungsmode (gleich groß wie das GR-Signal bei θ = 90°), eine Longitudinalmode (doppelt so
  groß) und Vektormoden. Das ist der quantitative Befund zu der in Paper 3 offen gelassenen Frage: Ohne Tensorfeld ist
  das Modell im Strahlungssektor eine Skalar-Vektor-Theorie. Die LIGO/Virgo-Polarisationstests und der Doppelpulsar
  schließen das aus. Das Sonnensystem allein (PPN) sieht den Unterschied nicht.

## Was das Gitter hier (nicht) zeigt

Gittereigene und Kontinuumssymbole ergeben dieselben Zahlen. Eine gitterspezifische Verletzung der Eichstruktur von
der Ordnung (Ωh)²/24 ≈ 0,6 % liegt unter dem Reflexionsrest von ≈ 5 % und wird hier nicht aufgelöst. Die Felder
gehorchen *eingeführten* Wellengleichungen. Das Ergebnis lautet deshalb: Mit dem Bindungsfeld aus GW1 trägt das
Gitter auf linearer Ordnung genau die zwei Polarisationen der ART, und der Detektor (GW2) misst sie korrekt. Woher die
Feldgleichungen kommen, bleibt dieselbe Lücke wie bei Φ und b.

## Nächster Schritt

GW4 – abgestrahlte Leistung: Rückwirkungskraft des retardierten Feldes auf die Quelle oder Energiefluss der
TT-Moden, im Vergleich mit der Quadrupolformel. Zum Vergleich: der Skalar-Vektor-Fluss des veröffentlichten Modells
(Doppelpulsar).
