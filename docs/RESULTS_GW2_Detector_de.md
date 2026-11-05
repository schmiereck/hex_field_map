# Gravitationswellen 2: der Detektor

Skript: `src/phase_bit_s4_open18.py` (Teile A–D, ~1 min), Daten: `data/phase_bit_s4_open18.npz`, Log `data/phase_bit_s4_open18.log`.
Vorgänger: [`RESULTS_GW1_BondTensor_de.md`](RESULTS_GW1_BondTensor_de.md).

## Frage

Reagiert ein Gitterphoton auf alle zehn Metrikkomponenten so, wie ein masseloses Teilchen in der ART? Das ist die
Voraussetzung dafür, dass ein Detektor im Gitter nur die TT-Polarisationen sieht und Eichanteile nicht.

Für ein Photon in Richtung n mit n^μ = (1, n) verlangt die ART (aus g^μν k_μ k_ν = 0):

  **δ ln ω = −½ h_μν n^μ n^ν = −½ (h₀₀ + 2h₀ₙ + hₙₙ).**

## Aufbau

Kompensierter H2-Walker (2O*, ε = 0,3, μ = 0). Hintergrund N_w = N_h = N₀ = 0,75 mit λ(N₀) = −2,627, daraus
c_eff = N₀² c′. Die Komponenten werden einzeln gestört, homogen im Raum (Bloch):

| Metrik | Gitter | Kalibrierung |
|---|---|---|
| h₀₀ | Wait-Gate N_w = N₀(1 + φ) | h₀₀ = −2φ |
| h₀ᵢ | Mitnahme-Hop exp(−i dt b·p_lat), p_lat = Σ_s s sin(k·s)/(4\|s\|²) (Bloch-Symbol von drag3d) | b = −c_eff h₀ᵢ |
| h_ij | Stay-Gate pro Bindungsachse N_h,d = N₀(1 + δ_a) | 6×6-Abbildung δ → h_ij, bei k = 0,004 gefittet |

Ein Freiheitsgrad fehlt in der Tabelle: die Normierung der Bindungsabbildung. Sie folgt aus dem Fit, nicht aus
einer Regel.

## A. Statische Antworten: Gitter / ART

Mittel und Bereich über 13 Richtungen:

| Komponente | k = 0,01 | 0,03 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| Lapse h₀₀ | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Mitnahme h₀ₓ | 1,0000 | 1,0000 | 0,9996 | 0,9984 | 0,9965 |
| Bindung Spur | 1,0013 | 1,0057 | 1,0203 | 1,0399 | 1,0577 |
| Bindung E_g | 1,0013 | 1,0057 | 1,0205 | 1,0404 | 1,0584 |
| Bindung T_2g | 1,0013 | 1,0057 | 1,0206 | 1,0407 | 1,0593 |

- Die Lapse ist bei jedem k exakt.
- Der Mitnahme-Hop weicht um O(k²) ab (p_lat gegen das Band).
- Das Bindungs-Gate weicht um ≈ 0,2·k ab, also **linear in k**. Das ist derselbe Term, der schon in Runde 5 (C3,
  Kopplung über die Münze) und bei der Bindungslast (GW1 B) auftauchte.

## B. Was eine Welle in Richtung n einem Photon zeigt

Alle Werte relativ zum TT-Signal (ART: transversal −½, entlang n 0, Eichanteile 0):

| n | k | TT transversal | TT entlang n | Eichwelle ξ₀ | Eichwelle ξₙ |
|---|---|---|---|---|---|
| x | 0,01 | −0,5007 | 8e-8 | 2e-6 | 2,6e-3 |
| x | 0,1 | −0,5099 | 5e-8 | −2,6e-4 | 3,9e-2 |
| x | 0,3 | −0,5273 | 4e-8 | −2,9e-3 | 1,1e-1 |
| (111) | 0,01 | −0,5007 | −6e-9 | −5e-6 | 2,6e-3 |
| (111) | 0,3 | −0,5297 | −1e-8 | −7,6e-3 | 1,2e-1 |

- Ein mitlaufendes Photon sieht die TT-Welle nicht. Das stimmt bis auf 1e-7.
- Die Eichwelle ξ₀ (Lapse und Mitnahme) bleibt bis auf O(k²) unsichtbar.
- Die Eichwelle ξₙ (Mitnahme und Bindung hₙₙ) hinterlässt einen Rest von ≈ 0,4·k. Er stammt direkt aus dem linearen
  k-Term des Bindungs-Gates.

Für einen kompakten Puls ist jede lineare Antwort auf eine Eichwelle entlang eines *nicht* mitlaufenden Wegs eine
totale Ableitung und integriert sich zu null. Das gilt unabhängig von den Kopplungskonstanten. Nur das mitlaufende
Photon prüft also das Verhältnis von Lapse, Mitnahme und Bindungs-Gate. Das tut die Tabelle.

## C. Echtzeit: Interferometer in h₊(t)

Homogenes h₊(t) = h₀ sin(Wt) mit h₀ = 1e-3 und W = 0,02, über eine halbe Periode. Die Welle läuft entlang x, also
h_yy = −h_zz. Gemessen wird die Phase gegen die freie Entwicklung:

| \|k\| | (y − z) / ART | Phase x | Leckage |
|---|---|---|---|
| 0,03 | 1,0056 | −8,66e-5 | 8e-12 |
| 0,1 | 1,0199 | −8,60e-5 | 5e-11 |
| 0,3 | 1,0545 | −8,45e-5 | 4e-11 |

- Das Differenzsignal der Arme entspricht der ART mit demselben linearen k-Term wie in A.
- Die Phase des Photons entlang x ist k-unabhängig und **exakt zweiter Ordnung**: −1,096 h₀² ∫sin² dt, gleich für
  h₀ = ±1e-3 und 2e-3. Das ist die Vakuumverschiebung O(h²) des Bindungs-Gates aus GW1. Sie verschiebt alle Photonen
  gleich und fällt in der Armdifferenz heraus.
- Es gibt keine Übergänge aus dem Band (Leckage ≤ 5e-11).

## D. Woher der lineare k-Term kommt: λ pro Bindung nachführen

λ steht fest bei λ(N₀). Eine Bindungsverzerrung ändert deshalb den Gitter-Hintergrund −αk², und der relative Effekt
ist O(k). Als Abhilfe wird λ bindungslokal nachgeführt: Die W90-Kopplung des Richtungspaars (d, d′) läuft bei
λ(√(N_d N_d′)), der Gegenterm pro Richtung bei λ(N_d). Danach wird die Abbildung neu kalibriert.

| Komponente | k = 0,01 | 0,03 | 0,1 | 0,3 |
|---|---|---|---|---|
| Bindung Spur | 1,0000 | 1,0001 | 0,9998 | 0,9947 |
| Bindung E_g | 1,0000 | 1,0001 | 1,0005 | 1,0006 |
| Bindung T_2g | 1,0048 | 1,0207 | 1,0761 | 1,2325 |

Für Spur und E_g ist der lineare Term weg; es bleibt O(k²). Damit ist auch die Eichwelle ξₓ für eine Welle entlang
einer Würfelachse auf 5e-5 (k = 0,01) bzw. 1,9e-4 (k = 0,1) unsichtbar. Für T_2g hilft die Paarregel nicht, die
Abweichung wird sogar größer (≈ 0,5·k). Der Grund ist geometrisch: Die beiden Achsen eines T_2g-Musters, etwa
(1,1,0) und (1,−1,0), bilden selbst ein 90°-Paar mit entgegengesetzter Störung. Ihr Produkt N_d N_d′ ändert sich in
erster Ordnung nicht. Wellen entlang der Diagonalen behalten deshalb einen Eichrest von ≈ 0,5–0,9·k.

## Bewertung

1. **Bei k → 0 ist der Detektor kovariant.** Alle zehn Komponenten folgen δ ln ω = −½ h_μν n^μ n^ν mit einer
   einzigen Kalibrierung. Ein mitlaufendes Photon sieht weder TT- noch Eichwellen, ein transversales sieht −h₊/2.
2. **Gitterrest:** Bei endlichem k bleibt eine Verletzung der Eichinvarianz von relativer Größe O(ka) (≈ 0,2–0,5·k).
   Sie kommt von der Kopplung der Bindungs-Gates an den Gitter-Hintergrund −αk². Für optische Photonen bei Planck-a
   ist das 1e-28 und damit unbeobachtbar. Begrifflich ist es aber eine echte Gittersignatur: Hochenergetische Photonen
   sähen einen Bruchteil ~ka der Eichanteile als physikalisches Signal (Zusatz-Polarisationen, abhängig von der Energie).
3. Mit λ pro Bindungspaar sinkt der Rest auf O(k²), aber nur für Spur und E_g. Für T_2g fehlt noch eine lokale
   Nachführregel. Kandidat ist eine Kopplung der 60°-Paare an die Bindungsverzerrung (wie die Münzgewichte s_d s_d′
   aus Runde 5).
4. Die Polarisationsfrage entscheidet sich damit nicht im Detektor, sondern in den Feldern (GW3). Ein kovarianter
   Detektor sieht genau das, was die Felder tragen. Erfüllen die Felder die harmonische Bedingung nicht, entstehen
   physikalische Nicht-TT-Moden (Atmungs-, Longitudinalmode), und der Detektor misst sie.

## Nächster Schritt

GW3: Die drei Wellengleichungen (Φ, b, Bindungsfeld) laufen auf dem 3D-Gitter mit einer erhaltenen rotierenden
Quadrupolquelle. Geprüft werden Fernfeld, harmonische Bedingung, Winkelverteilung (1 + cos²θ)/2 und cos θ sowie die
Eardley-Klasse. Zum Vergleich läuft das veröffentlichte Modell ohne Bindungsfeld mit, das eine Atmungsmode zeigen muss.
