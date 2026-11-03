# S4, sechste Runde: Quelle T₀₀ + T_ii aus der Rechenlast, kausale Gravitation, Spiegel für Masse, Landau-Zweige

*Paper 3. Code: `src/phase_bit_s4_open6.py` (Teile `A`, `B`, `C`, `D`) und
`src/phase_bit_s4_open6_figs.py`. Daten: `data/phase_bit_s4_open6.npz`.
Abbildung: `figures/phase_bit_s4_open6.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open5_de.md`](RESULTS_PhaseBit_S4_Open5_de.md).*

Speicher: Alles läuft Schritt für Schritt mit dünn besetzten Matrizen (x-Ketten mit
bis zu 1200 Lagen) oder mit 12×12-Blochmatrizen und braucht wenige 100 MB. Frei waren
3.8 GB.

Modell wie in Runde 4/5: **2O\*** (ε = 0.3) mit Vakuum-Gegenterm und Warte-Puffer
(N₀ = 0.75). Teil D nutzt minimales SU(2).

## 1. Eine Quelle wie in der ART: der Bewegungsanteil der Rechnung zählt doppelt

In Runde 5 war die Quelle die Last e(x), also die Energiedichte. Damit zog eine
bewegte Quelle mit E, die ART verlangt aber E(1 + v²), weil im schwachen Feld
T₀₀ + T_ii die Quelle von g₀₀ ist. Für ein Dirac-Teilchen ist T_ii = ψ†(α·p)ψ der
kinetische Anteil, also gilt T₀₀ + T_ii = 2E − (Massenanteil).

**Massen-Last:** Der Massenanteil der Energie ist μ·∂E/∂μ (Hellmann–Feynman). Er ist
lokal messbar:

```
e_M(x) = μ · Re[ i (Uψ)†(x) · (∂U/∂μ ψ)(x) ] / Δt        ∂U/∂μ: Nachbarschaftsoperator
Quelle s(x) = 2 e(x) − e_M(x)
```

Für einen Dirac-Zustand ergibt das E + p·v = E(1 + v²), Licht zählt also doppelt.
Anschaulich: Der Teil der Rechnung, der den Ort ändert, zählt zweimal, der Teil, der
nur die innere Uhr dreht (die Masse), einmal.

Aufbau wie in Runde 5: x-Kette mit 1000 Lagen, Testteilchen in Ruhe (μ = 0.03), Quelle
in Querbewegung (Bloch-k_y), 1D-Green-Funktion, 200 Takte. Das Lastintegral Q_M
stimmt mit N₀m²/E auf 1 bis 3 %, bei Licht ist es exakt 0.

| Quelle | E | Quelle 2e − e_M / Quelle e | ART 1 + v² |
|---|---|---|---|
| ruhend (m = 0.06) | 0.060 | 1.013 | 1 |
| k_y = 0.3 | 0.182 | **1.887** | 1.892 |
| k_y = 0.6 | 0.344 | **1.964** | 1.970 |
| Licht, k_y = 0.2 | 0.115 | **1.991** | 2 |
| Licht, k_y = 0.4 | 0.229 | **1.990** | 2 |

Relativ zur ruhenden Quelle: 5.62 / 11.02 (massiv) und 3.75 / 7.44 (Licht) gegen
E(1 + v²)/m = 5.74 / 11.31 und 3.84 / 7.63. Die 2 bis 2.5 % kommen größtenteils aus
der Normierung: Bei der ruhenden Quelle liegt Q_M 1 % unter Q, damit ist s dort 1.3 %
zu groß. **Die Quelle der linearisierten ART folgt aus der Rechenlast**, gemessen mit
zwei lokalen Größen.

## 2. Kausale Gravitation: eine lokale Feldgleichung

Bisher kam Φ aus der instantanen Green-Funktion. Jetzt lebt Φ auf den Knoten und
folgt einer lokalen Wellengleichung (Leapfrog, Courant-Zahl 0.47), beginnend bei Φ = 0:

```
Φ_tt − c_g² Φ_xx = −c_g² κ s(x),        c_g = c′
```

Aufbau: Testteilchen μ = 0.06, Quelle μ = 0.12, Abstand D = 159 (260 Lagen),
Pakete mit Breite 60 Lagen, x-Kette mit 1200 Lagen, 900 Takte.

| t | 100 | 200 | 300 | 400 |
|---|---|---|---|---|
| Verschiebung, instantanes Feld | +0.905 | +3.63 | +8.15 | +14.46 |
| Verschiebung, Wellengleichung | +0.005 | +0.101 | +0.626 | +2.18 |
| Vorhersage a·P(\|Δx\| < c′t) | +0.031 | +0.166 | +0.749 | +2.38 |

- **Das Testteilchen spürt die Quelle erst, wenn das Feld ankommt.** Ein Fit mit
  scharfem Einsatz gibt t₀ = 245, bei D/c′ = 276. Die Vorhersage mit den endlichen
  Paketbreiten (Beschleunigung mal Anteil der Quelle innerhalb des Lichtkegels) trifft
  den Verlauf mit einem rms-Rest von 0.12, das instantane Feld liegt um 7.1 daneben.
  Der Rest von etwa 8 % bei t = 400 kommt vermutlich von der Gitterdispersion der
  Wellengleichung, die kurze Wellen etwas langsamer als c′ führt.
- Umgekehrt spürt die schwere Quelle das leichte Teilchen ebenfalls ab t₀ ≈ 242.
- Spät (t > 360) liegt die Beschleunigung bei 0.885 des instantanen Werts, weil ein
  Teil der breiten Quelle noch außerhalb des Kegels liegt.
- **Grenze 1D:** Das retardierte Feld einer statischen Quelle wächst in 1D linear in
  der Zeit (Φ ∝ c′t innerhalb des Kegels). Der Gradient ist richtig, aber der
  gleichförmige Anteil würde den Lapse überall senken. Hier wird Φ am Ort des
  jeweiligen Teilchens referenziert, damit der Gradient allein wirkt. Erst in 3D fällt
  das Feld wie 1/r ab.

## 3. Der Spiegel für massive Teilchen

Aufbau wie Runde 5, Teil A2: N fällt von 0.9 (x > 40) auf N_min = 0.04 (x < −40). Ein
massives Teilchen fällt aus der Ruhe bei x = 20.

| Teilchen | E′ | Band-Maximum | Umkehr vorhergesagt | Simulation |
|---|---|---|---|---|
| μ = 0.15 | 0.203 | 1.395 | N = 0.142, x = −30.5 | tiefster Punkt −30.2 (t = 590), zurück bis +18.8 (t = 1175), Start +20; 85 % wieder oben |
| μ = 0.03 | 0.046 | 1.635 | N = 0.026 < N_min: **keine Umkehr** | läuft ins Innere, 98 % jenseits; bei t = 2000 bei −58 |
| masselos | 0.155 | 1.672 | N = 0.089, x = −35.5 | tiefster Punkt −39.7, zurück, 26 % am Start |

- **Springender Ball:** Das schwere Teilchen fällt, wird an der Gittergrenze gespiegelt
  und steigt fast wieder auf die Starthöhe (Energieerhaltung, 98 % der Fallhöhe). Dann
  fällt es erneut (zweites Minimum bei t ≈ 1760).
- **Kriterium:** Gespiegelt wird nur, wer f_{N_min}(E_max) ≥ E′ erreicht, grob
  E′ > N_min·E_max. Ein Teilchen mit wenig Koordinatenenergie bleibt unter der
  Bandkante und läuft in das Gebiet mit kleinem N, wo es sehr langsam wird. Ein
  massives Teilchen bleibt dabei kohärenter als ein masseloses Paket: Das masselose Paket zerfällt
  am Umkehrpunkt stärker (26 % gegen 85 % kehren zurück).

## 4. Landau-Zweige: Es gibt keine globale Lücke

In Runde 5 war die Windung von det U(k_z) null, aber der Verlauf des einzelnen Zweigs
blieb offen. Die Bandstruktur des Helizitätssektors +1 ohne Feld (48³ Punkte in der
Zone, Minima verfeinert) zeigt, warum diese Frage keinen eigenen Inhalt hat:

- Die 12 Bänder überdecken den ganzen Quasienergie-Kreis. Bei jeder Quasienergie liegen
  1 bis 7 Bänder. Die Bereiche reichen von [−6.28, −2.08] (Band 0) bis [+1.26, +6.28]
  (Band 11).
- Benachbarte Bänder berühren sich: Die kleinste Lücke ist 0 bei 9 von 12 Paaren und
  höchstens 6·10⁻³ bei den übrigen, auch über den Zonenrand. Band 11 erreicht oben und
  Band 0 unten genau ±2π, eine Lücke am Zonenrand gibt es also nicht.
- Eine Zählung der Weyl-Knoten mit Chiralität (Berry-Fluss) war nicht sauber. Viele
  Berührungen liegen auf Knotenlinien oder an mehrfach entarteten Punkten (die
  vierfachen Niveaus bei −0.57 und +1.26), dort ist der Fluss eines einzelnen Bandes
  nicht definiert. Diese Liste wird nicht berichtet.

**Folgerung:** Ohne Lücke ist nur der Netto-Spektralfluss eine topologische Größe, und
der ist null (Runde 5). Die Landau-Zweige der beiden Knoten hängen über ein Netz von
Bandberührungen zusammen. „Direkt oder über den Zonenrand“ ist keine Frage, die das
Spektrum eindeutig beantwortet: Bei einer Fortsetzung durch das Netz entscheidet an
jeder Berührung die Durchfahrgeschwindigkeit (Landau–Zener), welchen Weg der Zustand
nimmt. Der Punkt ist damit abgeschlossen.

## Antwort

1. **ART-Quelle:** s = 2e − e_M, Rechenlast doppelt minus Massenanteil, beide lokal
   gemessen. Eine bewegte Quelle zieht mit E(1 + v²) (1.887 / 1.964 gegen 1.892 / 1.970),
   Licht doppelt (1.991). Das ist T₀₀ + T_ii.
2. **Kausale Gravitation:** Mit einer lokalen Wellengleichung für Φ wirkt die Quelle erst
   nach D/c′. Die Vorhersage mit Paketbreiten trifft die Simulation.
3. **Spiegel für Masse:** Ein fallendes massives Teilchen wird an der Gittergrenze
   reflektiert und springt fast auf die Starthöhe zurück (−30.2 gegen −30.5 als
   Umkehrpunkt). Mit zu wenig Energie gibt es keine Umkehr, das Teilchen läuft ins
   Innere.
4. **Landau-Zweige:** Der Helizitätssektor hat keine globale Quasienergie-Lücke, die
   Bänder berühren sich paarweise. Topologisch bleibt nur der Netto-Fluss null.

## Offene Punkte

*Weiter mit der Hypothese H1 und der Eigenzeit: [`RESULTS_H1_Antimatter_ProperTime_de.md`](RESULTS_H1_Antimatter_ProperTime_de.md).*

- **3D:** Die lokale Feldgleichung in drei Dimensionen (1/r-Potential, keine linear
  wachsende Konstante) mit der Quelle s = 2e − e_M. Damit erst ein echter Test von Newton
  plus Retardierung. Der Speicher ist der Engpass: Ein 3D-Gitter mit 48 Zuständen pro
  Knoten braucht bei 64³ etwa 200 MB pro Zustand.
- **Selbstkonsistente räumliche Metrik:** A = 1 − Φ aus derselben Quelle für bewegte
  Testteilchen. Dann wäre die Lichtablenkung durch eine bewegte Quelle (Gravitomagnetismus)
  zugänglich.
- **Dynamische Gravitationswellen:** Die Coin-Gewichte h als eigenes Feld mit
  Wellengleichung und einer Quadrupol-Quelle aus der Rechenlast.
- Der masselose Spiegel zerfällt stärker als der massive: Anteil der Konversion in
  andere Bänder an der Bandkante.
