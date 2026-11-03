# Beobachter im Gitter, Newton und Periheldrehung, Lense–Thirring

*Paper 3. Code: `src/phase_bit_s4_open13.py` (Teile `A`, `B1`, `B2`, `C1`, `C2`) und
`src/phase_bit_s4_open13_figs.py`. Daten: `data/phase_bit_s4_open13.npz`.
Abbildung: `figures/phase_bit_s4_open13.png`.
Vorgänger: [`RESULTS_Gravity3D_de.md`](RESULTS_Gravity3D_de.md).*

Modell: **2O\*** (ε = 0.3, λ\*, Vakuum-Gegenterm). Die 3D-Läufe nutzen 64³-Boxen
(complex64, 0.05 GB pro Register) und laufen je 2–8 Minuten.

## 1. Michelson–Morley im Gitter (Teil A)

Ein Beobachter im Gitter definiert Längen mit langwelligem Licht (Radar). Messbar ist
für ihn nur, ob zwei Sonden übereinstimmen, wenn er den Aufbau dreht. Das Verhältnis
seiner Radarzeiten mit zwei Lichtenergien ist Q(n̂) = v(k, n̂)/v(k_tief, n̂). Gerechnet
ist es über 2000 Richtungen, mit der Gruppengeschwindigkeit des oberen Weyl-Dubletts
(Hellmann–Feynman, exakt).

| k | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| Richtungsabhängigkeit von v (max − min)/Mittel | 1.9·10⁻⁵ | 1.1·10⁻⁴ | 3.6·10⁻⁴ | 8.7·10⁻⁴ | 1.5·10⁻³ | 5.7·10⁻² |
| Energieabhängigkeit 1 − ⟨v⟩/c′ | 6.2·10⁻⁵ | 3.9·10⁻⁴ | 1.6·10⁻³ | 6.4·10⁻³ | 2.7·10⁻² | 0.12 |
| Winkel zwischen Strahl und Wellenvektor (max) | 0.0006° | 0.004° | 0.013° | 0.04° | 0.04° | 1.0° |

- **Richtungsabhängigkeit:** Im gedrehten Aufbau ändert sich Q um 9·10⁻⁵ / 3.4·10⁻⁴ /
  8.5·10⁻⁴ bei k = 0.05 / 0.1 / 0.2 (gegen k = 0.02). Bei kleinem k geht das wie
  k^1.83 ≈ k², passend zu einem kubischen Term k³·Y₄ in E(k). Bei k ≈ 0.3 tauschen die
  schnellste und die langsamste Richtung ([111] und [100]) die Rollen.
- **Energieabhängigkeit, auch ohne Drehen messbar:** v = c′(1 − ξ(ka)²) mit ξ = 0.42 im
  Mittel und einem richtungsabhängigen Anteil von ±0.048 (etwa 11 %). Hochenergetisches
  Licht kommt später an, wie bei der Suche nach Lorentz-Verletzung mit Gammablitzen.
- **Für den Beobachter heißt das:** Mit langwelligem Licht findet er keine
  Vorzugsrichtung, bis auf (ka)². Seine „Kugeln“ aus Radarzeiten sind für alle
  langwelligen Signale dieselben. Erst Licht nahe der Gitterskala verrät das Gitter,
  zuerst durch die Energie- und danach durch die Richtungsabhängigkeit.

## 2. Newton und Periheldrehung

### B1: Kraftgesetz im 3D-Walker

Ein massives Paket (μ = 0.15, m = 0.3, σ = 5) ruht im Abstand r von einer Punktmasse
U = GM/√(r² + r_c²) mit GM = 0.25, r_c = 2 und N = e^{−U}. Die Masse darf außerhalb der
Box liegen, weil U analytisch ist. Gemessen wird F = d⟨k_x⟩/dt, mit Zitterbewegung bei
2E mitgefittet, gemittelt über beide Kramers-Partner.

| r | 6 | 9 | 13 | 18 | 26 |
|---|---|---|---|---|---|
| W: Simulation / lokale Kraft | 1.034 | 1.019 | 1.001 | 0.991 | 0.989 |
| H2: Simulation / lokale Kraft | 1.010 | 1.003 | 0.998 | 0.996 | 0.998 |
| H2 / W | 1.188 | 1.180 | 1.176 | 1.181 | 1.187 |

- **Der Walker folgt der lokalen Kraft der Metrik** −⟨∂H/∂x⟩ mit
  H = N√(m² + (N^n c′k)²) auf ≤ 3.4 %, ab r = 13 auf ≤ 1 %. Die lokale Kraft ist dabei
  über Dichte und Impulsbreite gemittelt.
- **Über Newton liegt die Kraft, weil die Bewegungsenergie mitgravitiert.** Das
  „ruhende“ Paket hat bei σ = 5 eine Impulsbreite mit v² ≈ 0.18. Mit der Quelle
  E(1 + v²) zieht es deshalb 8 % (W) bzw. 28 % (H2) stärker als m·GM/r². Daher auch
  H2/W = 1.18: Der Faktor (1 + v²) wirkt in H2 auf die Testmasse.
- Die Steigung von ln F gegen ln r ist −1.86 statt −1.96. Das Paket zerläuft in den
  32 Zeiteinheiten von σ = 5 auf etwa 8, damit ist der Vergleich mit dem Punktgesetz
  nicht sauber. Die lokale Kraft enthält das und stimmt.

### B2: Periheldrehung (klassische Grenze)

**Ein Wellenpaket auf einer Bahn ist auf dem Gitter nicht machbar.** Für eine
klassische Bahn muss der Drehimpuls L = m·r·v groß gegen ħ sein. Bei m = 0.3 ergibt
sich nur L ≈ 0.8 ħ (r = 10, GM = 0.25) bzw. 2 ħ (r = 30, GM = 0.5), und das Paket zerläuft
in 10–90 Zeiteinheiten bei einer Umlaufzeit von 700–2500. Es wäre ein
„Gravitationsatom“, keine Bahn.

Gerechnet sind deshalb die klassischen Bahnen des effektiven Hamiltonians des Modells,
H = N(r)·√(m² + c′²p²/A(r)²) (Integrator DOP853, Toleranz 10⁻¹²):

| Variante | γ, β | a = 400: Drehung / PPN | a = 1600: Drehung / PPN | Drehung / ART |
|---|---|---|---|---|
| H2: N = e^{−U}, A = 1/N | 1, 1 | 1.0106 | 1.0026 | 1.01 / 1.00 |
| nur Warten: N = e^{−U}, A = 1 | 0, 1 | 1.0042 | 1.0011 | 0.33 |
| linearer Lapse: N = 1 − U, A = 1/N | 1, 1/2 | 1.0160 | 1.0039 | 1.18 |

PPN-Formel: (2 + 2γ − β)/3 · 6πGM/p. Alle drei Varianten folgen ihr. Der Rest fällt wie
1/a, also wie die nächste nachnewtonsche Ordnung. **Die H2-Regel mit der Produktregel
(N = e^{−U}) gibt die volle ART-Periheldrehung von Merkur.** Reines Warten gäbe 1/3,
ein linear überlagerter Lapse 7/6.

## 3. Lense–Thirring

### C1: Feld einer rotierenden Kugel, Knotendrehung

- **Mitnahmefeld:** Eine starr rotierende Kugel (Radius 5) gibt aus der 3D-Feldgleichung
  für b (Quelle Impulsdichte, Faktor 4) den Dipol b = 2c′²(κ/4π)·J × r/r³, also den
  ART-Koeffizienten. Bei r = 8 / 10 / 12 / 15 ist das Verhältnis 0.991 / 0.981 / 0.958 /
  0.923; außen wirken die Spiegel der periodischen Box.
- **Knotendrehung:** Eine klassische Bahn (a = 300, e = 0.2, Neigung 60°) mit
  H ⊃ b·p und b = B·J × x/r³ dreht ihren Knoten mit dem Lense–Thirring-Wert
  B/(a³(1 − e²)^{3/2}), auf **1.001**. Dafür müssen a und e der tatsächlichen
  (relativistischen) Bahn eingesetzt werden; mit denen der newtonschen Startbedingung
  waren es 1.04.

### C2/C3: Mitnahme im 3D-Walker, und der Kreisel

**Lokaler Mitnahme-Operator in 3D:** Die Mitnahme ist eine Ziegelmauer aus reellen
Bindungsdrehungen zwischen Nachbarn entlang der 6 FCC-Richtungen. Der Winkel einer
Bindung ist Δt·(b·s)/(8ℓ²), ausgewertet in der Bindungsmitte. Der Operator ist für alle
inneren Zustände gleich, exakt unitär und strikt lokal, und für gleichförmiges b gibt
er H ⊃ b·κ(k) mit κ ≈ k.

**Test:** starr rotierendes Mitnahmefeld b = Ω ẑ × r mit Ω = 0.004. Ein ruhendes Paket
(μ = 0.15, σ = 4, Spin entlang +x) startet bei R = 8, 60 Takte.

| | Bahnwinkel-Rate | Spin-Winkel-Rate |
|---|---|---|
| nur skalare Mitnahme | 0.00406 | **0.00026** |
| Mitnahme + innere Drehung exp(−iΔt ½∇×b·J) | 0.00406 | **0.00399** |
| ART (mitgeführtes Bezugssystem) | Ω = 0.004 | Ω = 0.004 |

- **Das Paket wird mitgeführt** (0.00406 gegen 0.004, Norm 0.974; bei längeren Läufen
  erreicht das zerlaufende Paket den offenen Rand).
- **Ohne Zusatzregel dreht der Spin nicht mit.** Die skalare Mitnahme b·p ist für eine
  starre Drehung Ω·L: Sie koppelt nur an den Bahndrehimpuls. Ein Kreisel würde den
  Gitterraum als rotierend erkennen. Die ART verlangt die Kopplung an J = L + S
  (gemessen von Gravity Probe B).
- **Die Regel:** Mitnahme ist Transport mit einem Strömungsfeld. Ein mitgeführtes
  Element dreht sich dabei mit der halben Wirbelstärke ½∇ × b. Dazu braucht es den
  inneren Drehgenerator.
  - **Der Spin-Index allein reicht nicht.** P(1⊗σ/2)P auf dem ruhenden Dublett hat für
    das Teilchen g = −1/3: Es dreht mit einem Drittel und im falschen Sinn. Für das
    Antiteilchen ist g = +1. Der größte Teil des Teilchenspins sitzt in den Richtungen
    (Headings).
  - **Der Gitter-eigene Generator funktioniert:** J_a = (2i/π)·log R₉₀,a, wobei R₉₀ die
    innere Wirkung der 90°-Drehung des Gitters ist (Heading-Permutation ×
    Spin-Drehung, vertauscht exakt mit der Münze). Er gibt auf **jedem**
    Spin-½-Dublett g = 1, für Teilchen und Antiteilchen.
  - Mit dieser Regel dreht der Spin mit 0.00399, also Ω auf 0.3 %.

## Antwort

1. **Beobachter im Gitter:** Mit langwelligem Licht findet er weder eine
   Vorzugsrichtung noch eine Energieabhängigkeit, bis auf (ka)². Seine
   Radar-„Kugeln“ sind für alle langwelligen Signale dieselben. Isotropie im Sinn der
   Physik heißt genau das.
2. **Newton:** Der 3D-Walker folgt der lokalen Kraft der Metrik auf ≤ 1 % (r ≥ 13), und
   die Bewegungsenergie gravitiert mit. **Periheldrehung:** Die klassischen Bahnen der
   Gitter-Metrik geben (2 + 2γ − β)/3 der ART-Drehung, mit H2 genau den ART-Wert.
3. **Lense–Thirring:** Das Mitnahmefeld rotierender Massen ist der ART-Dipol, und die
   klassische Knotendrehung stimmt auf 0.1 %. Im Walker wird ein Paket lokal mitgeführt.
   **Ein Kreisel braucht eine zusätzliche Regel**, die innere Drehung mit ½∇ × b. Sie
   funktioniert nur mit dem Gitter-eigenen Drehgenerator, nicht mit dem Spin-Index.

**Was das nicht zeigt:**
- Die Periheldrehung ist die klassische Grenze des effektiven Hamiltonians, keine
  Wellenpaket-Rechnung.
- Die Kreiselregel ist für das ruhende Teilchen exakt (g = 1). Für bewegte Zustände
  vertauscht J nicht mit dem Schritt; der Fehler ist dort von der Ordnung der
  Gitteranisotropie und nicht gemessen.
- In C2 ist ½∇ × b gleichförmig (starre Drehung). Ein ortsabhängiges ∇ × b, etwa im
  Dipolfeld einer rotierenden Masse, ist nicht im Walker gerechnet.

## Offene Punkte

*Weiter mit dem Gravitationsatom: [`RESULTS_GravAtom_de.md`](RESULTS_GravAtom_de.md).*

- **Gravitationsatom:** Die Quantenversion der Periheldrehung ist die Aufspaltung der
  l-Entartung im 1/r-Potential. Messbar über Zeitbereichs-Spektroskopie wie bei den
  Landau-Niveaus.
- Kreisel im Dipolfeld einer rotierenden Masse (Gravity-Probe-B-Geometrie) im Walker.
- Ob sich die Kreiselregel aus der Budget-Idee ableiten lässt („der Knoten reicht den
  Zustand samt seiner Ausrichtung weiter“), statt sie zu setzen.
