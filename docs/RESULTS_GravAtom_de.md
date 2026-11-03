# Das Gravitationsatom: die Quantenversion der Periheldrehung

*Paper 3. Code: `src/phase_bit_s4_open14.py` (Teile `A`, `B`) und
`src/phase_bit_s4_open14_figs.py`. Daten: `data/phase_bit_s4_open14.npz`.
Abbildung: `figures/phase_bit_s4_open14.png`.
Vorgänger: [`RESULTS_Observer_Perihelion_LT_de.md`](RESULTS_Observer_Perihelion_LT_de.md).*

## Frage

In einem reinen 1/r-Potential hängen die Energien nur von n ab (Entartung in l,
geschlossene Kepler-Bahnen). Die relativistischen Terme der Metrik heben diese Entartung
auf. Für große Quantenzahlen ist die Aufspaltung E(n, l+1) − E(n, l) die
Präzessionsfrequenz der Bahn, also die Quantenversion der Periheldrehung. Die Frage ist,
ob das Modell sie liefert, und mit welchem Faktor für die verschiedenen Regeln.

## Warum nicht direkt im 3D-Walker

Für die Aufspaltung zwischen 2s und 2p muss der Bohr-Radius viele Gitterabstände groß
sein, etwa 10. Das 2p-Orbital reicht dann bis rund 60 Längeneinheiten. Das wären etwa
200³ Knoten mit 96 Komponenten (rund 11 GB pro Zustand) und für eine Aufspaltung von etwa
10⁻⁴ rund 10⁴ Takte. Das passt nicht auf diesen Rechner.

Daher zwei Schritte:
- **A:** Das Gitteratom in 1D (billig) prüft den effektiven Hamiltonian des Modells.
- **B:** Das 3D-Atom wird mit diesem geprüften Hamiltonian gerechnet.

Effektiver Hamiltonian (aus der Bandebene von H2: Lücke N·m, Lichtgeschwindigkeit
c′N^{1+n}), symmetrisch geordnet:

```
H = N^{1/2} √(m² + c′² Π²) N^{1/2},     Π² = N^{n/2} (−∇·N^n ∇) N^{n/2}
n = 1: H2 (A = 1/N),  n = 0: nur Warten (A = 1);   N = e^{−U} oder N = 1 − U
```

## A. Das 1D-Gitteratom

Aufbau:
- x-Kette mit 512 Lagen, Bloch-k_y = k_z = 0.
- μ = 0.08 (m = 0.16), Potential U = GM/√(x² + r_c²) mit GM = 0.6, a₀ = 21.7, r_c = a₀/2,
  U(0) = 0.055.
- Schemata: W bzw. H2 mit kompensierter Kopplung λ(N).
- Gerade und ungerade Startzustände, 60 000 Takte. Die Niveaus kommen aus dem
  Zeitsignal ⟨ψ₀|ψ(t)⟩ (Blackman-Harris-Fenster, Spitzen verfeinert).

| Niveau (E − m, 10⁻³) | W: Gitter | W: effektiv | H2: Gitter | H2: effektiv |
|---|---|---|---|---|
| gerade, Grundzustand | −5.161 | −5.172 | −5.259 | −5.268 |
| ungerade, 1. | −1.624 | −1.610 | −1.706 | −1.690 |
| gerade, 2. | −0.782 | −0.760 | −0.824 | −0.803 |

- **Die gebundenen Zustände des Walkers folgen dem effektiven Hamiltonian** auf
  0.2 % (Grundzustand) bis 3 % (zweites angeregtes Niveau). Für den effektiven
  Hamiltonian ist dabei die Dispersion des freien Gitterbands verwendet.
- **Die relativistische Verschiebung H2 − W** ist im Gitter −0.098 / −0.083 / −0.042 und
  effektiv −0.096 / −0.080 / −0.043, also auf etwa 3 % gleich.
- **Messfallen:**
  - Ein Hann-Fenster täuschte eine Linie unter dem Potentialminimum vor; das war seine
    erste Nebenkeule (−31 dB, 2.4 Bins neben dem Grundzustand).
  - Mit 20 000 Takten verschmolzen benachbarte Linien in der Hauptkeule des Fensters.
    Die angeregten Zustände lagen dadurch 8–13 % daneben.

## B. Das 3D-Gravitationsatom

Radialgleichung für l = 0…3, exakt diagonalisiert:
- 3500 Punkte bis 70 a₀, Einheiten m = c = 1.
- U = GM/√(r² + r_c²) mit r_c = 0.1 a₀, also ein schwaches Feld überall (U(0) ≤ 0.1).
- Die Abrundung spaltet l schon newtonsch auf. Dieser Anteil wird mit derselben
  Rechnung nichtrelativistisch (Schrödinger) bestimmt und abgezogen.

**Relativistischer Teil der l-Aufspaltung** (GM = 0.1, Werte in 10⁻⁶):

| n, l → l+1 | H2 (γ = β = 1) | nur Warten (γ = 0) | linearer Lapse (β = 1/2) | W / H2 | linear / H2 |
|---|---|---|---|---|---|
| 2, 0 → 1 | +46.28 | +10.13 | +54.58 | 0.219 | 1.179 |
| 3, 1 → 2 | +3.486 | +0.978 | +4.043 | 0.281 | 1.160 |
| 4, 0 → 1 | +5.770 | +1.279 | +6.786 | 0.222 | 1.176 |
| 4, 1 → 2 | +1.470 | +0.413 | +1.704 | 0.281 | 1.159 |
| 4, 2 → 3 | +0.576 | +0.179 | +0.669 | 0.311 | 1.163 |
| klassisch (Periheldrehung) | | | | **1/3** | **7/6** |

- **Die Metrik hebt die l-Entartung auf, und höheres l ist schwächer gebunden.** Das
  ist der Sinn der vorwärts laufenden Periheldrehung.
- **Die Verhältnisse der Regeln** gehen mit wachsendem l gegen die klassischen Faktoren
  der Periheldrehung: W/H2 = 0.22 → 0.28 → 0.31 gegen 1/3, linear/H2 = 1.16–1.18 gegen
  7/6. Bei kleinem l hängen die einzelnen relativistischen Terme verschieden von l ab
  (Quanteneffekt), bei großem l zählt nur die Bahn.
- **Skalierung:** Halbiertes GM verkleinert die Aufspaltung um den Faktor 16, also wie
  GM⁴ = (v/c)² × Bindungsenergie.
- **Korrespondenz:** Verglichen mit der klassischen Präzessionsfrequenz
  ω_p = 3GM⁴/(L²n³) (L = l + 1) liegt die H2-Aufspaltung um den Faktor 1.23–1.25
  (l = 0→1, 1→2) bzw. 1.10 (l = 2→3) darüber, bei GM = 0.05 um 1.14–1.22 bzw. 1.09. Das
  passt zu kleinen Quantenzahlen. Wie es für großes l gegen 1 geht, ist hier nur
  angedeutet (n ≤ 4).

## Antwort

1. **Die Quantenversion der Periheldrehung ist im Modell da.** Im Gravitationsatom
   hebt die H2-Metrik die l-Entartung des 1/r-Potentials auf. Die Aufspaltungen
   verhalten sich zwischen den Regeln wie die klassischen Periheldrehungen (1 : 1/3 :
   7/6), für großes l genau, für kleines l mit Quantenkorrekturen.
2. **Der effektive Hamiltonian beschreibt auch gebundene Quantenzustände des Gitters.**
   Das 1D-Gitteratom stimmt auf 0.2–3 % mit ihm überein, einschließlich der
   relativistischen Verschiebung zwischen H2 und Warten.

**Was das nicht zeigt:**
- Das 3D-Atom ist mit dem effektiven Hamiltonian gerechnet, nicht im Walker. Geprüft ist
  der Hamiltonian in 1D, wo es kein l gibt. Die Ordnung der Operatoren in 3D (etwa
  Π² = N^{n/2}(−∇·N^n∇)N^{n/2}) ist eine plausible Fortsetzung.
- **Ohne Spin.** Das Teilchen des Modells hat Spin ½. Die gravitative Spin-Bahn-Kopplung
  (Feinstruktur j = l ± ½) fehlt. Den Hinweis darauf gab es schon: die spinabhängige
  Kraft (±8 %) auf einen einzelnen Kramers-Partner, der sich quer im Gradienten bewegt
  (g₀ᵢ-Runde).

## Offene Punkte

- **Gravitative Spin-Bahn-Kopplung:** die spinabhängige Kraft im 1D-Gradienten
  quantitativ gegen die Dirac-Gleichung in der Metrik (de-Sitter-/Thomas-Anteile) prüfen.
  Damit wäre die Feinstruktur des Gravitationsatoms vorhergesagt.
- Ein kleines 3D-Gitteratom (1s, 2p) mit einer harmonischen Inversion statt FFT, um mit
  weniger Takten auszukommen. Das wäre die direkte Prüfung der 3D-Ordnung.
