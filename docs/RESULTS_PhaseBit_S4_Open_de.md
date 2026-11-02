# S4, offene Punkte: Symmetrien der Masse, 2D-Schutz, Untergrund, Dirac-Regime

*Paper 3, Hauptlinie. Code: `src/phase_bit_s4_open.py` und `src/phase_bit_s4_open_figs.py`.
Daten: `data/phase_bit_s4_open.npz`. Abbildung: `figures/phase_bit_s4_open.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_de.md`](RESULTS_PhaseBit_S4_de.md).*

S4 hat vier Fragen offen gelassen. Alle vier sind hier beantwortet. Dabei hat sich
eine Korrektur an S4 ergeben (Abschnitt 1).

Speicher: Alle Rechnungen brauchen unter 1 GB. Die Dynamik im Dirac-Regime läuft
mit einer exakten k-Raum-Methode (Abschnitt 4) statt auf einem Ortsgitter.
Ein Ortsgitter wäre für σ = 40 etwa 300³ groß gewesen, das sind rund 5 GB pro
Zustandsvektor.

## 1. Was ist Zeitumkehr? Korrektur und exakte Gitter-Chiralität

**Korrektur an S4.** Der in S4 verwendete antiunitäre Operator Θ = (1 ⊗ iσ_y)K
lässt den Impuls k unverändert, weil er die Richtung d nicht umkehrt. Er ist
deshalb **PT** und nicht T. Genau das erklärt die Kramers-Verdopplung *bei
jedem* k: Sie folgt aus PT mit (PT)² = −1. Die echte Zeitumkehr ist
**T = Π·Θ** (Richtung umkehren plus Spin-Flip). Geprüft wurde das im
symmetrischen Zeitrahmen W(k) = C^½ ph(k) C^½: T W(k)* T⁻¹ = W(−k)⁻¹ gilt auf
8·10⁻¹⁶, in beiden Modellen.

Damit lassen sich die Terme aus dem S4-Scan neu einordnen:

| Term | P | T | PT (Kramers) | Chiralität H | Kontinuums-Analogon |
|------|---|---|--------------|--------------|---------------------|
| μΠ (Umkehr ohne Spindrehung) | + | + | + | antivertauscht | m ψ̄ψ (β) |
| 180° i u_d·σ = Π·(iH) | − | **−** | + | antivertauscht | m ψ̄ iγ₅ ψ |
| Knoten-Helizität u_d·σ = H | − | + | **−** | vertauscht | chirales chemisches Potential μ₅ |

μΠ erhält also wirklich P, T und PT, und die Aussage von S4 bleibt richtig. Die
**paritätsungerade Masse bricht dagegen P und T** (sie erhält nur PT). Das ist
genau wie bei ψ̄iγ₅ψ, dem CP-verletzenden θ-Term der Teilchenphysik. In S4 war
sie fälschlich als „T-erhaltend“ eingetragen.

**Exakte chirale Symmetrie auf dem Gitter.** Die Helizität
H = Σ_d |d⟩⟨d| ⊗ u_d·σ (Spin entlang der Bewegungsrichtung) vertauscht mit dem
masselosen Schritt: [G, H] = 6·10⁻¹⁷ (minimale SU(2)) bzw. 0 (2O). Mit der
Verschiebung vertauscht sie ohnehin, weil sie richtungsdiagonal ist. Der Grund:
Der Spin-Transport erhält die Helizität. Π antivertauscht mit H, {Π, H} = 0. Also gilt:

- Die paritätsungerade Masse ist exakt Π·(iH) (Differenz 0).
- Die Drehung X = exp(iθH/2) bildet μΠ exakt auf μ(cos θ·Π + sin θ·Π iH) ab (4·10⁻¹⁶).
- Das Spektrum hängt deshalb **überhaupt nicht** von θ ab, max |E(θ) − E(0)| =
  5·10⁻¹⁵ über das ganze Band (Abbildung b).

**Antwort auf die offene Frage:** Physikalisch ist die paritätsungerade Masse für
ein freies Teilchen dieselbe Masse wie μΠ, nur in einer chiral gedrehten Basis.
Der Winkel θ ist unbeobachtbar, wie im Kontinuum. Beobachtbar würde er erst
durch eine Kopplung, die H bricht.

**Fermion-Verdopplung (Nielsen–Ninomiya).** In jedem Helizitätssektor ist der
Dirac-Punkt ein Weyl-Knoten mit Steigung c/3. Der Partner mit entgegengesetzter
Chiralität sitzt **bei derselben Impulslage k = 0, aber bei einer anderen
Quasienergie**. Bei minimaler SU(2) liegen die Knoten bei E/ε = −6.928 und
+2.309, beim 2O-Modell bei ±5.657. Die Chiralitäten summieren sich in jedem
Sektor zu null. Der „Doppler“ wohnt also in der Energie, nicht im Impuls.

## 2. 2D: Warum es dort keine T-erhaltende Masse gibt

Für den Z₁₂-Walker aus S2 ist die Zeitumkehr **T = S³K**: Heading +180° und
komplexe Konjugation (symmetrischer Rahmen 2·10⁻¹⁶). Es gilt T² = S⁶ = R₆, also −1
im Spin-Sektor. Der Hopping-Term G₀ ist T-gerade. Der chirale Umkehrterm G₃, der in
S2 die Masse lieferte, ist exakt T-ungerade.

Der Grund für das Fehlen einer T-erhaltenden Masse ist **Kramers selbst**:

- Der 2D-Dirac-Punkt liegt bei k = 0, einem zeitumkehrinvarianten Impuls. Er ist nur
  zweifach und damit **ein einziges Kramers-Paar**.
- Jeder T-erhaltende Term lässt dieses Paar entartet, kann also keine Lücke öffnen.
  Getestet mit je 500 Zufallstermen:

  | Zufallsterme | Aufspaltung der drei k=0-Paare |
  |---|---|
  | T-gerade | **höchstens 2.9·10⁻¹⁵** |
  | T-ungerade | immer mindestens 6·10⁻³ |

  Abbildung (a) zeigt die Verteilung.
- In 3D ist der Dirac-Punkt vierfach, also zwei Kramers-Paare. Eine Masse trennt
  dort Paar von Paar, und das ist erlaubt.

Das 2D-Spin-Wirbel-Kegel ist damit vom gleichen Typ wie der Oberflächenzustand
eines topologischen Isolators. Er ist durch die Zeitumkehr geschützt, nicht nur
„zufällig“ masselos.

## 3. Der Untergrund −αk²

**Am untersten Dirac-Punkt hat der Untergrund ein festes Vorzeichen.** Gemessen
wurde α·ε = 0.130, 0.130, 0.124 (minimale SU(2), ε = 0.05 / 0.1 / 0.3) und
0.196, 0.196, 0.191 (2O). α ∝ 1/ε zeigt, dass er aus Niveauabstoßung in zweiter
Ordnung kommt. Der unterste Punkt ist ein extremales Niveau des Generators, also
schieben ihn alle anderen Niveaus in dieselbe Richtung. Ein lokaler Coin-Term
kann ihn nur verkleinern, aber nicht umdrehen. Beispiel: Im 2O-Modell vertauscht der
90°-Transportterm mit G und erhält damit alle Symmetrien inklusive E ↔ −E.
Mit 2ε·W₉₀ sinkt α·ε nur von 0.196 auf 0.055 (ε = 0.05). Ein Nulldurchgang bei
großem λ tritt nur durch das Umklappen der Floquet-Zone auf und hängt dann von ε
ab. Das ist kein struktureller Ausgleich.

**Am inneren Dirac-Punkt lässt er sich exakt kompensieren.** Das
minimale-SU(2)-Modell hat einen zweiten vierfachen Dirac-Punkt (E/ε = +2.309, Steigung
c/3) mit Niveaus auf beiden Seiten. Mit dem Zusatzterm

> λ*·ε·W₉₀,  W₉₀ = Kopplung der 90°-Richtungspaare mit Spin-Transport,  **λ* = −0.40827**

verschwindet der Untergrund: α·ε = 5·10⁻⁶, −3·10⁻¹¹, −2·10⁻⁶ bei ε = 0.05 / 0.1 /
0.3, also strukturell und unabhängig von ε (Abbildung c). Dabei bleibt alles
erhalten:

- W₉₀ erhält P, T, PT und die Chiralität H.
- Die Steigung bleibt exakt c/3, die Clifford-Algebra gilt auf 10⁻¹⁵.
- μΠ öffnet weiterhin eine Lücke von exakt 4μ.
- Kramers gilt auf 5·10⁻¹⁵, und max |v_g| = 1.30 < c.

Krümmung d²E/dk² der beiden massiven Dubletts (ε = 0.3):

| μ (μ/ε) | ohne Kompensation unten / oben | kompensiert unten / oben | Dirac ±c′²/m |
|---------|------------------|-------------|------------|
| 0.015 (0.05) | −11.90 / +10.25 | **−11.05 / +11.05** | ±11.11 |
| 0.03 (0.1) | −6.31 / +4.65 | **−5.44 / +5.44** | ±5.56 |
| 0.15 (0.5) | −1.67 / −0.31 | −0.35 / +0.35 | ±1.11 |

Nach der Kompensation sind Teilchen- und Antiteilchenzweig spiegelsymmetrisch, und
die effektive Masse stimmt bis auf 0.5 % (μ/ε = 0.05) bzw. 2 % (μ/ε = 0.1) mit Dirac
überein. **Grenze:** Bei μ/ε = 0.5 ist auch die Aufspaltung selbst nicht mehr
Dirac-artig (0.35 statt 1.11). μ ≪ ε bleibt also nötig, die Kompensation
entfernt nur den gemeinsamen Untergrund. Für das 2O-Register gibt es keinen
inneren vierfachen Dirac-Punkt (seine achtfachen Niveaus sind anisotrope
Mehrfachkegel), dort bleibt der Untergrund.

## 4. Dynamik im Dirac-Regime (μ/ε = 0.1)

**Methode.** Ein bandbegrenztes Paket auf dem unendlichen FCC-Gitter wird exakt
im k-Raum entwickelt, nur auf dem Träger des Pakets (Würfel k₀ ± 5/σ, 40³ Punkte).
Ein Gitterschritt ist ψ̂(k) ← ph(k)·(C ψ̂(k)), die Ortshülle ist die inverse FFT.
Validierung gegen die 81³-Ortsraumläufe von S4:

| Größe | k-Raum-Methode | 81³-Ortsgitter |
|---|---|---|
| rms-Wachstum in Ruhe | 0.0508 | 0.051 |
| v bei k = 0.2 | −0.0875 | −0.088 |
| Zitterbewegung ω | 0.6450 | 0.645 |

**Läufe:** ε = 0.3, μ = 0.03, m = 0.06, σ = 40 (m·σ/c′ = 4.2). Verglichen werden das
Modell ohne Kompensation (unterster Punkt) und das kompensierte (innerer Punkt).

- **Ruhend** (Abbildung f): Die Drift bleibt bei 2·10⁻¹⁰. Das rms wächst von 49.6
  auf 79.6 (kompensiert) bzw. 72.4 (ohne Kompensation) bei t = 400. Ein ballistisches
  Dirac-Teilchen gleicher Breite erreicht 80.5.
- **Bewegt** (Abbildung e), Schwerpunktsgeschwindigkeit:

  | k | ohne Kompensation | kompensiert | Dirac c′²k/E |
  |---|------|-------------|--------------|
  | 0.025 | +0.105 | +0.124 | +0.135 |
  | 0.05 | +0.193 | +0.231 | +0.250 |
  | 0.1 | +0.297 | +0.369 | +0.400 |
  | 0.2 | +0.308 | +0.425 | +0.512 |

  Im Dirac-Regime läuft das obere Dublett jetzt **in Richtung k** (in S4 bei
  μ/ε = 0.5 lief es dagegen). Die Kompensation bringt die Geschwindigkeit
  deutlich näher an Dirac. Der verbleibende Rest von 5 bis 8 % kommt etwa zu einem Drittel aus
  der Mittelung über die k-Breite des Pakets (die Band-v_g liegt bei 0.380 statt
  0.369 bei k = 0.1), zu zwei Dritteln aus der Gitterkrümmung des Bandes selbst.
- **Zitterbewegung** bei k₀ = 0.03 (Abbildung g, h): ω = 0.1275 (kompensiert) bzw.
  0.1277 (ohne). Der Bandabstand beträgt 0.1248 und Dirac 2√(m² + (c/3)²k²) = 0.1249,
  das sind 2 % Abweichung. Die Schwingung hält über sechs Perioden an. Ihre Amplitude wurde nicht
  quantitativ mit Dirac verglichen.

## Antwort

1. **Untergrund:** Am untersten Dirac-Punkt lässt er sich mit lokalen Termen nicht
   kompensieren, das Vorzeichen ist fest. Am inneren Dirac-Punkt gelingt es exakt
   und strukturell mit einem einzigen spin-transportierenden 90°-Term
   (λ* = −0.408), ohne eine Symmetrie zu brechen.
2. **Dirac-Regime:** Mit der exakten k-Raum-Methode ohne großes Gitter berechnet.
   Das massive Paket ruht, zerfließt wie ein Dirac-Teilchen, läuft in Richtung k
   mit fast Dirac-Geschwindigkeit und zittert mit 2E.
3. **Paritätsungerade Masse:** Sie ist die chiral gedrehte Form von μΠ. Die
   Gitter-Helizität H ist eine exakte chirale Symmetrie, θ ist für freie Teilchen
   unbeobachtbar. Sie bricht P und T (nicht nur P), wie ψ̄iγ₅ψ. In S4 war sie
   fälschlich als T-erhaltend eingetragen, weil Θ dort PT war.
4. **2D:** Der Dirac-Punkt ist ein einziges Kramers-Paar bei k = 0. T-erhaltende
   Terme können ihn grundsätzlich nicht öffnen. In 3D sind es zwei Paare.

## Offene Punkte

- Ein innerer, kompensierbarer Dirac-Punkt im 2O-Register fehlt. Gibt es eine
  2O-Variante (andere Transportwahl, Zusatzterme) mit E ↔ −E *und* α = 0?
- Kopplungen, die H brechen (z. B. ortsabhängige Terme), würden θ beobachtbar
  machen: eine Gitterversion der chiralen Anomalie?
- Die Amplitude der Zitterbewegung ist noch nicht mit der Dirac-Theorie verglichen.
