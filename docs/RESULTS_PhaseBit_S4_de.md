# Experiment S4: Masse für den 3D-Spin-Wirbel unter Erhaltung der Zeitumkehr

*Paper 3, Hauptlinie. Code: `src/phase_bit_mass_s4.py` und `src/phase_bit_mass_s4_figs.py`.
Daten: `data/phase_bit_mass_s4.npz`. Abbildung: `figures/phase_bit_mass_s4.png`.
Vorgänger: [`RESULTS_PhaseBit_S3_de.md`](RESULTS_PhaseBit_S3_de.md), korrigiertes FCC-Modell:
[`RESULTS_FCC_3D_de.md`](RESULTS_FCC_3D_de.md).*

## Frage

Der 3D-Spin-½-Walker ist masselos. Sein unterstes Niveau ist ein vierfacher
Dirac-Punkt mit Steigung c/3, sowohl mit minimaler SU(2) als auch mit dem
endlichen 2O-Register. **Welcher lokale Term öffnet eine Massenlücke, ohne die
Zeitumkehr (Kramers) zu brechen?** In 2D ging das nicht (S2: Masse nur mit
T-Bruch).

## Methode: Symmetrieanalyse am Dirac-Punkt

1. **Effektiver Dirac-Hamiltonoperator.** Sei P der vierdimensionale Eigenraum
   des Dirac-Punkts. Dann gilt H_eff(k) = E₀ + Σᵢ kᵢ Γᵢ mit Γᵢ = P†(Δrᵢ/Δt)P
   (entartetes Hellmann–Feynman). Geprüft: {Γᵢ, Γⱼ} = 2(c/3)² δᵢⱼ auf 2·10⁻¹⁵ in
   beiden Modellen. Es ist ein echter Dirac-Punkt.
2. **Bedingung für eine Masse.** Ein Term V im Generator, C = expm(i(εG + μV)),
   öffnet genau dann eine Lücke, wenn der spurfreie Teil von P†VP mit **allen**
   Γᵢ antivertauscht. Vertauscht er mit ihnen, spaltet er den Kegel nur auf.
3. **Zeitumkehr.** V muss mit Θ = (1 ⊗ iσ_y)K vertauschen.
4. **Parität.** Die Inversion Π : d → −d (Spinor unverändert) unterscheidet eine
   gewöhnliche Dirac-Masse (Π-gerade, „β“) von einer paritätsungeraden
   („iβγ₅“).

## Ergebnisse

### 1. Scan aller lokalen Terme (beide Modelle identisch klassifiziert)

Kopplungen zwischen Richtungen im Winkel 60°, 90°, 120° und 180°, mit den
Spin-Faktoren *Transport*, *Identität*, i·u_d·σ (Achse der Ausgangsrichtung) und
i·n·σ (Normale), jeweils symmetrisch (K+K†) und antisymmetrisch (i(K−K†)),
dazu der Knotenterm u_d·σ:

| Term | Wirkung am Dirac-Punkt | Π | Θ |
|------|------------------------|---|---|
| 60°/90°/120° mit **Spin-Transport** | nur Verschiebung (∝ 1) | + | erhalten |
| 60°/90°/120° mit **Spin-Identität** | **Masse** | + | erhalten |
| 60°/90°/120° mit i·n·σ | **Masse** | + | erhalten |
| **180° mit Spin-Identität = Π** | **Masse** | + | erhalten |
| 60°/90°/120°/180° mit i·u_d·σ (K+K†) | **Masse** (paritätsungerade) | − | erhalten |
| … mit i·u_d·σ, i(K−K†) | spaltet den Kegel | − | **gebrochen** |
| Knoten u_d·σ (Helizität) | spaltet den Kegel | − | **gebrochen** |
| alle übrigen i(K−K†) | keine Wirkung | | |

Es gibt also eine ganze Familie T-erhaltender Massen, gewöhnliche (Π-gerade) und
paritätsungerade. Gemeinsam ist ihnen: **Der Spin wird beim Richtungswechsel
nicht mitgedreht** (oder um die falsche Achse). Terme, die den Spin korrekt
transportieren, sind am Dirac-Punkt nur eine Konstante (der 60°-Transportterm ist
der Generator G selbst). Masse entsteht, wo die Kopplung von Spin und Richtung
gelockert wird.

### 2. Der natürlichste Massenterm ist die Inversion selbst

Der Term „180°-Umkehr ohne Spindrehung“ ist **exakt der Inversionsoperator Π**
(Differenz 0.0). Das entspricht genau der Dirac-Theorie, in der der Massenterm m·β
lautet und β der Operator der inneren Parität ist. Auf dem Gitter heißt das:

> **Masse = Amplitude, die Richtung umzukehren, ohne den Spin zu drehen.**

Warum ist die Lücke exakt 4μ? Π vertauscht mit G (in beiden Modellen [Π, G] = 0),
also zerfällt der Dirac-Raum in Paritätsräume (+1 und −1, je zweidimensional). Die
Energien sind exakt E = −(ελ ± μ)/Δt. Bei Δt = ½ ergibt das eine Lücke von 4μ, und
zwar bei k = 0 in allen Ordnungen.

### 3. Prüfung an der vollen Bandstruktur (ε = 0.3)

| Modell + Term | Lücke/μ (μ = 0.01 … 0.1) | Kramers-Aufspaltung | E ↔ −E |
|---------------|--------------------------|---------------------|--------|
| min. SU(2) + μΠ | 4.000000 (alle) | 6.2·10⁻¹⁵ | nein (0.88) |
| min. SU(2) + μ·(90° Identität) | 4.000000 | 6.4·10⁻¹⁵ | nein |
| min. SU(2) + μ·(180° i u_d·σ) | 4.000000 | 4.7·10⁻¹⁵ | nein |
| **2O + μΠ** | 4.000000 | 4.7·10⁻¹⁵ | **ja, 4.4·10⁻¹⁵** |
| 2O + μ·(90° Identität) | 4.000000 | 5.6·10⁻¹⁵ | ja, 3.9·10⁻¹⁵ |
| 2O + μ·(180° i u_d·σ) | 4.000000 | 4.4·10⁻¹⁵ | ja, 4.9·10⁻¹⁵ |

**Mit dem 2O-Register und μΠ gibt es zum ersten Mal ein 3D-Spin-½-Teilchen, das
gleichzeitig massiv ist, die Kramers-Entartung bei jedem k hat und
teilchen-antiteilchen-symmetrisch ist, alles aus einem endlichen Register.**

### 4. Wie Dirac-artig ist das massive Quartett?

Zerlegung E = E_c(k) ± Δ(k) (Mittelwert und halbe Aufspaltung der beiden Dubletts):

- **Aufspaltung Δ(k)** folgt √(m² + (c/3)²k²) bei kleinem k, isotrop:

  | μ | m | k = 0.05 | 0.1 | 0.2 | 0.4 |
  |---|---|----------|-----|-----|-----|
  | 0 | 0 | −0.1 % | −0.4 % | −1.6 % | −6.4 % |
  | 0.05 | 0.10 | −0.2 % | −0.6 % | −2.0 % | −7.0 % |
  | 0.15 | 0.30 | −0.2 % | −0.7 % | −2.7 % | −9.4 % |

- **Gemeinsamer Untergrund** E_c(k) − E_c(0) = −α·k² mit α·ε ≈ 0.12 bis 0.13
  (ε = 0.1: 1.30; 0.2: 0.64; 0.3: 0.41; 0.5: 0.22). Dieser Term ist schon ohne
  Masse vorhanden. Er krümmt beide Dubletts gleich, wie ein Wilson-artiger
  Gitterterm. Abbildung (a) zeigt, dass er das obere Dublett abflacht.
  Bei der Compton-Impulsskala k ~ 3m/c ist er relativ zur Masse von der Ordnung
  3αm ~ 0.4·μ/ε. Das saubere Dirac-Regime ist also **μ ≪ ε**. In den Läufen unten
  (μ/ε = 0.5) ist der Untergrund deutlich sichtbar.

### 5. Dynamik in 3+1D (81³ Gitter, ε = 0.3, μ = 0.15, m = 0.30)

- **Ruhend:** Das massive Paket bleibt am Ort (Schwerpunktsdrift 1.4·10⁻⁸, Norm
  1.00000) und zerfließt langsam (rms-Wachstum 0.051). Das masselose Paket wächst
  mit 0.194. Das ist weniger als c/3, weil der Kegel über die k-Breite des Pakets
  (σ_k = 0.25) schon stark gekrümmt ist (Abbildung a). ⟨Π⟩ bleibt konstant
  (−0.7477).
- **Bewegt** (oberes Dublett, bandprojiziert):

  | k | v Schwerpunkt | v_g des Bandes |
  |---|---------------|----------------|
  | 0.1 | −0.041 | −0.033 |
  | 0.2 | −0.088 | −0.078 |
  | 0.4 | −0.212 | −0.217 |

  Alle Werte liegen weit unter c = 1.73. Das obere Dublett läuft **gegen** k: Bei
  μ/ε = 0.5 überwiegt der Untergrund −αk² die Dirac-Krümmung, die Masse ist
  effektiv negativ. Bei kleinerem μ/ε dreht das Vorzeichen.
- **Zitterbewegung:** Eine Überlagerung aus unterem und oberem massivem Zustand
  bei k₀ = 0.3 zittert quer zur Bewegung mit ω = 0.645. Der Bandabstand bei k₀
  beträgt 0.653 (1.3 % Abweichung). ⟨Π⟩ eignet sich dafür nicht, weil Π in der
  Bandbasis fast diagonal ist (Übergangsgewicht 0.012).

## Antwort

Der gesuchte Term existiert und ist sehr einfach: **μ·Π, die Umkehr der Richtung
ohne Drehung des Spins.** Er ist die Gitterversion von m·β, öffnet den
Dirac-Punkt mit einer Lücke von exakt 4μ und erhält die Zeitumkehr (Kramers bei
jedem k). Im 2O-Register bleibt außerdem die Teilchen-Antiteilchen-Symmetrie
exakt erhalten. Daneben gibt es weitere T-erhaltende Massen (gewöhnliche und
paritätsungerade). Gemeinsam ist ihnen, dass sie die Kopplung von Spin und
Richtung lockern.

Damit sind die Spin-Wirbel aus Phasen-Bits in 3D vollständig. Ein endliches
Register (2O) liefert Spin ½, Kramers, Belt-Trick und Teilchen-Antiteilchen-
Symmetrie. Ein einziger zusätzlicher Term (μΠ) liefert die Masse.

## Offene Punkte

- **Untergrund −αk².** Lässt er sich durch einen zusätzlichen Term kompensieren,
  analog zum Wilson-Term der Gitter-QCD, sodass auch das obere Dublett bei
  größerem μ/ε ein normales massives Band wird?
- **Dynamik im Dirac-Regime μ ≪ ε** braucht größere Gitter (σ ≳ 3/m). Möglich wären
  Ausnutzen der FCC-Parität oder eine k-Raum-Formulierung.
- **Paritätsungerade Masse** (i u_d·σ): Was unterscheidet sie physikalisch von μΠ?
  Eine Kombination beider ergibt eine chirale Phase der Masse (θ-Winkel).
- **2D-Gegenstück:** Warum gibt es dort keine T-erhaltende Masse? Vermutlich, weil
  der 2D-Kramers-Kegel nur zweikomponentig ist (2+1D-Paritätsanomalie).
