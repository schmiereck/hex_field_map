# Roadmap Paper 3: Diskrete Zustände, Phasen-Bits und Gravitation als Rechenzeit

*Stand: 2. November 2026. Ideengrundlage: Gemini-Gespräch
[`SOURCE_Paper3_Gemini_Thread_de.md`](SOURCE_Paper3_Gemini_Thread_de.md).
Dieses Dokument trennt die tragfähigen Teile von den nicht belastbaren und legt
die Experimente fest.*

---

## 1. Kritische Einordnung des Quell-Threads

### Was hält

| Aussage | Status |
|---------|--------|
| Kuboktaeder-Schale n hat N(n) = 10n² + 2 Knoten | korrekt |
| Tight Frame (1/6)·Σ\|k⟩⟨k\| = I₂, ⟨k\|k̄⟩ = 0, \|Ψ_k\|² + \|Ψ_k̄\|² = 1/6 | korrekt, gilt aber für **jede** punktsymmetrische Richtungsmenge |
| Bloch-Vektor aus Betragsquadraten: s = 3·Σ\|Ψ_k\|²·r̂_k | korrekt (Kuboktaeder ist ein sphärisches 3-Design) |
| Masse = Zitterbewegung, v = p/E < c | korrekt, aber genau die Kernaussage von **Paper 1**, dort exakt gerechnet |
| Schale n = alle Punkte im Graph-Abstand n | korrekt, und das ist genau die **Lichtkegel-Front** des FCC-Walkers (`quantum_fcc_3d.py`) |

### Was nicht hält

1. **Ein Z₄-Wirbel auf einem 4er-Ring ist kein Spin ½.** Windung 1 mit e^{iφ} ist
   ganzzahliger Drehimpuls (m = 1 mod 4). Spin ½ verlangt das Vorzeichen −1 nach
   einer 2π-Drehung, und das kann kein Feld leisten, das an jedem Knoten nur einen
   Wert hat. Zudem hängt die Wirbel-Phase in „Weg A“ von der Phasenkonvention der
   Referenzspinoren ab. Richtig gemacht ist das bereits im Repo:
   SU(2)-Transport in `quantum_fcc_3d.py`, wo die kürzeste geschlossene Schleife
   (L = 6) −1 liefert (Belt-Trick).
2. **Superposition ist kein Bit-Verhältnis.** Ein Anteil von 70 % „+1“-Bits ergibt
   höchstens ein klassisches Gemisch. Interferenz braucht sich auslöschende Amplituden.
3. **Ganzzahlig, linear und umkehrbar schließt Masse aus.** In einer unitären Matrix
   mit Einträgen in Z[i] hat jede Spalte die Norm 1, und die Betragsquadrate sind
   ganze Zahlen. Also hat jede Spalte genau einen Eintrag vom Betrag 1, und die
   Matrix ist eine Permutation mit Phasen. Es mischt nichts, also entstehen weder
   Masse noch Dispersion. Nichtlineare „Schwellen“-Regeln opfern das
   Superpositionsprinzip.
4. **1/r und 1/r²** brauchen ein eigenes Feld mit eigener Dynamik und eine Kopplung.
   Aus dem bloßen Abzählen von Schalen folgen sie nicht.
5. Die „Ergebnisse“ des Threads (stabiler Spin-Wirbel, 1/n-Abfall, sichtbare
   Gravitationsablenkung) sind **Vorhersagen ohne Regel**. Die Kollisionsregel wird
   nie konkret angegeben.

### Echte Verbindungen zum bestehenden Projekt

- **Das Phasen-Bit steckt in Paper 1.** Ein Richtungswechsel bringt den Faktor iε,
  die Pfadphase ist also i^T mit einem Z₄-Zähler T. Die Amplitude ist
  Σ_T N_T(x,t)·(iε)^T mit **ganzzahligen** Pfadzahlen N_T. Die Dynamik ist damit
  reine Ganzzahl-Kombinatorik, ε kommt erst beim Auslesen hinzu.
- **Im 2D-Hexgitter ist „×i pro 60°-Drehung“ Spin ½.** Das entspricht α = 3/2,
  und weil α nur modulo 1 zählt, ist das dasselbe wie α = ½. Nach dem Whitney-Satz
  gilt Σn = 6w für geschlossene Schleifen, also Phase i^{6w} = (−1)^w. In 3D gilt
  das nicht mehr, weil die Holonomie nicht quantisiert ist
  (`quantum_fcc_holonomy.py`). Dafür braucht man SU(2).
- **Schalen als Auflösung, Bezug zu Paper 2.** Die 12 Richtungen zerlegen sich
  unter O_h in A1g ⊕ Eg ⊕ T2g ⊕ T1u ⊕ T2u. Ohne die s-Welle bleibt das 11-fach
  entartete Band aus Paper 2.

---

## 2. Experimente

### Experiment S: Spin-Wirbel aus Phasen-Bits (Hauptfokus)

Leitfrage: **Lassen sich „Spin-Wirbel“ aus Phasen-Bits tatsächlich sehen?**

Die korrigierte Version der Idee aus dem Thread: Ein Walker trägt neben seiner
Richtung d ∈ Z₆ ein Phasen-Bit p ∈ Z₄. Bei einer Drehung um n·60° gilt p → p + n.
Gerade weiter lässt p unverändert. Von (d, p) = (0, 0) aus sind nur die Paare mit
d ≡ N (mod 6) und p ≡ N (mod 4) erreichbar, wobei N der Netto-Drehzähler ist. Das
sind 12 Zustände, also Z₁₂, die **Doppelüberlagerung** der 6 Richtungen. Das ist
das diskrete Gegenstück zu SU(2) → SO(3). Nach einer vollen 360°-Drehung ist die
Richtung wieder dieselbe, aber das Phasen-Bit ist um i⁶ = −1 gekippt. Erst nach
720° ist alles wieder wie zu Beginn.

Der „Spin-Wirbel“ ist damit ein **Halbquanten-Wirbel im Richtungsraum**: ein
Phasenmuster um die 6 Kanten eines Knotens, das sich bei einem Umlauf um π statt
um 2π windet. Analytisch ist das das α = ½-Spinor-Ergebnis des Turning-Modells
(Phase ×i pro 60° entspricht α = 3/2, und α zählt nur mod 1). Neu wäre, es als
reinen Ganzzahl-Automaten zu bauen und den Wirbel direkt sichtbar zu machen.
Gemini hatte ihn falsch platziert, als Z₄-Windung auf einem 4er-Ring.

| Stufe | Inhalt | Prüfgröße | Status |
|-------|--------|-----------|--------|
| S1 | 2D-Phasen-Bit-Automat mit **ganzzahligen** Pfadzahlen N[x, d, p, T] | exakte Übereinstimmung mit dem Fließkomma-Turning-Modell (bis auf die Eichung); Vorzeichen −1 nach 360°, +1 nach 720°; Wirbelmuster der Kramers-Moden | **erledigt**, siehe [`RESULTS_PhaseBit_S1_de.md`](RESULTS_PhaseBit_S1_de.md) |
| S2 | Dynamik: Wirbel als Wellenpaket, ruhend und bewegt | Halbwindung exakt erhalten (Superauswahl, 8e-16); ohne Umkehr masselos (Ring mit c/2); chirale Umkehr gibt Masse 8μ (bricht Kramers); v < c; Zitterbewegung j=7↔5 bei ω = Bandabstand (0.5 %) | **erledigt**, siehe [`RESULTS_PhaseBit_S2_de.md`](RESULTS_PhaseBit_S2_de.md) |
| S3 | 3D mit **endlichem** Spin-Register: binäre Oktaedergruppe 2O (48 Elemente, Clifford-Gruppe) statt kontinuierlicher SU(2) im FCC-Walker | Vierteldrehung um Würfelachse pro 60°-Kurve; 48 von 576 Labels (Z₄-Bit pro Richtung); Holonomie Dreieck π, Quadrat 2π; Kramers bei jedem k (6e-15); **E ↔ −E-Symmetrie exakt** (neu); mehr Anisotropie | **erledigt**, siehe [`RESULTS_PhaseBit_S3_de.md`](RESULTS_PhaseBit_S3_de.md) |
| S4 | Masse für den 3D-Spin-Wirbel unter Erhaltung der Zeitumkehr | Symmetrieanalyse am Dirac-Punkt; **μ·Π** (Umkehr ohne Spindrehung = Inversion = Gitter-β) gibt Lücke exakt 4μ, Kramers bei jedem k; mit 2O bleibt E ↔ −E exakt; weitere Kramers-erhaltende Massen (auch paritätsungerade); Untergrund −αk², sauberes Dirac-Regime μ ≪ ε | **erledigt**, siehe [`RESULTS_PhaseBit_S4_de.md`](RESULTS_PhaseBit_S4_de.md) |
| S4+ | Offene Punkte von S4 | Θ aus S4 ist PT, echte Zeitumkehr T = ΠΘ; Helizität H = exakte Gitter-Chiralität, paritätsungerade Masse = chiral gedrehtes μΠ (θ unbeobachtbar, bricht P und T); 2D: Dirac-Punkt = ein Kramers-Paar, T-erhaltende Masse unmöglich; Untergrund am inneren Dirac-Punkt exakt kompensierbar (λ* = −0.408, 90°-Transportterm); Dirac-Regime μ/ε = 0.1 mit exakter k-Raum-Methode: v nahe Dirac, Zitterbewegung 2E auf 2 % | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open_de.md`](RESULTS_PhaseBit_S4_Open_de.md) |
| S4++ | Neue offene Punkte | 2O + λ*·ε·W₉₀ (λ* ≈ −5): E ↔ −E und α = 0 ohne Verdopplung, bei ε = 0.3 auf 2 % Dirac bis k = 0.4 (μ/ε = 0.5); θ beobachtbar neben helizitätsbrechender Masse (Lücke 4μ\|1+e^{iθ}\|); Massen-Domänenwand μΠ → −μΠ trägt T-geschützten masselosen 2D-Dirac-Kegel (c/3), chirale Wand gegapt: 2D-Spin-Wirbel = Randzustand des 3D-Wirbels; Zitterbewegungs-Amplitude = Bandtheorie auf 0.6 % | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open2_de.md`](RESULTS_PhaseBit_S4_Open2_de.md) |
| S4+++ / 1b | Gravitation auf 2O*, Z₂, Eichfeld | Vakuum-Gegenterm nötig (Floquet-Offset gravitiert); R′ = 0.997 für Teilchen und Antiteilchen (ohne Kompensation 0.73/1.20); Uhren-Lapse = Nordström (Licht unabgelenkt, κ = 1/γ²), voller Lapse ½{N,H} = Einstein 1911 (Licht fällt wie langsame Materie); Wandkegel mit 1 Kramers-Paar robust gegen T-gerade Störung, mit 2 Paaren vermiedene Kreuzung (Lücke 4·10⁻³); [H, W_B] = 0: θ bleibt mit Eichfeld unbeobachtbar, Anomalie betrifft Knoten-Chiralität | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open3_de.md`](RESULTS_PhaseBit_S4_Open3_de.md) |
| S4⁴ / 1c | Lokaler Lapse, räumliche Metrik, Gegenterm, innerer Knoten | Warte-Puffer pro Knoten (Phase π, lokale Mischung R(a)): exakt unitär, Spektrum um E = 0 mal N = (1+cos a)/2, Puffer hält 1 − N → lokaler voller Lapse = Einstein 1911; Puffer-Lapse N/A + Masse μA = schwache ART-Metrik: Licht 2× (1.999), ruhende Masse 1×, bewegte 1 + v²/c′²; Gegenterm erzwungen (R′ = 1 ± δ/m, Vakuum = Fixpunkt der Warte-Regel); E ∥ B in der Zeit: innerer Knoten +0.70…0.85 c′ gegen −0.99 c′ unten | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open4_de.md`](RESULTS_PhaseBit_S4_Open4_de.md) |
| S4⁵ / 1d | Starker/zeitabhängiger Lapse, Lapse aus der Last, anisotrope Metrik, globale Landau-Zweige | Ein Puffer reicht für 0 < N ≤ 1, Lapse = aktiver Anteil des Vakuum-Fixvektors; Photon vor N → 0 blauverschoben bis zur Gittergrenze, dann Umkehr (Spiegel statt Horizont); Last e(x) = lokale Energiedichte (Vakuum: 0) als Quelle: aktive = passive Masse = E, drittes Gesetz bis 2 %, aber bewegte Quelle zieht mit E statt E(1+v²); Warten je Richtung mittelt sich weg, Coin-Kopplungen je Richtung geben jede spurfreie Metrik (2 kubische Konstanten), GW-Interferometer = ART für k → 0; det U(k_z) windet nicht (kein Floquet-Fluss über den Zonenrand) | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open5_de.md`](RESULTS_PhaseBit_S4_Open5_de.md) |
| S4⁶ / 1d | ART-Quelle, kausale Gravitation, Spiegel für Masse, Bandstruktur des Helizitätssektors | Quelle s = 2e − e_M (Rechenlast doppelt minus Massen-Last μ∂E/∂μ, beide lokal) = T₀₀ + T_ii: bewegte Quelle 1 + v² (1.887/1.964 gegen 1.892/1.970), Licht 2× (1.991); lokale Wellengleichung für Φ: Wirkung erst nach D/c′; fallendes massives Teilchen wird an der Gittergrenze gespiegelt und springt fast auf die Starthöhe zurück; Helizitätssektor ohne globale Lücke → nur Netto-Fluss (0) topologisch | **erledigt**, siehe [`RESULTS_PhaseBit_S4_Open6_de.md`](RESULTS_PhaseBit_S4_Open6_de.md) |

Zu S3: Die 60°-Drehungen zwischen benachbarten Kuboktaeder-Richtungen liegen
**nicht** in der Oktaedergruppe O. Ein Element von O, das d auf d′ abbildet,
existiert immer (z. B. eine 120°-Drehung um eine Raumdiagonale), aber es ist nicht
die minimale Drehung. Ob ein solcher endlicher Transport trotzdem ein
konsistentes Spin-½ ergibt, ist die eigentliche offene Frage. Wenn ja, gäbe es
Spin ½ aus einem endlichen Phasen-Bit-Register auch in 3D.

### Experiment 1: Gravitation als Rechenzeit (Nebenlinie)

Ursprüngliche Idee: Jeder Knoten hat ein **begrenztes Rechenbudget**. Sind dort
viele Felder aktiv, laufen **alle** Berechnungen langsamer, auch die Ausbreitung.
Eine Welle, die an einer stark belasteten Zone vorbeiläuft, wird daher abgelenkt.
In der Stufenfolge unten sind das 1b (alles langsamer) und 1d (Verlangsamung aus
der lokalen Last). 1a hat gezeigt, dass es **nicht** reicht, nur das Abbiegen zu
verlangsamen. Das stützt die Idee, dass auch die Ausbreitung mitbetroffen sein muss.

Leitfrage: **Welche Art von „langsamer rechnen“ reproduziert die Gravitation?**
Die Stufen sind so gewählt, dass jede eine historisch konkurrierende Theorie
nachbildet und eine falsifizierbare Vorhersage hat.

| Stufe | Was wird verlangsamt | Erwartung | Status |
|-------|----------------------|-----------|--------|
| **1a** | nur die innere Uhr (Abbiegerate ε → ε·N) | Nordström (skalar): universeller Fall, **keine** Lichtablenkung | **erledigt**, siehe [`RESULTS_Gravity_Lapse_de.md`](RESULTS_Gravity_Lapse_de.md) |
| 1b | die ganze Rechnung, also auch die Ausbreitung (c → N·c) | Einstein 1911: halbe ART-Ablenkung, Fall wie 1a | **erledigt**: global in [`RESULTS_PhaseBit_S4_Open3_de.md`](RESULTS_PhaseBit_S4_Open3_de.md), lokal (Warte-Puffer) in [`RESULTS_PhaseBit_S4_Open4_de.md`](RESULTS_PhaseBit_S4_Open4_de.md) |
| 1c | zusätzlich die räumliche Metrik (Gitterabstand / Brechungsindex) | volle ART-Ablenkung (Faktor 2 gegenüber 1b) | **erledigt**: Puffer-Lapse N/A + Masse μA, Licht 1.999× 1b, bewegte Masse 1 + v²/c′², siehe [`RESULTS_PhaseBit_S4_Open4_de.md`](RESULTS_PhaseBit_S4_Open4_de.md) |
| 1d | Rückwirkung: Lapse aus der lokalen Teilchendichte | selbstkonsistentes Feld, Poisson-Gleichung? | offen, spekulativ |

**Ergebnisse 1a (Kurzfassung):** Der Lapse-Schritt ist exakt unitär. Gitter und
Halbklassik stimmen auf 10⁻³ überein. Der freie Fall ist im Kontinuumslimes
universell (Äquivalenzprinzip, Abweichung O(ε²)), aber mit R = (m₀/m*)/c² → 2
statt 1. Die Ursache ist die Bandkrümmung des Turning-Modells (m* = m₀/2c²).
Bewegte Teilchen koppeln mit κ ≈ 1/γ² und entkoppeln bei hoher Energie. Das ist
das Nordström-Verhalten.

**Nächste Schritte 1b:**
- Lokale Lichtgeschwindigkeit unitär realisieren. Kandidaten:
  (i) ein Ruhe-Kanal (der „gerade/zeitartige“ Zug aus Paper 1) mit lokaler
  Mischung η(r), kalibriert so, dass c_eff = N·c **und** m = N·m₀ gelten;
  (ii) In/Out-Register pro Knoten mit teilweisem Transfer pro Link
  (Brechungsindex, Fresnel-Reflexion an Gradienten mitmessen).
- Vorhersage prüfen: Ablenkwinkel eines masselosen Pakets (α = 1/2) an einem
  Lapse-Profil N(r) = 1 − GM/(c²r), frequenzunabhängig, 1911-Wert.
- Vorher klären: Welches Modell hat R = 1? Kandidaten sind das hexagonale Modell
  aus Paper 1 (relativistisch angepasst, aber nicht unitär) und der FCC-Walker mit
  SU(2)-Münze.

### Experiment 2: Ganzzahlige Formulierung (geht in S1 auf)

- Propagatoren exakt über ganzzahlige Pfadzahlen N_T(x,t) rechnen (memoisiert,
  siehe `NOTES_Caching_de.md`).
- Messen, wie viele Bits pro Knoten und Schritt die exakte Dynamik braucht. Das
  beantwortet quantitativ, wie endlich der Automat sein kann.
- Exakt unitäre Münzen ohne Gleitkomma: Cayley-Transformation
  C = (1 + iεG/2)(1 − iεG/2)⁻¹ mit rationalem ε, Einträge in Q(i), z. B. (3+4i)/5.
  Wachstum der Nenner messen.

### Experiment 3: Isotropie der Schalen über n (Priorität 3)

- Lichtkegel-Front des FCC-Walkers auf Schale n nach Kugelflächenfunktionen bzw.
  O_h-Darstellungen zerlegen.
- Messen, wie schnell die Anisotropie ab l ≥ 4 mit n abnimmt. Das ist die
  quantitative Antwort auf die ursprüngliche Bloch-Kugel-Frage.

### Experiment 4: Feldabstrahlung und 1/r (zurückgestellt)

- Braucht ein zweites, masseloses Feld und eine Kopplung an das Teilchen.
  Sinnvoll erst nach 1d.

---

## 3. Mögliche Gliederung von Paper 3

1. Motivation: diskret auf allen Ebenen; Phasen-Bit = Z₄-Abbiegezähler aus Paper 1
2. Ganzzahlige Pfadzahlen und die Grenze „unitär ⇒ Permutation“ (Exp. 2)
3. Gravitation als Rechenzeit: Nordström → Einstein 1911 → ART (Exp. 1a–1c)
4. Äquivalenzprinzip auf dem Gitter: R = m₀/(m*c²) als Prüfstein
5. Ausblick: Schalen-Auflösung (Exp. 3), selbstkonsistentes Feld (Exp. 1d, 4)
