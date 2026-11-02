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

### Experiment 1: Gravitation als Rechenzeit (Priorität 1)

Leitfrage: **Welche Art von „langsamer rechnen“ reproduziert die Gravitation?**
Die Stufen sind so gewählt, dass jede eine historisch konkurrierende Theorie
nachbildet und eine falsifizierbare Vorhersage hat.

| Stufe | Was wird verlangsamt | Erwartung | Status |
|-------|----------------------|-----------|--------|
| **1a** | nur die innere Uhr (Abbiegerate ε → ε·N) | Nordström (skalar): universeller Fall, **keine** Lichtablenkung | **erledigt**, siehe [`RESULTS_Gravity_Lapse_de.md`](RESULTS_Gravity_Lapse_de.md) |
| 1b | die ganze Rechnung, also auch die Ausbreitung (c → N·c) | Einstein 1911: halbe ART-Ablenkung, Fall wie 1a | offen |
| 1c | zusätzlich die räumliche Metrik (Gitterabstand / Brechungsindex) | volle ART-Ablenkung (Faktor 2 gegenüber 1b) | offen |
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

### Experiment 2: Ganzzahlige Formulierung (Priorität 2)

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
