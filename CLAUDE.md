# CLAUDE.md — Projektübersicht: hex_field_map

## Projektziel

Implementierung und Analyse von **diskreten Quanten-Pfadintegral-Modellen** auf verschiedenen Gittern.
Kernfrage: Welche Gittergeometrie reproduziert eine relativistische Quantenmechanik
(Dirac-Gleichung) mit korrekter Lichtgeschwindigkeit c, physikalischer Masse m ∝ ε und Kausalität?

### Amplitudenregel (alle Modelle)
- Selbe Richtung wie vorheriger Schritt → Faktor **1**
- Richtungswechsel → Faktor **iε** (ε ist der Masseparameter)
- Gesamtamplitude = Produkt aller Schrittfaktoren
- Wahrscheinlichkeit = |Summe über alle Pfade|²

---

## Modelle (chronologisch entwickelt)

### 1+1D Modelle (`quantum_path_integral.py`, `quantum_dispersion_phys.py`)

| Modell | Datei | Moves | Kantenlänge | c | m |
|--------|-------|-------|-------------|---|---|
| Feynman Checkerboard | `quantum_path_integral.py` | (±1, +1) | √2 | 1 | ≈ ε |
| Quadratisch + Rest | `quantum_path_integral.py` | (−1,+1),(0,+1),(+1,+1) | √2,1,√2 | ≈1 | ≈ ε |
| Gleichseitiges Dreieck | `quantum_dispersion_phys.py` | (±√3/2,+½),(0,+1) | **1** (alle gleich) | **√3** | ≈ ε |

Das gleichseitige Dreiecksgitter ist das physikalisch natürlichste: alle Kanten gleich lang,
c = √3 geometrisch exakt, keine Skalierungsartefakte.

### 2+1D Modell (`quantum_hex_2d.py`) ← **aktuelles Hauptmodell**

Erweiterung auf 2 Raumdimensionen mit hexagonalem Gitter:
- **7 Richtungen**: 6 diagonal (0°, 60°, 120°, 180°, 240°, 300°) + 1 geradeaus
- Alle Kantenlängen = 1
- Δt = 0.5 für diagonale Moves, Δt = 1.0 für geraden Move
- **14×14 Transfermatrix** (je 7 Amplituden für aktuelle und vorherige Zeit)

---

## Dateien

### Python-Skripte

| Datei | Beschreibung |
|-------|--------------|
| `quantum_path_integral.py` | 1+1D Simulation (3 Modelle), erzeugt Vergleichsplots |
| `quantum_dispersion.py` | Dispersionsanalyse (ältere Version) |
| `quantum_dispersion_phys.py` | Physikalisch korrekte Dispersionsanalyse (1+1D) |
| `quantum_phase_patterns.py` | Phasenmuster-Analyse |
| `quantum_lattice_viz.py` | Gittervisualisierung |
| `quantum_hex_2d.py` | **2+1D hexagonales Modell** (Hauptdatei) |

### Ergebnisdateien

| Datei | Beschreibung |
|-------|--------------|
| `RESULTS.md` | Ergebnisse der 1+1D Modelle |
| `RESULTS_2D.md` | Ergebnisse des 2+1D hexagonalen Modells (mit Abbildungen) |

### Generierte Abbildungen (2+1D)

| Datei | Inhalt |
|-------|--------|
| `lattice_geometry_2d.png` | Gittergeometrie, Kantenlängen |
| `spacetime_spread_2d.png` | \|ψ(x,y,t)\|² mit Lichtkegel r=√3·t |
| `dispersion_relation_2d.png` | E(k) + 2D-Heatmap |
| `group_velocity_2d.png` | Gruppengeschwindigkeit |
| `epsilon_sweep_2d.png` | m(ε)-Abhängigkeit |

---

## Kernphysik: 2+1D Hexagonalmodell

### Transfermatrix (`TM14_half`)
```
M_half = [[A, B],   (14×14)
          [I7, 0]]

A[d,d'] = exp(i·kx·Δx[d] + i·ky·Δy[d]) · C[d,d']   (Diagonalbewegungen)
B[d,6]  = exp(i·kx·Δx[6] + ...) · C[d,6]              (Gerader Move)
C[d,d'] = iε falls d≠d', sonst 1                       (Amplitudenregel)

M_full = M_half @ M_half   (ein ganzer Zeitschritt)
```

### Schlüsselfunktionen
- `simulate_hex_2d(T, eps)` — Zeitentwicklung via Rekurrenz auf (Nx,Ny,7)-Array
- `TM14_half(kx, ky, eps)` — einzelner k-Punkt
- `TM14_full_batch(kx_arr, ky_arr, eps)` — gebatcht (n_k,n_k,14,14)
- `fit_rel_2d_direct(eps)` — misst c=√3 (geometrisch) und m=arctan(2ε/(1−ε²))
- `physical_band_2d(...)` — wählt physikalisches Band via E_ref-Selektor

### Physikalische Masse
Das propagierende Band startet bei k=0 am **5-fach entarteten Eigenwert**:
```
m_phys = arctan(2ε / (1−ε²))  ≈  2ε   (für kleine ε)
```
Nicht der einzelne Eigenwert bei ε (das ist ein nicht-propagierender Mode).

---

## Bestätigte Ergebnisse (2+1D)

| Eigenschaft | Wert | Bemerkung |
|-------------|------|-----------|
| c | √3 = 1.7321 | Geometrisch exakt, Simulation bestätigt |
| m(ε=0.1) | 0.1993 ≈ 2ε | 5-fach entarteter k=0-Eigenwert |
| Isotropie-Fehler | 0.0000 | Bei \|k\| ≤ 0.4, 6-fache Symmetrie |
| max\|v_g\| | 1.88 ≈ c | Leichter Gitterartefakt an Zonengrenze |
| Kausalität | strikt | Lichtkegel r=√3·t eingehalten |

---

## Entwicklungszweig

Aktuelle Arbeit läuft auf: `claude/quantum-path-integral-simulator-GZNoz`

---

## Hinweise für zukünftige Entwicklung

- **Band-Tracking**: Das 5-fach entartete Band bei m=2ε spaltet sich für k>0 in Sub-Bänder.
  Der isotrope Sub-Band ist das physikalisch relevante. Bestes Tracking:
  `E_guide = 0.5*(E_ref(m=2ε) + E_prev)` mit Filter `|vg| < 1.5c`.

- **Isotropie**: Nur bei |k| ≤ 0.4 exakt; Gitterkorrekturen für größere k normal.

- **3+1D Erweiterung**: Nächster logischer Schritt wäre ein 3+1D kubisch-flächenzentriertes
  Gitter (FCC) mit analoger Amplitudenregel.
