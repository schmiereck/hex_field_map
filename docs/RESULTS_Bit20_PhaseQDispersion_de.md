# Bit-Test 20 — Kontrollierte q-abhängige Paketantwort auf dem 4D-D4-Gitter

Stand: 11. November 2026. **Ergebnis: numerische Beobachtung einer nichtmonotonen, q-abhängigen frühen Schwerpunktbewegung; keine nachgewiesene Einmoden-Dispersion, kein gebundenes Teilchen.**

## Forschungsfrage

Beeinflusst die diskrete verborgene Wellenzahl q die räumliche Ausbreitung und das Kohärenzverhalten einer phasenkodierten, zunächst lokalisierten Störung? Ist die Asymmetrie der alten Bit-6-Kollision unter einer mikroskopisch exakt B4-symmetrischen Kollisionsregel verschwunden?

## Rechenmodell und Kontrollen

* Exakte Bit-8-Zwei-Bit-Austauschregel mit unikaler gegenseitiger Kanten-Orbit-Wahl; Integer-Realisierung im Numba-Compiler. **Keine neue Mikrodynamik**, keine stochastischen Zeitschritte.
* Dimensionen: `(48,4,4,16)` in Gitterkoordinaten, **6.144 physische D4-Knoten** (gerade Parität). Periode in x von 32 auf 48 vergrößert; die kurzen y/z-Perioden von 4 bleiben erhalten.
* Sichtbarer Detektor: `w=0`, daneben Fourierkanal der versteckten Richtung mit `q=2π m/16`, wobei `m=1,2,3`. `k_x=0,+3·2π/48,-3·2π/48`, also 0 und ±π/8.
* Für jedes (q,k) werden acht **unabhängige Seed-Mikrozustände** aus endlichen 16-Bit-Schwellen präpariert. Vier reelle Trajektorien (Cosinus/Sinus × Vorzeichen) je Seed; insgesamt **72 Seed-Fälle, 288 Trajektorien und je 24 Ticks**; je eine kollisionsfreie Vergleichsdynamik mit denselben Schwellen.
* Komplexe Quadratur, Fourier-Projektion, Mittelwerte, Bootstrap-Intervalle und Phasenfits finden **ausschließlich außerhalb** der Bit-Dynamik statt.
* Die Präparation ist ein dokumentierter **Nachbau der Bit-6-Idee**, kein wiederhergestellter Originaleigenmode: Referenzgewichte verwenden die akustische D4-Geschwindigkeit `ω_ref=√((k²+q²)/2)` **nur zur Initialisierung**.

## Prüfung der exakten Mikrodynamik

* 1430 lokale Zustände stimmen bitgenau mit der vorherigen Bit-10-Implementierung überein; Involution und alle fünf additiven Ladungen geprüft.
* Auf 512 physischen Knoten: 4 vollständige Vorwärts- und Rückwärtsticks bitgenau zur unabhängigen Bit-10-Referenz.
* Auf 6.144 Knoten: 6 Ticks von `S` und `mirror_x(S)` mit exakter Spiegeläquivalenz; vollständiger bitgenauer Rücklauf. Dasselbe auch für eine konkret phasenkodierte q=π/4-Störung.
* Für alle 288 realen Vorwärtsläufe stimmen die fünf Ladungen am Start und nach Tick 24; vollständige Rückläufe wurden nicht für alle 288 Trajektorien gerechnet.

## Gemessene Paketbewegung: Fenster Tick 2 bis 8

Die beobachtete Geschwindigkeit ist **die Steigung des Intensitätsschwerpunkts des ensemble-gemittelten Quadraturfelds**. Dies ist keine unabhängige Messung der Gruppengeschwindigkeit eines nachgewiesenen Spektraleigenmodus.

| q | Träger | q-Kanal, Bit8 | sichtbare w=0-Schicht | q-Kanal, nur Streaming | Amplitude bei Tick 24, Bit8/frei |
|---|---|---:|---:|---:|---:|
| π/8 | 0 | +0,025 | +0,197 | +0,063 | 24,8 % / 14,4 % |
| π/8 | +π/8 | +0,631 | +0,541 | +0,591 | 23,2 % / 19,3 % |
| π/8 | −π/8 | -0,638 | -0,604 | -0,601 | 23,9 % / 19,1 % |
| π/4 | 0 | +0,015 | +0,171 | -0,009 | 35,4 % / 57,5 % |
| π/4 | +π/8 | -0,369 | -0,414 | +0,336 | 20,4 % / 42,7 % |
| π/4 | −π/8 | +0,399 | +0,386 | -0,382 | 20,7 % / 42,5 % |
| 3π/8 | 0 | -0,005 | -0,106 | +0,063 | 24,4 % / 14,3 % |
| 3π/8 | +π/8 | +0,697 | +0,569 | +0,753 | 17,3 % / 19,4 % |
| 3π/8 | −π/8 | -0,666 | -0,669 | -0,764 | 16,9 % / 19,1 % |

**Auffällig:** Bei q=π/8 und q=3π/8 bewegen sich die positiven Träger früh nach +x, bei q=π/4 dagegen nach −x (für negative k entsprechend gespiegelt). Ohne Kollision bleibt die q=π/4-Bewegung mit +k nach +x. Auch der sichtbare w=0-Kanal wechselt im Ensemble-Mittel die Bewegungsrichtung.

## Reproduzierbarkeit der Vorzeichen und Symmetrie

Die *individuell* ermittelten Seed-Schwerpunkt-Steigungen können von der Steigung des ensemble-gemittelten Feldes abweichen (Mittelung und Betragsquadrat vertauschen nicht). Die folgende Tabelle gibt für individuelle q-Kanal-Seeds die Bootstrap-95-%-Intervalle des Mittelwerts und das Even-Drift-Intervall für ±k an:

| q | +k, Seedmittel 95 % | −k, Seedmittel 95 % | gemeinsame even-Drift, 95 % |
|---|---:|---:|---:|
| π/8 | [+0,581; +0,639] | [-0,638; -0,590] | [-0,019; +0,017] |
| π/4 | [-0,397; -0,311] | [+0,318; +0,402] | [-0,029; +0,033] |
| 3π/8 | [+0,626; +0,672] | [-0,654; -0,587] | [-0,005; +0,034] |

Alle even-Drift-Intervalle enthalten 0. Es gibt in diesem Versuch somit keine belastbare Evidenz für eine bevorzugte **absolute** Richtung, während die *vorzeichensymmetrische* q-abhängige Umkehr deutlich ist.

## Phasen und Kohärenz (kein Spektraleigenmode nachgewiesen)

Die linearen Fits der demodulierten **Phasenänderung pro Tick** sind nur ein lokaler Frequenzindikator. Sie unterscheiden sich teilweise stark zwischen Zeitfenstern, weshalb ein eindeutiges Dispersionsgesetz `ω(k,q)` aus ihnen **nicht** abgeleitet werden darf.

| q | k | Phase/Tick 2–8 | Phase/Tick 9–16 | Phase/Tick 17–24 | SNR-artiger Kohärenz-/Seedvarianz-Wert bei Tick 8 |
|---|---|---:|---:|---:|---:|
| π/8 | 0 | -0,296 | -0,298 | -0,350 | 2.99 |
| π/8 | +π/8 | -0,418 | -0,382 | -0,391 | 27.91 |
| π/8 | −π/8 | -0,417 | -0,385 | -0,400 | 27.56 |
| π/4 | 0 | -0,986 | -0,869 | -0,819 | 46.75 |
| π/4 | +π/8 | -0,872 | -0,785 | -0,829 | 4.96 |
| π/4 | −π/8 | -0,872 | -0,785 | -0,828 | 5.52 |
| 3π/8 | 0 | -0,429 | -1,144 | -1,184 | 1.91 |
| 3π/8 | +π/8 | -1,273 | -1,246 | -1,205 | 17.12 |
| 3π/8 | −π/8 | -1,272 | -1,244 | -1,204 | 15.39 |

Besonders `q=3π/8, k=0` wechselt die Phase/Tick-Schätzung stark (früh gegen mittleres Fenster); das spricht für Modenmischung oder instabile Auswertung, nicht eine klar identifizierte ruhende Eigenfrequenz.

## Lokalisierung und Dämpfung

| q | Träger | Intensitätsbreite zu Beginn | Intensitätsbreite Tick 24 | verbliebene q-Amplitude Tick 24 |
|---|---|---:|---:|---:|
| π/8 | 0 | 5.30 | 11.80 | 24,8 % |
| π/8 | +π/8 | 5.31 | 15.34 | 23,2 % |
| π/8 | −π/8 | 5.31 | 14.80 | 23,9 % |
| π/4 | 0 | 5.31 | 12.30 | 35,4 % |
| π/4 | +π/8 | 5.33 | 13.37 | 20,4 % |
| π/4 | −π/8 | 5.32 | 13.11 | 20,7 % |
| 3π/8 | 0 | 5.30 | 11.84 | 24,4 % |
| 3π/8 | +π/8 | 5.29 | 15.69 | 17,3 % |
| 3π/8 | −π/8 | 5.29 | 15.72 | 16,9 % |

Alle Präparationen verlieren einen erheblichen Anteil ihrer kohärenten Trägeramplitude und verbreitern sich. **Kein stabiles lokalisiertes Teilchen** wurde identifiziert. Die Stärke der Kollision im Vergleich zum freien Streaming hängt ebenfalls nichtmonoton von q ab.

## Einordnung und Limitationen

1. Nur 24 Ticks; Phasen- und Schwerpunktmessungen können von **mehreren kollektiv angeregten Moden** beeinflusst sein. Eine negative Schwerpunkt-Steigung ist weder automatisch negative Gruppengeschwindigkeit noch ein rückwärts fließendes Teilchen.
2. Das D4-Gitter ist periodisch, insbesondere `y=z=4`. Nach wenigen Ticks können sich diese Richtungen selbst überlagern und die effektive Dispersion gegenüber einem ausgedehnten 4D-Raum verändern.
3. Die Bit-6-Spektraleigenvektoren konnten nicht wiederhergestellt werden; die anfänglichen akustischen Richtungsgewichte treffen **nicht zwingend** einen Eigenmode der Bit-8-Kollision.
4. 8 unabhängige Seeds pro (q,k), aber **eine** Präparationsfamilie, **eine** Gittergeometrie und nur die drei q-Werte und k-Werte. Bootstrap-Intervalle quantifizieren Seed-Streuung, nicht Modellunsicherheit durch andere Gitterdimensionen.
5. Die q-Projektion ist ein **Messoperator** auf dem vollständigen 4D-Mikrozustand; die q-Mode-Bits sind keine eigenständige zusätzliche lokale Dynamik. Das w=0-Signal ist nur eine einzelne 3D-Schicht, nicht eine systematisch abgeleitete 4D→3D-Projektion.
6. Eine noch weitergehende Deutung als D3/D4/D8-Übergang, Massengap oder Elementarteilchen bleibt spekulativ.

## Nächste entscheidende Falsifikationsprüfung

Die q=π/4-Vorzeichenumkehr auf **gleicher physikalischer Wellenzahl** auf einem 4D-Gitter mit verborgener Periode `w=32` wiederholen (`q_mode=4` statt 2) und außerdem die kleinen y/z-Perioden vergrößern. Zusätzlich zwei kleinere sichtbare k-Werte und explizite getrennte Anregungsäste testen. Nur wenn Richtung und Phasenverlauf unter diesen Änderungen robust bleiben, ist eine echte Band-/Dispersionsinterpretation gerechtfertigt.

## Dateien, Kommando und Provenienz

* `src/phase_bit_s4_open39_numba_collision.py`: exakt äquivalenter beschleunigter Ganzzahl-Kern.
* `src/phase_bit_s4_open39_dispersion.py`: reproduzierbarer kohärenter Vier-Trajektorien-Versuch je Seed.
* `src/phase_bit_s4_open39_merge.py`: führt q-Teilserien deterministisch zusammen.
* `src/phase_bit_s4_open39_analyze.py`: q-/w0-Antwort, Bootstrap, Phase, Breite und Abbildung.
* `src/phase_bit_s4_open39_validate.py`: Referenzgleichheit, Ladungen, Spiegelung, Roundtrip, Seed-Integrität.
* `src/phase_bit_s4_open39_plot.py`: gegenläufige Schwerpunktsgeschwindigkeit über q.
* `data/bit20_raw.npz`: **vollständige einzelne Rohtrajektorien (komplexe externe Quadraturauswertung)**; auch die für alle neun (q,k) identisch präparierten kollisionsfreien Kontrollen.
* `data/bit20_metadata.json`, `data/bit20_results.json`, `data/bit20_validation.json`: maschinenlesbare Provenienz und Messwerte.
* `figures/bit20_q_center_velocities.svg`, `figures/bit20_q_amplitudes.svg`: Auswertungsgrafiken.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python src/phase_bit_s4_open39_validate.py
python src/phase_bit_s4_open39_dispersion.py --qs 1 --out data/bit20_q1.npz --metadata data/bit20_q1_metadata.json
python src/phase_bit_s4_open39_dispersion.py --qs 2 --out data/bit20_q2.npz --metadata data/bit20_q2_metadata.json
python src/phase_bit_s4_open39_dispersion.py --qs 3 --out data/bit20_q3.npz --metadata data/bit20_q3_metadata.json
python src/phase_bit_s4_open39_merge.py
python src/phase_bit_s4_open39_analyze.py
python src/phase_bit_s4_open39_plot.py
```

*Hinweis:* Der volle Rohdatenbestand wird im ZIP und in der Projekt-Dateibibliothek bewahrt. Die zugehörigen GitHub-Dateien mit Quellcode, Ergebnissen und Abbildern enthalten keine approximierten Mikrodynamikdaten.
