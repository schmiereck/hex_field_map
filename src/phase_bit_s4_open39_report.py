#!/usr/bin/env python3
"""Render source-driven German research report from machine-readable Bit20 data."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
r=json.loads((root/'data/bit20_results.json').read_text())
m=json.loads((root/'data/bit20_metadata.json').read_text())
v=json.loads((root/'data/bit20_validation.json').read_text())
def row(q,k):return next(x for x in r['rows'] if x['q_mode']==q and x['k_mode']==k)
def f(x):return f'{x:+.3f}'.replace('.',',')
def pc(x):return f'{x*100:.1f}'.replace('.',',')+' %'
lines=['# Bit-Test 20 — Kontrollierte q-abhängige Paketantwort auf dem 4D-D4-Gitter',
'', 'Stand: 11. November 2026. **Ergebnis: numerische Beobachtung einer nichtmonotonen, q-abhängigen frühen Schwerpunktbewegung; keine nachgewiesene Einmoden-Dispersion, kein gebundenes Teilchen.**',
'', '## Forschungsfrage',
'', 'Beeinflusst die diskrete verborgene Wellenzahl q die räumliche Ausbreitung und das Kohärenzverhalten einer phasenkodierten, zunächst lokalisierten Störung? Ist die Asymmetrie der alten Bit-6-Kollision unter einer mikroskopisch exakt B4-symmetrischen Kollisionsregel verschwunden?',
'', '## Rechenmodell und Kontrollen',
'', '* Exakte Bit-8-Zwei-Bit-Austauschregel mit unikaler gegenseitiger Kanten-Orbit-Wahl; Integer-Realisierung im Numba-Compiler. **Keine neue Mikrodynamik**, keine stochastischen Zeitschritte.',
'* Dimensionen: `(48,4,4,16)` in Gitterkoordinaten, **6.144 physische D4-Knoten** (gerade Parität). Periode in x von 32 auf 48 vergrößert; die kurzen y/z-Perioden von 4 bleiben erhalten.',
'* Sichtbarer Detektor: `w=0`, daneben Fourierkanal der versteckten Richtung mit `q=2π m/16`, wobei `m=1,2,3`. `k_x=0,+3·2π/48,-3·2π/48`, also 0 und ±π/8.',
'* Für jedes (q,k) werden acht **unabhängige Seed-Mikrozustände** aus endlichen 16-Bit-Schwellen präpariert. Vier reelle Trajektorien (Cosinus/Sinus × Vorzeichen) je Seed; insgesamt **72 Seed-Fälle, 288 Trajektorien und je 24 Ticks**; je eine kollisionsfreie Vergleichsdynamik mit denselben Schwellen.',
'* Komplexe Quadratur, Fourier-Projektion, Mittelwerte, Bootstrap-Intervalle und Phasenfits finden **ausschließlich außerhalb** der Bit-Dynamik statt.',
'* Die Präparation ist ein dokumentierter **Nachbau der Bit-6-Idee**, kein wiederhergestellter Originaleigenmode: Referenzgewichte verwenden die akustische D4-Geschwindigkeit `ω_ref=√((k²+q²)/2)` **nur zur Initialisierung**.',
'', '## Prüfung der exakten Mikrodynamik',
'', f"* {v['local_collision_audit']['cases']} lokale Zustände stimmen bitgenau mit der vorherigen Bit-10-Implementierung überein; Involution und alle fünf additiven Ladungen geprüft.",
'* Auf 512 physischen Knoten: 4 vollständige Vorwärts- und Rückwärtsticks bitgenau zur unabhängigen Bit-10-Referenz.',
'* Auf 6.144 Knoten: 6 Ticks von `S` und `mirror_x(S)` mit exakter Spiegeläquivalenz; vollständiger bitgenauer Rücklauf. Dasselbe auch für eine konkret phasenkodierte q=π/4-Störung.',
'* Für alle 288 realen Vorwärtsläufe stimmen die fünf Ladungen am Start und nach Tick 24; vollständige Rückläufe wurden nicht für alle 288 Trajektorien gerechnet.',
'', '## Gemessene Paketbewegung: Fenster Tick 2 bis 8',
'', 'Die beobachtete Geschwindigkeit ist **die Steigung des Intensitätsschwerpunkts des ensemble-gemittelten Quadraturfelds**. Dies ist keine unabhängige Messung der Gruppengeschwindigkeit eines nachgewiesenen Spektraleigenmodus.',
'', '| q | Träger | q-Kanal, Bit8 | sichtbare w=0-Schicht | q-Kanal, nur Streaming | Amplitude bei Tick 24, Bit8/frei |',
'|---|---|---:|---:|---:|---:|']
for q in (1,2,3):
 for k in (0,3,-3):
  z=row(q,k)
  lines.append(f"| {['','π/8','π/4','3π/8'][q]} | {['','', '', '+π/8'][k] if k==3 else ('−π/8' if k==-3 else '0')} | {f(z['early_center_slope_2_to_8'])} | {f(z['visible_w0_early_center_slope_2_to_8'])} | {f(z['early_free_center_slope_2_to_8'])} | {pc(z['sample_ratio_tick_last'])} / {pc(z['free_sample_ratio_tick_last'])} |")
lines.extend(['', '**Auffällig:** Bei q=π/8 und q=3π/8 bewegen sich die positiven Träger früh nach +x, bei q=π/4 dagegen nach −x (für negative k entsprechend gespiegelt). Ohne Kollision bleibt die q=π/4-Bewegung mit +k nach +x. Auch der sichtbare w=0-Kanal wechselt im Ensemble-Mittel die Bewegungsrichtung.',
'', '## Reproduzierbarkeit der Vorzeichen und Symmetrie',
'', 'Die *individuell* ermittelten Seed-Schwerpunkt-Steigungen können von der Steigung des ensemble-gemittelten Feldes abweichen (Mittelung und Betragsquadrat vertauschen nicht). Die folgende Tabelle gibt für individuelle q-Kanal-Seeds die Bootstrap-95-%-Intervalle des Mittelwerts und das Even-Drift-Intervall für ±k an:',
'', '| q | +k, Seedmittel 95 % | −k, Seedmittel 95 % | gemeinsame even-Drift, 95 % |',
'|---|---:|---:|---:|'])
for q in (1,2,3):
 p=row(q,3);n=row(q,-3);qq=r['q_comparison'][q-1]
 def ci(z):return '['+f(z[0])+'; '+f(z[1])+']'
 lines.append(f"| {['','π/8','π/4','3π/8'][q]} | {ci(p['seed_center_slope_95ci'])} | {ci(n['seed_center_slope_95ci'])} | {ci(qq['independent_seed_even_drift_95ci'])} |")
lines.extend(['','Alle even-Drift-Intervalle enthalten 0. Es gibt in diesem Versuch somit keine belastbare Evidenz für eine bevorzugte **absolute** Richtung, während die *vorzeichensymmetrische* q-abhängige Umkehr deutlich ist.',
'', '## Phasen und Kohärenz (kein Spektraleigenmode nachgewiesen)',
'', 'Die linearen Fits der demodulierten **Phasenänderung pro Tick** sind nur ein lokaler Frequenzindikator. Sie unterscheiden sich teilweise stark zwischen Zeitfenstern, weshalb ein eindeutiges Dispersionsgesetz `ω(k,q)` aus ihnen **nicht** abgeleitet werden darf.',
'', '| q | k | Phase/Tick 2–8 | Phase/Tick 9–16 | Phase/Tick 17–24 | SNR-artiger Kohärenz-/Seedvarianz-Wert bei Tick 8 |',
'|---|---|---:|---:|---:|---:|'])
for q in (1,2,3):
 for k in (0,3,-3):
  z=row(q,k)
  lines.append(f"| {['','π/8','π/4','3π/8'][q]} | {('0' if k==0 else ('+' if k>0 else '−')+'π/8')} | {f(z['early_phase_slope_2_to_8'])} | {f(z['mid_phase_slope_9_to_16'])} | {f(z['late_phase_slope_17_to_24'])} | {z['coherent_power_over_seed_variance_tick_8']:.2f} |")
lines.extend(['', 'Besonders `q=3π/8, k=0` wechselt die Phase/Tick-Schätzung stark (früh gegen mittleres Fenster); das spricht für Modenmischung oder instabile Auswertung, nicht eine klar identifizierte ruhende Eigenfrequenz.',
'', '## Lokalisierung und Dämpfung',
'', '| q | Träger | Intensitätsbreite zu Beginn | Intensitätsbreite Tick 24 | verbliebene q-Amplitude Tick 24 |',
'|---|---|---:|---:|---:|'])
for q in (1,2,3):
 for k in (0,3,-3):
  z=row(q,k)
  lines.append(f"| {['','π/8','π/4','3π/8'][q]} | {('0' if k==0 else ('+' if k>0 else '−')+'π/8')} | {z['width_tick0']:.2f} | {z['width_tick_last']:.2f} | {pc(z['sample_ratio_tick_last'])} |")
lines.extend(['','Alle Präparationen verlieren einen erheblichen Anteil ihrer kohärenten Trägeramplitude und verbreitern sich. **Kein stabiles lokalisiertes Teilchen** wurde identifiziert. Die Stärke der Kollision im Vergleich zum freien Streaming hängt ebenfalls nichtmonoton von q ab.',
'', '## Einordnung und Limitationen',
'', '1. Nur 24 Ticks; Phasen- und Schwerpunktmessungen können von **mehreren kollektiv angeregten Moden** beeinflusst sein. Eine negative Schwerpunkt-Steigung ist weder automatisch negative Gruppengeschwindigkeit noch ein rückwärts fließendes Teilchen.',
'2. Das D4-Gitter ist periodisch, insbesondere `y=z=4`. Nach wenigen Ticks können sich diese Richtungen selbst überlagern und die effektive Dispersion gegenüber einem ausgedehnten 4D-Raum verändern.',
'3. Die Bit-6-Spektraleigenvektoren konnten nicht wiederhergestellt werden; die anfänglichen akustischen Richtungsgewichte treffen **nicht zwingend** einen Eigenmode der Bit-8-Kollision.',
'4. 8 unabhängige Seeds pro (q,k), aber **eine** Präparationsfamilie, **eine** Gittergeometrie und nur die drei q-Werte und k-Werte. Bootstrap-Intervalle quantifizieren Seed-Streuung, nicht Modellunsicherheit durch andere Gitterdimensionen.',
'5. Die q-Projektion ist ein **Messoperator** auf dem vollständigen 4D-Mikrozustand; die q-Mode-Bits sind keine eigenständige zusätzliche lokale Dynamik. Das w=0-Signal ist nur eine einzelne 3D-Schicht, nicht eine systematisch abgeleitete 4D→3D-Projektion.',
'6. Eine noch weitergehende Deutung als D3/D4/D8-Übergang, Massengap oder Elementarteilchen bleibt spekulativ.',
'', '## Nächste entscheidende Falsifikationsprüfung',
'', 'Die q=π/4-Vorzeichenumkehr auf **gleicher physikalischer Wellenzahl** auf einem 4D-Gitter mit verborgener Periode `w=32` wiederholen (`q_mode=4` statt 2) und außerdem die kleinen y/z-Perioden vergrößern. Zusätzlich zwei kleinere sichtbare k-Werte und explizite getrennte Anregungsäste testen. Nur wenn Richtung und Phasenverlauf unter diesen Änderungen robust bleiben, ist eine echte Band-/Dispersionsinterpretation gerechtfertigt.',
'', '## Dateien, Kommando und Provenienz',
'', '* `src/phase_bit_s4_open39_numba_collision.py`: exakt äquivalenter beschleunigter Ganzzahl-Kern.',
'* `src/phase_bit_s4_open39_dispersion.py`: reproduzierbarer kohärenter Vier-Trajektorien-Versuch je Seed.',
'* `src/phase_bit_s4_open39_merge.py`: führt q-Teilserien deterministisch zusammen.',
'* `src/phase_bit_s4_open39_analyze.py`: q-/w0-Antwort, Bootstrap, Phase, Breite und Abbildung.',
'* `src/phase_bit_s4_open39_validate.py`: Referenzgleichheit, Ladungen, Spiegelung, Roundtrip, Seed-Integrität.',
'* `src/phase_bit_s4_open39_plot.py`: gegenläufige Schwerpunktsgeschwindigkeit über q.',
'* `data/bit20_raw.npz`: **vollständige einzelne Rohtrajektorien (komplexe externe Quadraturauswertung)**; auch die für alle neun (q,k) identisch präparierten kollisionsfreien Kontrollen.',
'* `data/bit20_metadata.json`, `data/bit20_results.json`, `data/bit20_validation.json`: maschinenlesbare Provenienz und Messwerte.',
'* `figures/bit20_q_center_velocities.svg`, `figures/bit20_q_amplitudes.svg`: Auswertungsgrafiken.',
'', '```bash', 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1', 'python src/phase_bit_s4_open39_validate.py', 'python src/phase_bit_s4_open39_dispersion.py --qs 1 --out data/bit20_q1.npz --metadata data/bit20_q1_metadata.json', 'python src/phase_bit_s4_open39_dispersion.py --qs 2 --out data/bit20_q2.npz --metadata data/bit20_q2_metadata.json', 'python src/phase_bit_s4_open39_dispersion.py --qs 3 --out data/bit20_q3.npz --metadata data/bit20_q3_metadata.json', 'python src/phase_bit_s4_open39_merge.py', 'python src/phase_bit_s4_open39_analyze.py', 'python src/phase_bit_s4_open39_plot.py', '```', '', '*Hinweis:* Der volle Rohdatenbestand wird im ZIP und in der Projekt-Dateibibliothek bewahrt. Die zugehörigen GitHub-Dateien mit Quellcode, Ergebnissen und Abbildern enthalten keine approximierten Mikrodynamikdaten.', ''])
report=root/'docs/RESULTS_Bit20_PhaseQDispersion_de.md';report.parent.mkdir(parents=True,exist_ok=True)
report.write_text('\n'.join(lines),encoding='utf-8')
print('generated',report,'chars',report.stat().st_size)
