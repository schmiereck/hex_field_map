# Bit-Test 22 — Dynamischer Strukturfaktor auf dem D4-Bitgitter: Dichte und Richtungsströme

**Stand:** 11. November 2026. **Status:** Ausgeführt, mit unabhängiger Streaming-Kontrolle und exakter mikroskopischer Rückwärtsprüfung. Ausgangspunkt sind Bit-Test 20/21 (rekonstruierte, phasenkodierte Pakete). **Die Bit-8-Kollision und das D4-Streaming sind unverändert.** Alle Fouriertransformierten und komplexen Zahlen sind ausschließlich externe Diagnostik.

## 1. Forschungsfrage und begriffliche Trennung

Bit 20/21 zeigten für schmale, phasenkodierte **Paketantworten** die frühe scheinbare Richtungsfolge +/−/+ bei sichtbarem Träger $k_x=\pi/8$ und verborgenen Wellenzahlen $q=\pi/8,\pi/4,3\pi/8$. Insbesondere wechselte der Schwerpunkt des +$k_x$-Pakets bei $q=\pi/4$ auf die −x-Seite. Der Begriff „Gruppengeschwindigkeit einer einzelnen Eigenmode“ war dafür ausdrücklich **nicht** gerechtfertigt.

Dieser Test misst zum ersten Mal einen **stationären Gleichgewichts-Strukturfaktor** aus unbeeinflussten, gleichverteilten 24-Bit-Zuständen, getrennt nach Dichte $\rho$, x-Richtungsstrom $J_x$ und verborgenem w-Richtungsstrom $J_w$. Die endlichen, extern phasenkodierten **getriebenen Dichteantworten** aus Bit 20 werden lediglich ergänzend spektral dargestellt: $|\mathrm{FFT}_t\langle\delta\rho_{k,q}(t)\rangle_{\rm seed}|^2$ ist **kein** stationärer Gleichgewichts-Strukturfaktor. In Bit 20 wurde noch **kein** Strom der getriebenen Pakete gespeichert.

Die Nullhypothese ist: **Mehrere Frequenzanteile liegen bereits in der freien D4-Richtungsbewegung**, und die lokale Kollision verschiebt oder verbreitert deren Gewichte. Ein neuer, isolierter kollektiver Eigenzweig darf nur behauptet werden, wenn er unabhängig davon und mit gut aufgelösten Frequenzen identifiziert wurde.

## 2. Messdefinition

Jeder physische D4-Knoten besitzt 24 richtungsabhängige Bits $b_d(\mathbf{x},t)\in\{0,1\}$ mit ganzzahligen D4-Wurzeln $\mathbf v_d$. Die drei lokalen Messgrößen sind

$$
\rho=\sum_{d=1}^{24}b_d,\qquad J_x=\sum_d v_{d,x}b_d,\qquad J_w=\sum_d v_{d,w}b_d.
$$

Räumliche Fourierkoeffizienten werden für die vorab festgelegten physikalischen $(k_x,q)$ gebildet:

$$
\widehat A(k_x,q,t)=\frac{1}{\sqrt{N}}\sum_{\mathbf{x}\in D_4} A(\mathbf{x},t)\,e^{-i(k_x x+qw)}.
$$

Mit vorab entferntem zeitlichem Mittel je Seed und Hann-Fenster $h_t$ lautet das gemessene **Auto-/Kreuzspektrum**

$$
S_{AB}(k_x,q,\omega_n)=\frac{1}{N_s\sum_t h_t^2}\sum_{s=1}^{N_s}
\mathcal F_t[h_t(\widehat A_s-\overline{\widehat A_s})]_n\,\mathcal F_t[h_t(\widehat B_s-\overline{\widehat B_s})]_n^*.
$$

Neben den Diagonalen $S_{\rho\rho}$, $S_{J_xJ_x}$ und $S_{J_wJ_w}$ wird der longitudinale Strom $J_\parallel=(k_xJ_x+qJ_w)/\sqrt{k_x^2+q^2}$ ausgewertet, einschließlich normierter $\rho$–$J_\parallel$-Kohärenz. **Eine hohe Kohärenz ist wegen der mikroskopischen Erhaltungsgleichungen erwartbar; allein ist sie kein Eigenmodenbeweis.** Die FFT verwendet die NumPy-Vorzeichenkonvention $e^{-i\omega_n t}$; die Frequenzvorzeichen sind relativ zu dieser Konvention angegeben. Das Zeitmittelabziehen schwächt sehr tiefe Frequenzen, weshalb die Kategorisierung der freien Streaming-Linien $|\omega|>0{,}16$ verwendet.

### Versuche

| Fall | D4-Form $(x,y,z,w)$ | physische Knoten | unabhängige Seeds | Ticks vorwärts + Rücklauf |
|---|---|---:|---:|---|
| W32 | $(48,4,4,32)$ | 12.288 | 8 | 80 Vorwärts, erster Seed 80 rückwärts |
| y/z8 | $(48,8,8,32)$ | 49.152 | 4 | 80 Vorwärts, erster Seed 80 rückwärts |

Je Seed: 24 unabhängige faire Besetzungsbits pro Knoten, ohne Paketpräparation (exakt uniformes stationäres Ensemble, da die vollständige Dynamik bijektiv ist). Dieselben Mikrozustände laufen parallel mit **freier** Richtungsverschiebung ohne Kollision. Untersucht: $q=\pi/8,\pi/4,3\pi/8$ und $k_x=0,\pm\pi/12,\pm\pi/8,\pm\pi/6$, zusammen 21 räumliche Fourierkanäle. Jeder Verlauf hat 81 Messzeitpunkte, daher Frequenzabstand $\Delta\omega=2\pi/81\approx0{,}07757$ pro Tick. Die Werte für Haupt- und Quergitter dürfen wegen verschiedener Knotenzahlen nur nach der angegebenen Normierung verglichen werden; die Quergitter-Stichprobe ist kleiner.

## 3. Ein analytisch prüfbarer Streaming-Nullfall

Bei ausgeschalteter Kollision ist jede Richtungsbesetzung unabhängig und hat exakt die Zeitfrequenz

$$
\omega_d(k_x,q)=k_x v_{d,x}+q v_{d,w}\pmod{2\pi}.
$$

Für $(k_x,q)=(\pi/8,\pi/4)$ ergeben sich die verschiedenen Vorzeichenklassen

$$
0,\quad \pm\pi/8,\quad \pm\pi/4,\quad\pm3\pi/8.
$$

Die **vollständigen freien 81-Punkt-Zeitreihen aller 21 Kanäle, inklusive Dichte, $J_x$ und $J_w$**, wurden für den ersten Seed aus den 24 D4-Wurzeln analytisch nachgerechnet. Maximaler absoluter Unterschied zum separat simulierten freien Streaming: **2,39×10⁻⁷** (12.288 Knoten) bzw. **2,57×10⁻⁷** (49.152 Knoten). Die Abweichung ist mit der Speicherung der Diagnostik als `complex64` vereinbar. Damit sind die frei transportierten Spektrallinien **nicht erst aus dem Fit erfunden**.

## 4. Zentraler Befund bei $q=\pi/4$, $k_x=\pi/8$

Auf 12.288 Knoten, acht Seeds, sind die vier starken Dichtepeaks bei $\omega\approx+0{,}388,-0{,}776,-0{,}388,+0{,}776$; sie liegen alle höchstens eine FFT-Breite von den exakt erlaubten **freien** D4-Richtungsphasen entfernt und treten in beiden Seed-Hälften auf.

Zwei davon sind in den Stromkanälen sehr deutlich verschieden:

| gemessene $\omega$ | $S_{\rho\rho}$ | $S_{J_xJ_x}$ | $S_{J_wJ_w}$ | dominierender Richtungsstrom |
|---:|---:|---:|---:|---|
| +0,3879 | 62,2 | 58,3 | 1,77 | **x** |
| −0,7757 | 58,5 | 0,65 | 55,7 | **w** |

Der Strom-Polarisationsindex $P=(S_{J_xJ_x}-S_{J_wJ_w})/(S_{J_xJ_x}+S_{J_wJ_w})$ hat dort die Werte **+0,941 [Bootstrap 95 %: +0,841…+0,977]** und **−0,977 [−0,989…−0,955]**. Auf dem größeren Quergitter bleiben die Unterschiede erhalten: **+0,915 [0,477…0,989]** und **−0,963 [−0,971…−0,943]**, bei nur vier Seeds entsprechend unsicherer.

**Interpretation:** Die Dichteantwort enthält gut unterscheidbare, gerichtete spektrale Beiträge. Ein einziger Skalar-Dispersionszweig erklärt die beobachtete Struktur **nicht ohne zusätzliche Moden-/Polarisationsannahmen**. Allerdings erscheinen entsprechende Frequenzen und Richtungsströme **auch ohne Kollision**. Wir haben daher **keinen neu entstandenen, stabilen kollektiven Eigenzweig** nachgewiesen; eine Änderung der bereits kinematisch möglichen Streaming-Mischung ist die sparsamere derzeitige Erklärung. Die exakte Auswertung der Reversibilität beweist weiterhin keine lange Lebensdauer eines kollektiven Pakets.

## 5. q-Skala und kollisionsbedingte spektrale Verbreiterung

Für $k_x=\pi/8$ wird der Anteil des Dichtespektrums in Fenstern von ±1,15 FFT-Bins um die analytischen freien Richtungsfrequenzen gemessen, ohne $|\omega|\le0{,}16$. Diese gewählte **Operationalisierung** ist kein universeller Modenanteil; wegen der endlichen Zeit und des Hann-Fensters können auch reine Streaming-Linien etwas außerhalb liegen.

| $q$ | Bit 8, W32 | Frei, W32 | Bit 8, y/z8 | Frei, y/z8 |
|---|---:|---:|---:|---:|
| $\pi/8$ | 86,7 % | 100,0 % | 86,6 % | 100,0 % |
| $\pi/4$ | 82,7 % | 99,5 % | 87,2 % | 98,8 % |
| $3\pi/8$ | 67,9 % | 96,2 % | 77,4 % | 96,4 % |

Bei $q=\pi/4$, W32: **Differenz (Bit 8 minus frei) −16,7 Prozentpunkte**, gepaarter Seed-Bootstrap-95-%-Bereich **−20,3 bis −14,0 Prozentpunkte**; auf dem größeren Quergitter **−11,6 Punkte [−13,7; −9,2]**. Bei $q=3\pi/8$ ist die Verschiebung noch deutlicher (W32 −28,3 Punkte). Die Kollision verteilt spektrale Leistung von scharfen freien Streaming-Linien auf benachbarte Frequenzen. Wegen dieser Glättung und nur 80 Ticks kann daraus **keine** separate Lebensdauer oder exakte Dispersionskurve bestimmt werden.

Die Zeitfrequenzkonzentration hängt von $q$ ab; die beiden Quergitter zeigen qualitativ das gleiche Muster, quantitativ jedoch unterschiedliche Anteile. Ein von der Geometrie unabhängiges neues Naturgesetz wird damit nicht behauptet.

## 6. Vergleich mit den tatsächlich angeregten Paketen aus Bit 20/21

Die archivierten Bit-20-Arrays liefern 25 komplexe Diagnostik-Zeitpunkte (24 Ticks), jedoch nur **Dichteantworten** pro Seed in drei Beobachtungskanälen. Für die kohärente, über Seeds gemittelte und im sichtbaren $k_x$ projizierte $q$-Antwort ergibt sich bei $q=\pi/4$, $k_x=\pi/8$ ein dominanter Peak nahe $\omega=-0{,}754$. Auch die zugehörige **freie** Paketantwort hat ihren dominanten Peak nahe dieser Frequenz. Die grobe Frequenzauflösung ist **$2\pi/25\approx0{,}251$**, nicht $2\pi/81$. Die Spektren sind wegen unterschiedlicher Anfangsmaße und Observablen nicht punktweise gleichzusetzen.

Das Ergebnis passt zu der Möglichkeit, dass die in Bit 20/21 gemessene **negative Schwerpunktbewegung** bei $q=\pi/4$ aus veränderter Überlagerung und Gewichtung mehrerer x-/w-orientierter Beiträge hervorgeht. Es **beweist die Kausalität nicht**; selbst der kollisionsfreie Puls besitzt mehrere Frequenzen, bewegt sich jedoch früh in die andere Richtung. Die Ableitung einer einzigen $\partial\omega/\partial k_x$ aus dem stärksten Spektralpeak bleibt daher unzulässig.

## 7. Prüfungen und Grenzen

- **Mikrophysik:** unveränderte Bit-8-Kollision in der zuvor gegen die Python-Referenz getesteten Numba-Version. In allen 12 neuen Gleichgewichtsläufen blieben Teilchenzahl und alle vier Impulskomponenten erhalten. In jeder der beiden Geometrien hat der erste Seed **80 exakte Rückwärtsschritte** zum originalen 24-Bit-Feld bestanden.
- **Streaming-Nullfall:** direktes analytisches Nachrechnen aller drei Sonden, aller 21 Raumkanäle und aller 81 Zeitpunkte mit Fehler $<3\cdot10^{-7}$.
- **Unabhängige Seeds:** je acht bzw. vier anfängliche uniforme Bitfelder; explizite Train/Held-out-Kontrolle der zentralen Peakpositionen. Bootstrap-Bereiche sind beschreibend, nicht um Mehrfachtests oder Zeit-Korrelationen korrigiert.
- **Nur zwei endliche Gitter:** keine systematische thermodynamische Extrapolation, keine Verschiebung der $k$-Gitterauflösung oder der 80-Tick-Frequenzauflösung. Sehr tiefe Frequenzen sind durch die Auswertungsdefinition unterdrückt.
- **Gleichgewicht vs. getriebene Antwort:** Der stationäre $S_{\rho\rho}$ und die selektive Paketantwort messen unterschiedliche Dinge. Ein Gleichgewichtsmaximum ist nicht automatisch eine selektiv angeregte Mode. Die 24 reellen Bitfelder pro Paket-Seed existieren im Bit-20-Lauf nur als jeweils vier mikroskopische Quadraturtrajektorien, nicht als ein komplexer physischer Zustand.
- **Keine mikroskopische D3-Rückkopplung, kein D8-Übergang:** Beobachtet wird ausschließlich die Projektion der D4-Dynamik auf Wellenzahlen und die Schicht $w=0$. Die weitergehende Dimensionshypothese wird hier nicht getestet.

## 8. Schlussfolgerung / nächster physikalischer Test

**Die q-abhängige Paketantwort ist real im Sinne der ausgeführten endlichen D4-Bit-8-Simulation.** Bit-Test 22 liefert eine **mehrkomponentige, strompolarisierte Spektralstruktur**, deren stärkste Frequenzen bereits durch die **freien D4-Richtungen** vorgegeben sind; Kollisionen verteilen und verbreitern diese spektrale Leistung. Ein neuer einzelner langlebiger Eigenzweig ist derzeit **nicht belegt**. Das ist eine Einschränkung der kollektiven-Massenzweig-Interpretation, aber ein konkretes Werkzeug zur Untersuchung der 4D→3D-Projektion.

**Empfohlener Bit-Test 23:** Den linearen Antwortoperator der 24 diskreten Richtungsbesetzungen **um die stationäre Halbfüllung** bestimmen (lokaler Kollisions-Jacobi aus gepaarten Bit-Ensembles, danach Fourier-Streaming), auf dieser Grundlage mehrere Eigenzweige und ihre x-/w-Polarisation unabhängig identifizieren. Anschließend gezielt **nur einen** vorhergesagten Zweig statt des akustischen Mischprofils anregen und unter Bit 8 prüfen, ob dessen Gruppenbewegung die beobachteten $q$-abhängigen Vorzeichen reproduziert. Die linearisierte Diagnose ändert wiederum keine mikroskopischen Regeln.

## 9. Reproduktion und Archiv

```bash
# Unterstützende Module aus dem unveränderten Bit-20-Backup als support/src bereitstellen.
export PYTHONPATH=src:support/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python src/phase_bit_s4_open41_structure_factor.py --shape 48 4 4 32 --ticks 80 --seeds 8 \
  --out data/bit22_equilibrium_w32.npz --meta data/bit22_equilibrium_w32_metadata.json
python src/phase_bit_s4_open41_structure_factor.py --shape 48 8 8 32 --ticks 80 --seeds 4 --seed0 412900 \
  --out data/bit22_equilibrium_yz8.npz --meta data/bit22_equilibrium_yz8_metadata.json
python src/phase_bit_s4_open41_validate.py --equilibrium data/bit22_equilibrium_w32.npz data/bit22_equilibrium_yz8.npz \
  --metadata data/bit22_equilibrium_w32_metadata.json data/bit22_equilibrium_yz8_metadata.json --out data/bit22_validation.json
python src/phase_bit_s4_open41_spectral_analysis.py --equilibrium data/bit22_equilibrium_w32.npz data/bit22_equilibrium_yz8.npz \
  --metadata data/bit22_equilibrium_w32_metadata.json data/bit22_equilibrium_yz8_metadata.json \
  --bit20-dir support --out data/bit22_spectral_summary.json
python src/phase_bit_s4_open41_evidence.py --equilibrium data/bit22_equilibrium_w32.npz data/bit22_equilibrium_yz8.npz \
  --metadata data/bit22_equilibrium_w32_metadata.json data/bit22_equilibrium_yz8_metadata.json --out data/bit22_key_findings.json
python src/phase_bit_s4_open41_plot.py
```

**Artefakte:** Quellcode in `src/`, Originalmesszeitreihen und Metadaten in `data/`, Einzeldiagnosen und Diagramme in `figures/`, wissenschaftlicher Bericht hier. Im vollständigen ZIP sind die erforderlichen **Bit-20-Vorgängermodule und Bit-20-Rohdaten** unter `support/` sowie das vollständige Bit-21-Backup unter `previous/` enthalten. Die alten Feldgrafiken und andere Projektartefakte bleiben auf `main` unberührt.
