# Bit-Test 21 — Falsifikationsversuch der q-abhängigen Bewegungsumkehr

**Stand:** 11. November 2026  
**Status:** abgeschlossen; unabhängige Geometriekontrollen bestehen, physikalische Eigenmodeninterpretation weiterhin offen.

## 1. Fragestellung und Vorhersagen

Bit-Test 20 beobachtete bei festem sichtbaren Träger \(k_x=\pm\pi/8\) die frühe Paket-Schwerpunktbewegung \(+,-,+\) für die verborgenen Wellenzahlen \(q=\pi/8,\pi/4,3\pi/8\). Bei \(q=\pi/4\) bewegte sich das Paket bei positivem \(k_x\) sogar *entgegen* der kollisionsfreien Referenz. Es war unklar, ob dies ein Artefakt der kleinen periodischen Box \((48,4,4,16)\), eine sich verformende Modenüberlagerung oder eine kollektive Dispersionserscheinung ist.

**Prüfungen:** (i) Nur die verborgene Periode 16→32 verdoppeln, dabei **physikalisch dasselbe** \(q\) durch Verdoppelung des ganzzahligen Fourierindex behalten; (ii) separat die transversalen Perioden \(y,z\): 4→8 vergrößern; (iii) beide Veränderungen gleichzeitig bei \(q=\pi/4\) prüfen; (iv) die sichtbare Wellenzahl \(k_x\) bei festem \(q\) variieren. Die Entwicklungsregel bleibt exakt Bit 8.

**Vorab unterscheidbar:** Verschwindet die Umkehr bei einer der Vergrößerungen, ist eine einfache Geometrie-unabhängige Interpretation widerlegt. Bleibt sie erhalten, sind diese beiden konkreten Klein-Box-Artefakte nicht hinreichend, um den Effekt zu erklären. Das beweist noch keine Eigenmode oder Gruppengeschwindigkeit.

## 2. Durchführung

- **Exakte Boolesche Mikrodynamik:** 24 Besetzungsbits pro physischem D4-Knoten, lokale reversible B4-symmetrische Bit-8-Kollision und Streaming. Unveränderte Numba-Implementierung aus Bit-Test 20.
- **Präparation:** rekonstruierte Bit-19/20-Phasenpräparation mit diskreten 16-Bit-Schwellen. Je Seed werden vier *reelle* Trajektorien (Cosinus/Sinus × positive/negative Störung) kombiniert; komplexe Größen treten erst in der **Auswertung** auf.
- **Zentrum:** x-Länge 48, Gaußapertur \(\sigma_x=48/6{,}4\), Amplitude 0,105. Für die Hauptversuche: \(k_x=\pm2\pi(3)/48=\pm\pi/8\). Verborgene Wellenzahlen \(q=\pi/8,\pi/4,3\pi/8\) als ganzzahlige Moden \((1,2,3)\) bei W=16 bzw. \((2,4,6)\) bei W=32.
- **Beobachtungsfenster:** 12 Vorwärtsticks, Schwerpunktfit Tick 2–8; 8 unabhängige Seeds je \((q,k_x)\) in den beiden einzeln vergrößerten Geometrien, 4 je Richtung bei gemeinsamer Vergrößerung, zusätzlich 8 je sichtbarem Träger für den \(k\)-Scan. Jede Konfiguration besitzt eine kollisionsfreie Referenz mit denselben Schwellen und Präparationen.
- **Datenumfang:** insgesamt 120 Seed-Konfigurationen × vier Realkurven = **480 kollidierende reelle Trajektorien**, plus 480 kollisionsfreie Referenztrajektorien. Der volle 24-Bit-Zustand ist bei jeder Berechnung mikroskopisch; gespeichert wurden per-Seed diagnostische Antwortkanäle, nicht alle vollständigen Mikrozustände jedes Ticks.
- **Definition der gemessenen Bewegung:** Schwerpunktfit der Intensität \(|\langle\delta\rho_q(x,t)\rangle_{\rm seed}|^2\) im x-Raum. Dies ist **keine** aus Eigenfrequenzen abgeleitete Gruppengeschwindigkeit. Zusätzlich werden die Schwerpunkte pro Seed gemessen; Bootstrap-Intervalle betreffen den Seed-Mittelwert. Ein unabhängiger Spiegeltest ist keine Schätzung statistischer Drift.

| Konfiguration | Form \((x,y,z,w)\) | D4-Knoten | Seeds je \((q,k)\) | physikalische \(q\) |
|---|---|---:|---:|---|
| Referenz Bit 20 | (48,4,4,16) | 6.144 | 8 | alle drei |
| W verdoppelt | (48,4,4,32) | 12.288 | 8 | alle drei |
| y und z verdoppelt | (48,8,8,16) | 24.576 | 8 | alle drei |
| beides | (48,8,8,32) | 49.152 | 4 | \(\pi/4\) |

## 3. Ergebnisse: Frühbewegung der gegenläufigen Pakete

Alle Werte in **Gittereinheiten pro Tick**, Fit Tick 2–8. Ein positives Vorzeichen ist eine Verschiebung des kohärenten Antwortschwerpunkts nach +x.

| Geometrie | \(q\) | +\(k_x\) | −\(k_x\) |
|---|---|---:|---:|
| Bit 20, W16 y=z4 | \(\pi/8\) | +0,631 | −0,638 |
| | \(\pi/4\) | −0,369 | +0,399 |
| | \(3\pi/8\) | +0,697 | −0,666 |
| W32, y=z4 | \(\pi/8\) | **+0,611** | **−0,620** |
| | \(\pi/4\) | **−0,395** | **+0,409** |
| | \(3\pi/8\) | **+0,642** | **−0,696** |
| W16, y=z8 | \(\pi/8\) | **+0,634** | **−0,628** |
| | \(\pi/4\) | **−0,381** | **+0,390** |
| | \(3\pi/8\) | **+0,644** | **−0,666** |
| W32, y=z8 | \(\pi/4\) | **−0,372** | **+0,388** |

Die Vorzeichenfolge **+ / − / +** bei +\(k_x\) bleibt in beiden unabhängigen 8-Seed-Geometriekontrollen erhalten. Bei \(q=\pi/4\) bleibt die Umkehr sogar unter *gleichzeitiger* Verdoppelung beider Geometriemerkmale bestehen (4 Seeds pro Richtung).

### Statistische Kontrolle bei q = π/4

Die **Mittelwerte der Einzel-Seed-Schwerpunktslopes**, mit Bootstrap-95-%-Intervallen, sind:

| Geometrie | +\(k_x\), pro Seed | −\(k_x\), pro Seed | mittlere ungerade Komponente \((v_+-v_-)/2\) |
|---|---|---|---|
| W32 y=z4 | −0,384 [−0,424; −0,347] | +0,384 [+0,336; +0,430] | −0,384 [−0,414; −0,354] |
| W16 y=z8 | −0,379 [−0,409; −0,348] | +0,385 [+0,360; +0,409] | −0,382 [−0,402; −0,362] |
| W32 y=z8 | −0,368 [−0,403; −0,343] | +0,386 [+0,361; +0,409] | −0,377 [−0,399; −0,358] |

Die geraden Komponenten \((v_++v_-)/2\) sind bei \(q=\pi/4\) mit null verträglich. Dies entspricht der mikroskopischen Spiegelungssymmetrie. Es handelt sich um beschreibende Bootstrap-Intervalle aus den vorliegenden Seeds, nicht um einen Nachweis eines asymptotischen Naturgesetzes.

### Kollisionsfreie Kontrollserie

Bei \(q=\pi/4\) bewegt sich der +\(k_x\)-Träger kollisionsfrei in den neuen Geometrien jeweils **nach +x**, mit frühen Schwerpunktslopes +0,364 (W32), +0,369 (y=z8), +0,365 (beides). Der kollidierende Träger hat jeweils **negatives** Vorzeichen. Somit lässt sich die Umkehr nicht allein auf die Winkel des Startprofils, die Fourierfilterung oder freie Richtungsbewegung zurückführen.

### Sichtbares w=0

Bei \(q=\pi/4\) zeigt auch die einzelne 3D-Sichtschicht \(w=0\) für +\(k_x\) in allen drei größeren Kontrollen negative Schwerpunktbewegung (−0,356; −0,305; −0,364). Der genaue Betrag ist von der gewählten Projektion abhängig. Das ist **keine** Rückkopplung aus 3D nach 4D, lediglich die Beobachtung der höherdimensionalen Dynamik.

## 4. Zusätzlicher Gegenversuch: sichtbare k-Abhängigkeit

Bei W32, y=z4 und festem \(q=\pi/4\) mit acht Seeds je positivem Träger:

| physikalisches \(k_x\) | \(k\)-Index | Bit-8-Schwerpunktslope | frei | kohärente Amplitude bei Tick 12 / Anfang |
|---|---:|---:|---:|---:|
| \(\pi/12\) | 2 | −0,272 | +0,114 | 0,360 |
| \(\pi/8\) | 3 | −0,395 | +0,364 | 0,197 |
| \(\pi/6\) | 4 | +0,183 | +0,544 | 0,064 |

**Wesentliche Einschränkung:** Der Vorzeichenwechsel bei \(q=\pi/4\) ist **nicht universell für alle sichtbaren \(k_x\)**. Das Signal bei \(k_x=\pi/6\) ist bei Tick 12 besonders schwach; sein frühes Richtungszeichen (Fit 2–8) ist in der Seed-Stichprobe dennoch konsistent positiv. Dies ist mit Dispersion, Modenüberlagerung oder amplitudenabhängiger Antwort vereinbar; keine der Erklärungen ist hier isoliert nachgewiesen.

Ein weiterer Warnhinweis: Aus den *frühen demodulierten Phasenfrequenzen* an den Trägern \(k=2,4\) ergäbe ein naiver Frequenz-Differenzenquotient in der Bit-8-Antwort etwa −0,778. Aber auch beim **freien Streaming** ergibt der gleiche Quotient etwa −1,000, obwohl sich dessen Paket nach +x bewegt. Folglich darf dieser Quotient **nicht als Gruppengeschwindigkeit einer isolierten Eigenmode** interpretiert werden. Die aktuelle Phasen-/Paketprojektion vermischt offensichtlich mehr Struktur, als eine einzige Frequenzkurve erklären würde.

## 5. Korrektheit und Reversibilität

`data/bit21_validation.json` enthält:

- 310 zusätzlich gegen Bit-8/Bit-10 abgeglichene lokale Zustände; lokale Involution und alle fünf additiven Erhaltungsgrößen bestanden.
- Eine *phasenkodierte* \(q=\pi/4\)-Präparation mit vollständiger x-Spiegelung auf allen drei neuen Geometrien. Spiegelungsgleichheit **an jedem** der vier Vorwärtsticks, vier exakte Rückwärtsticks und fünf Erhaltungsgrößen jeweils bestanden.
- Die Bit-20-Kernprüfung mit weiteren 1.430 Testzuständen und die früheren algebraischen Symmetrieargumente bleiben bestehen. Die endlichen Stichproben allein sind keine vollständige Enumeration aller \(2^{24}\) lokalen Zustände.

## 6. Schlussfolgerung und nächste Falsifikation

**Geometrische Nullhypothesen teilweise widerlegt:** Der q-abhängige Richtungswechsel ist weder ein bloßer Effekt von W=16 noch allein durch y=z=4 bedingt. Er überlebt beide Änderungen getrennt (acht Seeds) und zusammen (vier Seeds). **Keine stabile räumliche Bindung**, keine Massengleichung und keine einzelne kollektive Eigenmode wurden damit nachgewiesen. Ein Verhalten der beobachteten Paketantwort auf einem D4-Gitter ist nicht bereits eine Erklärung des dimensionalen Übergangs D3→D4→D8.

**Nächster physikalischer Test:** Einen phasen- und stromaufgelösten dynamischen Strukturfaktor \(S(k_x,q,\omega)\) messen, zunächst auf dem robusten W32/y=z8-Gitter. Dazu mehrere schmale k-Präparationen und die korrelierte Dichte-/Richtungsstromantwort verwenden; getrennte spektrale Zweige auf Trainings- und unabhängigen Seeds identifizieren. Erst eine aus *derselben* Modenbranche gewonnene \(\partial\omega/\partial k_x\) kann gegen die Paket-Schwerpunktbewegung getestet werden. Bei starker Modenmischung ist eine einzelne Gruppengeschwindigkeit nicht sinnvoll.

## 7. Reproduktion

Alle neuen Skripte befinden sich in `src/phase_bit_s4_open40_*.py`. Zusätzlich werden die bereits auf `main` vorliegenden Bit-8- und Bit-20-Module benötigt. Beispiel:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python src/phase_bit_s4_open40_falsify_q.py --shape 48 4 4 32 --q 2 --modes 3 -3 --ticks 12 --seeds 8 --symmetry --out data/bit21_w32_q2.json --raw data/bit21_w32_q2.npz
python src/phase_bit_s4_open40_falsify_q.py --shape 48 8 8 16 --q 2 --modes 3 -3 --ticks 12 --seeds 8 --out data/bit21_yz8_q2.json --raw data/bit21_yz8_q2.npz
python src/phase_bit_s4_open40_validate.py --out data/bit21_validation.json
python src/phase_bit_s4_open40_analyze.py --dir data --bit20 data/bit20_results.json --out data/bit21_summary.json --figure figures/bit21_geometry.svg --kfigure figures/bit21_k_dependence.svg
```

**Datenkonvention:** `data/bit21_*_q*.json` enthält Einzel-Seed-Schwerpunktslopes, Bootstrap-Intervalle und weitere Metriken. Die entsprechenden `npz`-Dateien enthalten je Seed/Träger/Tick die drei komplexen *Analysekanäle*, jeweils rückführbar auf vier echte Boolesche Trajektorien. Das vollständige ZIP sichert zusätzlich die unveränderten Bit-20-Vorgänger. Der Bericht und die zusammengefassten Metadaten sind auf `main` vorgesehen; die binären Rohantworten bleiben im ZIP bzw. in der Dateiablage.
