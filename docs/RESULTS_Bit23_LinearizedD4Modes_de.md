# Bit-Test 23 — Statistischer 24×24-Kollisionsoperator und gezielte D4-Eigenmoden

**Datum:** 11. November 2026  
**Repository:** `schmiereck/hex_field_map`, Fortsetzung von Bit 20–22.  
**Status:** Berechnungen und unabhängige Kontrollen bestanden. **Kein Nachweis einer stabilen gebundenen Teilchenstruktur oder eines neuen isolierten Modus jenseits der D4-Bewegung.**

## 1. Forschungsfrage und Ergebnis

Bit 20/21 zeigte für phasenkodierte Pakete mit `k_x=+π/8` bei zunehmendem verborgenem `q=π/8,π/4,3π/8` die frühe Schwerpunkt-Bewegungsfolge **positiv, negativ, positiv**. Bit 22 identifizierte Dichte/Strom-Peaks nahe `ω=±π/8` und `±π/4`, die schon im freien Streaming auftreten. Bit 23 fragt, ob ein aus der **unveränderten lokalen Bit-8-Kollision** abgeleiteter linearer Antwortoperator diese Beobachtungen quantitativ erklären und getrennte Moden vorhersagen kann.

**Ja, in den geprüften frühen Zeitfenstern.** Aus 800.000 unabhängigen lokalen Bit-Zuständen wurde eine 24×24-Matrix gewonnen. Ohne Anpassung an die alten Paketgeschwindigkeiten reproduziert sie alle sechs positiven/negativen Bewegungsfälle der W32-Geometrie mit **maximal 0,0263**, mittlerem absolutem Fehler **0,0169** Gittereinheiten/Tick. Gezielt einzeln präparierte Eigenmoden folgen 10 Ticks in der unveränderten reversiblen Bitdynamik mit hoher Reinheit. Eine explizite Zerlegung der linearen Antwort zeigt, dass die negative Schwerpunktbewegung bei `q=π/4` durch **Interferenz mehrerer gerichteter Moden** zustande kommt: Ohne deren Intensitäts-Kreuzterme wechselt das Vorzeichen zurück nach positiv.

## 2. Exakte Mikrodynamik versus statistische Linearisierung

Ein einzelner Knoten hat 24 Boolesche Besetzungsbits `X_i`. Das lokal deterministische und reversible Bit-8-Kollisionsabbild ist `Y=C(X)`. Es wird **nicht** geändert. Um den unabhängigen halbgefüllten Produktzustand (`P(X_i=1)=1/2`) ist die Ableitung des *mittleren* Ausgangsbits bezüglich einer Eingangsbit-Wahrscheinlichkeit wohldefiniert:

\[
K_{ij}=\left.\frac{\partial\mathbb E[Y_i]}{\partial p_j}\right|_{p=1/2}
=\mathbb E[(2Y_i-1)(2X_j-1)].
\]

Die zweite Gleichheit gilt bei unabhängigen uniformen Eingangsbits exakt. Die **Stichprobenwerte** von `K` sind natürlich nur numerisch geschätzt; weder die Bits noch die Kollisionsregel führen Gleitkommazahlen als mikroskopische Zustände ein.

Das Kollisionsabbild ist eine Involution und bewahrt das uniforme Maß, daher ist der exakte Erwartungsoperator symmetrisch. Die B4-Gruppe besitzt 384 signierte Achsenpermutationen; durch Gruppenmittelung wird die entsprechende Äquivarianz exakt erzwungen. Die fünf erhaltenen lokalen Ladungen (Zahl und 4 Impulse) liefern den Projektor `P` auf die zugehörigen Richtungsvektoren. In der finalen Auswertung gilt `K_proj=P+(I-P) K_sym (I-P)`. Das erzwingt ausschließlich **mathematisch bekannte** lineare Identitäten, keine auf Messdaten gefittete zusätzliche Dispersionsregel.

Für `k=(k_x,0,0,q)` ergibt sich die testbare **Einpunkt-/Produktzustands-Näherung**

\[
\delta \widehat f_{t+1}=M(k_x,q)\delta\widehat f_t,
\quad M(k_x,q)=\operatorname{diag}\big(e^{-i(k_x v_{i,x}+qv_{i,w})}\big)\,K_\mathrm{proj}.
\]

**Grenze:** Nach mehreren exakten Ticks können räumliche Korrelationen entstehen. Deshalb ist diese 24×24-Dynamik kein exakter reduzierter Liouville-Operator für beliebige endliche Zeiten. Das Experiment prüft ihre Vorhersagekraft gesondert an der exakten Dynamik.

## 3. Monte-Carlo-Schätzung und Kontrollsymmetrien

- Acht unabhängige Zufallsstichproben mit je 100.000 uniformen 24-Bit-Zuständen, insgesamt **800.000** Zustände.
- Bei rund **58,9 %** der zufälligen lokalen Zustände verändert die Kollision den Bitzustand.
- Der 24×24-Operator besitzt fünf exakte Ladungseigenrichtungen mit Eigenwert 1. Die übrigen 19 Eigenwerte liegen ungefähr im Bereich **0,289 bis 0,996**, einschließlich Entartungen.
- Die maximale Standardunsicherheit eines Matrixeintrags in der acht-Seed-Schätzung beträgt rund **0,000371**. Die unkorrigierte Ladungsabweichung liegt bei rund `8,24×10⁻⁴`; nach struktureller Projektion unter `6×10⁻¹⁶`.
- Unabhängiger Regelvergleich: **1.826** lokale Testfälle mit der früheren Bit-10-Implementierung stimmen exakt überein; lokale Rückwärtsregel und fünf Erhaltungsgrößen ebenfalls. Zusätzlich 100 B4-Generator-Transformationstests.

## 4. Zwei getrennte spektrale Moden bei q=π/4

Bei `k_x=π/8`, `q=π/4`:

| Mode | Eigenwert-Betrag | Zeitphase/Tick | dominierender Strom |
|---|---:|---:|---|
| X-polarisiert | 0,97331 | −0,3915 | `J_x` |
| W-polarisiert | 0,97359 | −0,7845 | `J_w` |

Das stimmt mit den dominanten frequenzpolarisierten Anteilen aus Bit 22 überein. Bei `q=π/8` gibt es eine stark gemischte Dichteantwort bei Phase ca. −0,3914; bei `q=3π/8` trennen sich X- und W-Zweige wieder.

## 5. Falsifikation durch gezielte Eigenvektor-Anregung in der exakten Bitdynamik

Vier unabhängige Modi (`q2_x`, `q2_w`, `q1_mixed`, `q3_x`) wurden **nicht** durch kontinuierliche Zustände simuliert, sondern mit gepaarten 16-Bit-Schwellen in jeweils zwei komplementären Bit-Präparationen realisiert: `p_i(x)=1/2 ± 0,075 Re(u_i exp(i(k_x x+q w)))`. Je Mode 8 Seeds, **10 Vorwärtsticks**, vollständige Bit-8-Kollision und Streaming, fünf Ladungen erhalten, erster Seed jeder Mode exakt rückwärts überprüft. Die komplexen Fourierwerte sind nur externe Diagnostik.

| Präparation | Fehler des Ein-Tick-Vektors | Amplitude nach 10 Ticks, exakt | 24×24-Vorhersage | außer Modenrichtung verbleibender Energieanteil, Tick 10 |
|---|---:|---:|---:|---:|
| q=π/4, X | 1,70 % | 75,60 % | 76,30 % | 0,475 % |
| q=π/4, W | 1,60 % | 76,22 % | 76,52 % | 0,363 % |
| q=π/8, gemischt | 1,66 % | 83,84 % | 84,12 % | 0,248 % |
| q=3π/8, X | 1,58 % | 76,26 % | 76,12 % | 0,959 % |

Diese Reinheit betrifft die **ensemblegemittelte Fourierantwort**, nicht jeden einzelnen verrauschten Mikrozustand. Sie stellt keine langzeitstabile Teilchenbindung dar.

## 6. Unabhängige Geschwindigkeitsvorhersage für Bit 21 (kein Fit)

Aus derselben 24×24-Matrix wurde die original rekonstruierte, räumlich gaußförmige Akustikreferenz-Präparation aus Bit 20 in der `shape=(48,4,4,32)`-Geometrie linear fortentwickelt. Der Anfangswert verwendet **keine** aus Bit 21 angepassten Modenkoeffizienten oder Geschwindigkeiten. Schwerpunktmethode und Fitfenster entsprechen Bit 21 (Tick 2–8).

| verborgenes q | k_x | exakte Bit-21-Geschwindigkeit | Bit-23-Vorhersage |
|---|---|---:|---:|
| π/8 | +π/8 | +0,611 | +0,629 |
| π/8 | −π/8 | −0,620 | −0,631 |
| π/4 | +π/8 | **−0,395** | **−0,397** |
| π/4 | −π/8 | +0,409 | +0,389 |
| 3π/8 | +π/8 | +0,642 | +0,667 |
| 3π/8 | −π/8 | −0,696 | −0,669 |

Maximale absolute Abweichung: **0,0263**; mittlere: **0,0169**; RMS **0,0189** Gittereinheiten/Tick. Die unabhängigen acht Teilstichproben der lokalen Matrix ergeben für `q=π/4, +k_x` einen Vorhersage-Standardfehler von etwa **0,0017**. Das ist **nicht** das gesamte physikalische Unsicherheitsbudget: Es kommt die Seed-Streuung der exakten Paketmessung und Modellfehler durch nichtlineare Korrelationen hinzu.

## 7. Kontrollierte Aufspaltung: Was kehrt sich wirklich um?

Für `q=π/4, k_x=+π/8` wurde die vollständige lineare Paketlösung in jeder räumlichen Fourierkomponente nach Strompolarisation in disjunkte X-, W- und Rest-Eigenmodenmengen aufgeteilt. Die Summe rekonstruiert die unzerlegte lineare Rechnung auf `2,2×10⁻¹³` genau (Schwerpunktabweichung).

| Messgröße, Fitfenster Tick 2–8 | Bewegung (Gitter/Tick) |
|---|---:|
| X-dominierte Moden allein | +0,971 |
| W-dominierte Moden allein | −0,013 |
| Restmoden allein | −0,307 (nur 3,5 % der gruppierten Intensität) |
| **Alle Moden kohärent addiert** | **−0,397** |
| **Dieselben Moden ohne Intensitäts-Kreuzterme** | **+0,189** |

Die X-dominierte Gruppe trägt rund 33,1 % der gruppierten Intensität, W 63,4 %, Rest 3,5 % (Tick 2–8). Dass der Schwerpunkt nur im **kohärenten** Signal negativ ist, zeigt einen kausalen Interferenzeffekt **innerhalb der linearen Beschreibung**. Es handelt sich um eine Umkehr des **Schwerpunkts der gemessenen Dichteantwort**, nicht um den Beweis, dass eine einzelne Energietransport- oder Gruppengeschwindigkeit negativ ist. Die Modengruppen werden nach ihrer Strompolarisation klassifiziert und sind keine neu entdeckten Erhaltungssektoren.

## 8. Bedeutung für 4D/3D und offene Hypothesen

Die Resultate liefern eine konkrete, testbare Wirkung des verborgenen `w`-Wellenzahlkanals auf eine in 3D beobachtbare Dichteantwort. **Nicht** gezeigt wurden ein physikalischer Dimensionsübergang von D4 zu D3, die Bedeutung von D8, Kugeloberflächen als elementare Objekte, stabile massetragende Pakete oder eine emergente relativistische Dispersionsgleichung.

Aus dem Erfolg der frühen linearen Beschreibung folgt auch keine exakte langsame Feldreduktion auf nur 24 Richtungsamplituden: Bei langen Zeiten, höherer Anregung und inhomogenem Hintergrund können Korrelationen und Modenwechselwirkungen entscheidend werden.

**Nächster sinnvoller Test (Bit 24):** Gezielte Kollision zweier *einzeln präparierter* Modenpakete und Messung der ausgehenden X/W-Kanäle, mit systematischer Variation von Amplitude, räumlicher Separation und verborgener Phasendifferenz. Besonders wichtig ist die Frage, ab welcher Amplitude die 24×24-Einpunkt-Näherung merklich versagt und ob eine kleine Zahl nichtlinearer Modenkopplungen genügt.

## 9. Reproduktion, Dateien und Backup

Die numerische Kollision findet in `phase_bit_s4_open39_numba_collision.py` statt, unverändert gegenüber Bit 20 und nach Bit-10-Referenz geprüft. Alle neuen Quelldateien beginnen mit `src/phase_bit_s4_open42_`. Der unabhängige Bit-21-Referenzvergleich verwendet `data/bit21_w32_q{1,2,3}.json` aus dem historischen ZIP; die drei Referenzen sind im aktuellen Archiv ebenfalls enthalten. Das vollständige Bit-22-Backup und die originale historische Grafik wurden zusätzlich im Bit-23-ZIP konserviert.

Reproduktionsreihenfolge (innerhalb des entpackten Archivs, aus dem Projektverzeichnis):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_linear_collision.py --out data/bit23_collision_operator.json --npz data/bit23_collision_operator.npz
for mode in q2_x q2_w q1_mixed q3_x; do
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_eigenprobe.py \
    --seeds 8 --ticks 10 --selected "$mode" \
    --out "data/bit23_eigenprobe_${mode}.json" --npz "data/bit23_eigenprobe_${mode}.npz"
done
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_predict_packets.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_spectral_decomp.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_mode_interference.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_validate.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open42_plots.py
```

Der Standardmodus des Eigenprobe-Skripts ist eine zusammengefasste `bit23_eigenprobe.npz`/JSON. Für ressourcenschonende Einzelläufe wurden die vier Fälle einzeln gespeichert (`data/bit23_eigenprobe_q2_x.*` usw.). Reproduktion setzt NumPy, SciPy, Numba und für die Abbildungen Matplotlib voraus. Es wurde kein realzeitfähiger 3D/4D-Teilchen-Solver entwickelt.
