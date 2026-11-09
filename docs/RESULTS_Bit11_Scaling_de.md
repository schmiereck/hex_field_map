# Bit-Test 11 — Skalierung des gepackten symmetrischen D4-Kerns

Stand 9. November 2026. Fortsetzung von Bit 10 (`phase_bit_s4_open29_engine.py`). **Unveränderte** lokale Bit-8/10-Kollisionsregel; keine neue Physik und kein lokaler Pakettest.

## Aufbau

* Separater Python-Prozess je Gittergröße: `(4,4,4,4)`, `(8,4,4,8)`, `(16,4,4,16)`, `(32,4,4,32)`, `(64,4,4,64)` und `(256,4,4,32)`.
* D4-Knoten sind Positionen mit gerader Koordinatensumme; deshalb die Hälfte des 4D-Boxvolumens. Pro Knoten genau 24 variable Richtungsbits in `uint32`.
* Reproduzierbare Zufallsbelegungen mit **genau zwölf Bits je Anfangsknoten** (Seed 3011), periodische Grenzen. 2 Vorwärtsticks je Größe; beim Zielgitter drei. Danach exakt so viele Rückwärtsticks.
* Jeder Vorwärtstick umfasst tatsächliche symmetrische Kollision und vektorisiertes Streaming. Alle fünf Erhaltungsgrößen werden nach jedem Tick geprüft, ebenso Null-Padding und exakte Rückkehr zum Startzustand.
* `maxrss` ist gemessener maximaler **gesamter Python-Prozessspeicher**, einschließlich NumPy, Tabellen und Python-Runtime. Zeiten sind Einzellaufmessungen, keine kalibrierten Mehrfachmedianwerte.

## Halbfüllung — gemessene Zeiten

| Physische Knoten | Tick 1 (s) | Tick 2 (s) | Tick 3 (s) | Peak RSS (MiB) | Cache nach Vorwärtsticks (Einträge) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0,012631 | 0,014089 | — | 106,9 | 407 |
| 512 | 0,051152 | 0,050790 | — | 107,0 | 1.641 |
| 2.048 | 0,239494 | 0,239405 | — | 109,1 | 6.554 |
| 8.192 | 0,882929 | 0,857071 | — | 117,5 | 26.164 |
| 32.768 | 3,302261 | 3,304455 | — | 145,0 | 103.911 |
| 65.536 | 6,914178 | 7,063474 | 6,837175 | 179,7 | 309.172 |

Bei 65.536 Knoten entstehen allein in den drei Vorwärtsticks **103.356 / 102.924 / 102.892** neue Einträge im Engine-Cache. Die Anfangsbelegung enthält 64.733 unterschiedliche Zustandswörter. Die Rückrechnung über drei Schritte war mit bereits gefülltem Cache in 0,118 s vollständig; ein erneuter Tick über die ursprünglichen Anfangsbitmuster benötigte 0,0236 s. **Diese warmen Zeiten repräsentieren keine fortlaufende neue Trajektorie.**

Die Messwerte zeigen in diesem Bereich grob lineares Zeitwachstum, etwa 0,10–0,12 Millisekunden je physischem Knoten und frischem Tick. Es gibt noch keine Langzeitmessung oder einen Nachweis asymptotisch linearer Skalierung.

## Besetzungsdichte — 8.192 Knoten, zusätzlicher Kontrolllauf

Für die Dichtekontrolle wurden Bits unabhängig mit Wahrscheinlichkeit `density/24` gezogen (nicht exakt festgelegte lokale Besetzungszahl). Identischer Seed, gleicher Kern und dieselbe Gittergröße.

| Erwartete Bits je Knoten | Tick 1 (s) | Tick 2 (s) |
|---:|---:|---:|
| 3 | 0,233746 | 0,171212 |
| 6 | 0,670913 | 0,651968 |
| 12 (exakt belegt, andere Präparation) | 0,882929 | 0,857071 |
| 18 | 0,713476 | 0,639941 |

Die Kostendifferenz hängt wesentlich von der Zahl und Komplexität erstmalig auftretender lokaler 24-Bit-Zustände ab; die Stichproben sind kein kontrollierter physikalischer Gleichgewichtstest.

## Cache-/Speicherproblem

`VectorOrbitD4.cache` ist ein **unbegrenztes Python-Dictionary** mit ggf. zwei Einträgen pro berechnetem Zustand. Die `functools.lru_cache(maxsize=100000)` in `collide_fast` bzw. `preference_fast` begrenzen **nicht** dieses Dictionary. Mit 309.172 Einträgen nach nur drei Schritten ist die Skalierung über sehr viele Ticks nicht gesichert. Die untere Schranke aus Roharraydaten (7,25 MiB auf dem größten Gitter) unterschätzt den tatsächlichen Python-Speicher deutlich.

Für lange Läufe: ausdrücklich begrenzter Engine-Cache (z. B. LRU), dessen Einträge nur Performance und niemals Mikrodynamik beeinflussen dürfen; danach Profiling mit langen, repräsentativen Paketpräparationen. Nicht die Orbit-Ergebnisse approximieren oder abschneiden.

## Physikalischer Durchsatz

Bei ungefähr 6,9 s pro neuem Tick auf dem Zielgitter ergäben 55.296 Gesamtticks unter **unrealistisch konstanten Bedingungen** bereits rund 106 Stunden (4,4 Tage) CPU-Wandzeit. Das ist nur eine lineare Überschlagsrechnung; nach langem Lauf können Cache, Temperatur, Populationen und Speicherverbrauch abweichen. Die ursprüngliche Bit-6-Kampagne umfasste viele einzelne Realisationen, die hier **nicht** wiederholt wurden.

**Fazit:** Kurze 4D-Läufe bis zur ursprünglichen Zielgröße sind mit verifiziertem Roundtrip und moderatem Prozess-RSS möglich. Für die lokale Paketsimulation ist vor allem die Kollision, die unbeschränkte Cache-Größe und die lange Gesamtrechenzeit noch nicht gelöst. Nächste technische Prüfung: Kapazitätsbegrenzung mit exaktem Vergleich gegen unbeschränkte Engine über längere Trajektorien.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open30_scaling.py --shape 256 4 4 32 --ticks 3 --out data/size65536.json
```

Alle Messdaten in `data/size*.json` und `data/density*_8192.json`. Die Skripte `src/phase_bit_s4_open29_*.py`, `src/phase_bit_s4_open28_fast4d.py` und ihre Vorgänger sind als Eingaben für einen eigenständigen Archivlauf enthalten. Die historischen Projektdaten und die vom Nutzer hochgeladene Feldgrafik bleiben auf GitHub unverändert.