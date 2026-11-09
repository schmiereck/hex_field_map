# Bit-Test 13 – Cache-Strategien bei strukturierten 4D-Anfangszuständen

Stand: 9. November 2026. Vergleich der unveränderten Bit-8-Kollision mit dem vektorisierten Bit-10-Kern und dem begrenzten Engine-Cache aus Bit 12.

## Ziel und Grenzen

Verglichen werden (1) kein **Engine**-Cache, (2) ein 256-Einträge-LRU und (3) ein statischer Hotset-Cache der bis zu 256 häufigsten Wörter des Anfangszustands. Cache-Misses rechnen immer mit `collide_fast` nach. Die getrennten internen Funktions-Caches von Bit 10 bleiben bei allen Varianten aktiv. Der Hotset-Cache wird **nur auf Basis des Anfangszustands** befüllt und nicht nachträglich angepasst.

Die strukturierte Anfangsbelegung ist ausdrücklich **kein vollständiger Nachbau von Bit-Test 6**: Die vorherige vierfache Cosinus/Sinus- und Vorzeichenpräparation und die verschiedenen Detektorkanäle fehlen. Für diese reine Performance-Diagnose werden sechs reproduzierbar erzeugte halbgefüllte Bitwörter (jeweils zwölf gesetzte Bits) räumlich anhand einer Gaußhülle in x und eines Cosinusträgers in x/w verteilt; y/z sind vorhanden. Verglichen wird eine uniforme Zufallspräparation mit genau zwölf gesetzten Bits pro Knoten.

## Messergebnisse

Ein vollständiger Tick besteht aus einer lokalen Kollision und anschließendem Richtungs-Streaming. Die Dauer enthält acht bzw. sechs Vorwärtsticks, nicht die Rückläufe; die angegebenen Cache-Zähler enthalten Vorwärts- und Rückwärtsphase. Die drei Strategien laufen seriell und teilen sich Prozessallocator und importierte Tabellen. Einzelmessungen sind daher **nicht** ausreichend, um kleine Laufzeitunterschiede als statistisch belastbar einzustufen.

| Anfangszustand | Knoten | Ticks vor/zurück | Ohne Engine-Cache | LRU 256 | Hotset 256 | LRU-Treffer | Hotset-Treffer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strukturiertes Paket-Surrogat | 512 | 6 / 6 | 0,249 s | 0,277 s | 0,251 s | 0 | 8 |
| Uniforme 12-Bit-Zufallsbelegung | 512 | 6 / 6 | 0,358 s | 0,290 s | 0,276 s | 0 | 351 |
| Strukturiertes Paket-Surrogat | 2.048 | 8 / 8 | 1,198 s | 1,253 s | 1,232 s | 0 | 12 |
| Uniforme 12-Bit-Zufallsbelegung | 2.048 | 8 / 8 | 1,676 s | 1,616 s | 1,647 s | 0 | 352 |

**Korrektheit:** In allen vier Versuchen ergaben alle drei Varianten exakt denselben Endzustand. Alle durchgeführten Rückläufe landeten bitgenau beim Anfangszustand. Teilchenzahl und sämtliche vier Impulskomponenten blieben erhalten; das Padding blieb Null. Es gibt keine Änderung an der Mikrodynamik.

**Wichtig:** Nur sechs unterschiedliche lokale Wörter im strukturierten *Anfangszustand* bedeuten nicht, dass es später nur sechs Zustände gibt. Streaming erzeugt schnell viele neue Kombinationen. Das erklärt die niedrige Hotset-Trefferzahl. Die 256-Einträge-LRU hat in allen vier Versuchen **null Treffer**.

## Interpretation

Die Daten liefern **keine belastbare Beschleunigung durch einen Engine-LRU**. Ein statischer Hotset kann einzelne Wiederholungen abfangen, die Auswirkung auf die Laufzeit ist in diesem Umfang nicht robust. Für die nächste Optimierung bietet sich zunächst das **Abschalten des Engine-Caches bei beibehaltener begrenzter interner Funktions-Memoisierung** an. Das reduziert Verwaltungsaufwand und verhindert unkontrolliertes Wachstum. Weitere Geschwindigkeitssteigerungen müssen eher direkt an der Kollision ansetzen.

Nicht getestet: volle Bit-6-Paketpräparation, 65.536-Knoten-Paketlauf, Langzeit-RSS in getrennten Prozessen, langfristige Hotset-Anpassung oder vollständige Verteilung von Einzelfallzeiten. Das Skript kann diese Parameter für zusätzliche kurze Versuche variieren.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open32_packet_cache.py --shape 8 4 4 8 --ticks 6 --out data/bit13_packet512.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open32_packet_cache.py --shape 16 4 4 16 --ticks 8 --kind random12 --out data/bit13_random2048.json
```