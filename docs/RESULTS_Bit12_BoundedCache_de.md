# Bit-Test 12 — Begrenzter Kollisions-Cache für lange 4D-Läufe

Stand: 9. November 2026. Ausgangspunkt: Bit 10/11 (`VectorOrbitD4`). **Die mikroskopische Bit-8-Kollision bleibt unverändert.** Dies ist eine technische Speicher-/Korrektheitsstudie, keine Paketmessung.

## Implementierung

`BoundedOrbitD4` benutzt ein `OrderedDict` als LRU-Cache mit konfigurierbarer Maximalzahl lokaler 24-Bit-Eingabezustände. Ein Eintrag `s -> C(s)` kostet einen Cache-Slot; die Umkehrbelegung kann einen eigenen Slot verbrauchen. Beim Cache-Miss wird **immer die vollständige unveränderte `collide_fast(s)`** ausgeführt. Es gibt keine Approximation und kein abgeschnittenes Symmetrieorbit. `--capacity 0` deaktiviert diesen Engine-Cache. Der alte unbegrenzte Dictionary-Cache wird nicht mehr verwendet.

Zusätzlich enthält Bit 10 zwei **separate** `functools.lru_cache(maxsize=100000)` für `collide_fast` und `preference_fast`. Diese bleiben begrenzt und können unabhängig Einträge verdrängen. Die Speichergrenze bezieht sich auf diese drei Caches — nicht auf den gesamten Python-Prozess, die temporären NumPy-Arrays oder mögliche externe Verbraucher.

## Messungen

Reproduzierbare Anfangszustände, jeweils 12 von 24 Bits pro Knoten. Ein vollständiger Tick besteht aus Kollisionsauswertung und Richtungs-Streaming. Die ersten vier Testfälle vergleichen jeden vorwärts/rückwärts berechneten Zustand **bitgenau** mit der ursprünglichen unbeschränkten Bit-10-Engine. Die längeren Fälle prüfen Roundtrip, die fünf additiven Erhaltungsgrößen und Null-Padding; ein zeitgleicher vollständiger Referenzvergleich wurde dort aus Ressourcengründen nicht durchgeführt.

| Test | Knoten | LRU-Limit | Vorwärts + Rückwärts | Unveränderte Referenz | Engine-Cache am Ende | Peak RSS (MiB) |
|---|---:|---:|---:|---|---:|---:|
| kleiner Grenzwert | 512 | 64 | 12 + 12 | bitgenau | 64 | 110.5 |
| mittlere Box | 2.048 | 1.024 | 24 + 24 | bitgenau | 1.024 | 136.6 |
| größere Box | 8.192 | 4.096 | 12 + 12 | bitgenau | 4.096 | 156.8 |
| ohne Engine-Cache | 512 | 0 | 16 + 16 | bitgenau | 0 | 111.6 |
| länger, 512 | 512 | 256 | 160 + 160 | nicht parallel | 256 | 149.4 |
| länger, 8.192 | 8.192 | 4.096 | 16 + 16 | nicht parallel | 4.096 | 156.6 |

*Die Ressourcenmessung `ru_maxrss` enthält in den vier Parallelvergleichsfällen auch den unbegrenzten **Referenz-Cache**: Diese RSS-Werte sind deshalb **keine** isolierten Speichermessungen der neuen Engine. In den zwei Fällen ohne Referenz wird nur die neue Engine betrieben.*

Im Test mit 512 Knoten und 160 Ticks wurden **163.584 Engine-Einträge verdrängt**, während das Dictionary zu keinem Zeitpunkt 256 Einträge überschritt. Die internen Funktions-Caches erreichten jeweils ihre Grenze von 100.000 Einträgen. Der Prozess-Peak lag bei rund 149 MiB. Auf 8.192 Knoten und 16 Vorwärtsticks: 257.952 Verdrängungen und maximal 4.096 Engine-Einträge.

## Wesentliche Einschränkungen

1. **Die Begrenzung ist bewiesen für die Anzahl der Engine-Einträge** (Assertion direkt bei jeder Kollisionsauswertung) und durch `maxsize` für die inneren Funktions-Caches. Sie ist **keine feste Obergrenze der RSS-Nutzung**. Temporäre Arrays, Python-Allocator, NumPy-Tabellen und die Gittergröße beeinflussen den Arbeitsspeicher zusätzlich.
2. Der einfache LRU zeigte in diesen Zufallsläufen **null Engine-Cache-Treffer**: Die Zahl frischer lokaler Zustände je Tick übersteigt den kleinen Cache erheblich. Ein LRU löst also das Speicherproblem, **beschleunigt aber nicht automatisch**. Bei großen Gittern und langem Lauf entsteht zusätzliche Rechenlast durch wiederholte Orbit-Auswertung.
3. Ein auf 8.192 Knoten geplanter 60-Tick-Versuch überschritt die äußere Laufzeitgrenze; auch ein 3-Tick-Versuch mit 65.536 Knoten wurde abgebrochen. **Für das Zielgitter wurde hier deshalb kein vollständiger neuer Langzeittest validiert.** Die vorherigen Bit-11-Messungen auf dieser Größe betrafen den unbegrenzten Cache.
4. Die probabilistische Seedauswahl gehört ausschließlich zur Präparation; Updates und Cache-Verdrängung verändern weder die Bitregel noch die Invertierbarkeit.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open31_longrun.py --shape 16 4 4 16 --ticks 24 --capacity 1024 --out data/bit12_bounded1024_2048.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/phase_bit_s4_open31_longrun.py --shape 8 4 4 8 --ticks 160 --capacity 256 --no-reference --out data/bit12_long160_512.json
```

## Nächster technischer Schritt

Ein noch kleinerer Engine-Cache oder `--capacity 0` kann bei dieser gleichverteilten Präparation günstiger sein als eine größere LRU-Tabelle. Für die Paketkampagne sollten als Nächstes **Cache-Abschaltung und Cache-Limit gegen gemessene Zustandswiederholung** sowie ein echter mehrminütiger Speicherverlauf bei kontrolliertem CPU-Budget verglichen werden. Vorher darf die vorherige Hochrechnung für 65.536 Knoten nicht als gesicherter Durchsatz verwendet werden.