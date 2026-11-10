# Bit-Test 17 — Exakte lokale Ereignisfortschreibung auf dem D4-Gitter

Stand: 10. November 2026. Folge von Bit-Test 16 (Korrekturfronten). **Unveränderte mikroskopische Bit-8-Kollision** und dieselbe gerichtete D4-Streaming-Permutation wie in Bit-Test 9/10. Quellcode: `src/phase_bit_s4_open36_event_solver.py`.

## Forschungsfrage

Kann die Entwicklung einer lokalisierten Störung um einen bekannten Hintergrund vollständig aus *lokalen* Kollisionsereignissen berechnet werden, ohne an jedem Knoten die gestörte Welt neu zu kollidieren?

## Exakte Konstruktion

Mit ungestörtem Hintergrund `B_t`, gestörtem Zustand `P_t`, Differenz `D_t = P_t XOR B_t`, lokaler Kollisionsinvolution `C` und Richtungs-Streaming `S` gilt

`D_(t+1) = S( C(B_t XOR D_t) XOR C(B_t) )`.

Wegen der streng lokalen Kollision ist die Differenz **außerhalb des Trägers von `D_t` genau null**. Deshalb wird die Kollisionsdifferenz ausschließlich für Knoten mit `D_t[x] != 0` ausgewertet. Das resultierende 24-Bit-Wort wird bitweise auf die 24 gerichteten Zielknoten gestreamt. Überlappende Beiträge werden per XOR kombiniert. Es wird weder ein Radius geschnitten noch probabilistisch approximiert.

Im räumlich homogenen Spezialfall `B_t[x] = b_t` genügt ein einziges 24-Bit-Hintergrundwort: `b_(t+1) = C(b_t)`, weil die gerichtete Translation einer überall gleichen Belegung diese nicht verändert. Das Hintergrundwort darf bei der Kollision wechseln; **es ist kein notwendigerweise statischer Hintergrund**. Wir verwenden ein bei Bit 8 tatsächlich kollisionsaktives Wort `0x8cd1ae`.

Bei räumlich inhomogenem Hintergrund wird dieser separat als vollständiges Feld fortgeschrieben. Die sparse Technik spart nur die **zusätzliche** gestörte Vollsimulation, nicht die Hintergrundkosten.

## Verifikation

Alle Tests vergleichen nach **jedem** Tick den aus dem Hintergrund und dem Sparse-Dictionary rekonstruierten Zustand bitgenau mit der vollständig ausgerechneten gestörten Simulation. Invarianten (Teilchenzahl + vier Impulskomponenten) bleiben erhalten; Referenzhintergrund und Referenzstörung werden nach allen Vorwärtsschritten vollständig rückwärts zum Anfang entwickelt. Jede lokale Kollision stammt von `collide_fast`, der Bit-8-Regel von Bit-Test 10.

| Fall | Knoten | Ticks | Sparse-Kollisionsknoten im letzten Tick | Nach letztem Tick betroffene Knoten | Nach letztem Tick neu erzeugte Kollisionsquellen | Referenzvergleich / Roundtrip |
|---|---:|---:|---:|---:|---:|---|
| Leerer Hintergrund | 512 | 8 | 24 | 1 | 0 | bestanden |
| Homogen, kollisionsaktiv | 512 | 8 | 282 | 321 | 77 | bestanden |
| Homogen, kollisionsaktiv | 2.048 | 9 | 640 | 835 | 164 | bestanden |
| Homogen, kollisionsaktiv | 8.192 | 8 | 476 | 643 | 112 | bestanden |
| Homogen, kollisionsaktiv | 65.536 | 6 | 239 | 325 | 46 | bestanden |
| Zufällige Halbfüllung | 512 | 6 | 112 | 190 | 50 | bestanden |
| Zufällige Halbfüllung | 2.048 | 6 | 108 | 213 | 47 | bestanden |

Die Tabelle zählt **alle aktuell veränderten Eingangsknoten**, nicht nur die Quellen `Q_t`. Dieser Unterschied ist entscheidend: selbst wenn `Q_t` relativ klein bleibt, muss die Interaktion der ganzen aktuellen Störung mit dem Hintergrund ausgewertet werden.

## Laufzeit und Messgrenzen

Im Test auf 65.536 Knoten (homogener Hintergrund, sechs Ticks) summierten sich die gemessenen Zeiten des Sparse-Schritts auf etwa **0,00957 s**, die der Vollreferenz für Hintergrund **und** Störung auf **0,09714 s**. Das ist ein realer Unterschied **in genau diesem Versuch**, aber **kein allgemeiner Geschwindigkeitsfaktor**: Beide Varianten teilen sich NumPy- und Funktions-Caches; die Referenz profitiert davon, dass einige Eingaben bereits vorab im Sparse-Lauf ausgewertet wurden. Die Fälle repräsentieren nur kurze, stark strukturierte Präparationen. Hintergrund-, Diagnose-, Initialisierungs- und Rückwärtszeiten sind aus den Einzel-Tick-Zeiten unterschiedlich ausgeklammert. Keine belastbare Aussage über Langzeitdurchsatz und realistische Bit-6-Paketpräparation.

Auf räumlich zufälligem Hintergrund sind die Hintergrund-Vollticks in dieser Python-Implementierung weiterhin der dominante Aufwand. Die aktiv wachsenden Sparse-Dictionaries benötigen zudem Metadaten und können bei ausgedehnten Störungen ungünstiger sein als `uint32`-Arrays.

## Erkenntnis

**Ja:** Das exakte nichtlineare Störungsproblem ist auf lokalen aktiven Gebieten geschlossen, **sofern ein exakt bekannter Hintergrund bereitsteht**. Man kann die differenzielle Evolution ohne vollständige gestörte Feldkollisionen berechnen. Für räumlich homogene Hintergründe ergibt das einen echten autonomen Sparse-Solver (Hintergrundwort + Sparse-Störung).

**Nein, bisher nicht bewiesen:** dass die aktiven Gebiete bei vielen Kollisionen klein bleiben, dass die Anzahl geometrischer Schalenobjekte gering bleibt oder dass der Solver für inhomogene, langzeitstabile Wellenpakete schneller ist. Die Ereignisse wachsen, und der Kurztest misst keinen asymptotischen Gewinn.

## Nächste Prüfungen

1. Aktive Gebiete langfristig (40–100 Ticks) auf größerer 4D-Box verfolgen, Wachstum und Kreuzungspunkt zum dichten Feld messen.
2. Ereignis-Zustände als gerichtete Bitlisten, Front-Geometrie oder adaptive kompakte Bereiche kodieren und den **gesamten** Speicherbedarf vergleichen.
3. Für nichtuniforme, aber periodische/kompressible Hintergrundmuster eine exakte symbolische Hintergrunddynamik entwickeln.
4. Erst danach ein glaubwürdiges D3-Projektions-/D4-/D8-Oberflächenmodell ableiten; hierfür gibt es in Bit 17 keinen direkten Beleg.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open36_event_solver.py --kind uniform_half --shape 256 4 4 32 --ticks 6 --out data/bit17_uniform65536.json
OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open36_event_solver.py --kind random_half --shape 16 4 4 16 --ticks 6 --out data/bit17_random2048.json
```