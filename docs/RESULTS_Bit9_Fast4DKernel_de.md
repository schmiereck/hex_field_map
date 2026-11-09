# Bit-Test 9: Gepackter 4D-Kern für die symmetrische Bit-8-Kollision

Stand: 9. November 2026. **Neuer, unabhängig getesteter Zwischenschritt**, keine Wiederholung der verloren gegangenen Bit-6-Paketkampagne.

## Ziel und Methode

Die exakt reversible, B4-äquivariante Kollisionsregel aus Bit-Test 8 wird **unverändert** auf einem endlichen D4-Gitter ausgeführt. Physische Knoten haben 24 Richtungsbits; die 8 hohen Bits der `uint32`-Wörter bleiben null. D4-Knoten sind die Koordinaten mit gerader Koordinatensumme. Die vier räumlichen Richtungen und ein externer synchroner Tick bleiben unverändert.

Das neue Modul `src/phase_bit_s4_open28_fast4d.py` verwendet:

1. Vorberechnete Streaming-Quellindizes für die 24 festen D4-Richtungen.
2. Vektorisiertes, exakt invertierbares Richtungs-Streaming auf `numpy.uint32`.
3. Eine dünne, **exakte** Kollisions-Cache-Schicht um `phase_bit_s4_open27_active_collision.collide`. Nur verschiedene Knotenwörter werden erstmals ausgewertet; die inverse Paarung wird gleichzeitig gecacht.
4. Dieselbe Regel beim Rückwärtsschritt: inverse Translation, dann lokale involutive Kollision.

**Keine neue Mikrophysik:** Die beschleunigte Implementierung ändert weder die Zustandsübergänge noch die Auswertungsvorschrift. Der Cache speichert **nur** korrekte, deterministische Resultate. Quantenphysikalische Folgerungen entstehen daraus nicht.

## Endliche Tests

| Gitter | physische Knoten | Startbesetzung | Vorwärts/Rückwärts | Bitvergleich mit separater Python-Referenz | Erhaltungen |
|---|---:|---:|---:|---|---|
| 4×4×4×4 | 128 | 12 Bits/Knoten | 3 / 3 Ticks | bitidentisch | exakt |
| 8×4×4×8 | 512 | 12 Bits/Knoten | 2 / 2 Ticks | bitidentisch | exakt |
| 4×4×4×4 | 128 | 6 Bits/Knoten | 2 / 2 Ticks | bitidentisch | exakt |

Alle drei Tests bestätigen Null-Padding, vier Impulskomponenten plus Teilchenzahl und Rückkehr zum gesamten Ausgangszustand. Diese endlichen Stichproben sind keine vollständige Enumeration von 2^24 lokalen Zuständen und kein neuer Nachweis der allgemeinen Symmetrie; die theoretische Konstruktion stammt aus Bit-Test 8.

## Geschwindigkeitsmessungen

Auf der aktuellen CPU / Python-Umgebung (Richtwerte, nicht hardwareunabhängig):

- Reines 4D-Streaming auf 512 Knoten: ca. **0,000080 s/Tick** gegenüber **0,007758 s/Tick** in der unabhängigen Python-Schleife, ca. **97-fach schneller** (30 versus 3 Messungen). Dies misst **nur Transport**, keine lokale Kollision.
- Auf 512 Knoten kostet ein frischer Vorwärtstick mit noch unbekannten Kollisionszuständen ca. **0,48–0,54 s**.
- Beim 128-Knoten-Test ca. **0,15–0,22 s** pro Vorwärtstick.
- Bereits vollständig gecachte Kollisionen können für 512 Knoten in ca. **0,0001–0,00015 s** erneut ausgewertet werden. Dies ist **kein** realistischer Wert für lange neue Zufallstrajektorien.

Die B4-Kanonisierung der neuen **unbekannten** Kollisionszustände bleibt der Engpass. Das jetzige System ist ein verifizierter, schneller **Transportkern** mit exaktem Cache, aber noch **kein ausreichend schneller vollständiger Kollisionskern** für die große Paketkampagne mit 65.536 physischen Knoten. Diese Unterscheidung ist wichtig.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open28_fast4d.py --ticks 3
OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open28_fast4d.py --shape 8 4 4 8 --ticks 2
OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open28_fast4d.py --ticks 2 --occupation 6
```

Die Implementierung importiert aus dem bestehenden Code `phase_bit_s4_open27_active_collision.py`, `phase_bit_s4_open26_symmetric_collision.py`, `phase_bit_s4_open25_symmetry_recovery.py`. Im Backup sind alle drei enthalten. Die JSON-Dateien in `data/` enthalten die konkreten Messungen.

## Nächster sinnvoller Schritt

Die **Kollisionskanonisierung** selbst muss ohne 384 Symmetriedurchläufe für jeden neuen lokalen Zustand implementiert werden (z. B. orbitbasierte Vorberechnung, effiziente Integer-Tabellen, kompakterer Algorithmus oder ein beweisbar gleichwertiger symmetrischer Gatterentwurf). Nach dem Vergleich mit der Bit-8-Referenz sollen mehrere Gittergrößen, lange vollständige Umkehrläufe und erst danach die 4D-Paketversuche folgen. Speicher- und Laufzeitbudgets müssen begrenzt werden.

Die historische `figures/phase_bit_s4_open25.png` auf `main` wird nicht geändert oder entfernt.