# Bit-Test 8: Aktive symmetrische D4-Kollision

Stand: 9. November 2026. Ausgangspunkt: Bit-Test 7 (`phase_bit_s4_open26_symmetric_collision.py`).

## Ziel und Befund

Die genau symmetrische Bit-7-Referenzregel hatte in 100 zufällig geprüften halbgefüllten Zuständen keine Kollision. Diese Variante erhöht die Aktivität ohne eine zusätzliche Taktvariable und ohne stochastische Mikrodynamik.

**Numerisches Screening:** Bei 24 unabhängigen, deterministisch aus Seed 2709 gezogenen 12-Bit-Konfigurationen bewegen sich **14 von 24** (58,3 %). Bei allen 13 geprüften Besetzungszahlen von 2 bis 22 treten Kollisionsänderungen auf. Insgesamt 312 Startzustände, wobei die zusätzlichen Symmetrieprüfungen transformierte Zustände untersuchen. Der Rang der beobachteten Kollisionsdifferenzen ist **19**. Diese Stichprobe ist weder ein Mixing- noch ein Langzeittransporttest.

## Endliche Regel

Lokale Zustände sind 24-Bit-Wörter mit Richtungen auf der D4-Wurzelschale. Zulässige Kandidaten `t` entstehen aus `s` ausschließlich durch den Austausch zweier besetzter gegen zwei unbesetzte Richtungen bei identischem Impuls; es gibt 282 solcher Gattertypen.

Für jedes ungerichtete Kandidatenpaar `{s,t}`:

1. Bilde das Austausch-Bitmuster `m=s xor t`.
2. Berechne den kleinsten Wert von `g(m)` über alle 384 vorzeichenbehafteten Koordinatenpermutationen `g` (B4).
3. Bei gleichem Rang entscheide über den kleinsten kanonischen ungerichteten Kantenwert `min_g (min(g(s),g(t)), max(g(s),g(t)))`.
4. Wenn mehr als ein Nachbar den minimalen Schlüssel besitzt, wähle keinen.
5. Tausche `s` und `t` genau dann, wenn sie sich gegenseitig eindeutig auswählen; andernfalls bleibe bei `s`.

Alle Bewertungen beruhen auf endlichen Bitpermutationen und Ganzzahlvergleichen; die Makroanalyse benutzt weiterhin Zahlen, aber die lokale Kollisionsentscheidung keine Zufallszahlen und keine kontinuierlichen Parameter.

## Korrektheit

- **Teilchenzahl und vier Impulse exakt erhalten:** Jeder einzelne zulässige Tausch erhält diese Größen; Selbstabbildung ebenfalls.
- **Involution / Reversibilität für alle 2^24 Zustände:** Die Regel tauscht ausschließlich Paare gegenseitig ausgewählter Zustände und fixiert alle übrigen.
- **B4-Äquivarianz für alle Zustände:** Kandidatenrelation und Schlüssel sind unter der Gruppenwirkung invariant; die eindeutige Wahl (oder Enthaltung) transformiert äquivariant. Dies wird ergänzend an acht Generatoren für die Stichprobe getestet.
- **Genau fünf additive lineare Erhaltungsgrößen:** 19 explizite, vom Test verifizierte linear unabhängige Kollisionsdifferenzen liefern Rang mindestens 19. Da Teilchenzahl und vier unabhängige Impulskomponenten immer erhalten sind, ist Rang höchstens 19. Die vollständige additiv-lineare Invariantendimension ist deshalb exakt 5.

Diese Aussagen betreffen die **lokale Kollisionsregel**. Die Transportpermutation des bekannten D4-Kerns ist mit der Geometrie kompatibel. Ein Test längerer zusammengesetzter 4D-Trajektorien dieser neuen Variante steht noch aus.

## Stichprobe

| Besetzte Bits | Geprüft | Veränderung |
|---:|---:|---:|
| 2 | 24 | 12 |
| 3 | 24 | 16 |
| 4 | 24 | 14 |
| 6 | 24 | 13 |
| 8 | 24 | 14 |
| 10 | 24 | 13 |
| 12 | 24 | 14 |
| 14 | 24 | 12 |
| 16 | 24 | 17 |
| 18 | 24 | 10 |
| 20 | 24 | 10 |
| 21 | 24 | 13 |
| 22 | 24 | 9 |

## Reproduktion

```bash
python src/phase_bit_s4_open27_active_collision.py --samples 24 --out data/phase_bit_s4_open27_active_collision_validation.json
```

Benötigt `phase_bit_s4_open26_symmetric_collision.py` und `phase_bit_s4_open25_symmetry_recovery.py` im selben `src`-Verzeichnis. Der Vorgängerkern `phase_bit_s4_open24.py` bleibt unverändert.

## Grenzen und nächster Test

Die Kandidatenregeln sind lokal, aber die naive Orbitkanonisierung mit 384 Gruppenpermutationen ist **deutlich zu langsam** für die ursprüngliche 65.536-Knoten-Kampagne. Ein größerer Screeninglauf musste wegen Laufzeit abgebrochen werden. Daher ist dies ein korrektheitsorientierter Prototyp, keine einsatzbereite schnelle Kollisionsimplementierung. **Nächster Schritt:** Orbitentscheidung vorab für die tatsächlich besuchten lokalen Zustände cachen, mit Zeit-/Speichergrenzen benchmarken und eine kurze inverse Vorwärts-Rückwärts-Trajektorie auf kleinen periodischen D4-Boxen testen; erst danach Bit-6-Paketexperiment wiederholen. Keine Aussage über Quantenteilchen, Born-Regel, Lorentzsymmetrie oder stabile Einzelpakete.

Die historische Feldgrafik `figures/phase_bit_s4_open25.png` gehört zum älteren Bit-6-Versuch und bleibt ausdrücklich auf `main` erhalten.