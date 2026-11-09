# Bit-Test 7 – exakt symmetrische reversible D4-Kollision (Referenzentwurf)

Stand: 9. November 2026. **Neu konstruiert, nicht Bestandteil des verlorenen Commits `283283b`.**

## Konstruktion

24 Boolesche Richtungsbits auf einem D4-Knoten. Für jedes lokale Zustandspaar `s,t` sind elementare Kandidaten nur solche **2-in/2-out-Swaps** auf vier verschiedenen Richtungen, bei denen die beiden Eingangsrichtungen dieselbe ganzzahlige Viererimpulssumme wie die beiden Ausgangsrichtungen haben. Es gibt 282 mögliche vierstellige Gate-Typen; pro Zustand ist nur ein Teil davon zulässig.

Die Gruppe der 384 vorzeichenbehafteten Achsenpermutationen (hyperoctahedrale D4-Wurzel-Symmetrie `B4`) wirkt auf allen 24 Richtungsplätzen. Definiere `canon(s)` als das kleinste ganzzahlige 24-Bit-Muster im Orbit von `s`. Jeder Zustand wählt den Nachbarn mit **eindeutig minimalem** `canon(t)`. Bei Gleichstand wählt er keinen. Nur wenn zwei Zustände **einander gegenseitig wählen**, tauschen sie ihre Zustände, ansonsten bleiben sie unverändert.

Formell: `p(s)=argmin_unique {canon(t): t ist zulässiger Nachbar von s}`; `C(s)=p(s)` genau dann, wenn `p(s)=t` und `p(t)=s`, sonst `C(s)=s`.

Das ist eine deterministische lokale Regel am 24-Bit-Gitterknoten ohne Wahrscheinlichkeiten oder stetige Mikrowerte. Die kanonische Berechnung nutzt ausschließlich endliche Bitpermutationen, Integer-Vergleiche und feste Tabellen. **Sie ist allerdings wesentlich teurer als die alte geordnete Gatterfolge.**

## Eigenschaften und Beweis

1. **Bijektivität/Reversibilität:** Jede aktive Kante ist ein gegenseitiges Paar. Die übrigen Zustände sind Fixpunkte. Daher `C(C(s))=s` für sämtliche `2^24` möglichen Zustände, ohne vollständige Enumeration.
2. **Teilchenzahl und D4-Impuls:** Jede mögliche Kante tauscht genau zwei besetzte mit zwei unbesetzten Richtungen identischer Summe; also werden Teilchenzahl und alle vier ganzzahligen Impulskomponenten exakt erhalten.
3. **Volle B4-Symmetrie:** Jede vorzeichenbehaftete Koordinatenpermutation bildet zulässige 2-in/2-out-Swaps auf solche Swaps ab. `canon(g·s)=canon(s)`, eindeutige Minima und Gegenseitigkeit werden erhalten. Daraus folgt exakt `C(g·s)=g·C(s)` für alle 384 Gruppenoperationen. Transport entlang D4-Richtungen ist ebenfalls äquivariant, wenn Orts- und Richtungskoordinaten gemeinsam transformiert werden.
4. **Genau fünf additive lineare Erhaltungsgrößen:** Die fünf Größen (Anzahl plus vier Impulse) liefern eine obere Grenze 19 für den Rang aller Kollisionsdifferenzen. Das JSON-Protokoll enthält 19 konkrete Zustandspaare mit aktiver Kollision, deren 24-komponentige Differenzvektoren über den rationalen Zahlen unabhängig sind. Damit ist der Rang **exakt 19** und der Raum der additiven linearen Invarianten **exakt fünfdimensional**.

Die Argumente 1–3 sind Strukturbeweise für sämtliche `2^24` Zustände. Die 19 überprüfbaren Zeugen zusammen mit den strukturell bewiesenen Erhaltungen schließen zusätzliche *additive lineare* Erhaltungsgrößen aus; nichtlineare Invarianten oder dynamische Fragmentierung sind damit nicht ausgeschlossen.

## Tatsächliche Tests

Das Python-Programm prüft alle 2.325 Bitzustände mit Besetzung 0 bis 3 und außerdem deterministisch gezogene weitere Zustände (insgesamt 2.745 Eingaben). Für alle überprüften Eingaben bestehen exakte Erhaltung, Rücklauf und die acht Generatoren (Inversion, vier Spiegelungen, drei Koordinatentausche). Für 19 aktive Zustandsübergänge werden unabhängige Differenzvektoren protokolliert. Es wurde **kein vollständiger 16.777.216-Zustände-Test** ausgeführt; die Allgemeingültigkeit der drei ersten Eigenschaften folgt aus dem oben beschriebenen Konstruktionsbeweis, nicht aus dem Stichprobentest.

### Entscheidendes negatives Ergebnis: Aktivitätsdichte

Bei zufälliger Halbfüllung wurden **0 von 100** getesteten Zuständen verändert; ebenso 0 von 100 bei 10 und 14 gesetzten Bits. Bei nur drei gesetzten Bits bewegten sich 480 von 2.024 Zuständen. Die Aktivitätsraten sind bei den zufälligen Stichproben noch ungenau, aber ein klares Warnzeichen: Die reversible Gegenwahl ist sehr restriktiv. Eine produktive pakettragende Dynamik ist **nicht nachgewiesen**.

**Fazit:** Erfolgreiche mathematische **Referenzregel** mit allen geforderten linearen Erhaltungen und räumlichen Symmetrien, aber als physikalischer Nachfolgekern noch nicht akzeptiert. Ein Folgetest muss den Anteil aktiver Kollisionen bei relevanter Dichte, Relaxationszeiten und spektrale Moden untersuchen. Erst eine ausreichend dynamische symmetrische Regel rechtfertigt die Wiederholung des teuren lokalen Pakettests.

## Reproduktion

```bash
python src/phase_bit_s4_open26_symmetric_collision.py --out data/phase_bit_s4_open26_symmetric_collision_validation.json
```

Benötigt `src/phase_bit_s4_open25_symmetry_recovery.py` aus dem zuvor gesicherten Recovery-Zweig. Keine Python-Drittbibliotheken erforderlich. Der Prototyp nutzt Caches im Speicher und ist ausdrücklich **kein skalierter 4D-Produktionskernel**.