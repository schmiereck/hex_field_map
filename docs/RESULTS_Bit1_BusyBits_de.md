# Bit-Test 1: Überlebt die Lapse, wenn ein Knoten pro Tick wirklich belegt oder frei ist?

Skript: `src/phase_bit_s4_open20.py` (< 1 min), Daten: `data/phase_bit_s4_open20.npz`, Log `data/phase_bit_s4_open20.log`.

## Frage

Im Modell ist die Lapse ein kontinuierlicher Gate-Winkel a = arccos(2N − 1). Das Wait-Gate R(a) = cos a + i sin a σ_x
dreht ein Flag-Qubit „aktiv/wartend“ jedes Zustands, und R(π/2) = iσ_x ist ein NOT. Der kontinuierliche Winkel ist also
ein kohärentes „teilweises NOT“: eine Überlagerung von belegt und frei.

Bit-Frage: Entsteht die Lapse auch, wenn der Knoten in jedem Tick wirklich belegt (1) oder frei (0) ist? Dann ist der
Gate-Winkel binär, a ∈ {0, π/2}.

Aufbau: 2O*, ε = 0,3, λ*, nur Wait-Gate, Bloch, k = 0,01 und μ = 0,03 für Geschwindigkeit und Lücke. Der physikalische
Ast wird über den Überlapp mit dem freien Dirac-Dublett ausgewählt. Der Puffersektor faltet bei manchen Mustern
(Phase (−1)^m pro Periode) auf E ≈ 0. Zeilen mit Überlapp 0 sind Fehlzuordnungen und werden nicht gewertet.

## A1 – NOT in jedem belegten Tick

| Muster (1 = belegt) | p | Lücke / (2μ) | v / c′ | 1 − p |
|---|---|---|---|---|
| 0 0 1 | 0,333 | 0,5000 | 0,5000 | 0,667 |
| 1 | 1 | 0,5000 | 0,5000 | 0 |
| 0 0 1 1 | 0,5 | 0,7500 | 0,7500 | 0,5 |
| 0 0 1 1 1 1 | 0,667 | 0,6667 | 0,6667 | 0,333 |

Zwei aufeinanderfolgende NOTs heben sich auf. Ein langes belegtes Intervall lässt die Amplitude nur zwischen aktiv und
wartend hin und her wechseln, und die Verlangsamung ist höchstens 1/2. **Das ist keine Lapse 1 − p.**

## A2 – Toggle: NOT beim Wechsel des Bits (Inhalt parken, solange belegt)

| Muster | p | Lücke / (2μ) = v / c′ | Sektor |
|---|---|---|---|
| 0 0 1 | 0,333 | 0,6667 | 1 − p |
| 0 1 | 0,5 | 0,5000 | – |
| 0 0 0 1 | 0,25 | 0,2500 | p |
| 0 1 1 1 | 0,75 | 0,7500 | p |
| 0 0 0 0 0 0 0 1 | 0,125 | 0,1250 | p |
| 0 0 0 1 1 1 1 1 | 0,625 | 0,6250 | p |

Lücke und Geschwindigkeit skalieren **exakt** gleich: eine reine Lapse, wie beim kontinuierlichen Wait-Gate (γ = 0).
Es gibt genau zwei Sektoren. Die Amplitude, die in den freien Ticks aktiv ist, hat N = 1 − p. Die im Puffer
gestartete Amplitude ist in den belegten Ticks aktiv und hat N = p. Welcher Sektor ausgewählt wird, hängt nur von der
Lage des Toggles im Periodenanfang ab. Sind die Puffer anfangs leer, ist nur der Sektor N = 1 − p besetzt.

## A3 – Zufällige, unabhängige Bits (Wahrscheinlichkeit p pro Tick und Knoten)

Die kohärente Amplitude entwickelt sich exakt mit der gemittelten Eintick-Abbildung (1 − p)U(0) + pU(π/2). Diese ist
proportional zum kontinuierlichen Gate mit tan a = p/(1 − p), mal |λ| = √((1−p)² + p²) pro Tick:

| p | kohärente Lapse | kontinuierliches Gate, tan a = p/(1−p) | \|λ\| pro Tick | Kohärenz zweier Knoten pro Tick |
|---|---|---|---|---|
| 0,02 | 0,9999 | 0,9999 | 0,980 | 0,961 |
| 0,05 | 0,9993 | 0,9993 | 0,951 | 0,905 |
| 0,1 | 0,9969 | 0,9969 | 0,906 | 0,820 |
| 0,25 | 0,9743 | 0,9743 | 0,791 | 0,625 |
| 0,5 | 0,8535 | 0,8536 | 0,707 | 0,500 |

Die Lapse ist nur zweiter Ordnung in p, die Dekohärenz dagegen erster Ordnung. Schon bei p = 0,05 verliert die
Kohärenz zwischen zwei Knoten mit unabhängigen Bits 10 % pro Tick. Ein Teilchen wäre nach wenigen Ticks lokalisiert.
**Eine zufällige, unkorrelierte Belegung ist ausgeschlossen.**

## Bewertung

1. Die Lapse als Bit-Physik funktioniert, aber nur mit einer bestimmten Semantik. Ein Knoten **parkt** seinen Inhalt
   für die Dauer der Belegung; NOT erfolgt an Anfang und Ende des belegten Intervalls. Dann ist N = 1 − p exakt (der
   Anteil freier Ticks), ohne kontinuierlichen Winkel.
2. Die Belegung muss **deterministisch** (oder räumlich korreliert) sein. Zufällige Belegung erzeugt kaum Lapse, aber
   starke Dekohärenz. Das ist eine scharfe Aussage über den Rechner: Die Last darf kein Rauschen sein.
3. Das kontinuierliche Wait-Gate ist eine kohärente Überlagerung von belegt und frei. In seiner Bit-Form ist es ein
   festes Toggle-Muster.

## Offen: der entscheidende Test (Bit 2)

Im Bloch-Test sind alle Knoten gleichzeitig belegt. Bei einer räumlich variierenden Belegung schickt ein freier
Nachbar Amplitude in einen belegten Knoten. Beim Toggle wird diese dort zunächst weiter gerechnet und erst beim
nächsten Toggle geparkt. Ob daraus im Mittel eine ortsabhängige Lapse wird (freier Fall, gleiche Beschleunigung für
alle Teilchen), und ob ein Bit-Stay-Gate (Hop nur, wenn beide Knoten frei sind) γ = 1 ergibt, prüft erst eine x-Kette
mit deterministischem Toggle-Muster pro Schicht. Vergleich: die kontinuierlichen Gates aus `phase_bit_s4_open8.py` B.
