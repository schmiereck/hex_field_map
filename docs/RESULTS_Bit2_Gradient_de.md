# Bit-Test 2: Freier Fall bei räumlich variierendem Belegungsfahrplan

Skript: `src/phase_bit_s4_open21.py` (~10 s), Daten: `data/phase_bit_s4_open21.npz`, Log `data/phase_bit_s4_open21.log`.
Vorgänger: [`RESULTS_Bit1_BusyBits_de.md`](RESULTS_Bit1_BusyBits_de.md).

## Frage

Bit-Test 1 hat gezeigt: Bei gleichförmiger Belegung entsteht die Lapse exakt als N = 1 − p, wenn ein Knoten seinen
Inhalt parkt, solange er belegt ist (Toggle). Gravitation braucht aber einen Gradienten von p. Dann sind benachbarte
Knoten zu verschiedenen Ticks belegt, und Amplitude läuft zwischen freien und belegten Knoten. Bleibt der freie Fall
erhalten?

## Aufbau

- x-Kette wie `phase_bit_s4_open8.py` B: 300 Schichten, Bloch in y, z, 2O*, ε = 0,3, λ*, Paket σ = 16, 160 Ticks.
- Lapse N(x) = N₀(1 + g x) mit N₀ = 0,75 und g = 2e-3, also Belegungsanteil p(x) = 1 − N(x).
- Ein belegter Tick bei x toggelt in t und t + 1: Inhalt, der in Tick t bei x ankommt, setzt Tick t + 1 aus. Für p < ½
  sind alle belegten Ticks isoliert, und die Phase pro Belegung ist +1 (Bit-Test 1).
- Startpaket: freier Walker im aktiven Register. Puffer sind leer, alle Knoten in t = 0 frei.
- Gemessen wird die Beschleunigung a_x relativ zur Semiklassik N(x)·E_frei(k). Bei gleichförmigem Fahrplan ist das die
  exakte Bandenergie der Bit-Regel.
- Leckage = Gewicht außerhalb des aktiven Registers auf Schichten, die in diesem Tick nicht parken. Im richtigen Sektor
  ist sie 0.

Drei Fahrpläne mit demselben Langzeitanteil p(x):

| Fahrplan | Konstruktion | Fehlanpassung benachbarter Schichten pro Tick |
|---|---|---|
| gemeinsame Phase | Sigma-Delta b = ⌊(t+1)p + φ⌋ − ⌊tp + φ⌋, φ = 0 überall | 0,036 (t < 40) → 0,26 (t ≥ 120), wächst ∝ t·∇p |
| Zufallsphase | dasselbe mit zufälligem φ pro Schicht | 0,38 |
| geschachtelt | Ereignis k (ungerade Ticks) belegt, wo u_k < 2p(x); **eine** gemeinsame Folge u_k = frac(k·φ_gold) | 0,001 ≈ ∇p, konstant |

Beim geschachtelten Fahrplan ist die Belegung einer Schicht mit kleinerem p eine Teilmenge der Nachbarbelegung. Die
Nachbarn unterscheiden sich nur, wenn u_k zwischen ihren Schwellen liegt. Jedes solche Ereignis ist eine „Front“ bei
x*: Links von x* setzt ein Tick aus, rechts nicht.

## A – Bloch: Wait- und Stay-Toggle zusammen

| W-Muster | H-Muster | Lücke / (2μ) | v / c′ | Vorhersage |
|---|---|---|---|---|
| 1 0 0 0 | – | 0,7500 | 0,7500 | 0,75 / 0,75 |
| – | 1 0 0 0 | 1,0000 | 0,7503 | 1 / 0,75 |
| 1 0 0 0 | 0 0 1 0 | 0,7500 | 0,5018 | 0,75 / 0,5 |
| 1 0 0 0 0 | 0 0 1 0 0 | 0,8000 | 0,6003 | 0,8 / 0,6 |
| 1 0 0 0 0 0 | 0 0 1 0 1 0 | 0,8333 | 0,4970 | 0,833 / 0,5 |
| 1 0 0 0 | 0 1 0 0 | 0,7500 | 0,3746 | Sektoren gemischt (Überlapp 0,71) |
| 1 0 0 0 | 0 0 0 1 | – | – | Fehlzuordnung (Überlapp 0) |

- Das Stay-Toggle ist die Bit-Form des Stay-Gates: Der Hop wird verweigert, die Münze läuft. Das ergibt
  Lücke = 1 − p_w und v = 1 − p_w − p_h. Der Rest O(k²) beträgt höchstens 0,4 %.
- Die Ereignisse eines Knotens müssen mindestens zwei Ticks auseinanderliegen, damit die vier Toggle-Ticks verschieden
  sind. Sonst vermischen sich die Sektoren.
- Folgerung für die Budgetregel auf Bit-Ebene: Der Hop-Anteil ist P(beide Knoten frei). Er hängt von der Korrelation
  der Fahrpläne benachbarter Knoten ab:

  | Nachbarn | Hop-Anteil | γ |
  |---|---|---|
  | synchron | 1 − p | 0 |
  | unkorreliert | (1 − p)² | 1, wie H2 |
  | antikorreliert | 1 − 2p | 1 im schwachen Feld |

## B – x-Kette, nur Wait-Toggle

a_x / Semiklassik:

| Fall | kontinuierlich 1b | ideale Sprungphase | gemeinsame Phase | Zufallsphase | geschachtelt | Leckage (geschachtelt, max) |
|---|---|---|---|---|---|---|
| μ = 0,03, Ruhe | 0,855 | 0,990 | 0,39 | −0,22 | **0,57** | 0,047 |
| μ = 0,03, k_y = 0,2 | 0,961 | 0,991 | 0,42 | 0,25 | **0,63** | 0,045 |
| μ = 0,15, Ruhe | 0,989 | 0,985 | 0,11 | 0,22 | **0,82** | 0,048 |
| masselos, k_y = 0,2 | 0,952 | 0,992 | 0,37 | 0,15 | **0,65** | 0,045 |
| masselos, k_y = 0,4 | 0,984 | 0,971 | 0,05 | −0,29 | **0,79** | 0,049 |

Leckage bei gemeinsamer Phase: 0,27 bei t = 40, 0,41 am Ende. Bei Zufallsphase: 0,38 bei t = 40.

Die Spalte „ideale Sprungphase“ ist die Kontrolle. Sie nimmt den freien Walker ohne Puffer und gibt den belegten
Schichten des geschachtelten Fahrplans nur die Phase e^{+iE dt} eines ausgesetzten Ticks. Sie trifft die Semiklassik auf
1–3 %, obwohl jede Front scharf ist (eine Schicht breit).

B2, geschachtelter Fahrplan, μ = 0,03 Ruhe / masselos k_y = 0,2:

| Variante | Bit | ideale Sprungphase | Leckage |
|---|---|---|---|
| u_k golden, u₀ = 0 | 0,569 / 0,646 | 0,990 / 0,992 | 0,047 |
| golden, u₀ = 0,5 | 0,547 / 0,583 | 0,957 / 0,987 | 0,049 |
| √2 | 0,608 / 0,660 | 0,985 / 1,000 | 0,047 |
| π (schlechte Diskrepanz) | 0,496 / 0,722 | 0,833 / 0,751 | 0,048 |
| σ = 8 | 0,398 / 0,610 | 1,007 / 1,001 | 0,047 |
| σ = 32 | 0,629 / 0,661 | 0,995 / 0,991 | 0,047 |
| g = 1e-3 | 0,684 / 0,730 | 0,996 / 0,977 | 0,024 |

## Was passiert

1. **Ein Gradient desynchronisiert zwangsläufig.** Knoten mit verschiedenem p sammeln verschieden viele belegte Ticks.
   Der Unterschied wächst wie t·∇p, und das ist die Zeitdilatation selbst. Ein Fahrplan mit gemeinsamer Startphase
   verliert die Synchronität deshalb immer schneller. Nach 40 Ticks ist ein Viertel des Pakets im falschen Sektor, und
   der freie Fall bricht ein (0,05–0,42). Zufallsphasen sind von Anfang an asynchron.
2. **Geschachtelte Fahrpläne halten die Fehlanpassung minimal.** Sie bleibt konstant ~∇p. Das Eigenzeitfeld steckt
   dann korrekt im Fahrplan: Die Kontrolle mit der idealen Sprungphase liefert den vollen freien Fall.
3. **Der Park-Mechanismus überträgt es nicht.** An einer Front setzt der linke Knoten aus, der rechte läuft weiter. Was
   im Freigabe-Tick vom rechten Nachbarn ankommt, trifft auf einen Slot, den der zurückkehrende geparkte Inhalt
   belegt. Unitär bleibt nur der Puffer. Das ist ein Stau, und die Amplitude landet im Komplementärsektor (N = p), wo
   sie nur in belegten Ticks läuft. Pro Front ist das etwa der linkslaufende Fluss über eine Schicht (~0,5 %).
   Zusammen sind es 5 % am Ende, proportional zu g·t. Weil der Fehler in der Amplitude steckt (√0,005 ≈ 7 % pro Front),
   ist er groß gegen die eigentliche Phase E·dt ≈ 0,03 einer Front.
4. **Der Verlust skaliert nicht mit E.** Das Defizit ist am größten für langsame, leichte Teilchen (0,57 bei E = 0,06)
   und am kleinsten für große E (0,82 bei E = 0,30). Damit ist auch die **Universalität verletzt**: Verschiedene
   Teilchen fallen verschieden schnell, ausgerechnet im Newtonschen Grenzfall am schlechtesten. Gegen 1 konvergiert es weder
   für σ → ∞ (0,40 / 0,57 / 0,63 bei σ = 8 / 16 / 32) oder kleineres g (0,57 → 0,68 bei halbem Gradienten und gleicher Leckage pro Front).

## Bewertung

- Die Lapse als echtes Bit (belegt/frei, Inhalt parken) funktioniert nur ohne Gradienten. Mit Gradient braucht ein
  unitärer Park-Mechanismus Synchronität mit den Nachbarn. Die Zeitdilatation zerstört sie aber zwangsläufig, und der
  dadurch entstehende Stau kostet 20–60 % des freien Falls und die Universalität.
- Der kontinuierliche Gate-Winkel löst genau diesen Stau. Der Puffer hält kohärent den Anteil 1 − N, und
  Nachbarschaften mit verschiedenem N mischen glatt. Auf Bit-Ebene müsste der Fahrplan **kohärent** sein, also eine
  Überlagerung belegter und freier Ticks statt einer Abfolge. Das ist das Ergebnis von Bit-Test 1 aus der anderen
  Richtung: Dort war Zufall ausgeschlossen (Dekohärenz), hier ist auch ein deterministischer Fahrplan ausgeschlossen
  (Stau).
- Was im Bit-Bild übrig bleibt: Der **Fahrplan** kann klassisch sein und trägt das Eigenzeitfeld exakt
  (Sprungphasen-Kontrolle 0,99). Das **Aussetzen** selbst muss aber als Phase wirken, also als e^{−i dt Σ(1−b)H}.
  Ein physisches Parken der Amplitude leistet das nicht. Das entspricht der vollen Lapse ½{N, H} aus Runde 3 mit
  binärem N(x, t). Sie braucht den Generator H, ist also in der Walker-Form nicht strikt lokal.
- Die Stay-Variante (Bit-H2) wurde deshalb nicht auf der Kette gerechnet. Sie hat denselben Toggle-Mechanismus und
  damit denselben Stau.

## Offen

- Gibt es eine lokale unitäre Parkregel ohne Stau? Kandidat: Der freigebende Knoten nimmt im Freigabe-Tick keine
  Ankünfte an, sondern reflektiert sie (Spiegel wie in Runde 5). Das kostet Rückstreuung statt Sektorwechsel.
- Den Fahrplan mit Ereignissen so legen, dass Fronten nur bei geparkten Nachbarn auftreten (Fronten „wandern“ mit dem
  Takt). Ob das mit t·∇p verträglich ist, ist offen.
