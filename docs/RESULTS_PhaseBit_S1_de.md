# Experiment S1: Spin-Wirbel aus Phasen-Bits (ganzzahliger Automat, 2D)

*Paper 3, Hauptlinie. Code: `src/phase_bit_automaton.py` und `src/phase_bit_automaton_figs.py`.
Daten: `data/phase_bit_automaton.npz`. Abbildung: `figures/phase_bit_automaton.png`.
Einordnung: [`ROADMAP_Paper3_de.md`](ROADMAP_Paper3_de.md).*

## Frage

Lassen sich „Spin-Wirbel“ aus Phasen-Bits tatsächlich bauen und sehen, und zwar
mit einer Dynamik, die nur aus ganzen Zahlen besteht?

## Der Automat

Ein Walker auf dem Dreiecksgitter trägt drei diskrete Größen:

| Größe | Werte | Bedeutung |
|-------|-------|-----------|
| d | Z₆ | Richtung (0°, 60°, …, 300°) |
| p | Z₄ | **Phasen-Bit**: +1, +i, −1, −i |
| T | 0, 1, 2, … | Zahl der elementaren 60°-Drehungen bisher |

Ein Schritt dreht um n ∈ {−2, …, +2} Einheiten von 60° (ohne Umkehr) und springt dann:

```
d → d + n (mod 6),    p → p + n (mod 4),    T → T + |n|
```

Der Zustand ist eine Tabelle **nichtnegativer ganzer Pfadzahlen** N[x, y, d, p, T].
In der Dynamik kommen keine Phasen, kein ε und keine Gleitkommazahlen vor, nur
Additionen von ganzen Zahlen.

Erst beim **Auslesen** werden Zahlen angeheftet:

```
A_ab(x, y, d; ε) = Σ_{p,T} N[x,y,d,p,T] · ε^T · i^(a·p + b·T)
```

- a = 1: Das Phasen-Bit wird als Phase gelesen (Spin-Auslesung).
- a = 0: Das Phasen-Bit wird ignoriert (Boson-Auslesung).
- b = 1: Jede elementare Drehung trägt zusätzlich den Faktor i, das ist die Regel
  „Abbiegen = iε“ aus Paper 1.

## Ergebnisse

### 1. Ganzzahl-Tabelle = Fließkomma-Walk, exakt

Gegenprobe: der Fließkomma-Walk mit der Münze W[d′,d] = ε^|n| · i^(a·n + b·|n|).
Über t ≤ 24, ε ∈ {0.1, 0.3, 0.7} und alle vier Auslesungen (a, b) beträgt die
größte relative Abweichung **3.2·10⁻¹¹**. Das ist reine Rundung bei der Summation
großer ganzer Zahlen.

Die Auslesung (a, b) = (1, 1) ist genau das „graded“ Turning-Modell bei α = 3/2.
Ihr Spektrum stimmt bei zufälligen k und ε mit dem bei **α = ½** auf
**4.7·10⁻¹⁵** überein (Eichäquivalenz, α zählt nur mod 1). Das Phasen-Bit ist
also exakt der Spinor des Turning-Modells.

### 2. Die Doppelüberlagerung ist sichtbar

Von den 24 möglichen Paaren (d, p) werden **genau 12** erreicht, nämlich alle mit
d ≡ p (mod 2). Das sind die Werte des Netto-Drehzählers mod 12, also Z₁₂, die
zweifache Überlagerung der 6 Richtungen. Das ist das diskrete Gegenstück zu
SU(2) → SO(3). Abbildung (a) zeigt es am einfachsten Pfad, 12 Linksdrehungen:
Nach dem ersten Umlauf ist die Richtung wieder dieselbe, das Bit aber −1. Erst
nach dem zweiten Umlauf ist es wieder +1.

### 3. Bit = −1 genau bei ungerader Windung

Gezählt werden geschlossene Pfade, die mit Richtung 0 zum Ursprung zurückkehren.
Der Vergleich mit einer unabhängigen Aufzählung aller geschlossenen Pfade nach
Windungszahl w ergibt (L = Schrittzahl):

| L | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| Automat, Bit −1 (p = 2) | 2 | 4 | 10 | 48 | 168 | 696 | 3128 |
| Aufzählung, w ungerade   | 2 | 4 | 10 | 48 | 168 | 696 | 3128 |
| Automat, Bit +1 (p = 0) | 0 | 0 | 0 | 26 | 112 | 548 | 2712 |
| Aufzählung, w gerade     | 0 | 0 | 0 | 26 | 112 | 548 | 2712 |

Die Übereinstimmung ist **exakt** (ganze Zahlen), mit 0 Verletzungen des
Whitney-Satzes. Das Phasen-Bit misst also nicht irgendeine Phase, sondern
topologisch die Parität der Windungszahl. Ein Bit, das nur ±1 annimmt
(p ∈ {0, 2}), reicht bei geschlossenen Pfaden für die gesamte Spin-Information.

### 4. Die Spin-Wirbel: Halbquanten-Wirbel im Richtungsraum

Bei k = 0 ist der Ganzzahl-Übergang auf Z₁₂ zirkulant. Seine Eigenmoden sind daher
exakt die Wirbel e^{2πi·j·N/12}, j = 0 … 11 (analytisch und numerisch gleich auf
1.7·10⁻¹⁵). Eine volle 360°-Drehung (N → N + 6) multipliziert die Mode j mit (−1)^j:

- **ungerades j:** Die Phase windet sich bei einem Umlauf um die 6 Kanten nur um
  π (j = 1) bzw. 3π und 5π. Das sind **Halbquanten-Wirbel**, die sich erst nach
  720° schließen. Das ist Spin ½, Abbildung (b).
- **gerades j:** gewöhnliche Wirbel mit ganzzahliger Windung, Abbildung (c).

Die **Spin-Auslesung (a = 1) behält genau die ungeraden j**, die Boson-Auslesung
genau die geraden (geprüft durch Projektion; die 6×6-Spektren stimmen auf
10⁻¹⁵ mit den jeweiligen j-Werten überein). Die Eigenwerte sind

```
λ_j = 1 + 2ε·i^b·cos(πj/6) + 2ε²·i^{2b}·cos(πj/3)
```

Weil j und 12 − j dasselbe λ haben, sind alle Spin-Niveaus **Kramers-Dubletts**
((1,11), (3,9), (5,7)). Die Boson-Seite hat dagegen 4 verschiedene Niveaus
(j = 0 und 6 einfach). Beispiel ε = 0.3, b = 1:

| Sektor | j | λ_j |
|--------|---|-----|
| Spin | 1, 11 | 0.91 + 0.520i |
| Spin | 3, 9 | 1.18 |
| Spin | 5, 7 | 0.91 − 0.520i |
| Boson | 0 | 0.82 + 0.6i |
| Boson | 2, 10 | 1.09 + 0.3i |
| Boson | 4, 8 | 1.09 − 0.3i |
| Boson | 6 | 0.82 − 0.6i |

### 5. Gleiche ganze Zahlen, zwei Welten

Die Abbildungen (d) und (e) zeigen |A|² nach 24 Schritten aus **derselben**
Ganzzahl-Tabelle, einmal mit Spin-Auslesung und einmal mit Boson-Auslesung. Die
Interferenzmuster unterscheiden sich deutlich. Ob ein Teilchen Spin ½ oder Spin 0
hat, steckt in dieser Formulierung nicht in der Dynamik, sondern in der Frage, wie
man das Phasen-Bit liest. Die Dynamik zählt nur Pfade.

### 6. Was das für „diskret auf allen Ebenen“ heißt

- Die Pfadzahlen wachsen: Die größte Zahl braucht nach 24 Schritten 42.9 Bit, also
  **1.79 Bit pro Schritt**. Ein exakter Automat mit **endlich** vielen Zuständen
  pro Knoten ist damit nicht möglich. Die Information wächst linear in der Zeit.
- Die hier verwendete Regel (graded, Paper-1-Typ) ist nicht unitär. Eine exakt
  unitäre Version mit ganzzahligen Einträgen ist unmöglich, weil sie eine
  Permutation wäre und nicht mischt (siehe Roadmap). Unitär **und** endlich geht
  nur mit einem ε, das erst beim Auslesen eingeht, wie hier, oder mit rationalen
  Münzen (Experiment 2).

## Bezug zum Thread

Gemini hatte den Spin-Wirbel als Z₄-Windung auf einem 4er-Ring des Kuboktaeders
beschrieben. Das ist ganzzahliger Drehimpuls. Der Spin-Wirbel existiert, aber an
anderer Stelle: als Halbquanten-Wirbel im **Richtungsraum** eines Knotens, getragen
vom Phasen-Bit, das die Doppelüberlagerung zählt.

## Nächste Schritte

- **S2:** Wirbel als Wellenpaket in Bewegung. Bleibt die Halbwindung stabil? Ist
  v < c? Ist die Zitterbewegung im Richtungsraum sichtbar? Dafür den unitären Coin
  verwenden, der im Spin-Sektor dieselben Wirbelmoden hat.
- **S3:** 3D mit endlichem Spin-Register (binäre Oktaedergruppe 2O). Hier ist der
  Ausgang offen, weil die 60°-Drehungen nicht in O liegen.
