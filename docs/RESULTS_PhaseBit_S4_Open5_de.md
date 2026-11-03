# S4, fünfte Runde: starker und zeitabhängiger Lapse, Lapse aus der Last, anisotrope Metrik, Landau-Zweige

*Paper 3. Code: `src/phase_bit_s4_open5.py` (Teile `A`, `B`, `C`, `D`) und
`src/phase_bit_s4_open5_figs.py`. Daten: `data/phase_bit_s4_open5.npz`.
Abbildung: `figures/phase_bit_s4_open5.png`.
Vorgänger: [`RESULTS_PhaseBit_S4_Open4_de.md`](RESULTS_PhaseBit_S4_Open4_de.md).*

Speicher: Die Teile A bis C rechnen Schritt für Schritt mit dünn besetzten Matrizen
und brauchen wenige MB. Das Größte war Teil D, die Determinanten der dichten
Sektormatrix der Dimension 4032 (etwa 260 MB pro Matrix). Frei waren rund 4.2 GB.

Modell wie in Runde 4: **2O\*** (ε = 0.3, λ\* = −3.2036) mit Vakuum-Gegenterm,
Warte-Puffer mit Phase π. Teil D nutzt minimales SU(2) wie in Runde 3/4.

## 1. Starke Felder: Ein Puffer reicht, und es gibt eine allgemeine Regel

**Ein Puffer deckt 0 < N ≤ 1 ab.** In Runde 4 stand, ein Puffer reiche nur bis
N ≥ 1/2. Das stimmt nicht: Der Mischwinkel a darf auch über π/2 gehen. Die Formel
f_a(θ) = θ/2 + arcsin(cos a · sin(θ/2)) bleibt exakt (gegen das 48×48-Spektrum
2.6·10⁻¹⁵ bei N = 0.4, 0.2, 0.05).

| N | 0.4 | 0.2 | 0.1 | 0.05 | 0.02 |
|---|---|---|---|---|---|
| Quartett-Verhältnis | 0.40000 | 0.20001 | 0.10001 | 0.05000 | 0.02000 |
| Pufferanteil | 0.600 | 0.800 | 0.900 | 0.950 | 0.980 |

Die Abweichung von der Linearität bleibt auch bei N → 0 beschränkt. Die Entwicklung
gibt f = Nθ − c(1 − c²)θ³/48 mit c = cos a, relativ also höchstens etwa θ²/12. Was
schrumpft, ist nur der Abstand der beiden Zweige am Zonenrand, 2(π − a) ≈ 4√N.
Mehrere Puffer braucht man also nicht.

**Allgemeine Regel (Hellmann–Feynman):** Für jedes lokale Warte-Schema U = M·[S C ⊕ D]
mit beliebiger unitärer Mischung M und festen Pufferphasen D gilt: Der Lapse
(Steigung des Kegels) ist der **aktive Anteil des Vakuum-Fixvektors**. Begründung:
dU/dk = −i (R M_akt R†) U, und R† v = D v für den Fixvektor. Geprüft mit sechs
zufälligen Schemata mit zwei Puffern (Steigung/aktiver Anteil):
0.0151/0.0151, 0.2379/0.2381, 0.4896/0.4903, 0.6180/0.6182, 0.4270/0.4275,
0.5497/0.5507. Der Rest ist die Krümmung bei k = 0.01. „Lapse = Bruchteil der Zeit,
in der gerechnet wird“ gilt damit für jede lokale Warte-Regel, nicht nur für die
gewählte.

## 2. Annäherung an einen Horizont: Auf dem Gitter wird er zum Spiegel

Aufbau: x-Kette mit 360 Lagen, Puffer-Lapse N(x), der von 0.9 (x > 40) linear auf
N_min = 0.04 (x < −40) fällt. Ein masseloses Paket (k = −0.3, Breite 12 Lagen)
läuft von x = 60 in Richtung kleiner N.

| t | ⟨x⟩ Sim. | Strahl, exakte Dispersion | Strahl, Kegel c′N(x) | N(⟨x⟩) | k Sim. (Strahl) |
|---|---|---|---|---|---|
| 50 | +34.60 | +34.60 | +34.23 | 0.84 | −0.33 (−0.32) |
| 150 | −0.69 | −0.75 | −1.82 | 0.46 | −0.59 (−0.59) |
| 300 | −24.66 | −24.72 | −27.21 | 0.21 | −1.45 (−1.44) |
| 600 | −35.88 | −35.05 | −41.38 | 0.08 | −5.72 (−5.63) |

- **Koordinatengeschwindigkeit c′·N(x) und Blauverschiebung:** Die Koordinatenenergie
  E′ bleibt erhalten (Drift 2.5·10⁻³). Daher wächst der Gitterimpuls wie k = E′/(N c′),
  und das Paket wird in x gestaucht (Breite 7.8 → 1.2).
- **Die Gittergrenze wirkt als Spiegel:** Bei k = −π/a (= −5.13) erreicht das
  Photonband sein Maximum E_max = 1.67. Dort wird die Gruppengeschwindigkeit null und
  kehrt dann um. Der Umkehrpunkt liegt bei f_N(E_max) = E′, also N = 0.089 bzw.
  x = −35.5. Das simulierte Paket kommt bis −39.7 (sein Schwerpunkt bei t = 865) und
  läuft dann zurück. Am Ende sind 22 % wieder bei x > 60, der Rest ist breit verteilt.
  Der Strahl mit exakter Dispersion folgt der Simulation bis zum Umkehrpunkt.
- **Deutung:** Ein Horizont (N → 0) ist auf dem Gitter nicht erreichbar. Die
  Blauverschiebung endet an der Gitterauflösung, und das Photon wird reflektiert,
  statt einzufrieren. Dieses Analogon zum Trans-Planck-Problem ist keine Zusatzannahme,
  es folgt aus der beschränkten Bandbreite des Walkers. Photonen mit höherer Energie
  dringen tiefer ein (N_turn ≈ E′/E_max).

## 3. Zeitabhängiger Lapse

Gleichförmige Rampe N: 0.9 → 0.3 bei k = (0.2, 0, 0) und μ = 0.03 (massiv):

| Rampe (Takte) | 4 | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|
| verlässt den Bandzustand | 2.9·10⁻⁴ | 1.5·10⁻⁵ | 5.1·10⁻⁷ | 2.4·10⁻⁸ | 1.2·10⁻⁹ | 2.9·10⁻¹⁰ |
| Phase − Σ f_N(t)(θ) | 4.9·10⁻⁴ | 2.0·10⁻⁴ | 6.9·10⁻⁵ | 2.1·10⁻⁵ | 6.9·10⁻⁶ | 2.1·10⁻⁶ |

Der Verlust geht in den π-Zweig des Puffers. Er fällt etwa wie T⁻³, schon eine Rampe
über 4 Takte verliert nur 3·10⁻⁴. Die Phase ist Σ N(t)·E·Δt, bis auf einen Rest
∝ 1/T. **Ein zeitabhängiger Lapse ist also eine reine Umparametrisierung der Zeit,**
ohne Teilchenerzeugung. Für Gravitationswellen braucht man dagegen die anisotrope
Metrik (Abschnitt 5).

## 4. Exp. 1d: der Lapse aus der lokalen Last

**Regel:** Die Last eines Knotens ist die Energiedichte, und sie ist lokal messbar:

```
e(x) = Re ψ†(x) · i (Uψ − ψ)(x) / Δt          (summiert über aktives und Puffer-Register)
```

U ist ein Nachbarschaftsoperator, also ist e(x) lokal. Für einen stationären Zustand
gibt es sin(E′Δt)/Δt·|ψ|². Das Vakuum hat wegen des Gegenterms (Fixpunkt E = 0) die
Last **null**. Der Lapse folgt aus N = N₀(1 + Φ) mit Φ″ = κ e. Gerechnet wird mit der
1D-Green-Funktion Φ = (κ/2)∫|x − x′| e(x′) dx′, einer konstanten Kraft unabhängig vom
Abstand. Jedes Paket bewegt sich im Feld des anderen (Hartree ohne Selbstwechselwirkung).

Aufbau: x-Kette mit 1000 Lagen, zwei Pakete im Abstand von 245 Längeneinheiten,
Breite 60 Lagen, 200 Takte, κ so gewählt, dass eine ruhende Quelle mit E = 0.06 die
Steigung 10⁻³ erzeugt.

**(1) Zwei Massen** (μ = 0.03 und 0.06, also m = 0.06 und 0.12):

| | a Simulation | Halbklassik (k-Breite des Pakets) | Band-R′ × N₀²c′²κQ/2 |
|---|---|---|---|
| leicht | +3.603·10⁻⁴ | +3.619·10⁻⁴ | +3.736·10⁻⁴ |
| schwer | −1.835·10⁻⁴ | −1.834·10⁻⁴ | −1.850·10⁻⁴ |

- Das Lastintegral Q = Σe bleibt bei N₀E (0.0450 → 0.0451, 0.0900 → 0.0900).
- **Drittes Gesetz:** (E₁a₁ + E₂a₂)/(E₁|a₁|) = −0.019. Das passt zu den Fallfaktoren der
  beiden Pakete (Erwartung −0.014): Das leichte Paket fällt wegen seiner Impulsbreite
  mit 0.965, das schwere mit 0.978. Aktive und passive schwere Masse sind also beide
  die Energie, und Actio = Reactio gilt bis auf die Abweichung vom Äquivalenzprinzip
  pro Spezies.
- Falle bei der Wahl von κ: Bei doppelter Kopplung lief N in den Ausläufern der
  breiten Pakete über 1. Das Abschneiden dort verfälschte a₁ um 7 %.

**(2) Quelle in Querbewegung** (k_y = 0, 0.3, 0.6; Testteilchen in Ruhe):
a_Test/a_Test(Ruhe) = 1.000 / 3.017 / 5.688 gegen E/E(Ruhe) = 1.000 / 3.036 / 5.740.
**Die aktive Masse ist die Energie γm,** wie Q = N₀E auch direkt zeigt.

**Grenze:** In der linearisierten ART zieht eine bewegte Quelle ein ruhendes
Testteilchen mit E(1 + v²), weil T_ii mitwirkt. Bei k_y = 0.6 wäre das etwa 11.3
statt 5.7 (Abbildung d). Ein Licht-Strahl wirkte doppelt so stark. Die Last e(x) ist
eine skalare Quelle (Nordström-Typ auf der aktiven Seite). Damit die Quelle der ART
entspricht, müsste der Druck bzw. Impulsfluss mitgezählt und die räumliche Metrik A
mit angetrieben werden.

## 5. Variable Kantenlängen: anisotrope Metrik

**Kantenlänge ist für den Walker nur über Laufzeit und Phase sichtbar.** Ein
unregelmäßiges Gitter mit isotroper Kantenlänge A(x) ist in Gitterkoordinaten
genau die Konstruktion von Exp. 1c: Wartezeit ∝ Länge (Puffer-Lapse N/A) und
Massenphase pro Kante ∝ Länge (μA). Neu ist der anisotrope Fall (Scherung,
Gravitationswellen):

**(a) Wartezeiten je Richtung mitteln sich weg.** Der Puffer sitzt auf dem gerichteten
Kantenzustand, also kann a von der Richtung abhängen: N_d = N₀/L_d mit L_d = |A u_d|.
Für A = diag(1.05, 0.95, 1) ergeben sich Steigungen von 0.74956, 0.74949 und
0.74953 c′, also isotrop. Die ART bräuchte 0.714 / 0.789 / 0.750. Der Grund liegt in
der Regel aus Abschnitt 1: Der Zähler ⟨v|M_akt|v⟩ der Geschwindigkeit hängt nicht von
den Puffern ab, nur die Normierung, und die ist das Mittel 1/⟨1/N_d⟩ = 0.74953. Warten
ändert also nur den Lapse.

**(b) Richtungsabhängige Coin-Kopplungen wirken.** Gewichtet man im Coin-Generator jedes
Kantenpaar mit s_d s_d′, s_d = 1 + u_dᵀ S u_d, dann bleibt der Kegel ein sauberes
Ellipsoid: E² = c′² kᵀ M k mit Fit-Rest 10⁻⁶. Das Quartett bleibt entartet (Aufspaltung
≤ 10⁻⁶). Die Antwort hat **zwei** Konstanten, wie es die kubische Symmetrie verlangt:
E_g −1.81 und T_2g −1.11. Mit S = −2h_E/r_E − 2h_T/r_T trifft man jede spurfreie
Verzerrung, M = g⁻¹ mit g = 1 + 2h.

| zufällige Verzerrung \|2h\| | 0.017 | 0.023 | 0.023 | 0.030 | 0.037 |
|---|---|---|---|---|---|
| \|M − g⁻¹\| | 3.5·10⁻⁴ | 7.3·10⁻⁴ | 7.3·10⁻⁴ | 1.5·10⁻³ | 1.9·10⁻³ |

Der Rest ist quadratisch in h, die lineare Ordnung ist also exakt. Beispiel h_plus =
0.05: Lichtgeschwindigkeit in x, y, z = 0.949 / 1.049 / 1.003 c′ gegen ART 0.954 /
1.054 / 1.000. Das Vakuum verschiebt sich nur in zweiter Ordnung (−2.4·10⁻⁵ bei
h = 0.005, −1.5·10⁻³ bei 0.04). Der Gegenterm muss also erst bei h² nachgezogen werden.

**(c) Gravitationswellen-Interferometer.** h_plus(t) = 0.01·sin(0.05 t), gleichförmig;
Photonen entlang x und y; exakter Takt über 800 Schritte.

| k | Phasendifferenz x − y (max) | ART 1. Ordnung (c′) | ART mit Gitter-v_g |
|---|---|---|---|
| 0.03 | 0.01397 | 0.01385 | 0.01385 |
| 0.1 | 0.04788 | 0.04617 | 0.04608 |
| 0.3 | 0.15416 | 0.13850 | 0.13640 |

Der exakte Takt folgt dem adiabatischen Band auf 10⁻⁶, die Fidelity ist 0.99997. Die
Abweichung von der ART wächst linear mit k (0.9 %, 3.7 %, 11 %) und hängt nicht von h₀
ab (bei h₀ = 0.005 dieselben Prozente). Sie ist eine Dispersionskorrektur der Antwort
bei endlichem k und verschwindet für k → 0.

**Fazit:** Isotrope Metrik = Warten + Massenphase. Anisotrope Metrik = Gewicht der
Kopplung zwischen Richtungen, also eine Art „Leitwert“ der Kante. Beides sind lokale
Einstellungen, doch nur das Zweite kann eine Welle tragen. Die Ausbreitung von h selbst
(eine Feldgleichung für die Kopplungsgewichte) ist nicht Teil dieser Runde.

## 6. Globale Landau-Zweige: periodische Kette mit einem Flussquant

Aufbau: periodische x-Kette, Landau-Eichung mit kommensurablem Fluss
B = 2π/(N_x a²), also ein Flussquant pro Zelle; Helizitätssektor +1; minimales SU(2),
μ = 0. Ohne offene Enden gibt es keine Randzustände.

| N_x | B | Sektor-Dimension | Unitarität | Windung von det U(k_z) über die Zone |
|---|---|---|---|---|
| 168 | 0.0997 | 2016 | 1·10⁻¹⁵ | **0.000** |
| 336 | 0.0499 | 4032 | 9·10⁻¹⁶ | **0.000** |

- Für eine unitäre Familie U(k_z) ist der Netto-Spektralfluss durch **jede**
  Quasienergie gleich der Windung von det U. Er ist hier null, also auch durch den
  Zonenrand ±2π/Δt. **Es gibt keinen anomalen Floquet-Fluss über den Zonenrand.** Der
  abwärts laufende chirale Zweig des unteren Knotens (χ = +1) wird bei jeder
  Quasienergie durch aufwärts laufende Zustände ausgeglichen, im Einklang mit dem
  Gegenzweig am inneren Knoten aus Runde 4.
- Die Spektren hängen kaum von k_y ab (2.4·10⁻³ bei B = 0.1), wie bei einem Flussquant
  zu erwarten.
- **Nicht gelungen:** den einzelnen Zweig über die Zone zu verfolgen. Bei einem
  Flussquant liegen 2016 Zustände dicht in der Quasienergie. Eine Fortsetzung über den
  maximalen Überlapp (Shift-Invert, Schritt 0.01) schließt sich nicht: Überlapp bis
  0.70, Endenergie ≠ Startenergie. Auch die k_z-aufgelöste Flussdichte aus 97 k_z-Punkten
  ist zu verrauscht. Ob der Zweig die Knoten **direkt** über das Volumen verbindet, ist
  damit nicht gezeigt. Gezeigt ist nur, dass er es nicht über den Zonenrand tut.

## Antwort

1. **Starke Felder:** Ein Puffer reicht für 0 < N ≤ 1. Allgemein ist der Lapse jeder
   lokalen Warte-Regel der aktive Anteil des Vakuum-Fixvektors.
2. **Horizont:** Ein Photon läuft mit c′N(x) und wird blauverschoben, bis seine
   Wellenlänge die Gitterauflösung erreicht. Dann kehrt es um, bei f_N(E_max) = E′. Das
   Gitter hat keinen Horizont, sondern einen Spiegel.
3. **Zeitabhängiger Lapse:** Er ist adiabatisch (Verlust ∝ T⁻³) mit der Phase Σ N E Δt.
4. **Exp. 1d:** Last = lokale Energiedichte, das Vakuum hat die Last null. Aktive Masse
   = passive Masse = Energie, und das dritte Gesetz gilt bis auf 1–2 %, was der
   Äquivalenzprinzip-Abweichung pro Paket entspricht. Eine bewegte Quelle zieht mit E,
   nicht mit E(1 + v²) wie in der ART.
5. **Kantenlängen:** Isotrop über Warten und Massenphase (1c). Anisotrop **nicht** über
   Warten, das sich wegmittelt, aber über richtungsabhängige Coin-Kopplungen, mit zwei
   kubischen Antwortkonstanten, exakt in linearer Ordnung. Ein GW-Interferometer
   reproduziert die ART-Phasendifferenz für k → 0.
6. **Landau-Zweige:** Die Windung von det U(k_z) ist null bei B = 0.1 und 0.05. Es gibt
   also keinen Floquet-Fluss über den Zonenrand, die Zweige gleichen sich innerhalb der
   Quasienergie aus.

## Offene Punkte

- **Quelle der ART:** Die Last e(x) ist skalar. Für E(1 + v²) und die doppelte Wirkung
  von Licht muss die Quelle den Impulsfluss enthalten, und die räumliche Metrik A muss
  mit angetrieben werden. Gesucht ist eine lokale Größe des Walkers für T_ii.
- **Feldgleichung:** Bisher Φ über die instantane 1D-Green-Funktion. Eine lokale,
  kausale Regel für Φ (und für die Coin-Gewichte h) wäre der nächste Schritt; erst
  damit breiten sich Gravitationswellen aus. Das braucht 3D (in 1D wächst das
  retardierte Feld einer statischen Quelle linear in der Zeit).
- Der Spiegel am „Horizont“: Abhängigkeit von der Energie und der Bandstruktur (im
  massiven Fall), Anteil der Konversion in andere Bänder.
- Direkter Verlauf des chiralen Zweigs über die Zone: braucht mehrere Flussquanten mit
  aufgelöster Entartung oder eine Fortsetzung, die die Landau-Entartung respektiert.
