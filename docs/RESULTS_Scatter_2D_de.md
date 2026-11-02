# Ablenkungsstöße in 2+1D

*Dateien:* `quantum_scatter_2d.py`, `quantum_scatter_2d_figs.py`
*Schritt 4 aus:* `ROADMAP_QCD_3D_de.md`
*Abbildung:* `scatter_2d.png`

---

## 1. Was hier geprüft wird — und was nicht

Schritt 3 hatte gezeigt: in 1+1D ist Zweikörper-Kontaktstreuung **integrabel**.
Sie erzeugt nur eine Phasenverschiebung, nie eine Umverteilung der Impulse.
Für einen echten Stoß braucht es mindestens zwei Raumdimensionen.

**Erwartungshaltung, ehrlich gesagt:** dieser Schritt sucht nichts, was andere
Modelle nicht vorhersagen würden. Die Interferenz identischer Teilchen,
f(θ) ± f(θ+π), ist Lehrbuchquantenmechanik (Mott-Streuung). Sie *muss*
herauskommen. Der Schritt ist ein **Test der Maschinerie**. Interessant ist
nur, was das Gitter und die Modellstruktur daran ändern — und das stellte sich
als der eigentliche Inhalt heraus.

Zur Frage nach Austauschteilchen: der Kontaktterm ist kein Gegenentwurf zum
Bosonaustausch, sondern dessen Niederenergie-Grenzfall. Ein massiver
Vermittler gibt `g²/(q² + M²)`, und für `q ≪ M` wird daraus `g²/M²` —
eine Kontaktwechselwirkung (so wie Fermis Vier-Fermion-Theorie der
ausintegrierte W-Austausch ist). Was fehlt, ist der Vermittler als eigener
dynamischer Freiheitsgrad und die Retardierung.

---

## 2. Die Reduktion auf die Relativkoordinate

Die volle Zwei-Teilchen-Amplitude in 2D wäre `(N_Plätze · 6)²` — aussichtslos.
Der Kontaktterm ist aber translationsinvariant, also ist der Gesamtimpuls Q
erhalten:

```
Psi(r1,d1,r2,d2) = e^{-i Q.r2} phi(r1 - r2, d1, d2)
```

Ein Schritt verschiebt `r -> r + dr(d1) - dr(d2)` und multipliziert mit
`e^{i Q.dr(d2)}`. Der reduzierte Zustand ist `(N_Plätze × 36)` — wenige MB.

**Verifikation (der wichtigste Test):** ein aus `phi` rekonstruierter voller
Zwei-Teilchen-Zustand wurde mit der vollen Entwicklung propagiert und mit der
reduzierten verglichen. Übereinstimmung **exakt 0.000e+00**, sowohl mit als
auch ohne Kontaktterm.

Bei Q = 0 ist der Austauschoperator einfach `(X phi)(r,d1,d2) = phi(-r,d2,d1)`;
er vertauscht mit dem Schritt auf **1·10⁻¹⁷**.

*Fallstrick:* `phi[::-1]` ist **nicht** φ(−r), sondern um einen Index
verschoben. Richtig ist die Spiegelung um den Nullpunkt, `i -> (2·c - i) mod N`.
Mit der falschen Version vertauscht X nicht mit dem Schritt. Derselbe
Off-by-one war schon in Schritt 3 aufgetreten.

---

## 3. Findet Ablenkung statt?  Ja

ε = 0.5, α = 0, Kontaktstärke U = 2, Relativimpuls k = 0.3:

| Größe | Wert |
|---|---|
| gestreutes Gewicht ‖φ_U − φ_frei‖² | **0.0617** |
| Norm erhalten | 0.946 (Rest: Randverlust) |
| Vorwärts-Peak p(2.5°)/p(177.5°) | **7.98** |

Das ist genau das, was in 1+1D unmöglich war. Die Winkelverteilung ist
vorwärts gepeakt, mit deutlicher **sechszähliger Struktur** — die Streuwelle
läuft bevorzugt entlang der Gitterachsen (in der Abbildung als Strahlenkreuz
sichtbar). Das ist der einzige genuin gittereigene Effekt.

---

## 4. Die Mott-Interferenz — vorhanden, aber gedeckelt

Boson (symmetrisiert) und Fermion (antisymmetrisiert) unterscheiden sich klar:
das Fermion hat bei **90°** ein Minimum, das Boson ein Maximum, mit etwa einem
Faktor 10 zwischen beiden Kurven.

Quantifiziert über den Kontrast `C = (p_b − p_f)/(p_b + p_f)` bei 90°
(C = 1 wäre volle Mott-Interferenz, C = 0 gar keine):

| k | \|⟨u(k)\|u(−k)⟩\| | bester gemessener Kontrast | Verhältnis |
|---|---|---|---|
| 0.30 | 0.7943 | **+0.752** | 0.95 |
| 0.45 | 0.6439 | +0.568 | 0.88 |
| 0.60 | 0.5160 | +0.401 | 0.78 |
| 0.90 | 0.3345 | +0.374 | 1.12 |

**Der Überlapp der inneren Zustände ist die Obergrenze, und die Messung
erreicht sie fast.**

### Warum es eine Obergrenze gibt

Der Bandeigenvektor ist **an den Impuls gekoppelt** — er ist ein
helizitätsartiger Zustand. Zwei gegenläufige Teilchen haben deshalb nicht
denselben inneren Zustand, und die Austauschbranche interferiert nur mit dem
Gewicht `|⟨u(k)|u(−k)⟩|`:

| α | Überlapp bei kleinem k | bei großem k |
|---|---|---|
| 0 (massiv) | 0.992 | 0.170 |
| **1/2 (Spinor)** | **4·10⁻¹⁵** | **4·10⁻¹⁵** |

Bei **α = ½ ist der Überlapp exakt null** — dort ist überhaupt keine
Austauschinterferenz möglich. Zwei gegenläufige Teilchen sind in diesem Modell
durch ihren inneren Zustand unterscheidbar, so wie zwei Teilchen
entgegengesetzter Helizität es sind.

Das ist derselbe Sachverhalt, der schon viermal zugeschlagen hatte: 3.7·10⁻¹⁶
im 2D-Zweipaket-Test, 0.009 auf FCC, in Schritt 3 beim Austauschterm, und
jetzt hier. Er verdient es, als **zentrale Eigenschaft dieser Modellklasse**
festgehalten zu werden und nicht als wiederkehrende Störung.

---

## 5. Ein Auswertungsfehler, der beinahe zum falschen Schluss geführt hätte

Der erste Messwert war ein Kontrast von nur +0.28 — und bei größerem k sogar
mit **umgekehrtem Vorzeichen** (−0.40). Das sah nach „keine Mott-Interferenz"
aus. Ursache war die Auswertung, nicht die Physik:

Der Kontrast **dephasiert mit dem Radius**, weil das Wellenpaket breit ist und
die beiden Austauschbranchen unterschiedliche Phasen aufsammeln:

| Ringradius | 13–18 | 18–23 | 23–27 | 27–32 | 32–37 | 37–41 | 41–46 |
|---|---|---|---|---|---|---|---|
| Kontrast bei 90° | **+0.752** | +0.711 | +0.579 | +0.464 | +0.370 | +0.205 | +0.172 |

Ein breiter Messring mittelt das weg. Radial aufgelöst steigt der Kontrast von
+0.28 auf **+0.75**. Alle Zahlen in Abschnitt 4 sind deshalb radial aufgelöst.

Ein zweiter Fehler derselben Art: beim Test der Spiegelsymmetrie ist der
Partner von Bin i das Bin **71−i**, nicht 72−i. Mit der falschen Zuordnung
erschien eine Asymmetrie von 29 %, mit der richtigen sind es 8.4 % (Rest:
Randverlust und endliche Ringbreite). Das Modell selbst ist exakt
spiegelsymmetrisch — Münze 1.7·10⁻¹⁶, Startzustand und freie Entwicklung
exakt 0.

---

## 6. Was Schritt 4 zeigt

| Frage | Antwort |
|---|---|
| Ablenkungsstoß in 2D? | Ja, 6.2 % gestreut, Vorwärtsverhältnis 8.0 |
| Gitterspezifisch? | Sechszähliges Strahlenmuster der Streuwelle |
| Mott-Interferenz? | Ja: Fermion-Minimum bei 90°, Faktor ~10 zum Boson |
| Volle Mott-Nullstelle? | **Nein** — gedeckelt durch \|⟨u(k)\|u(−k)⟩\| |
| Bei α = ½? | **Gar keine** Austauschinterferenz (Überlapp exakt 0) |
| Reduktion korrekt? | exakt 0.000e+00 gegen die volle Entwicklung |

---

## 7. Offene Punkte

* **Die Helizitätskopplung** ist die interessanteste offene Frage. Sie ist
  keine Näherung, sondern folgt daraus, dass der innere Index (Heading) die
  Bewegungsrichtung *ist*. Ob ein Modell mit einem vom Impuls **unabhängigen**
  inneren Freiheitsgrad (echter Spin zusätzlich zum Heading, wie im
  FCC-Modell) die volle Mott-Nullstelle liefert, ist nicht geprüft — das wäre
  der nächste sinnvolle Test.
* **Randverlust.** Bei k = 0.3 gehen 5 % der Norm über den Gitterrand
  verloren. Absorbierende Ränder oder ein größeres Gitter würden die
  Restasymmetrie von 8 % weiter drücken.
* **Retardierung und echter Austausch.** Der Kontaktterm ist der
  ausintegrierte Vermittler. Ein Stoß durch tatsächlichen Quantenaustausch
  bräuchte die Rückwirkung des Walkers auf das Eichfeld — unquenched, und das
  ist ein eigenes Projekt.
