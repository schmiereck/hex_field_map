# Zwei-Teilchen-Raum: Ausschließlichkeit, Statistik, echte Stöße

*Dateien:* `quantum_two_particle.py`, `quantum_two_particle_figs.py`
*Schritt 3 aus:* `ROADMAP_QCD_3D_de.md`
*Abbildung:* `two_particle.png`

---

## 1. Was bisher fehlte

Alle bisherigen Modelle waren **lineare Einteilchen-Theorien**. Zwei „Teilchen"
waren zwei Wellenpakete in derselben Wellenfunktion — sie konnten sich nur
durchdringen und interferieren (`RESULTS_Turning_2D_de.md`, Abschnitt 8).
Ausschließlichkeit braucht einen echten Zwei-Teilchen-Zustand.

---

## 2. Teil A — Austauschstatistik, exakt und billig

Für ein **wechselwirkungsfreies** Paar faktorisiert alles: der Zustand

```
Psi(x1,x2) = [ a(x1) b(x2) + e^{i theta} a(x2) b(x1) ] / Norm
```

entwickelt sich, indem man die beiden Einteilchen-Orbitale einzeln
entwickelt. θ = 0 ist ein Boson, θ = π ein Fermion. Kein großes Objekt muss
gespeichert werden.

Mit x = (r, s) und über den inneren Index summiert:

```
P(r1,r2) = 1/2 [ rho_a(r1) rho_b(r2) + rho_a(r2) rho_b(r1)
                 + 2 Re( e^{-i theta} K(r1) K*(r2) ) ]
K(r) = sum_s a(r,s) b*(r,s)
```

Bei Koinzidenz: `P(r,r) = rho_a rho_b + cos(theta) |K|²`. Cauchy-Schwarz gibt
`|K|² ≤ rho_a rho_b`, also ist die Fermion-Koinzidenzdichte nicht-negativ und
verschwindet **exakt**, wo die beiden inneren Zustände parallel sind.

### Gemessen (3+1D FCC, Spin-½-Orbitale, ε = 0.1)

| Test | Fermion | Boson |
|---|---|---|
| Pauli-Prinzip: zwei Teilchen im **selben** Orbital, max \|P(x₁,x₂)\| | **1.9·10⁻⁹** (= 0, einfache Genauigkeit) | 1.9·10⁻² |
| Koinzidenzsumme bei t=0 (identische innere Zustände) | **1.8·10⁻¹³** | 2.30·10⁻⁵ |
| dieselbe, unterscheidbar | 1.15·10⁻⁵ | — |

Bei t = 0 ist das Verhältnis Fermion/unterscheidbar exakt **0** und
Boson/unterscheidbar exakt **2** — vollständige Pauli-Vertiefung bzw.
vollständiges Bunching. Im Lauf der Zeit werden die inneren Zustände an einem
gegebenen Ort verschieden (das eine Paket kam von links, das andere von
rechts), und die Vertiefung wird partiell: das Verhältnis pendelt sich bei
etwa 0.65 (Fermion) und 1.35 (Boson) ein.

Die Paarkorrelationskarte zeigt es direkt: entlang der Diagonale x₁ = x₂ hat
das Fermion ein Loch, das Boson einen Grat.

### Eine Falle, die man kennen muss

**Der Austauschterm braucht überlappende innere Zustände.** Zwei frontal
aufeinander zulaufende Pakete gleicher Energie haben nahezu orthogonale innere
Zustände — gemessen 0.009 auf FCC und 3.7·10⁻¹⁶ im 2D-Modell — und zeigen
deshalb **überhaupt keinen** Austauscheffekt. Das ist eine Eigenschaft des
Modells, nicht der Statistik. Alle Messungen oben benutzen deshalb entweder
ruhende, ineinander zerfließende Pakete (innerer Überlapp 1.0) oder eine
Kreuzung bei 60° (0.75).

---

## 3. Teil B — ein echter Stoß

Austauschstatistik ist noch keine Wechselwirkung. Für einen echten Stoß
braucht es einen Term, der wirkt, wenn sich die Teilchen treffen — und dafür
die **volle** Zwei-Teilchen-Wellenfunktion. In 1+1D ist das bezahlbar: mit
N = 201 Plätzen und 2 Richtungen sind das 1.6·10⁵ komplexe Zahlen. Modell ist
der unitäre Feynman-Schachbrett-Walk, also die 1+1D-Fassung derselben Münze:
`C = expm(i·ε·σ_x)`.

Kontaktterm: Phase `exp(iU)`, wenn beide Teilchen am selben Platz sind.

### Was ich dabei zuerst falsch gemessen habe

Mein erstes Observable war „Wahrscheinlichkeit, dass beide auf derselben Seite
landen". Es kam für **alle** U und alle Statistiken exakt derselbe Wert heraus
(0.001664). Das sah nach einem Fehler aus, war aber Physik:

* Bei identischen Teilchen sind Reflexion und Transmission **nicht
  unterscheidbar** — beide Ausgänge sind dieselbe Konfiguration.
* In 1+1D ist Zweikörper-Kontaktstreuung **integrabel**: die Impulsverteilung
  kann sich gar nicht ändern, es entsteht nur eine Phasenverschiebung.

Gemessen: bei U = 2.0 ändert sich der Zustand erheblich
(`|⟨Ψ₀|Ψ_U⟩|` fällt von 1 auf 0.48), aber die Ortsverteilung um lediglich
`max|ΔP(x₁,x₂)| = 2.4·10⁻⁵`. Das ist die richtige Aussage, und das richtige
Observable ist die **Streuphase**, nicht die Umverteilung.

### Der scharfe Befund: eine Wechselwirkung, die Fermionen nicht spüren

Ein Kontaktterm, der auf der **vollen** Koinzidenz (x,d) = (x,d) wirkt:

| U | 1.0 | 2.5 | 4.0 |
|---|---|---|---|
| Boson, ‖Ψ(U) − Ψ(0)‖ | 1.01·10⁻² | 2.23·10⁻² | 6.78·10⁻² |
| **Fermion** | **7.0·10⁻¹⁷** | **6.0·10⁻¹⁷** | **7.6·10⁻¹⁷** |

Das Pauli-Prinzip verbietet (x,d) = (x,d), also kann diese Wechselwirkung
Fermionen **buchstäblich nicht berühren** — auf Maschinengenauigkeit, bei
jeder Stärke.

---

## 4. Zwei-Körper-Bindungszustände

Ein Streulauf kann einen Bindungszustand nur finden, wenn der Anfangszustand
mit ihm überlappt — und tatsächlich fiel das Nahdiagonal-Gewicht in allen
Zeitläufen auf ~0.001 zurück. Das beweist nichts. Der saubere Test ist die
**Diagonalisierung im Relativkoordinaten-Sektor**:

```
Psi(x1,d1,x2,d2) = e^{-i Q x2} phi(r, d1, d2),   r = x1 - x2
```

Ein Schritt verschiebt `r -> r + s(d1) - s(d2)` und multipliziert mit
`e^{iQ s(d2)}`. Der Operator ist (4·N_r)-dimensional und exakt
diagonalisierbar.

**Verifikation:** das Spektrum der vollen Zwei-Teilchen-Entwicklung (N=9,
Dimension 324) stimmt mit der Vereinigung aller reduzierten Sektoren auf
**5.2·10⁻¹⁵** überein.

### Ergebnis (Q = 2π·17/120, ε = 0.35)

Lokalisierung ⟨|r|⟩ des am stärksten lokalisierten Zustands (freier Wert ≈ 24):

| Kontakt wirkt auf | Boson | Fermion |
|---|---|---|
| **Platz** (x₁ = x₂, beliebiges d) | 0.85 … 17.9 (bindet) | 1.78 … 23.9 (bindet) |
| **volle Koinzidenz** (x,d) = (x,d) | 1.15 … 17.8 (bindet) | **23.93 … 23.93** (konstant) |

Ein Platz-Kontakt bindet **beide** — der Richtungsindex gibt den Fermionen
Raum, am selben Platz zu sein, genau wie der Spin es bei einem Singulett-Paar
tut. Ein Koinzidenz-Kontakt bindet **nur Bosonen**; für Fermionen bewegt sich
die Lokalisierung bei keinem U vom freien Wert weg.

### Eine Nebenbemerkung zur Methode

Der Austauschoperator im Relativbild ist
`(Xφ)(r,d₁,d₂) = e^{−iQr} φ(−r,d₂,d₁)`. Er enthält explizit `e^{−iQr}` und ist
auf einem Ring von N_r Relativpositionen nur für **quantisiertes**
Q = 2πm/N_r eindeutig. Mit Q = 0.5 vertauscht er nicht mit dem Schritt
([M,X] = 1.5); mit quantisiertem Q schon (9·10⁻¹⁵). Meine erste Klassifikation
war deshalb wertlos.

---

## 5. Teil C — das Farbsingulett

Ein Quark wird von `U` transportiert, ein Antiquark von `U*`. Die Haar-Mittelung

```
<U_ac U*_bd> = delta_ab delta_cd / N
```

projiziert jede Farbwellenfunktion auf ihren Spuranteil:
`chi -> delta · tr(chi)/N`.

| Farbzustand | überlebende Norm nach Eichmittelung |
|---|---|
| **Singulett** `chi = 1/√3` | **1.000000** |
| Oktett (spurlos) | 0.003467 (reines Sampling-Rauschen) |

(Numerische Mittelung über 60 000 Haar-Elemente; `⟨U ⊗ U*⟩` trifft
`δδ/N` auf 2.9·10⁻³.)

**Nur Farbsinguletts überleben die Eichmittelung.** Ein farbiger Zustand hat
eine exakt verschwindende eichgemittelte Amplitude und kann deshalb nicht als
asymptotischer Zustand propagieren. Das ist die Zwei-Teilchen-Fassung dessen,
was `RESULTS_SU3_MC_de.md` als Flächengesetz gemessen hat.

---

## 6. Was Schritt 3 zeigt

| Frage | Antwort |
|---|---|
| Ausschließlichkeit ohne Wechselwirkung? | Ja — Pauli-Loch in der Paarkorrelation, exakt |
| Zwei Fermionen im selben Zustand? | Ψ ≡ 0 auf Maschinengenauigkeit |
| Bunching vs. Antibunching? | Verhältnis 2 : 0 bei vollem innerem Überlapp |
| Echter Stoß in 1+1D? | Nur Phasenverschiebung — Integrabilität |
| Wechselwirkung, die Fermionen nicht spüren? | Ja, Koinzidenz-Kontakt: 6·10⁻¹⁷ |
| Gebundene Paare? | Ja, exakt nachgewiesen; nicht für Fermionen bei Koinzidenz-Kontakt |
| Farbsingulett? | Einziger Zustand, der die Eichmittelung überlebt |

---

## 7. Offene Punkte

* **Deflexion braucht ≥ 2 Raumdimensionen.** In 1+1D ist Kontaktstreuung
  integrabel; um einen Stoß zu sehen, der Impulse *umverteilt*, muss die volle
  Zwei-Teilchen-Rechnung nach 2D oder 3D. Der Speicherbedarf skaliert wie
  (N_Plätze·n_intern)² — in 2D auf einem 25×17-Gitter wären das schon 100 MB,
  in 3D ist es ohne Relativkoordinaten-Reduktion aussichtslos. Die Reduktion
  auf feste Gesamtimpulse (hier für 1+1D gebaut und verifiziert) ist der Weg.
* **Spin-Statistik ist Eingabe, nicht Ergebnis.** Das Modell erzwingt Spin ∈
  {0, ½} (Schritt 2), aber die Zuordnung „Spin 0 ↔ symmetrisch, Spin ½ ↔
  antisymmetrisch" wird hier von Hand gesetzt. Ein Gittermodell kann das
  Spin-Statistik-Theorem nicht herleiten.
* **Anyonen.** In 2D erlaubt die Topologie ein kontinuierliches θ. Der Ansatz
  `Ψ = [a(1)b(2) + e^{iθ}a(2)b(1)]` ist für jedes θ ein konsistenter
  Zwei-Teilchen-Zustand, aber ein **echtes** Anyon braucht Zopf-Struktur
  (die Phase muss von der Windung der Relativkoordinate abhängen), und das ist
  hier nicht implementiert.
* **Farbsingulett dynamisch.** Bisher nur die Eichmittelung. Der nächste
  Schritt wäre ein q q̄-Paar mit verbindender Wilson-Linie im MC-Ensemble aus
  Schritt 1 — dann fällt seine Amplitude mit der Stringspannung, und man sieht
  Confinement aus der Teilchenperspektive.
