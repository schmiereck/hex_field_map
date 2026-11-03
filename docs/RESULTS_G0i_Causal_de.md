# Kausales g₀ᵢ: Lapse und Shift als lokale Wellengleichungen

*Paper 3. Code: `src/phase_bit_s4_open11.py` und `src/phase_bit_s4_open11_figs.py`.
Daten: `data/phase_bit_s4_open11.npz`. Abbildung: `figures/phase_bit_s4_open11.png`.
Vorgänger: [`RESULTS_G0i_Gravitomagnetism_de.md`](RESULTS_G0i_Gravitomagnetism_de.md).*

Modell wie in der g₀ᵢ-Runde:
- **2O\*** (ε = 0.3), H2-Schema mit Warte- und Bleibe-Gatter, kompensierte Kopplung
  λ(N), N = N₀·exp(Φ), Mitnahme-Sprung für b.
- x-Kette mit 640 Lagen. Die Münzen kommen aus einer Tabelle in N, weil sich N jetzt in
  jedem Takt ändert.
- Speicher: wenige 100 MB, frei waren 3.3 GB.

## Frage

In der g₀ᵢ-Runde waren Φ und b instantane Green-Funktionen. Jetzt leben beide Felder
auf den Knoten und folgen lokalen Wellengleichungen (Leapfrog, Courant-Zahl 0.30):

```
Φ_tt = c_g² (Φ_xx − κ s/Δx),        s = 2e − e_M          (Runde 6)
b_tt = c_g² (b_xx − κ_b p/Δx),      p = k_s ρ_s,   κ_b = −4c′²N₀⁴κ
c_g  = c′N₀²                         (lokale Lichtgeschwindigkeit in Gitterkoordinaten, H2: N/A = N²)
```

Die Quelle wird pro Knoten durch Δx geteilt, damit der statische Grenzfall genau die
Green-Funktionssumme der Vorrunde ist. Ohne diesen Faktor waren die Felder um
Δx = 0.612 zu schwach. Runde 6 hatte ihn richtig, nur das neue Skript hatte ihn
zuerst vergessen.

**Vorhersage:** Die retardierte 1D-Lösung hat am Testort den statischen Gradienten,
eingeschränkt auf den Teil der Quelle innerhalb des Lichtkegels. Für die Kraft gilt
also F(t) = F_statisch(t)·P(|x_T − x_S| < c_g t), gewichtet mit Test- und
Quellverteilung. F_statisch(t) ist zeitaufgelöst aus dem instantanen Lauf genommen,
weil sich das Paket dort weiterentwickelt.

**Der Test:** Für parallele Lichtstrahlen heben sich Lapse- und Shift-Kraft im
statischen Fall auf. Laufen Φ und b gemeinsam mit c_g, spürt der Teststrahl **zu
keiner Zeit** etwas. Wäre b instantan und Φ retardiert, würde er abgestoßen, bevor das
Lapse-Feld ankommt. Das wäre eine akausale Signatur.

Aufbau:
- Zwei Schichten bei ±46 (Abstand D = 92, also D/c_g = 249), Breite σ = 16, N₀ = 0.8.
- Schwacher Gradient d ln N/dx = 3·10⁻⁴, damit sich ⟨k_x⟩ viel weniger ändert als k_y.
- 800 Takte (t = 400), Kraft über ⟨k_x⟩(t), Mittel über beide Kramers-Partner.

Vier Varianten:
1. beide Felder instantan
2. Φ kausal, ohne b
3. Φ kausal, b instantan („gemischt“)
4. Φ und b kausal

## Ergebnisse

Alle Werte sind ⟨k_x⟩(t) − ⟨k_x⟩(0) in 10⁻³ κ. Die statischen Kräfte stammen aus dem
instantanen Lauf.

| Fall | statisch: gesamt / Lapse / Shift | **vor Ankunft (t < 94): kausal / gemischt** | t = 400 kausal (Vorhersage) | nur Φ (Vorh.) | gemischt (Vorh.) |
|---|---|---|---|---|---|
| Licht, parallel | −0.48 / −11.39 / +10.90 | **+0.011 / +1029.5** | −57 (−73) | −1612 (−1720) | +2829 (+2639) |
| Licht, antiparallel | −22.31 / −11.39 / −10.92 | −0.17 / −1029.7 | −3169 (−3371) | −1612 (−1720) | −6052 (−6086) |
| μ = 0.03, parallel | −0.80 / −11.71 / +10.91 | **+0.011 / +1029.6** | −90 (−120) | −1668 (−1754) | +2764 (+2608) |
| μ = 0.03, antiparallel | −22.64 / −11.71 / −10.93 | −0.12 / −1029.7 | −3249 (−3394) | −1668 (−1754) | −6102 (−6125) |

Mittlere Abweichung von der Lichtkegel-Vorhersage (rms/Maximum): 0.5–2.5 % für alle
Läufe. Ausnahme ist der fast vollständig aufgehobene parallele Fall (8 bzw. 10 % eines
sehr kleinen Maximums).

- **Vor Ankunft der Felder** spürt der Test im kausalen Fall nichts (≤ 0.2 gegen 1030
  im gemischten Fall).
- **Gemischter Fall:** Ist b instantan, werden parallele Strahlen sofort abgestoßen
  (+1030 bis t = 94). Die Anziehung des Lapse kommt erst nach D/c_g an und hebt dann
  die Abstoßung auf: Die Kurve wird flach, die Abstoßung bleibt aber erhalten
  (Abb. a, c). Mit einem instantanen Shift könnte man also ein Signal schneller als
  das Lapse-Feld schicken.
- **Kausaler Fall:** Für parallele Strahlen bleibt die Kurve zu allen Zeiten nahe
  null. Bei t = 400 sind es −57 gegen −1612 für Φ allein, also 3.5 %. Das entspricht
  dem statischen Rest von 4 % (Impulsbreite bei σ = 16).
- **Shift-Anteil allein** (kausal minus „nur Φ“): Er folgt F_Shift·∫P dt wie der
  Lapse-Anteil. Beide Felder laufen also tatsächlich gemeinsam mit c_g.

**Messfallen dieser Runde:**
- **Fehlender Faktor Δx** in der Quelle der Wellengleichung: Die Felder waren um 0.612
  zu schwach.
- **Zu starker Gradient:** Bei 3·10⁻³ änderte sich ⟨k_x⟩ um 0.4, mehr als k_y = 0.2.
  Das Paket verlässt dann den linearen Bereich.
- **Offener Kettenrand:** Das Lichtpaket läuft quer auseinander (σ ≈ 36 bei t = 400)
  und erreicht bei 400 Lagen den Rand. Dann steigt die Lapse-Kraft scheinbar und die
  Shift-Kraft fällt. Abhilfe: 640 Lagen.
- **Konstante statt zeitaufgelöste statische Kraft** in der Vorhersage: Das gab
  scheinbare Abweichungen von 15–50 %.

## Antwort

1. **g₀ᵢ ist kausal:** Lapse und Shift breiten sich als lokale Wellen mit derselben
   Geschwindigkeit c_g aus. Die Testkraft folgt F_statisch × Lichtkegel-Anteil auf
   wenige Prozent.
2. **Gemeinsame Ausbreitung ist nötig:** Nur wenn Φ und b gemeinsam laufen, spüren
   parallele Lichtstrahlen zu keiner Zeit etwas. Ein instantaner Shift neben einem
   retardierten Lapse gäbe eine Abstoßung vor Ankunft des Feldes, also ein
   überlichtschnelles Signal.
3. Damit hat das 1D-Modell alle Teile der linearisierten ART-Metrik: Lapse (γ = 1,
   β = 1), räumliche Metrik und Shift, alle lokal, alle kausal.

**Was das nicht zeigt:**
- Die Quelle ist eine plötzlich eingeschaltete, ruhende Schicht. Eine echte
  Energieerhaltung zwischen Feld und Quelle (Rückwirkung, Abstrahlung) ist nicht
  geprüft.
- In 1D gibt es keine Rotation von b. Lense–Thirring und Gravitationswellen mit
  Quadrupolquelle brauchen 2D/3D.
- Dass Φ und b dieselbe Geschwindigkeit haben, ist hier vorgegeben, nicht abgeleitet.

## Offene Punkte

- **3D:** Feldgleichungen für Φ und b in 3D (1/r statt linear wachsend), danach
  Lense–Thirring und Gravitationswellen mit den anisotropen Coin-Gewichten aus Runde 5.
- **Experiment 3** (Isotropie der Kuboktaeder-Schalen) ist der natürliche erste
  3D-Schritt und prüft zugleich die Isotropie, die eine 3D-Gravitation braucht.
- Woher die gemeinsame Geschwindigkeit und der Faktor 4 kommen: eine Budget-Regel für
  die Ausbreitung der Rechenlast selbst.
