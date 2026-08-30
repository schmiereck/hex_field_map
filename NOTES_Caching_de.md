# Cachen auf dem regulären Gitter — was geht und was nicht

*Datei:* `quantum_walk_cache.py`

Die Idee: das Gitter ist regelmäßig, also wiederholen sich Berechnungen in
verschiedenen Raumbereichen. Man hasht den Randbereich, legt eine Kette ab
(„aus Hash1 folgt Hash2, der einen Gitterschritt mehr abdeckt"), berücksichtigt
Spiegelungen und Drehungen, und die Gitterpunkte tragen nur noch Referenzen.

Die Idee ist richtig — aber sie greift in drei sehr verschiedenen Graden, je
nachdem, **was** man cacht.

---

## 1. Kombinatorik: greift, und zwar dramatisch

Die Wegaufzählungen im Projekt (Windungszahl-Familien, Gürteltrick) laufen als
Tiefensuche mit (Verzweigung)^L Aufwand. Auf einem regulären Gitter hängt das
Teilproblem „wie viele Wege von *n* weiteren Schritten führen von (Ort,
Richtung) zurück zum Ursprung, und mit welchen Winkelsummen?" **nur** von
(Ort, Richtung, Restschritte) ab — nicht davon, wie man dorthin kam.

```
f(r, d, n) = sum_s  shift_s  f(r + dr(d+s), d+s, n-1)
```

Das ist genau die beschriebene Kette: ein Eintrag erzeugt den nächsten, der
einen Schritt mehr abdeckt.

### Gemessen, Dreiecksgitter (5 erlaubte Drehungen pro Schritt)

| L | Tiefensuche | memoisiert | Speedup | Tabelle | gefundene Wege |
|---|---|---|---|---|---|
| 7 | 0.035 s | 0.003 s | 10× | 725 | 370 |
| 9 | 0.646 s | 0.007 s | 88× | 1 444 | 7 454 |
| 11 | 13.393 s | 0.015 s | **917×** | 2 523 | 154 874 |
| 13 | — | 0.023 s | — | 4 034 | 3 330 646 |
| 15 | — | 0.037 s | — | 6 157 | **72 961 242** |

Die Tabelle wächst etwa linear, die Zahl der Wege exponentiell. L = 15 mit 73
Millionen Wegen dauert 37 ms; die Tiefensuche bräuchte Stunden. Gegen die
bestehende Aufzählung verifiziert: **identisch** (2D, L = 9, 110× schneller).

### Wo es einbricht: FCC, nur 2×

Auf dem FCC-Gitter wird nicht eine kleine ganze Zahl akkumuliert (die
Winkelsumme), sondern ein **Quaternion**. Die Werte der Tabelle sind dann
Wörterbücher über einer großen, praktisch dichten Menge von SU(2)-Elementen,
und sie wachsen mit. Gemessen bei L = 10: Ergebnis identisch (128 Wege mit
SU(2) = −1), aber nur **2×** schneller bei 10 911 Einträgen statt 1 444.

**Die Ersparnis hängt daran, wie grob die akkumulierte Invariante ist.** Eine
kleine diskrete Größe (Winkelsumme mod 6) cacht hervorragend; eine
kontinuierliche (Quaternion) fast nicht. In 3D lässt sich das nicht durch
Vergröbern retten: die ℤ₂-Klasse eines geschlossenen Weges ist aus Teilstücken
nicht bestimmbar, ohne das volle Quaternion mitzuführen — dasselbe Ergebnis,
das in `ROADMAP_QCD_3D_de.md` schon aufgetreten war (es gibt in 3D keine
Windungszahl).

---

## 2. Wellenfunktionen: der Hash trifft nie

Der Randbereich eines Raumgebiets trägt bei der Zeitentwicklung **kontinuierliche
komplexe Amplituden**. Zwei Gebiete haben nie bitgleiche Randdaten, also gibt
es keine Treffer. In dieser wörtlichen Form funktioniert die Idee für die
Zeitentwicklung nicht.

### Die richtige Fassung: den Operator cachen, nicht den Zustand

Auf einem **homogenen** Gitter ist die Abbildung „Rand hinein → Rand hinaus"
für ein Gebiet gegebener Form für jede Verschiebung und jedes Symmetriebild
**dieselbe**. Man cacht also den Operator, und das Verschmelzen zweier Blöcke
zu einem doppelt so großen ist die Kette. Das ist die hierarchische bzw.
rekursive Green'sche-Funktionen-Methode (und, in der Tensornetz-Sprache, ein
Baum-Tensornetz).

Aufwand: ein Gebiet der Kantenlänge L hat eine Randdimension O(L·n_intern),
das Verschmelzen kostet also O((L·n)³). Es lohnt sich, wenn man **viele**
Zeitschritte, eine Resolvente oder wiederholte Lösungen mit verschiedenen
Quellen braucht. Für schlichtes Zeitschreiten lohnt es nicht — das ist schon
O(N) pro Schritt.

### Das Projekt nutzt bereits die stärkste Form davon

Wo das Gitter homogen ist, ist der **Transfermatrix-Zugang genau diese Idee im
Extremfall**: statt jedes Raumgebiets nur eine 6×6-Matrix pro k. Und die
**Relativkoordinaten-Reduktion** in `quantum_scatter_2d.py` ist derselbe
Gedanke für zwei Teilchen — sie macht aus (N_Plätze·6)² gerade N_Plätze·36,
indem sie ausnutzt, dass nur der *Abstand* zählt, nicht die absolute Lage.
Verifiziert auf 0.000e+00 gegen die volle Entwicklung.

Interessant wird das Cachen also genau dort, wo die Homogenität **gebrochen**
ist: Magnetfeld, Eichhintergrund, Wechselwirkung. Dort gibt es keine
Transfermatrix mehr, und die hierarchische Blockmethode wäre der richtige
nächste Schritt.

---

## 3. Symmetrien: real, aber ein konstanter Faktor

Die Punktgruppe des Dreiecksgitters hat 12 Elemente (D6), die des FCC-Gitters
48 (O_h). Kanonisiert man ein Gebiet unter der Untergruppe, die den
Bezugspunkt festlässt, schrumpft die Tabelle um bis zu diesen Faktor.

In der Wegaufzählung halbiert schon die Spiegelung durch die Startrichtung die
Tabelle: sie bildet (Ort, Richtung) auf das Spiegelbild ab und negiert die
Winkelsumme. Das ist ein Faktor, kein Skalenverhalten — nützlich, aber es
ändert die Komplexitätsklasse nicht.

---

## 4. Fazit

| Was | Greift die Idee? | Gemessen |
|---|---|---|
| Wegaufzählung, kleine diskrete Invariante | **ja, dramatisch** | 917× bei L=11 |
| Wegaufzählung, Quaternion-Invariante | kaum | 2× |
| Wellenfunktion, Rand-Hash | **nein** | Amplituden sind kontinuierlich |
| Operator pro Gebietsform (homogen) | ja, = hierarchische Methode | im Projekt als Transfermatrix schon genutzt |
| Symmetriereduktion | ja, konstanter Faktor | bis 12 (2D) bzw. 48 (3D) |

Der Kern: **cachen kann man, was diskret und wiederholt ist.** Zustände sind
es nicht, Operatoren und kombinatorische Invarianten schon.
