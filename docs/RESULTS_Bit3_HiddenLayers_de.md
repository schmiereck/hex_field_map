# Bit-Test 3: Verborgene Gitterebenen, exakte Bits und die 3D-Sicht

Stand: 8. November 2026. Ausgangsstand: `ffedd18`.

Skript: [`../src/phase_bit_s4_open22.py`](../src/phase_bit_s4_open22.py)
Ergebnisse: [`../data/phase_bit_s4_open22.json`](../data/phase_bit_s4_open22.json)
Lauf: `python src/phase_bit_s4_open22.py` (hier unter einer Sekunde).
Vorgänger: [Bit-Test 2](RESULTS_Bit2_Gradient_de.md).

## Frage und Ergebnis

Kann eine vollständig endliche, reversible höhere Gitterebene eine niedrigere
Sicht tragen, deren scheinbare Übergänge von verborgenen Zuständen abhängen,
ohne dass die Grundregel Übergangswahrscheinlichkeiten berechnet?

**EXAKT / positiv:** Ein klassisches D4-Gittergas mit 24 Bits pro Knoten lässt
sich aus lokalen bijektiven Bit-Gattern und Transport aufbauen. Teilchenzahl
und vierdimensionaler Gitterimpuls bleiben exakt erhalten. Ausblenden der vierten
Raumkoordinate ergibt eine 3D-Sicht, die nicht deterministisch geschlossen ist:
gleiche sichtbare Zustände können verschiedene sichtbare Folgezustände haben.

**EXAKT / positiv, separater Minimaltest:** Ein rückgekoppelter Drei-Bit-Kreis
hat unter dem stationären Gleichgewicht aller acht Mikrozustände eine sichtbare
Bitfolge, die keine Markov-Kette erster Ordnung ist. Alle bedingten Anteile folgen
aus endlichem Zählen, ohne Zufallsentscheidung im Update.

**Begrenzung:** Weder eine massive Dispersion noch eine Gravitationsdynamik,
Quanteninterferenz, Born-Regel oder Lorentz-Signatur wurde damit hergeleitet.
Das D4-Gittergas ist ein neuer klassischer Architekturtest, keine unmittelbare
Bit-Implementierung des bisherigen komplexwertigen 2O*-Walkers. Der Parkmechanismus
aus Bit-Test 2 wurde nicht durch eine funktionierende alternative Eigenzeitregel ersetzt.

## 1. Welche Dimension ist Zeit?

In diesem Versuch sind `(x,y,z,w)` **vier Raumkoordinaten**; ein separater diskreter
Tick beschreibt die zeitliche Entwicklung. Die 3D-Sicht blendet `w` aus. Ein
alternativer Ansatz mit `w` als Zeit wäre eine **3+1D-Ereignisgeometrie** und müsste
eigene kausale Übergangsregeln bekommen. Beide Deutungen dürfen nicht vermischt werden.

Eine endliche kompakte Zusatzrichtung kann immer als interne lokale Registerstruktur
über dem sichtbaren Gitter umgeschrieben werden. Die räumliche Interpretation ist
dann eine zusätzliche Lokalitäts- und Nachbarschaftsstruktur; sie vermehrt nicht
automatisch die Ausdrucksmacht gegenüber denselben Registern. Ein Nutzen höherer
Dimensionen muss sich aus einfacher werdenden Regeln, Symmetrien oder Messgrößen ergeben.

## 2. Exakter Dimensionsvergleich: mehr Dimensionen sind nicht automatisch isotroper

Untersucht wird die ungewichtete Schale nächster Nachbarn, nicht die gesamte Dynamik.
Für D_n sind dies alle Vektoren mit genau zwei Einträgen ±1. E8 wird durch ganzzahlige
Koordinaten dargestellt: 112 Vektoren mit zwei Einträgen ±2 sowie 128 Vektoren mit
acht Einträgen ±1 und gerader Anzahl negativer Vorzeichen. Das sind die mit zwei
multiplizierten kanonischen E8-Wurzeln; alle haben dieselbe Länge.

Für eine rotationsinvariante vierte Momentordnung muss

\[
\frac{\sum_v v_1^4}{\sum_v v_1^2v_2^2}=3
\]

gelten. Im Skript wird nicht nur dieses notwendige Verhältnis geprüft, sondern jede
unabhängige Komponente des zweiten, vierten und sechsten Momenttensors, mit exakten
rationalen Referenzwerten für die Kugel. Für E8 sind das 36, 330 und 1716 Komponenten.

| Schale | Dimension | Nachbarn | Verhältnis | Ordnung 2 | Ordnung 4 | Ordnung 6 |
|---|---:|---:|---:|---|---|---|
| D3 / FCC | 3 | 12 | 2 | isotrop | nicht isotrop | nicht isotrop |
| D4 | 4 | 24 | 3 | isotrop | isotrop | nicht isotrop |
| D5 | 5 | 40 | 4 | isotrop | nicht isotrop | nicht isotrop |
| D8 | 8 | 112 | 7 | isotrop | nicht isotrop | nicht isotrop |
| E8 | 8 | 240 | 3 | isotrop | isotrop | isotrop |

**EXAKT:** Für D_n ist das Verhältnis `n-1`; die vierte Ordnung ist deshalb innerhalb
dieser Familie genau bei n=4 isotrop. E8 bietet bis zur sechsten Ordnung eine stärkere
Schalensymmetrie. Ein allgemeines 8D-Gitter besitzt diesen Vorteil nicht automatisch.

**Keine Folgerung:** Dies ist weder eine Auswahl der Raumdimension der Natur noch ein
Nachweis relativistischer Dynamik. Insbesondere bricht die im folgenden Test bewusst
einfach gewählte Reihenfolge der Kollisionsgatter die vollständige D4-Schalensymmetrie.
Isotropie der Schale darf nicht als Isotropie dieses Automaten ausgegeben werden.

## 3. D4-Mikrodynamik: Bits, Kollision, Transport

Physische Knoten erfüllen `x+y+z+w = 0 mod 2`. Die periodische Box hat vier Zellen pro
Koordinatenrichtung: 128 physische Knoten mit jeweils 24 gerichteten Besetzungsbits,
insgesamt 3072 dynamische Bits. Die im Array mitgespeicherten ungeraden Koordinatensummen
sind leere technische Auffüllplätze und bleiben leer.

Ein Bit `n(x,d)` bedeutet ausschließlich besetzt oder leer. Es gibt keine lokale
Fließkommaamplitude, keinen reellen Coin-Winkel, keine Zufallsauswahl und keinen
anwachsenden lokalen Integer. Gitterkoordinaten und Routingtabellen sind feste Geometrie.

Die 24 Richtungen werden zu zwölf Gegenrichtungspaaren geordnet. Ein lokales Gatter
betrachtet drei solcher Paare, also sechs Bits:

1. Sind alle Paare entweder `00` oder `11`, werden die drei Paare zyklisch versetzt.
2. Enthält eines der Paare `01` oder `10`, bleibt das gesamte Sechserregister unverändert.

Beispiele:

```
11 00 00 -> 00 11 00 -> 00 00 11 -> 11 00 00
11 11 00 -> 00 11 11 -> 11 00 11 -> 11 11 00
10 00 11 -> 10 00 11
```

Die Bedingung bleibt durch die Rotation erhalten; die inverse Rotation ist daher
die eindeutige Rückwärtsregel. Das Gatter ist nicht eine feste Permutation einzelner
Kanalbits: Welche Bits bewegt werden, hängt vom gemeinsamen Besetzungszustand ab.

Zwei überlappende Dreierzerlegungen der zwölf Achsen koppeln die Gruppen miteinander.
Anschließend wandert jedes Richtungsbit genau eine D4-Kante entlang seiner Richtung.
Ein Rückwärtstick macht zuerst den Transport rückgängig und führt dann die inversen
Kollisionsgatter in umgekehrter Reihenfolge aus. Die Reihenfolge ist eine feste Regel,
kein dynamisch wachsender Speicher und keine Zufallsvariable.

### Erhaltung und Reversibilität

Jedes volle Gegenrichtungspaar hat Impuls null. In den unveränderten Fällen bleibt
auch jeder nichtverschwindende Paarimpuls erhalten. Daher bewahrt jedes Gatter
Teilchenzahl und Gitterimpuls. Transport bewahrt beide ebenfalls.

- **EXAKT:** Alle 64 lokalen Gatterzustände wurden vollständig geprüft: Bijektion,
  korrekte Inverse, gleiche Teilchenzahl und gleiche gerichtete Paardifferenzen.
- **EXAKT, analytisch:** Komposition dieser lokalen Bijektionen mit bijektivem
  Transport ergibt eine globale Bijektion für jeden gültigen Gesamtzustand.
- **NUMERISCH, bitgenau:** 32 zusätzliche dicht besetzte Testzustände, je 16 Vorwärts-
  und 16 Rückwärtsticks: vollständige Wiederherstellung aller Bits. Während der 512
  Vorwärtsticks: Teilchenzahlfehler 0, Impulsfehler 0, keine Belegung ungültiger Knoten.

Der Zufallsgenerator erzeugt nur externe Prüfeingaben. In der Mikrodynamik gibt es ihn nicht.
Eine positive Teilchenzahl ist hier eine konservierte Größe; sie wird nicht bereits als
physikalische Energie des bisherigen Walkers identifiziert.

## 4. Gleiche 3D-Sicht, verschiedene Zukunft

Die Projektion ist

\[
C(x,y,z,d)=\sum_w n(x,y,z,w,d).
\]

Die Richtung d bleibt sichtbar. Verloren geht ausschließlich die Zuordnung zu w.
Damit ist der Test stärker als bloßes Ausblenden der Geschwindigkeit. Die Summen sind
exakte Beobachtungsgrößen; der mikroskopische Knoten speichert weiterhin nur seine Bits.
Die Punktmenge dieser Koordinatenprojektion ist Z³, nicht die einzelne FCC-Schicht.
Für eine exakt FCC-basierte Sicht müsste eine andere Schnitt-/Beobachtungsregel gewählt werden.

Zwei entgegengesetzt gerichtete Besetzungen liegen bei denselben sichtbaren Koordinaten.
Ihre verborgenen Positionen können jeweils w=0 oder w=2 sein. Es gibt genau vier Anordnungen:

| w des ersten Bits | w des zweiten Bits | lokale Begegnung am Anfang |
|---:|---:|---|
| 0 | 0 | ja |
| 0 | 2 | nein |
| 2 | 0 | nein |
| 2 | 2 | ja |

Alle vier haben **exakt dieselbe** anfängliche 3D-Sicht C. Nach einem Tick gibt es
zwei verschiedene sichtbare Folgezustände, jeweils aus zwei Mikrozuständen.
Bei gleicher Gewichtung dieser vier untersuchten Anfangszustände wären die Anteile
`2/4` und `2/4`. Kein Knoten muss eine Wahrscheinlichkeit `0,5` berechnen.

Als Kontrolle wurde die Kollision weggelassen: Reiner gerichteter Transport ist unter
derselben Projektion geschlossen. Erst die verborgene Gleichzeitigkeit/Ko-Lokalität
der kollidierenden Bits macht einen Unterschied. Es wurde also nicht einfach eine
sichtbare Geschwindigkeit vergessen und deren Wirkung als neue Physik ausgegeben.

**EXAKTE Folgerung:** Es existiert keine deterministische Regel F allein für C, die
für alle Mikrozustände `C(T(s)) = F(C(s))` erfüllt. Verborgene Korrelationen tragen
zusätzliche Information über die Zukunft.

**Wichtige Begrenzung:** Das beweist für sich allein noch nicht, dass unter jedem
speziellen Anfangsensemble eine stochastische Markov-Beschreibung unmöglich ist.
In den 252 ausgewerteten Drei-Zeit-Historien dieser vier D4-Anfangszustände wurde kein
zusätzlicher bedingter Markov-Gegenbeleg gefunden (`first_order_memory_witness = null`).
Nichtgeschlossene deterministische Projektion und statistische Nicht-Markovianität
sind verschiedene Aussagen. Letztere wird im separaten folgenden Minimaltest gezeigt.

## 5. Exakter Minimaltest für Rückwirkung und Gedächtnis

Ein sichtbares Bit a und zwei verborgene Bits b,c erhalten die Regel:

1. Falls a=1: vertausche b und c (Fredkin-Gatter).
2. Versetze `(a,b,c)` zyklisch nach `(c,a,b)`.

Das sichtbare Bit steuert somit die verborgene Weiterleitung; einer der verborgenen
Werte wird im nächsten Schritt sichtbar. Dies ist eine vollständig definierte
Rückkopplung. Alle acht Mikrozustände wurden auf Bijektivität, Inverse und Erhaltung
der Einsenanzahl geprüft. Der Test ist ein eigenständiger kleiner Kreis, kein bereits
in den D4-Walker integriertes Zusatzgatter.

Die Gleichgewichtung aller acht Zustände bleibt unter der Permutation stationär.
Es werden alle Zustände einzeln fortgeschrieben, keine Trajektorien ausgelost.
Für das beobachtete Bit Y_t=a_t ergeben sich exakt:

| Bedingung | Fälle mit Y₂=1 | Fälle insgesamt | Anteil |
|---|---:|---:|---:|
| Y₁=1 | 2 | 4 | 1/2 |
| Y₀=0 und Y₁=1 | 0 | 2 | 0 |
| Y₀=1 und Y₁=1 | 2 | 2 | 1 |

**EXAKT:** Die Kenntnis der Vergangenheit verändert die Vorhersage trotz desselben
aktuellen sichtbaren Bits. Die beobachtete stationäre Folge ist keine Markov-Kette
erster Ordnung. Die gesamte Mikrodynamik bleibt deterministisch und endlich.

Das beweist ein Architekturprinzip und ist keine neue Behauptung über fundamentale
Quantenzufälligkeit. Insbesondere entstehen dadurch nicht automatisch komplexe
Amplituden, destruktive Interferenz oder die Born-Regel.

## 6. Wie zusätzliche 3D-Regeln zurückwirken dürfen

Eine Sicht `C=f(s)` ist zunächst eine Beobachtung desselben Systems, keine zweite
unabhängige Welt. Zusätzliche sichtbare Regeln brauchen eine definierte Umsetzung
auf dem Mikrozustand. Eine solche Umsetzung sollte

- aus reversiblen lokalen Gattern auf s bestehen;
- die Erhaltungsgrößen ausdrücklich prüfen;
- sagen, welche Mikrozustände trotz gleicher Sicht verschieden behandelt werden;
- verworfene Information in endlichen Zuständen erhalten statt Werte zu überschreiben.

Der Drei-Bit-Kreis erfüllt dies im Kleinen. Im D4-Gas wechseln die lokalen Kollisionen
bereits zwischen Richtungen mit und ohne w-Komponente: im Beispiel wechselt deren
Besetzungszahl von 2 nach 0. Das ist eine Kopplung der Richtungssektoren, keine zusätzliche
hergeleitete 3D-Feldgleichung. Eine unbegrenzt dissipative Sicht kann außerdem nicht
auf Dauer ihre gesamte Verlustinformation in einem festen kleinen Hilfsregister sammeln.

Für eine Hierarchie 8D -> 5D -> 4D -> 3D muss jede Stufe ihre Projektion und die erhaltenen
Gedächtnisvariablen benennen. Eine immer größere verborgene Ebene erklärt noch nicht,
warum die sichtbare Dynamik einfacher wird. Eine feste Anzahl kompakter Ebenen ist
ein überprüfbarer erster Ansatz; eine unbegrenzte Hierarchie wird hier nicht postuliert.

## 7. Konsequenzen für das Forschungsprogramm

**Weiterverfolgbar:** Endliche Mikrozustände plus verborgene Korrelationen sind eine
konkrete Alternative dazu, reelle Übergangswahrscheinlichkeiten als Grundregel zu
speichern. Kontinuierliche Analysegrößen dürfen eine effektive Beschreibung sein,
ohne dass die Mikrodynamik mit unendlich genauen Zahlen rechnet. Auch umgekehrt beweist
das Auftreten reeller Zahlen in einer Theorie noch keine solche Rechenweise der Natur.

**Noch offen:** Eine feste Einteilchen-Kanalpermutation mit Translation erzeugt weiter
die bekannten affinen Eigenphasenzweige. Mehr verborgene Routingkanäle allein lösen
dieses frühere Problem nicht. Das nichtlineare Mehrteilchengatter verlässt diese enge
Klasse; eine massive kollektive Mode muss nun aber erst wirklich nachgewiesen werden.
Im Einteilchensektor dieses neuen Gases ist die Kollision wirkungslos: dort gibt es
ausdrücklich noch keine erzeugte Masse.

### Nächste konkrete Tests

1. **Kollektive Moden statt einzelner Bahnen:** Auf einem fest spezifizierten besetzten
   Hintergrund eine kleine Störung verfolgen. Korrelationsfunktionen aus exakten
   Besetzungszählungen bilden; erst anschließend Fourieranalyse. Verlange einen
   langlebigen Zweig mit Lücke, verschwindender Gruppengeschwindigkeit bei k=0 und
   positiver Krümmung. Dämpfung, Floquet-Faltung und normale Gaskinetik getrennt ausweisen.
2. **Regelsymmetrie:** Die willkürliche Achsenreihenfolge durch einen kleinen reversiblen
   lokalen Orientierungszustand oder ein äquivariantes Kollisionsgesetz ersetzen.
   D4-Momentsymmetrie allein genügt nicht. Der Orientierungszustand zählt zum Bitbudget.
3. **Eigenzeit als Ereignisregel:** Erst eine endliche reversible Regel für unterschiedliche
   lokale Ereignisraten und gemeinsame Übergaben konstruieren. Der heutige Transport
   läuft synchron und löst das Parkstauproblem von Bit-Test 2 nicht.
4. **D4 gegen E8:** Nur bei identischer Diagnostik und benanntem Ressourcenbudget
   vergleichen. E8 hat einen exakten geometrischen Vorteil bis Ordnung sechs, aber
   240 statt 24 Nachbarkanäle. Dieser Aufwand muss einen dynamischen Nutzen liefern.
5. **Tensorfeld:** Später die Herkunft der Bindungsfeldgleichung aus solchen Regeln
   prüfen. Die richtigen linearen GW-Polarisationen bei vorgegebenen Gleichungen
   aus GW3 sind ein anderer Nachweis.

## 8. Literatur und Einordnung

Die Begriffe und Methoden sind nicht neu; neu in diesem Arbeitsstand ist ihre konkrete
Zusammenführung mit den vorliegenden Bit- und Projektionsdiagnosen. Keine Neuheitsbehauptung.

- Frisch, Hasslacher, Pomeau (1986), *Lattice-Gas Automata for the Navier-Stokes Equation*:
  deterministische boolesche Gittergase als mikroskopische Modelle effektiver Kontinuumsphysik.
  https://doi.org/10.1103/PhysRevLett.56.1505
- Chorin, Hald, Kupferman (2000), *Optimal prediction and the Mori-Zwanzig representation
  of irreversible processes*: systematische Berücksichtigung verborgener Variablen
  und Gedächtnisterme in reduzierten Beschreibungen.
  https://doi.org/10.1073/pnas.97.7.2968
- Nebe/Sloane, Gitterkatalog D4 und E8:
  https://www.math.rwth-aachen.de/homes/Gabriele.Nebe/LATTICES/D4.html
  und https://www.math.rwth-aachen.de/homes/Gabriele.Nebe/LATTICES/E8.html
- Elze (2014), *Action principle for cellular automata and the linearity of quantum mechanics*:
  eine andere Linie mit ganzzahligen Hamilton-Automaten. Ganzzahligkeit allein erfüllt
  unser zusätzliches Ziel fester kleiner lokaler Bitbudgets noch nicht.
  https://doi.org/10.1103/PhysRevA.89.012111
