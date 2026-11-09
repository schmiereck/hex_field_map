# Bit-Test 14 – Expandierende 4D-Schalen als holistische Zustandsbeschreibung

Stand: 9. November 2026. Bezug: Bit-8-Kollisionsregel, Bit-10-Engine und Bit-13-Cachebefund. **Neue Diagnose, keine veränderte Mikrodynamik.**

## Hypothese

Eine über mehrere 4D-Knoten verteilte Anregung könnte als wenige expandierende
Kugeloberflächen mit optionalem Oberflächenmuster beschrieben werden. Diese
„holistische“ Kodierung wäre nur dann rechnerisch hilfreich, wenn sich die Zahl
der Komponenten und ihrer Freiheitsgrade unter echten Kollisionen klein halten
ließe. Zunächst prüfen wir genau diesen ersten Aspekt, ohne ihn vorauszusetzen.

## Exakter Versuch und observierte Größe

Der unveränderte reversible 24-Bit-D4-Gas-Kern wird mit der symmetrischen
Bit-8-Kollision vorwärts entwickelt. Für einen Ausgangs- und einen Kontrolllauf
wird an jedem physischen Knoten nur die Differenz der **Teilchendichten**
`delta rho(x,t)=popcount(s_pulse(x,t))-popcount(s_background(x,t))`
betrachtet. Das ist **kein** exakter 24-Bit-Zustand und keine Quantenamplitude.

Zwei Präparationen: (A) 24 anfängliche Richtungsbits an einem Knoten in einem
sonst leeren Gitter; (B) dieselbe lokale Auffüllung auf einer reproduzierbar
zufällig halbgefüllten Belegung (12 von 24 Bits pro Knoten). Als Negativkontrolle
wird bei gleicher Anzahl zusätzlich gesetzter Bits dieselbe Anregungsmenge
räumlich auf verschiedene Knoten verteilt. Diese Kontrolle besitzt nicht
notwendigerweise identische Impulskomponenten; sie testet nur die räumliche
Kompressibilität, keinen dynamisch äquivalenten Anfangszustand.

Die 4D-Schalen sind **exakte diskrete Abstandsklassen** nach `r²` mit periodisch
kürzestem Abstand zum ursprünglichen Anregungszentrum. Die ideale radiale
Projektion hat einen frei gewählten Mittelwert pro Abstandsklasse. Sie ist
bereits flexibler als eine einzelne gaußförmige, kontinuierliche Kugelschale;
ihr Projektionserfolg ist daher eine *optimistische Obergrenze* für eine reine
Radialkodierung mit festem Zentrum. Als schärfere Kompressionsdiagnose werden
die vier energiereichsten Schalen gezählt.

Zusätzlich wird innerhalb der vier am besten passenden Schalen je ein
Winkelpolynom (Grad 1 bzw. 2 in den vier normierten Koordinaten) optimal
angepasst. Das ist eine **stärker parametrisierte obere Schranke**, kein
kompakter Schalen-Solver: Bei Grad 2 sind häufig 35–56 effektive
Koeffizienten nötig; damit wächst der Darstellungsaufwand spürbar.

Alle Fits erfolgen *nach* der Booleschen Evolution. „Erklärte Energie“ bedeutet
`1 - ||delta rho - fit||² / ||delta rho||²` und ist **keine erhaltene
mikroskopische Energie**.

## Ergebnis auf `(8,4,4,8)` (512 physische Knoten)

Drei Seeds `1409,1410,1411`; vier Ticks. Beim Vakuumpuls ist der Startzustand
identisch, Unterschiede liegen nur in der zufälligen Kontrollverteilung.

| Tick | Vakuum: radial | Halbfüllung: radial, Bereich | Halbfüllung: vier Schalen mit Winkelgrad 2, Bereich |
|---:|---:|---:|---:|
| 0 | 100 % | 100 % | 100 % |
| 1 | 100 % | 50 % | 75–83 % |
| 2 | 85,2 % | 26,8–39,4 % | 56–69 % |
| 3 | 17,8 % | 2,8–5,1 % | 25–37 % |
| 4 | 46,7 % | 5,6–19,2 % | 32–35 % |

Die Vakuumanregung entwickelt sich **ohne lokale Kollisionsänderung** (alle
entsprechenden Messwerte null); ihre schön beschreibbare Anfangsschale ist
somit ein kontrollierter Fall freier Richtungs-Translation, **kein positiver
Nachweis einer kollisionsstabilen Kugelwelle**. Der Anstieg des radialen Fits
bei Tick 4 ist durch diskrete Überlagerung und kleine periodische Boxen
beeinflusst; er belegt keine dauerhafte Wiederkehr.

Auf dem halbgefüllten Hintergrund sind pro Tick etwa 280–314 von 512 lokalen
Kollisionseingaben veränderlich. Die radiale Beschreibung verliert trotzdem
nach wenigen Ticks stark an Erklärungsleistung. Mit Winkelprofilen lässt sich
zusätzliche Struktur zurückholen, aber die Parametrisierung wird größer.

Die räumlich verstreute Negativkontrolle hat bei Halbfüllung für den rein
radialen Fit meistens nur wenige Prozent Energieanteil (je nach Seed und
Tick); die lokalisierte Anregung ist also am Anfang deutlich spezieller.
Die Kontrollmessung ist weder impulsangepasst noch ein Ensemblebeweis.

## Perioden-Kontrolle auf `(8,8,8,8)` (2.048 physische Knoten)

Für Seed 1409, drei Ticks, halbgefüllter Hintergrund: radial erklärt nach
Tick 1 / 2 / 3 **50,0 % / 43,5 % / 2,9 %**. Das deutet darauf hin, dass das
frühe Scheitern der Radialbeschreibung nicht allein von den besonders kurzen
periodischen y-/z-Richtungen der Hauptbox stammt. Es ersetzt keine
systematische Größen- oder Zeitfensterextrapolation.

## Korrektheit und Grenzen

- Jeder Lauf erhält die Teilchenzahl und alle vier Gitterimpulskomponenten der
  jeweiligen Trajektorie exakt, auch während der Vorwärtsentwicklung.
- Alle acht Läufe wurden über gleich viele Rückwärtsticks **bitgenau** auf
  ihre jeweiligen Anfangsbitwörter zurückgeführt.
- Der Engine-Cache ist ausgeschaltet. Nur die zwei begrenzten Funktions-Caches
  der bestehenden Kollisionsroutine bleiben aktiv; ein Cache-Miss ändert nie
  die Kollisionsregel.
- Die Dichteprojektion lässt die 24 Richtungsbits und deren Winkelzuordnung
  größtenteils außer Acht. Ein Fit kann daher trotz hoher radialer Energie
  **keinen vollständigen Mikrozustand rekonstruieren**.
- Es handelt sich um zwei künstliche, endliche Präparationen, nicht die
  verlorene phasenkontrollierte Bit-6-Ensemblekampagne.

## Forschungsnotiz: D3 ↔ D4 ↔ D8 (geometrische Hypothese, nicht getestet)

Die Vermutung, Kugelschalen seien Träger von Wechselwirkungen zwischen
verschiedenen Gitterdimensionen, ist sinnvoll als **offene Hypothese** zu
notieren. Mathematisch gilt für eine idealisierte Kugel in einem d-dimensionalen
Euklidischen Raum: Schneidet eine Hyperfläche eine (d−1)-dimensionale
Kugeloberfläche S^(d−1), ergibt sich – sofern der Schnitt nicht leer und
nicht tangential ist – eine (d−2)-dimensionale Kugeloberfläche. Ihre
orthogonale Projektion kann dagegen einen ausgefüllten Ball ergeben.

Als Hypothesen für künftige Modelltests kommen in Betracht: 4D-Schalen, deren
Schnitt bei `w=const` dreidimensionale Schalen ergibt; die Aggregation über
verborgene Koordinaten; oder die Einbettung von D4-Richtungen in geeignete
Teilräume eines D8-Gitters. **Die bisherige D4-Mikrodynamik enthält jedoch
weder ein D8-Gitter noch einen Kopplungsoperator zwischen D3, D4 und D8.**
Die D3/D4/D8-Wechselwirkung ist folglich derzeit weder berechnet noch
physikalisch hergeleitet. Dies ist insbesondere keine Aussage über emergente
Quantentheorie, Raumzeit oder eine neue Elementarteilchentheorie.

## Nächster Test

Eine **bewegte, richtungsabhängige Schalenfront** mit wenigen diskreten
Oberflächenmoden statt festem Radialprofil fitten, und Modellkosten explizit
gegen die benötigte Bitzahl messen. Entscheidend ist ein auf separaten Seeds
und größeren Boxen validierter **Fehler-Kosten-Vergleich**, nicht bloß ein
hoher In-Sample-Fit. Danach die ursprüngliche Bit-6-Paketpräparation
rekonstruieren und an der Oberflächen-Darstellung prüfen.

Reproduktion: `src/phase_bit_s4_open33_shells.py` und numerische Rohberichte in
`data/bit14_*.json`. Abbildung: `figures/bit14_shell_retained_energy.png`.