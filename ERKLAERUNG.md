# Thema 7 ganz einfach erklärt

Diese Datei erklärt das ganze Projekt Schritt für Schritt, ohne Vorwissen.
Wer sie einmal von oben nach unten liest, kann jede Frage im Referat beantworten.

---

## 0. Das Ganze in einem Satz

Wir bringen einem Computerprogramm bei, Geld auf neun verschiedene Fonds zu
verteilen – Aktien, Anleihen, Gold, Immobilien, Rohstoffe –, lassen es dabei für
jeden Kauf und Verkauf 1 € Gebühr zahlen wie bei Trade Republic, probieren das mit
1.000 €, 10.000 € und 1.000.000 € aus und schauen, ob es besser ist als jemand, der
einfach kauft und nichts mehr tut.

---

## 1. Die Geschichte als Spiel

Stell dir ein Brettspiel vor.

- **Der Spieler** heißt bei uns **Agent**. Das ist das Programm, das lernt.
- **Das Spielbrett** heißt **Umgebung** (englisch *Environment*). Das ist die
  Börse, nachgespielt mit echten Kursen aus der Vergangenheit.
- **Ein Spielzug** ist **ein Handelstag**. An jedem Tag schaut der Agent aufs Brett
  und entscheidet: kaufen, verkaufen oder nichts tun.
- **Die Punkte** heißen **Belohnung** (englisch *Reward*). Wird das Depot an
  einem Tag mehr wert, gibt es Pluspunkte. Wird es weniger wert, Minuspunkte.
- **Eine Runde** heißt **Episode**. Eine Runde geht einmal vom ersten bis zum
  letzten Tag des Übungszeitraums.

Am Anfang drückt der Agent wild auf die Knöpfe. Nach vielen, vielen Runden
merkt er sich, welche Züge meistens Punkte bringen. Das nennt man
**Reinforcement Learning** – Lernen durch Belohnung. Genauso lernt ein Hund
„Sitz", wenn er dafür ein Leckerli bekommt.

---

## 2. Die Zutaten

### 2.1 Die neun Fonds

Ein **ETF** ist ein Korb mit vielen Wertpapieren, den man wie eine einzige Aktie
kauft. Ein **ETC** ist fast dasselbe, nur für Rohstoffe wie Gold. Der Agent darf
diese neun kaufen:

| Kürzel | Was drin ist | Anlageklasse | Preis pro Anteil Anfang 2024 |
|---|---|---|---|
| `SXR8` | die 500 größten Firmen der USA | Aktien | 454 € |
| `XSX6` | 600 große Firmen aus Europa | Aktien | 115 € |
| `IQQJ` | große Firmen aus Japan | Aktien | 15 € |
| `IQQE` | Firmen aus Schwellenländern wie China, Indien, Brasilien | Aktien | 36 € |
| `EUNH` | Schulden von Euro-Staaten (Staatsanleihen) | Anleihen | 112 € |
| `D5BG` | Schulden von Euro-Firmen (Unternehmensanleihen) | Anleihen | 150 € |
| `4GLD` | echtes Gold im Tresor | Gold | 61 € |
| `IQQ6` | Immobilienfirmen weltweit | Immobilien | 21 € |
| `EXXY` | Öl, Metalle, Getreide und mehr | Rohstoffe | 24 € |

Alle werden an der deutschen Börse Xetra in Euro gehandelt. Es gibt also kein
Wechselkurs-Problem.

**Dazu kommt ein zehnter Fonds, den der Agent nicht kaufen darf:** `EUNL`, der
MSCI World mit rund 1.400 großen Firmen aus Industrieländern. Er ist unser
**„Markt"** (in unserer Planung: Benchmark B4, Aktien Welt). Mit ihm vergleichen
wir den Agenten nur.

### 2.2 Das Geld – drei Varianten

Wir spielen das Ganze dreimal durch: mit **1.000 €**, mit **10.000 €** und mit
**1.000.000 €** Startkapital. Am Anfang ist immer alles Bargeld.

### 2.3 Die Gebühr

Trade Republic verlangt **1 € pro Order**, egal ob man für 50 € oder für 5.000 €
kauft. Das ist der Knackpunkt unseres Themas.

### 2.4 Warum drei verschiedene Startkapitale?

Weil die feste Gebühr je nach Geldmenge ganz unterschiedlich wehtut. Stell dir
vor, jemand verteilt sein Geld gleichmäßig auf alle neun Fonds:

| Startkapital | pro Fonds | 1 € Gebühr ist davon |
|---|---|---|
| 1.000 € | 111 € | **0,9 %** |
| 10.000 € | 1.111 € | 0,09 % |
| 1.000.000 € | 111.111 € | 0,0009 % |

Bei 1.000 € frisst jede Order fast ein Prozent. Bei einer Million merkt man sie
überhaupt nicht.

Dazu kommt ein zweites Problem: **Es gibt nur ganze Anteile.** Ein Anteil `SXR8`
kostet 454 €. Wer 1.000 € gleichmäßig auf neun Fonds verteilen will, hat pro Fonds
nur 111 € – das reicht für keinen einzigen Anteil `SXR8`, `D5BG`, `XSX6` oder `EUNH`.
Mit wenig Geld kann man also gar nicht so streuen, wie man möchte.

### 2.5 Die Kurse

Kommen kostenlos von Yahoo Finance. Alle neun Fonds gibt es gemeinsam seit Mai 2010.
Wir benutzen Januar 2016 bis September 2026.

---

## 3. Wie der Agent lernt

### 3.1 Was der Agent sieht – der Zustand (*State*)

Jeden Tag bekommt der Agent eine Liste mit **28 Zahlen**. Mehr weiß er nicht über
die Welt:

| Wie viele | Zahl | Beispiel |
|---|---|---|
| 1 | Welcher Anteil meines Depots ist Bargeld? | 0,05 (also 5 %) |
| 9 | Welcher Anteil meines Depots steckt in jedem Fonds? | 0,30 in `SXR8`, 0,00 in `4GLD`, … |
| 9 | **RSI** von jedem Fonds, geteilt durch 100 | 0,58 |
| 9 | **MACD** von jedem Fonds, geteilt durch seinen Kurs | 0,004 |

Die zwei letzten Sorten nennt man **technische Indikatoren**. Sie fassen den
Kursverlauf zusammen:

- **RSI**: Ging es in den letzten 30 Tagen eher rauf oder eher runter? Über 0,7
  heißt „ging viel rauf", unter 0,3 heißt „ging viel runter".
- **MACD**: Ist der Kurs gerade schneller unterwegs als sonst? Positiv heißt
  „Schwung nach oben", negativ heißt „Schwung nach unten".

**Warum Anteile statt Euro?** Stell dir vor, der Agent sähe „Bargeld: 1.000.000".
Mit so riesigen Zahlen kann ein neuronales Netz schlecht umgehen, und es würde
bei einer Million ganz anders lernen als bei tausend Euro. Der Vergleich der drei
Startkapitale wäre dann kaputt. Mit Anteilen sieht der Agent bei jedem Kapital
dasselbe: „5 % Bargeld" ist bei 1.000 € und bei einer Million gleich.

Wichtig: Nur **der Agent** sieht Anteile. Gekauft, verkauft und abgerechnet wird
im Hintergrund weiter in echten Euro und ganzen Anteilen.

Der Agent weiß **nicht**, was morgen passiert. Er sieht nur diese 28 Zahlen von heute.

### 3.2 Was der Agent tut – die Aktion (*Action*)

Der Agent antwortet mit **9 Zahlen**, eine pro Fonds, jede zwischen −1 und +1.

- **+1** heißt: „So viel kaufen wie erlaubt."
- **−1** heißt: „So viel verkaufen wie erlaubt."
- **0** heißt: „Nichts tun."

Die Zahl wird mit **`hmax`** malgenommen und abgerundet. Das Ergebnis ist die Zahl
der Anteile. `hmax` hängt vom Startkapital ab:

| Startkapital | `hmax` | Aktion +0,5 heißt |
|---|---|---|
| 1.000 € | 11 | 5 Anteile kaufen |
| 10.000 € | 101 | 50 Anteile kaufen |
| 1.000.000 € | 10.002 | 5.001 Anteile kaufen |

`hmax` ist so gewählt, dass eine volle Aktion (+1) im billigsten Fonds ungefähr ein
Neuntel des Startkapitals bewegt. Mit einem festen `hmax` von 100 bräuchte der
Agent bei einer Million Wochen, bis das Geld investiert ist – bei 1.000 € könnte
er dagegen mit einem Zug alles verpulvern.

Reihenfolge an jedem Tag: **erst alle Verkäufe, dann alle Käufe.** So kann man
mit dem Geld aus einem Verkauf am selben Tag etwas anderes kaufen.

### 3.3 Die Totzone

Weil abgerundet wird, passiert bei kleinen Aktionen **gar nichts**:

| Startkapital | `hmax` | Ab welcher Aktion entsteht 1 Anteil? |
|---|---|---|
| 1.000 € | 11 | ab ±0,091 |
| 10.000 € | 101 | ab ±0,0099 |
| 1.000.000 € | 10.002 | ab ±0,0001 |

Bei 1.000 € ist also fast ein Zehntel des ganzen Knopfes „tot". Ein vorsichtiger
Agent, der nur ein bisschen kaufen möchte, kauft bei wenig Geld schlicht nichts.
Das haben wir beim Testen gesehen (siehe 11.3).

### 3.4 Die Punkte – die Belohnung (*Reward*)

> Belohnung = (Depotwert heute − Depotwert gestern) × 100 ÷ Startkapital

Das heißt einfach: **1 % Gewinn an einem Tag gibt 1 Punkt** – egal ob man 1.000 €
oder eine Million hat. So lernen alle drei Varianten gleich stark.

Wichtig: Die Belohnung zählt **nur Geld**. Ob das Depot dabei stark wackelt, ist
dem Agenten egal. Man sagt: Der Agent ist **risikoneutral**. Das erklärt später
einige Ergebnisse.

### 3.5 Die Lernregel – PPO

Der Agent ist ein kleines **neuronales Netz**: Oben kommen die 28 Zahlen rein,
unten kommen die 9 Zahlen raus, dazwischen liegen viele einstellbare Rädchen
(Gewichte).

**PPO** (*Proximal Policy Optimization*) ist die Regel, wie die Rädchen verstellt
werden:

1. Der Agent spielt 2.048 Tage lang und merkt sich, was er getan hat und wie
   viele Punkte es gab.
2. Züge, die mehr Punkte gebracht haben als erwartet, werden **etwas**
   wahrscheinlicher. Züge mit weniger Punkten werden **etwas** unwahrscheinlicher.
3. Das „etwas" ist der Trick von PPO: Die Rädchen dürfen sich pro Schritt nur
   **ein kleines Stück** drehen. Sonst verlernt der Agent alles, wenn er einmal
   Pech hatte.
4. Wieder von vorne.

Beim Lernen probiert der Agent **mit Zufall** aus – dann landen seine Aktionen oft
weit weg von 0 und er handelt. In der Prüfung dagegen nimmt er **ohne Zufall**
immer seine beste Idee. Ist diese Idee nur ein zaghaftes „ein bisschen kaufen",
fällt sie bei wenig Geld in die Totzone.

Wir benutzen PPO nicht selbst programmiert, sondern aus der Bibliothek
**Stable-Baselines3**. FinRL ist nur die Hülle drumherum.

### 3.6 Wie lange gelernt wird – Timesteps

Ein **Timestep** ist ein Spielzug, also ein Tag. Wir trainieren **60.000 Timesteps**.
Der Übungszeitraum hat 1.778 Handelstage. Der Agent spielt die Jahre 2016 bis
2022 also **ungefähr 34 Mal** komplett durch.

### 3.7 Der Zufall – Seeds

Beim Lernen ist Zufall im Spiel: Die Rädchen starten mit zufälligen Werten, und
der Agent probiert zufällig Dinge aus. Ein **Seed** ist die Startzahl für diesen
Zufall. Gleicher Seed heißt gleicher Zufall heißt exakt gleiches Ergebnis.

Wir trainieren jeden Versuch mit **8 verschiedenen Seeds** (42 bis 49).
Warum? Stell dir vor, du wirfst eine Münze einmal und sie zeigt Kopf. Daraus
folgt nicht, dass die Münze immer Kopf zeigt. Genauso kann ein einzelner Agent
Glück gehabt haben. Erst wenn **viele** Agenten dasselbe tun, ist es ein
Ergebnis.

Insgesamt sind es **6 Versuche** (3 Startkapitale × täglich/monatlich) mit je 8
Agenten, also **48 Agenten**.

---

## 4. Die Fixgebühr – unser eigener Beitrag

Das ist der Teil, den **wir selbst** gebaut haben. Er steckt in `tr_env.py`.

### 4.1 Das Problem

FinRL kennt nur **prozentuale** Gebühren, zum Beispiel 0,1 % vom Kaufbetrag.
Trade Republic nimmt aber **immer 1 €**. Das macht einen großen Unterschied:

| Kauf | FinRL mit 0,1 % | Trade Republic |
|---|---|---|
| 50 € | 0,05 € | **1,00 €** (2 % vom Kauf!) |
| 1.000 € | 1,00 € | 1,00 € |
| 5.000 € | 5,00 € | **1,00 €** (0,02 % vom Kauf) |

Kleine Käufe sind bei Trade Republic also **teuer**, große Käufe **billig**.
Wer oft kleine Mengen hin und her schiebt, zahlt drauf.

### 4.2 Die Lösung

FinRL hat zwei Stellen, an denen gekauft und verkauft wird: `_buy_stock()` und
`_sell_stock()`. Unsere Klasse `TradeRepublicEnv` erbt alles von FinRL und
ersetzt **diese beiden Stellen**.

**Kaufen, Schritt für Schritt:**

1. Ziehe zuerst die 1 € Gebühr vom Bargeld ab.
2. Rechne aus, wie viele ganze Anteile vom Rest bezahlbar sind.
3. Nimm das Kleinere von „so viele will der Agent" und „so viele kann er bezahlen".
4. Sind es 0 Anteile: **nichts tun und keine Gebühr verlangen.**
5. Sonst: Kaufpreis plus 1 € vom Bargeld abziehen, Anteile gutschreiben.

Beispiel: 700 € Bargeld, Kurs 120 €, der Agent will 10 Anteile.
700 − 1 = 699 €. 699 ÷ 120 = 5,8, also 5 Anteile. Kosten: 5 × 120 + 1 = **601 €**.
Übrig: 99 €.

**Warum Schritt 1 zuerst?** Würde man erst die Anteile ausrechnen und danach die
Gebühr abziehen, könnte das Konto ins Minus rutschen. Mit 600 € Bargeld und Kurs
120 € hätte man 5 Anteile für 600 € gekauft – und danach fehlt 1 € für die
Gebühr. Bei prozentualen Gebühren kann das nicht passieren, bei festen schon.
**Genau das ist der Unterschied, um den es im Referat geht.**

**Warum Schritt 4?** Würde der Agent auch fürs Nichtstun zahlen, würde er lernen,
dass Nichtstun Geld kostet, und hektisch handeln. Das wäre falsch gelernt.

**Verkaufen** funktioniert genauso: Erlös minus 1 €. Wäre der Erlös kleiner als
1 €, wird der Verkauf nicht gemacht, weil er nur Verlust brächte.

### 4.3 Handeln nur jeden 21. Tag – „semi-automatisch"

Die Aufgabe sagt „**semi-automatische** Portfolio-Selection". Unsere Deutung: Ein
Mensch soll mitkommen können. Deshalb gibt es den Schalter `--rebalance`.

- `--rebalance 1`: Der Agent darf **jeden Tag** handeln.
- `--rebalance 21`: Der Agent darf nur **jeden 21. Handelstag** handeln, also
  ungefähr einmal im Monat. An allen anderen Tagen wird sein Wunsch einfach
  ignoriert.

Der Agent **weiß das aber nicht**. Er entscheidet jeden Tag und wundert sich,
dass meistens nichts passiert. Das ist eine Schwäche, die wir offen zugeben.

---

## 5. Üben, Probe, Prüfung – die drei Zeiträume

Wie in der Schule:

| Zeitraum | Heißt | Tage | Wozu |
|---|---|---|---|
| 2016 – 2022 | **Training** | 1.778 | Hier übt der Agent. Nur diese Kurse sieht er beim Lernen. |
| 2023 | **Validierung** | 255 | Die Probeklausur. Neue Aufgaben, aber noch nicht die echte Prüfung. |
| 2024 – 08.09.2026 | **Test** | 678 | Die echte Prüfung. Diese Kurse hat der Agent nie gesehen. |

Gezählt werden nur Tage, an denen **alle neun** Fonds einen Kurs haben.

Warum so streng getrennt? Ein Schüler, der die Prüfungsaufgaben vorher kennt,
schreibt eine 1 – und hat trotzdem nichts gelernt. Das nennt man **Overfitting**
(auswendig lernen statt verstehen). Deshalb zählen für uns nur Ergebnisse aus der
Prüfung.

**Das Enddatum ist fest** (`--end 2026-09-09`). Ohne festes Datum würde die
Prüfung jeden Tag einen Tag länger, und Ergebnisse von heute und nächster Woche
wären nicht vergleichbar. Das Datum ist der 09.09., weil das Enddatum selbst
nicht mehr dazugehört – der letzte Prüfungstag ist also der 08.09.

---

## 6. Gegen wen der Agent antritt – die Benchmarks

Ein Ergebnis wie „17.000 € am Ende" sagt allein nichts. Es kommt darauf an, was
man **ohne** den Agenten gehabt hätte. Deshalb laufen im Prüfungszeitraum drei
einfache Vergleichs-Strategien mit – alle mit denselben Regeln: ganze Anteile,
1 € pro Order, dasselbe Startkapital.

| Name | Was die Strategie macht |
|---|---|
| **1/N Buy & Hold** | Am ersten Tag gleich viel Geld in jeden der neun Fonds, danach nichts mehr. |
| **Markt: MSCI World (`EUNL`)** | Am ersten Tag alles in den MSCI World, danach nichts mehr. |
| **1/N alle 21 Tage** | Jeden 21. Tag wieder auf gleich viel in jedem Fonds zurückschieben. |

„1/N" heißt einfach: gleich viel in jeden Topf. Klingt dumm, ist aber in der
Forschung erstaunlich schwer zu schlagen.

**Achtung bei 1.000 €:** Wie in 2.4 erklärt, reicht es dort nicht für alle neun.
„1/N" kauft mit 1.000 € nur fünf Fonds; die vier teuren bleiben leer. Die
Strategie heißt trotzdem so, sie kann es nur nicht besser.

---

## 7. Wie wir messen – die Kennzahlen

| Kennzahl | Kinderfrage | Genauer |
|---|---|---|
| **Endwert** | Wie viel Geld ist am Ende da? | Depotwert am letzten Prüfungstag. |
| **CAGR** | Wie viel Prozent gab es pro Jahr? | Die gleichmäßige Jahresrendite, die zum selben Endwert geführt hätte. |
| **Vola** | Wie stark wackelt es? | Schwankung der Tagesrenditen, auf ein Jahr hochgerechnet. |
| **Sharpe** | Wie viel Rendite gibt es pro Wackeln? | Durchschnittsrendite geteilt durch Vola, aufs Jahr gerechnet. Über 1 ist gut. Einen risikofreien Zins ziehen wir nicht ab. |
| **Sortino** | Wie viel Rendite pro Wackeln **nach unten**? | Wie Sharpe, aber nur schlechte Tage zählen als Wackeln. |
| **Max. Drawdown** | Wie tief war das tiefste Loch? | Größter Absturz vom bisherigen Höchststand. −20 % heißt: Irgendwann war das Depot 20 % unter seinem bisherigen Rekord. |
| **Orders / Gebühren** | Wie oft gehandelt, wie viel gezahlt? | Jede ausgeführte Order kostet 1 €. Bei verschiedenen Startkapitalen vergleichen wir die Gebühren **in Prozent** des Startkapitals. |

**Die wichtigste Zahl ist Sharpe.** Ein Agent, der 5 % mehr verdient, aber
doppelt so stark wackelt, ist nicht besser – er hat nur mehr Risiko genommen.
Aber Vorsicht: Wer viel Bargeld hält, wackelt wenig und bekommt dadurch auch einen
guten Sharpe, ohne viel zu verdienen. Man muss immer mehrere Zahlen anschauen.

---

## 8. Die Dateien – wer macht was

Stell dir eine Küche vor.

| Datei | In der Küche wäre das … | Aufgabe |
|---|---|---|
| `tr_env.py` | **das Rezept** | Die Spielregeln: Fixgebühr, Handelstakt, und was der Agent sieht. Unser eigener Beitrag. |
| `run_training.py` | **der Koch** | Holt die Zutaten, lässt den Agenten lernen, prüft ihn, vergleicht, schreibt alles auf. |
| `eval_saved.py` | **der Vorkoster** | Nimmt einen fertigen Agenten und lässt ihn unter anderen Gebühren spielen – ohne neu zu lernen. |
| `vergleich.py` | **der Kritiker** | Legt zwei Läufe nebeneinander und rechnet aus, ob der Unterschied echt oder Zufall ist. |
| `app.py` | **das Schaufenster** | Die Demo-Webseite, auf der man dem Agenten beim Handeln zusieht. |
| `finrl_shim.py` | **der Türsteher** | Sorgt dafür, dass FinRL startet, ohne Pakete zu verlangen, die wir nie benutzen. |
| `setup.ps1`, `setup.sh` | **der Einkauf** | Richten einmalig alles ein: Python-Umgebung, Pakete, FinRL. |
| `requirements.txt` | **der Einkaufszettel** | Welche Pakete in welcher genauen Version. |
| `runs/` | **das Fotoalbum** | Ein Ordner pro Lauf mit Einstellungen, Ergebnissen, Bild und den trainierten Agenten. |
| `data/` | **der Kühlschrank** | Die geladenen Kurse, damit man sie nicht jedes Mal neu holt. Nicht im Repo. |

### Was in einem Lauf-Ordner liegt

| Datei | Inhalt |
|---|---|
| `config.json` | Alle Einstellungen des Laufs – auch das ausgerechnete `hmax`, die Belohnungs-Skalierung und der Markt-Fonds. |
| `ergebnisse.csv` | Alle Kennzahlen für jeden Seed und jeden Benchmark. |
| `depotwert.png` | Bild: Depotwert aller Agenten und Benchmarks im Prüfungszeitraum. |
| `actions_seed42.csv` … | Für jeden Prüfungstag und jeden Fonds: wie viele Anteile wirklich gekauft (+) oder verkauft (−) wurden. |
| `ppo_seed42.zip` … | Der trainierte Agent selbst – die eingestellten Rädchen. |

---

## 9. Einmal alles durch: Was passiert bei `run_training.py`?

```bash
.venv\Scripts\python.exe run_training.py --seeds 42 43 44 45 46 47 48 49 --fee 1 --initial 10000 --rebalance 1 --end 2026-09-09 --tag k10000_daily
```

1. **Kurse holen.** Liegen sie schon in `data/`, werden sie von dort genommen,
   sonst von Yahoo geladen. Tage, an denen nicht alle neun Fonds gehandelt wurden,
   fliegen raus.
2. **Indikatoren ausrechnen.** RSI und MACD für jeden Fonds und jeden Tag.
3. **In drei Zeiträume schneiden.** Training, Validierung, Test.
4. **`hmax` ausrechnen** aus Startkapital und dem billigsten Fonds am ersten Übungstag.
5. **Ordner anlegen.** In `runs/` entsteht ein Ordner mit Datum, Uhrzeit und dem
   Namen hinter `--tag`. Die Einstellungen landen in `config.json`.
6. **Benchmarks ausrechnen.** Dafür werden auch die Kurse des Markt-Fonds `EUNL` geladen.
7. **Für jeden Seed:**
   1. Spielbrett mit Fixgebühr aufbauen, nur mit den Trainingskursen.
   2. Neuen Agenten mit zufälligen Rädchen erzeugen.
   3. 60.000 Tage lang spielen und lernen.
   4. Agenten als `ppo_seedXX.zip` speichern.
   5. Agenten die Probeklausur (2023) schreiben lassen – ohne Lernen, ohne Zufall.
   6. Agenten die Prüfung (2024–2026) schreiben lassen – ohne Lernen, ohne Zufall.
   7. Kennzahlen, Orders und Gebühren ausrechnen und ausgeben.
8. **Alles aufschreiben.** Tabelle in `ergebnisse.csv`, Bild in `depotwert.png`.

---

## 10. Die Demo-Seite

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Dann öffnet sich im Browser `http://localhost:8501`. Die Seite ist nur auf dem
eigenen Rechner erreichbar.

**Links** stellt man ein, welchen Agenten man sehen will: Lauf (also Startkapital
und Takt), Seed, Probeklausur oder Prüfung, und wer „der Markt" ist. Gebühr und
Handelstakt kann man auch verstellen – dann spielt **derselbe** Agent unter
anderen Regeln, und die Seite zeigt eine gelbe Warnung. Er hat unter diesen
Regeln nie gelernt.

**Oben** stehen vier große Zahlen: Endwert, Sharpe, tiefstes Loch und Orders –
jeweils mit dem Unterschied zum Markt. Darunter die Kennzahlen-Tabelle.

**Die vier Reiter:**

| Reiter | Was man sieht | Wofür im Referat |
|---|---|---|
| **Depotwert** | Wie sich das Geld von Agent, Markt und 1/N entwickelt. Darunter: Liegt der Agent vor oder hinter dem Markt? | „Ist er besser?" |
| **Trades** | Eine Tabelle mit Käufen, Verkäufen und Gebühren für jeden der neun Fonds. Darunter der Kursverlauf eines Fonds, den man auswählt, mit Dreiecken: ▲ blau = gekauft, ▼ rot = verkauft. Dann das Orderbuch mit jeder einzelnen Order. | „Was hat er getan?" |
| **Depot** | Wie viel Bargeld und wie viel in Aktien, Anleihen, Gold, Immobilien und Rohstoffen steckt, über die Zeit. Darunter der durchschnittliche Anteil jedes Fonds. | „Wie war er aufgestellt?" |
| **Was hat er gelernt?** | Siehe unten. | „Warum?" |

**Der Reiter „Was hat er gelernt?"** hat drei Teile:

1. **Kurzdiagnose.** Die Seite sortiert den Agenten in eine von vier Schubladen:
   - *Kein Handel*: keine einzige Order, alles blieb Bargeld.
   - *Praktisch Buy & Hold*: einmal gekauft, dann Ruhe.
   - *Dauerhandel*: im Schnitt jeden zweiten Tag eine Order pro Fonds oder mehr.
   - *Gelegentliches Umschichten*: alles dazwischen.

   Außerdem sagt sie, welcher Fonds im Schnitt die größte Position war, welcher
   Vergleichs-Strategie der Agent am ähnlichsten ist und wie breit die Totzone war.
2. **Punktwolke.** Jeder Punkt ist ein Tag. Nach rechts: ein Indikator eines Fonds,
   zum Beispiel RSI/100. Nach oben: was der Agent mit diesem Fonds tun wollte.
   Sieht man ein Muster – etwa „bei hohem RSI will er verkaufen" –, hat er etwas mit
   diesem Indikator gelernt. Sieht man keins, ist ihm der Indikator egal.
3. **Policy-Sonde.** Man nimmt einen Tag, friert alle 28 Zahlen ein und dreht nur
   an **einer** – zum Beispiel am RSI von `SXR8`. Die blaue Linie zeigt, wie der
   Agent beim gewählten Fonds darauf reagiert, die grauen Linien, wie er bei allen
   anderen Fonds reagiert. Flache Linie: Die Zahl interessiert ihn nicht. Steile
   Linie: Die Zahl ist ihm wichtig.

---

## 11. Drei Stolperfallen, die wir gelöst haben

### 11.1 FinRL will Pakete, die wir nicht brauchen

Wenn man FinRL startet, will es sofort Programme für echte Börsenkonten
(Alpaca), eine Uni-Datenbank (WRDS) und einen Browser-Fernsteuerer (Selenium)
laden. Wir brauchen nichts davon. Eines davon (`alpaca_trade_api`) würde sogar
eine uralte pandas-Version erzwingen und alles andere kaputt machen.

**Lösung:** `finrl_shim.py` ist ein Türsteher. Fragt FinRL nach einem dieser
Pakete, gibt der Türsteher eine leere Attrappe heraus. FinRL ist zufrieden und
startet. Würde jemand die Attrappe wirklich benutzen wollen, gäbe es beim ersten
Methodenaufruf einen Fehler. Deshalb muss `import finrl_shim` immer **vor** FinRL
geladen werden.

### 11.2 Die Gebühren waren plötzlich 0

Beim ersten Test stand „0 Orders, 0 € Gebühren" – obwohl das Depot um 71 %
gewachsen war. Das kann nicht sein.

**Der Grund:** FinRL packt das Spielbrett zum Prüfen in eine Schachtel
(`DummyVecEnv`). Die Schachtel räumt das Brett nach dem letzten Zug **automatisch
auf** – und beim Aufräumen werden die Zähler für Orders und Gebühren auf 0
gesetzt. Wer danach nachschaut, sieht 0.

**Lösung:** Unser Spielbrett schreibt sich die Zähler kurz vor dem Aufräumen auf
einen Zettel (`last_episode_cost`, `last_episode_trades`).

### 11.3 Ein ungelernter Agent handelt bei 1.000 € gar nicht

Beim Kurztest mit 1.000 € hat der Agent in der Prüfung **keine einzige Order**
gemacht. Kein Fehler, sondern die Totzone aus 3.3: Seine Aktionen lagen im
Schnitt bei ±0,026, für einen Anteil hätte er ±0,091 gebraucht. Beim Lernen mit
Zufall lagen dagegen 93 % seiner Aktionen über der Schwelle – er hat also sehr
wohl gehandelt, nur in der Prüfung ohne Zufall nicht mehr.

**Was wir daraus mitnehmen:** Wenn ein Agent mit wenig Kapital nicht handelt,
muss man nachsehen, ob er „nichts tun" gelernt hat – oder ob er nur zu zaghaft ist.

---

## 12. Die Ergebnisse

> *Wird eingetragen, sobald die sechs Läufe fertig sind.*

---

## 13. Was wir bewusst nicht können

Ehrlich sein gehört zum Referat. Diese Dinge sind vereinfacht oder fehlen:

- **Nur ganze Anteile.** Trade Republic kann auch Bruchstücke. Bei 1.000 € ist
  das eine echte Einschränkung, weil teure Fonds unerreichbar sind.
- **Die Totzone.** Bei wenig Kapital gehen zaghafte Entscheidungen verloren.
- **`hmax` zählt Anteile, nicht Euro.** Eine volle Aktion bedeutet bei `SXR8`
  (454 €) viel mehr Geld als bei `IQQJ` (15 €).
- **Kein Spread.** Beim echten Kauf ist der Kaufpreis immer etwas höher als der
  Verkaufspreis. Dafür gibt es keine kostenlosen historischen Daten.
- **Tage ohne Umsatz.** Manche Fonds wurden an einzelnen Tagen gar nicht gehandelt
  (`EUNH` an über 600 Tagen seit 2009). Yahoo meldet dann trotzdem einen Kurs.
  Wir tun so, als hätte man dort handeln können.
- **Keine Steuern.**
- **Nur eine Prüfung.** Besser wäre, mehrere Prüfungszeiträume hintereinander
  zu testen (Walk-Forward).
- **Der Agent kennt nur Geld, kein Risiko.** Die Belohnung zählt nur, ob das
  Depot wächst – nicht, wie sehr es dabei wackelt.
- **Beim Monatstakt weiß der Agent nicht, wann er handeln darf.**
- **Die Lern-Einstellungen sind Standardwerte** und nicht ausprobiert.

---

## 14. Mini-Wörterbuch

| Wort | Einfach gesagt |
|---|---|
| **Agent** | Das lernende Programm. |
| **Aktion** | Was der Agent an einem Tag tut: 9 Zahlen zwischen −1 und +1. |
| **Anlageklasse** | Die Sorte eines Fonds: Aktien, Anleihen, Gold, Immobilien, Rohstoffe. |
| **Anleihe** | Man leiht einem Staat oder einer Firma Geld und bekommt Zinsen. |
| **Benchmark** | Eine einfache Vergleichs-Strategie. |
| **Buy & Hold** | Einmal kaufen, liegen lassen. |
| **Depotwert** | Bargeld plus alle Anteile zum aktuellen Kurs. |
| **Episode** | Einmal den Übungszeitraum von vorne bis hinten durchspielen. |
| **ETF / ETC** | Ein Korb mit vielen Wertpapieren (ETF) oder mit einem Rohstoff wie Gold (ETC), den man wie eine Aktie kauft. |
| **FinRL** | Eine Bibliothek, die Börsen-Spielbretter für Reinforcement Learning bereitstellt. |
| **Fixgebühr** | Immer gleich viel Gebühr pro Order, egal wie groß. |
| **`hmax`** | Wie viele Anteile eine volle Aktion (+1) höchstens bedeutet. |
| **Indikator** | Eine Zahl, die den Kursverlauf zusammenfasst (RSI, MACD). |
| **Markt** | Bei uns der MSCI World (`EUNL`): alles in einen breiten Welt-Aktienfonds. |
| **Overfitting** | Auswendig lernen statt verstehen. |
| **Policy** | Die gelernte Strategie: „Bei diesen 28 Zahlen tue ich das." |
| **PPO** | Die Lernregel: kleine, vorsichtige Verbesserungen. |
| **Rebalancing** | Das Depot wieder auf die gewünschte Aufteilung zurückschieben. |
| **Reward** | Die Punkte: 1 Punkt pro 1 % Tagesgewinn. |
| **Seed** | Die Startzahl für den Zufall. Gleicher Seed, gleiches Ergebnis. |
| **Startkapital** | Das Geld am ersten Tag: bei uns 1.000 €, 10.000 € oder 1.000.000 €. |
| **State** | Die 28 Zahlen, die der Agent jeden Tag sieht. |
| **Timestep** | Ein Spielzug, also ein Tag. |
| **Totzone** | Aktionen, die so klein sind, dass abgerundet 0 Anteile herauskommen. |
| **Turnover** | Wie viel gehandelt wird. |
| **Vieltrader** | Ein Agent, der ständig handelt – bei Fixgebühren teuer. |
