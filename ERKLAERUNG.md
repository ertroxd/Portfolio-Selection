# Thema 7 ganz einfach erklärt

Diese Datei erklärt das ganze Projekt Schritt für Schritt, ohne Vorwissen.
Wer sie einmal von oben nach unten liest, kann jede Frage im Referat beantworten.

---

## 0. Das Ganze in einem Satz

Wir bringen einem Computerprogramm bei, 10.000 € auf zwei Aktienfonds zu verteilen,
lassen es dabei für jeden Kauf und Verkauf 1 € Gebühr zahlen – so wie bei Trade
Republic – und schauen dann, ob es am Ende besser ist als jemand, der einfach
kauft und nichts mehr tut.

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

**Das Geld:** 10.000 € Startkapital, am Anfang alles als Bargeld (Cash).

**Die zwei Fonds** (ETFs – Körbe mit vielen Aktien, die man wie eine Aktie kauft):

| Kürzel | Was drin ist | Rolle bei uns |
|---|---|---|
| `EUNL.DE` | iShares Core MSCI World – rund 1.400 große Firmen aus Industrieländern | Das ist unser **„Markt"** |
| `IS3N.DE` | iShares Core MSCI EM IMI – Firmen aus Schwellenländern wie China, Indien, Brasilien | Die zweite Möglichkeit |

Beide werden an der deutschen Börse Xetra in Euro gehandelt. Deshalb gibt es
kein Wechselkurs-Problem.

**Die Gebühr:** Trade Republic verlangt **1 € pro Order**, egal ob man für 50 €
oder für 5.000 € kauft. Das ist der Knackpunkt unseres Themas, dazu gleich mehr.

**Die Kurse:** kommen kostenlos von Yahoo Finance, Januar 2016 bis September 2026.

---

## 3. Wie der Agent lernt

### 3.1 Was der Agent sieht – der Zustand (*State*)

Jeden Tag bekommt der Agent eine Liste mit **9 Zahlen**. Mehr weiß er nicht über
die Welt:

| Nr. | Zahl | Beispiel |
|---|---|---|
| 1 | Wie viel Bargeld habe ich? | 312,50 € |
| 2–3 | Was kostet ein Anteil von jedem Fonds heute? | 127,14 € und 48,56 € |
| 4–5 | Wie viele Anteile habe ich von jedem Fonds? | 40 und 95 |
| 6–7 | **RSI** von jedem Fonds | 58 und 41 |
| 8–9 | **MACD** von jedem Fonds | 0,8 und −0,2 |

Die zwei letzten Sorten nennt man **technische Indikatoren**. Sie fassen den
Kursverlauf zusammen:

- **RSI** (0 bis 100): Ging es in den letzten 30 Tagen eher rauf oder eher runter?
  Über 70 heißt „ging viel rauf", unter 30 heißt „ging viel runter".
- **MACD**: Ist der Kurs gerade schneller unterwegs als sonst? Positiv heißt
  „Schwung nach oben", negativ heißt „Schwung nach unten".

Der Agent weiß **nicht**, was morgen passiert. Er sieht nur diese 9 Zahlen von heute.

### 3.2 Was der Agent tut – die Aktion (*Action*)

Der Agent antwortet mit **2 Zahlen**, eine pro Fonds, jede zwischen −1 und +1.

- **+1** heißt: „So viel kaufen wie erlaubt."
- **−1** heißt: „So viel verkaufen wie erlaubt."
- **0** heißt: „Nichts tun."

Die Zahl wird mal 100 genommen und abgerundet. **+0,37** wird zu „37 Anteile
kaufen". Mehr als 100 Anteile auf einmal gehen nicht (`hmax = 100`).

Reihenfolge an jedem Tag: **erst alle Verkäufe, dann alle Käufe.** So kann man
mit dem Geld aus einem Verkauf am selben Tag etwas anderes kaufen.

Es gibt nur **ganze Anteile**. Einen halben Anteil kann der Agent nicht kaufen,
obwohl Trade Republic das eigentlich könnte. Das ist eine bewusste Vereinfachung.

### 3.3 Die Punkte – die Belohnung (*Reward*)

> Belohnung = Depotwert heute − Depotwert gestern

Depotwert heißt: Bargeld plus alle Anteile zum heutigen Kurs. Die Zahl wird noch
mit 0,0001 malgenommen, damit sie klein und handlich ist – am Prinzip ändert
das nichts.

Wichtig: Die Belohnung zählt **nur Geld**. Ob das Depot dabei stark wackelt, ist
dem Agenten egal. Man sagt: Der Agent ist **risikoneutral**. Das erklärt später
einige Ergebnisse.

### 3.4 Die Lernregel – PPO

Der Agent ist ein kleines **neuronales Netz**: Oben kommen die 9 Zahlen rein,
unten kommen die 2 Zahlen raus, dazwischen liegen viele einstellbare Rädchen
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

Wir benutzen PPO nicht selbst programmiert, sondern aus der Bibliothek
**Stable-Baselines3**. FinRL ist nur die Hülle drumherum.

### 3.5 Wie lange gelernt wird – Timesteps

Ein **Timestep** ist ein Spielzug, also ein Tag. Wir trainieren **60.000 Timesteps**.
Der Übungszeitraum hat 1.779 Handelstage. Der Agent spielt die Jahre 2016 bis
2022 also **ungefähr 34 Mal** komplett durch.

### 3.6 Der Zufall – Seeds

Beim Lernen ist Zufall im Spiel: Die Rädchen starten mit zufälligen Werten, und
der Agent probiert zufällig Dinge aus. Ein **Seed** ist die Startzahl für diesen
Zufall. Gleicher Seed heißt gleicher Zufall heißt exakt gleiches Ergebnis.

Wir trainieren jeden Versuch mit **8 verschiedenen Seeds** (42 bis 49).
Warum? Stell dir vor, du wirfst eine Münze einmal und sie zeigt Kopf. Daraus
folgt nicht, dass die Münze immer Kopf zeigt. Genauso kann ein einzelner Agent
Glück gehabt haben. Erst wenn **viele** Agenten dasselbe tun, ist es ein
Ergebnis.

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
ersetzt **nur diese beiden Stellen**.

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
| 2016 – 2022 | **Training** | 1.779 | Hier übt der Agent. Nur diese Kurse sieht er beim Lernen. |
| 2023 | **Validierung** | 255 | Die Probeklausur. Neue Aufgaben, aber noch nicht die echte Prüfung. |
| 2024 – 08.09.2026 | **Test** | 680 | Die echte Prüfung. Diese Kurse hat der Agent nie gesehen. |

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
1 € pro Order.

| Name | Was die Strategie macht | Wie viele Orders |
|---|---|---|
| **1/N Buy & Hold** | Am ersten Tag je 5.000 € in beide Fonds, danach nichts mehr. | 2 |
| **100 % MSCI World Buy & Hold** | Am ersten Tag alles in `EUNL.DE`, danach nichts mehr. Das ist **„der Markt"**. | 1 |
| **1/N alle 21 Tage** | Jeden 21. Tag wieder auf halbe-halbe zurückschieben. | etwa 48 |

„1/N" heißt einfach: gleich viel in jeden Topf. Klingt dumm, ist aber in der
Forschung erstaunlich schwer zu schlagen.

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
| **Orders / Gebühren** | Wie oft gehandelt, wie viel gezahlt? | Jede ausgeführte Order kostet 1 €. |

**Die wichtigste Zahl ist Sharpe.** Ein Agent, der 5 % mehr verdient, aber
doppelt so stark wackelt, ist nicht besser – er hat nur mehr Risiko genommen.

---

## 8. Die Dateien – wer macht was

Stell dir eine Küche vor.

| Datei | In der Küche wäre das … | Aufgabe |
|---|---|---|
| `tr_env.py` | **das Rezept** | Die Spielregeln mit Fixgebühr. Unser eigener Beitrag. |
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
| `config.json` | Alle Einstellungen, mit denen der Lauf gestartet wurde. |
| `ergebnisse.csv` | Alle Kennzahlen für jeden Seed und jeden Benchmark. |
| `depotwert.png` | Bild: Depotwert aller Agenten und Benchmarks im Prüfungszeitraum. |
| `actions_seed42.csv` … | Für jeden Prüfungstag: wie viele Anteile wirklich gekauft (+) oder verkauft (−) wurden. |
| `ppo_seed42.zip` … | Der trainierte Agent selbst – die eingestellten Rädchen. |

---

## 9. Einmal alles durch: Was passiert bei `run_training.py`?

```bash
.venv\Scripts\python.exe run_training.py --timesteps 60000 --seeds 42 43 44 45 46 47 48 49 --fee 1 --rebalance 1 --end 2026-09-09 --tag daily_fee1
```

1. **Ordner anlegen.** In `runs/` entsteht ein Ordner mit Datum, Uhrzeit und dem
   Namen hinter `--tag`. Die Einstellungen landen in `config.json`.
2. **Kurse holen.** Liegen sie schon in `data/`, werden sie von dort genommen,
   sonst von Yahoo geladen. Tage, an denen nicht beide Fonds gehandelt wurden,
   fliegen raus.
3. **Indikatoren ausrechnen.** RSI und MACD für jeden Fonds und jeden Tag.
4. **In drei Zeiträume schneiden.** Training, Validierung, Test.
5. **Benchmarks ausrechnen.** Die drei Vergleichs-Strategien im Prüfungszeitraum.
6. **Für jeden Seed:**
   1. Spielbrett mit Fixgebühr aufbauen, nur mit den Trainingskursen.
   2. Neuen Agenten mit zufälligen Rädchen erzeugen.
   3. 60.000 Tage lang spielen und lernen.
   4. Agenten als `ppo_seedXX.zip` speichern.
   5. Agenten die Probeklausur (2023) schreiben lassen – ohne Lernen, nur spielen.
   6. Agenten die Prüfung (2024–2026) schreiben lassen – ohne Lernen, nur spielen.
   7. Kennzahlen, Orders und Gebühren ausrechnen und ausgeben.
7. **Alles aufschreiben.** Tabelle in `ergebnisse.csv`, Bild in `depotwert.png`.

---

## 10. Die Demo-Seite

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Dann öffnet sich im Browser `http://localhost:8501`. Die Seite ist nur auf dem
eigenen Rechner erreichbar.

**Links** stellt man ein, welchen Agenten man sehen will: Lauf, Seed, Probeklausur
oder Prüfung, und wer „der Markt" ist. Gebühr und Handelstakt kann man auch
verstellen – dann spielt **derselbe** Agent unter anderen Regeln, und die Seite
zeigt eine gelbe Warnung. Er hat unter diesen Regeln nie gelernt.

**Oben** stehen vier große Zahlen: Endwert, Sharpe, tiefstes Loch und Orders –
jeweils mit dem Unterschied zum Markt. Darunter die Kennzahlen-Tabelle.

**Die vier Reiter:**

| Reiter | Was man sieht | Wofür im Referat |
|---|---|---|
| **Depotwert** | Wie sich das Geld von Agent, Markt und 1/N entwickelt. Darunter: Liegt der Agent vor oder hinter dem Markt? | „Ist er besser?" |
| **Trades** | Kurse beider Fonds mit Dreiecken: ▲ blau = gekauft, ▼ rot = verkauft. Darunter das Orderbuch mit jeder einzelnen Order. Dazu, wie oft der Agent handeln **wollte**, es aber nicht ging. | „Was hat er getan?" |
| **Depot** | Wie viel Bargeld und wie viel in jedem Fonds steckt, über die Zeit. | „Wie war er aufgestellt?" |
| **Was hat er gelernt?** | Siehe unten. | „Warum?" |

**Der Reiter „Was hat er gelernt?"** hat drei Teile:

1. **Kurzdiagnose.** Die Seite sortiert den Agenten in eine von drei Schubladen:
   - *Praktisch Buy & Hold*: einmal gekauft, dann Ruhe.
   - *Dauerhandel*: im Schnitt jeden zweiten Tag eine Order pro Fonds oder mehr.
   - *Gelegentliches Umschichten*: alles dazwischen.

   Außerdem sagt sie, welcher Vergleichs-Strategie der Agent am ähnlichsten ist.
   Liegt er weniger als 1 % daneben, hat er im Grunde nur diese Strategie
   nachgebaut.
2. **Punktwolke.** Jeder Punkt ist ein Tag. Nach rechts: ein Indikator, zum
   Beispiel RSI. Nach oben: was der Agent tun wollte. Sieht man ein Muster –
   etwa „bei hohem RSI will er verkaufen" –, hat er etwas mit diesem Indikator
   gelernt. Sieht man keins, ist ihm der Indikator egal.
3. **Policy-Sonde.** Man nimmt einen Tag, friert alle 9 Zahlen ein und dreht nur
   an **einer** – zum Beispiel RSI von 20 bis 80. Die Linie zeigt, wie der Agent
   darauf reagiert. Flache Linie: Die Zahl interessiert ihn nicht. Steile Linie:
   Die Zahl ist ihm wichtig.

---

## 11. Zwei Stolperfallen, die wir gelöst haben

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

---

## 12. Die Ergebnisse

Zwei Läufe, beide mit 1 € pro Order und je 8 Agenten (Seeds 42 bis 49):

- **`daily_fee1`** – der Agent darf jeden Tag handeln.
- **`monthly_fee1`** – der Agent darf nur jeden 21. Handelstag handeln.

Alle Zahlen stammen aus der **Prüfung** (02.01.2024 bis 08.09.2026). Diese Kurse
hat kein Agent beim Lernen gesehen.

### 12.1 Vorweg: Gleiche Zutaten, gleicher Kuchen

Die neu trainierten Agenten im Tageslauf sind **exakt dieselben** wie die alten,
auf den Euro genau. Das ist kein Fehler. Gleiche Kurse, gleiche Einstellungen und
gleiche Seeds ergeben gleiche Rädchen – wie ein Rezept, das jedes Mal genau
gleich gelingt. Für das Referat ist das sogar gut: Jeder kann unsere Zahlen
nachrechnen und bekommt dasselbe heraus.

### 12.2 Die Zahlen

| | Sharpe | Endwert | Orders |
|---|---|---|---|
| **Agenten täglich** (Mittel von 8) | 1,22 | 16.007 € | 2 bis 681 |
| **Agenten monatlich** (Mittel von 8) | 1,23 | 15.235 € | 9 bis 41 |
| 1/N Buy & Hold | **1,32** | **16.144 €** | 2 |
| 100 % MSCI World | 1,26 | 15.435 € | 1 |

Alle 16 Agenten einzeln:

| Seed | täglich: Endwert | Sharpe | Orders | monatlich: Endwert | Sharpe | Orders |
|---|---|---|---|---|---|---|
| 42 | 17.204 € | 1,25 | 24 | 15.940 € | 1,13 | 9 |
| 43 | 16.037 € | 1,34 | 16 | 14.610 € | 1,20 | 11 |
| 44 | 15.446 € | 1,27 | 2 | 17.149 € | 1,30 | 24 |
| 45 | 16.902 € | 1,21 | 30 | 16.171 € | 1,24 | 39 |
| 46 | 16.446 € | 1,16 | 270 | 14.797 € | 1,16 | 9 |
| 47 | 15.008 € | 1,20 | 193 | 14.252 € | 1,11 | 40 |
| 48 | 15.822 € | 1,09 | 681 | 13.824 € | **1,50** | 41 |
| 49 | 15.191 € | 1,22 | 18 | 15.135 € | 1,21 | 12 |

### 12.3 Was das heißt – in sechs Sätzen

**1. Kein Agent ist verlässlich besser als „halbe-halbe kaufen und liegen lassen".**
Im Mittel haben die Agenten einen Sharpe von 1,22 bzw. 1,23, die langweilige
1/N-Strategie hat 1,32. Nur 2 von 16 Agenten liegen darüber, und einer davon nur
ganz knapp (1,34).

**2. Viele Agenten haben gelernt: „Alles in einen Topf."**
Täglich Seed 42 und Seed 48 hatten die ganze Zeit rund 98 % im
Schwellenländer-Fonds (`IS3N.DE`). Täglich Seed 44 hatte 99,5 % im MSCI World
(`EUNL.DE`). Das ist keine schlaue Handelsstrategie, sondern eine Wette auf
einen einzigen Fonds. Welcher Fonds es wird, entscheidet der Zufall beim Lernen.

**3. Der Zufall entscheidet mit.**
Mit genau denselben Einstellungen endet ein Agent bei 15.008 €, ein anderer bei
17.204 €. Nur der Seed ist anders. Wer nur einen einzigen Agenten zeigt, zeigt
also vor allem, ob er Glück hatte. Deshalb trainieren wir 8.

**4. Täglich handeln macht manche Agenten hektisch.**
Täglich Seed 48 hat 681 Mal gehandelt, also 681 € Gebühren gezahlt. Seed 46 und
47 kamen auf 270 und 193 Orders. Beim Monatstakt hat kein Agent mehr als 41 Mal
gehandelt.

**5. Was kostet die Gebühr wirklich?**
Um das zu messen, lassen wir **dieselben** Agenten die Prüfung noch einmal ohne
Gebühr spielen (`eval_saved.py --fee 0`). Der Unterschied ist das, was die Gebühr
gekostet hat:

| Seed (täglich) | Orders | So viel mehr Geld ohne Gebühr |
|---|---|---|
| 44 | 2 | 2 € |
| 43 | 16 | 40 € |
| 49 | 18 | 83 € |
| 42 | 24 | 53 € |
| 45 | 30 | 66 € |
| 47 | 193 | 467 € |
| 46 | 270 | 435 € |
| 48 | 681 | **842 €** |

Faustregel: Wer wenig handelt, merkt die Gebühr nicht. Wer viel handelt, verliert
Hunderte Euro. **Nicht die Größe des Depots entscheidet, sondern wie oft man
handelt.**

Beim Monatstakt sind die Gebühren klein. Trotzdem gibt es dort Überraschungen:
Monatlich Seed 47 zahlt nur 40 € Gebühren, hätte ohne Gebühr aber 603 € mehr.
Monatlich Seed 48 hätte ohne Gebühr sogar 421 € **weniger**. Wie geht das? Die
Gebühr bestimmt mit, welche Orders überhaupt stattfinden – eine Order, die sich
mit 1 € Gebühr gerade nicht lohnt, findet ohne Gebühr statt. Danach sieht das
ganze Depot anders aus, und dieser andere Weg kann besser oder schlechter
ausgehen. Das nennt man **Pfadabhängigkeit**.

**6. Der beste Sharpe ist nicht automatisch das meiste Geld.**
Monatlich Seed 48 hat den besten Sharpe von allen (1,50) – und gleichzeitig am
wenigsten Geld (13.824 €). Der Grund: Er hatte im Schnitt **40 % Bargeld** im
Depot. Bargeld wackelt nicht, also wackelt das Depot wenig, also ist der Sharpe
hoch. Aber Bargeld wächst auch nicht. Man muss immer mehrere Kennzahlen zusammen
anschauen.

### 12.4 Was man im Referat sagen kann

> Unser PPO-Agent schlägt eine einfache 1/N-Strategie nicht verlässlich. Die
> meisten Agenten lernen, einen einzelnen Fonds zu halten, und welcher das ist,
> hängt vom Zufall ab. Die feste Ordergebühr von Trade Republic wird erst teuer,
> wenn ein Agent viel handelt – täglicher Handel führt dazu, monatlicher nicht.

---

## 13. Was wir bewusst nicht können

Ehrlich sein gehört zum Referat. Diese Dinge sind vereinfacht oder fehlen:

- **Nur ganze Anteile.** Trade Republic kann auch Bruchstücke.
- **Kein Spread.** Beim echten Kauf ist der Kaufpreis immer etwas höher als der
  Verkaufspreis. Dafür gibt es keine kostenlosen historischen Daten.
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
| **Aktion** | Was der Agent an einem Tag tut: zwei Zahlen zwischen −1 und +1. |
| **Benchmark** | Eine einfache Vergleichs-Strategie. |
| **Buy & Hold** | Einmal kaufen, liegen lassen. |
| **Depotwert** | Bargeld plus alle Anteile zum aktuellen Kurs. |
| **Episode** | Einmal den Übungszeitraum von vorne bis hinten durchspielen. |
| **ETF** | Ein Korb mit vielen Aktien, den man wie eine Aktie kauft. |
| **FinRL** | Eine Bibliothek, die Börsen-Spielbretter für Reinforcement Learning bereitstellt. |
| **Fixgebühr** | Immer gleich viel Gebühr pro Order, egal wie groß. |
| **Indikator** | Eine Zahl, die den Kursverlauf zusammenfasst (RSI, MACD). |
| **Overfitting** | Auswendig lernen statt verstehen. |
| **Policy** | Die gelernte Strategie: „Bei diesen 9 Zahlen tue ich das." |
| **PPO** | Die Lernregel: kleine, vorsichtige Verbesserungen. |
| **Rebalancing** | Das Depot wieder auf die gewünschte Aufteilung zurückschieben. |
| **Reward** | Die Punkte: wie viel das Depot an einem Tag gewonnen oder verloren hat. |
| **Seed** | Die Startzahl für den Zufall. Gleicher Seed, gleiches Ergebnis. |
| **State** | Die 9 Zahlen, die der Agent jeden Tag sieht. |
| **Timestep** | Ein Spielzug, also ein Tag. |
| **Turnover** | Wie viel gehandelt wird. |
| **Vieltrader** | Ein Agent, der ständig handelt – bei Fixgebühren teuer. |
