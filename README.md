# Thema 7 — RL zur semi-automatischen Portfolio-Selection (FinRL + Trade Republic)

Referat „Introduction to Reinforcement and Deep Learning", Prof. Dr. Guericke.
Aufgabenstellung und Vorüberlegungen: [`kickoff_vorbereitung.md`](kickoff_vorbereitung.md).
Das ganze Projekt ohne Vorwissen erklärt: [`ERKLAERUNG.md`](ERKLAERUNG.md).

Ein PPO-Agent verteilt Kapital auf neun Xetra-ETFs/ETCs über die großen
Anlageklassen und zahlt dabei die **feste** Ordergebühr von Trade Republic
(1 € je Order) statt der prozentualen Kosten, die FinRL eingebaut hat. Das
Ganze läuft mit drei Startkapitalen – 1.000 €, 10.000 € und 1.000.000 € –, weil
eine Fixgebühr relativ zum Kapital unterschiedlich schwer wiegt. Verglichen wird
gegen einfache Benchmarks nach Rendite und Risiko.

| Ticker | Anlageklasse | | Ticker | Anlageklasse |
|---|---|---|---|---|
| `SXR8.DE` | Aktien USA | | `D5BG.DE` | Euro-Unternehmensanleihen |
| `XSX6.DE` | Aktien Europa | | `4GLD.DE` | Gold (ETC) |
| `IQQJ.DE` | Aktien Japan | | `IQQ6.DE` | Immobilien global |
| `IQQE.DE` | Aktien Schwellenländer | | `EXXY.DE` | Rohstoffe breit |
| `EUNH.DE` | Euro-Staatsanleihen | | `EUNL.DE` | **Markt-Benchmark (B4 Aktien Welt), nicht handelbar** |

## Schnellstart

Voraussetzungen: **Python 3.12** (nicht 3.13) und Internet. Ein lokaler
FinRL-Klon ist nicht nötig.

```bash
git clone https://github.com/ertroxd/-Portfolio-Selection.git
cd -Portfolio-Selection
```

Setup, einmalig:

| System | Befehl |
|---|---|
| Windows | `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| macOS / Linux | `bash setup.sh` |

Das Skript legt `.venv` an, installiert `requirements.txt` und danach FinRL in
einer festen Version (Commit `2334a5f`) als Archiv direkt von GitHub. Am Ende
prüft es, ob der Import klappt.

Danach, im Projektordner (Windows-Pfade; unter macOS/Linux `.venv/bin/python`):

```bash
# Demo-Seite: den trainierten Agenten beim Handeln ansehen
.venv\Scripts\python.exe -m streamlit run app.py

# Smoke-Test der Trainingspipeline, 1-2 Minuten, Ergebnis bedeutungslos
.venv\Scripts\python.exe run_training.py --timesteps 2000 --seeds 42 --end 2026-09-09 --tag smoke
```

Beim ersten Start werden die Kursdaten von Yahoo Finance geladen und unter
`data/` zwischengespeichert.

## Dateien

| Datei | Zweck |
|---|---|
| `tr_env.py` | **Kern der Arbeit.** `TradeRepublicEnv` — FinRLs `StockTradingEnv` mit fixer Ordergebühr, optionalem Handelstakt und normierter Beobachtung. |
| `run_training.py` | Pipeline: Daten → Indikatoren → Train/Valid/Test → PPO über mehrere Seeds → Backtest → Benchmarks → Kennzahlen und Plot. |
| `eval_saved.py` | Lässt eine fertig trainierte Policy unter einer anderen Gebühr laufen. Trennt den Kosteneffekt vom Trainingseffekt. |
| `vergleich.py` | Vergleicht zwei Läufe gepaart über die Seeds, inklusive Einstufung als Vieltrader. |
| `app.py` | Demo-Seite (Streamlit), siehe unten. |
| `finrl_shim.py` | Macht `import finrl` ohne Broker- und Scraper-Pakete möglich. |
| `setup.ps1`, `setup.sh` | Einmaliges Setup. |
| `requirements.txt` | Exakte Paketversionen. |
| `runs/<zeitstempel>_<tag>/` | Ein Ordner je Lauf: `config.json`, `ergebnisse.csv`, `depotwert.png`, Orderlisten `actions_seed*.csv` und Modelle `ppo_seed*.zip`. |
| `data/` | Lokaler Kurs-Cache. Nicht im Repo, weil die Nutzungsbedingungen von Yahoo keine Weitergabe der Daten erlauben. |

## Demo-Seite

`app.py` lädt ein gespeichertes Modell aus `runs/`, lässt es Tag für Tag über den
Validierungs- oder Testzeitraum handeln und protokolliert jeden Schritt. Links
wählt man Lauf, Seed, Zeitraum und Markt-Vergleich; Gebühr und Handelstakt
lassen sich abweichend vom Training einstellen.

- **Depotwert** — Agent gegen Markt (MSCI World) und 1/N Buy & Hold, dazu der Vorsprung gegenüber dem Markt.
- **Trades** — Käufe, Verkäufe und Gebühren je Titel, Kursverlauf eines wählbaren Titels mit markierten Orders, das vollständige Orderbuch.
- **Depot** — Zusammensetzung aus Cash und Anlageklassen über die Zeit, in € und in Anteilen, dazu der durchschnittliche Anteil je Titel.
- **Was hat er gelernt?** — Kurzdiagnose (kein Handel, Buy & Hold, Dauerhandel oder Umschichten; größte Position; ähnlichster Benchmark; Totzone), Aktion gegen Indikator je Handelstag und eine Policy-Sonde.

Die Seite spielt auch die ersten Läufe mit 2 ETFs und rohem State ab.

## Die Läufe

Alle mit den neun Titeln oben, 1 € je Order, 60.000 Timesteps, Seeds 42–49,
Train 2016–2022 (1.778 Tage), Validierung 2023 (255), Test 2024-01-02 bis
2026-09-08 (678).

| Ordner | Startkapital | Handelstakt | `hmax` |
|---|---|---|---|
| `*_k1000_daily` | 1.000 € | täglich | 11 |
| `*_k1000_monthly` | 1.000 € | alle 21 Handelstage | 11 |
| `*_k10000_daily` | 10.000 € | täglich | 101 |
| `*_k10000_monthly` | 10.000 € | alle 21 Handelstage | 101 |
| `*_k1000000_daily` | 1.000.000 € | täglich | 10.002 |
| `*_k1000000_monthly` | 1.000.000 € | alle 21 Handelstage | 10.002 |

Dazu die erste Generation mit 2 ETFs (`EUNL.DE` + `IS3N.DE`, 10.000 €, roher
State): `*_daily_fee1` und `*_monthly_fee1`.

Nachrechnen, am Beispiel 10.000 € (für die anderen Läufe `--initial` und `--tag` anpassen):

```bash
.venv\Scripts\python.exe run_training.py --seeds 42 43 44 45 46 47 48 49 --fee 1 --initial 10000 --rebalance 1 --end 2026-09-09 --tag k10000_daily
.venv\Scripts\python.exe run_training.py --seeds 42 43 44 45 46 47 48 49 --fee 1 --initial 10000 --rebalance 21 --end 2026-09-09 --tag k10000_monthly
.venv\Scripts\python.exe vergleich.py --a "runs/*_k1000_daily" --b "runs/*_k1000000_daily"
.venv\Scripts\python.exe eval_saved.py --run runs/<ordner> --fee 0
```

Das Training ist deterministisch: gleiche Einstellungen und gleiche Seeds ergeben
bitgenau dieselben Modelle.

**Befunde:** *folgen, sobald die Läufe durch sind.*

## Was man wissen muss

**1. FinRLs Import-Kette.** `import finrl` lädt `finrl.train` und `finrl.trade`,
und die ziehen `alpaca_trade_api`, `alpaca`, `wrds`, `selenium` und
`webdriver_manager` nach — Live-Broker-, Datenbank- und Scraper-Anbindung, von
der wir nichts benutzen. `alpaca_trade_api` pinnt zusätzlich `pandas < 2`.
FinRL wird deshalb mit `--no-deps` installiert, und `finrl_shim.py` hängt einen
Import-Finder vor `sys.meta_path`, der für diese Pakete Platzhalter-Module
liefert. Die Platzhalter erfüllen nur den Import: Klassen daraus lassen sich
anlegen, aber jeder Methodenaufruf bricht mit `AttributeError` ab.
`import finrl_shim` muss deshalb **vor** jedem `finrl`-Import stehen.

**2. `cost` und `trades` nach dem Backtest.** `DRL_prediction()` verpackt die
Umgebung in einen `DummyVecEnv`, und der setzt sie nach dem letzten Schritt
automatisch zurück — `StockTradingEnv.reset()` nullt dabei `cost` und `trades`.
Wer die Gebühren danach ausliest, bekommt still **0**.
`TradeRepublicEnv.reset()` sichert die Werte vorher nach `last_episode_cost` /
`last_episode_trades`.

**3. Vergleichbarkeit über Startkapitale.** Drei Stellschrauben sorgen dafür, dass
sich die Läufe nur im ökonomischen Gewicht der Gebühr unterscheiden und nicht darin,
wie gut das neuronale Netz mit der Zahlengröße zurechtkommt:

- **Beobachtung** (`normalize_obs`): Der Agent sieht Cash-Anteil, Depotanteil je Titel, RSI/100 und MACD/Kurs statt roher Euro-Beträge. Der interne State bleibt roh.
- **`hmax`** = ⌈Startkapital ÷ Zahl der Titel ÷ billigster Kurs am ersten Trainingstag⌉. Eine volle Aktion im billigsten Titel bewegt so etwa 1/N des Kapitals.
- **`reward_scaling`** = 100 ÷ Startkapital. 1 % Tagesgewinn ergibt überall Reward 1.

**4. Totzone.** Weil `aktion × hmax` abgerundet wird, entsteht erst ab |Aktion| ≥
1/`hmax` ein Anteil — bei 1.000 € ab 0,091. Ein Agent mit zaghaften Aktionen
handelt bei kleinem Kapital deterministisch gar nicht, obwohl er im Training mit
stochastischen Aktionen gehandelt hat.

## Benchmarks

- **1/N Buy & Hold** — einmal gleich viel in jeden Titel, liegen lassen. Mit 1.000 € sind nur 5 der 9 Titel bezahlbar; nicht ausgeführte Käufe kosten keine Gebühr.
- **Markt: 100 % MSCI World (`EUNL.DE`) Buy & Hold** — separat geladen, für den Agenten nicht handelbar.
- **1/N periodisch rebalanciert** — gleiche Gebühr wie der Agent, aber keinerlei Intelligenz.

Alle kaufen nur ganze Stücke und zahlen dieselbe Gebühr wie der Agent.

## Reproduzierbarkeit

- FinRL ist auf Commit `2334a5f` gepinnt, alle Pakete auf exakte Versionen, jeder Lauf auf feste Seeds.
- **`--end` immer fest setzen.** Der Default ist „heute", dann wandert das Testfenster mit dem Kalender. `data_split()` schneidet mit `date < end` ab: für einen Test bis einschließlich 08.09.2026 also `--end 2026-09-09`.
- Die Kursdaten werden beim ersten Lauf neu von Yahoo geladen. Yahoo korrigiert Historien gelegentlich nachträglich; Nachrechnungen können deshalb minimal von `runs/*/ergebnisse.csv` abweichen.
- Laufordner ohne `normalize_obs` in der `config.json` (erste Generation) werden weiter mit rohem State ausgewertet.

## Offene Punkte

- `--end` ist im Default noch „heute".
- Der Docstring von `finrl_shim.py` behauptet, die Platzhalter fielen schon beim Anlegen auf — tatsächlich erst beim Methodenaufruf.
- `hmax` gilt in Stück für alle Titel gleich; eine volle Aktion ist bei `SXR8` (≈ 450 €) ein Vielfaches des Betrags bei `IQQJ` (≈ 15 €).
- Tage ohne Umsatz (z. B. `EUNH.DE`) werden mit dem von Yahoo gemeldeten Kurs als handelbar behandelt.
- Nur ein Testfenster, keine Walk-Forward-Validierung.
- Der Reward ist `ΔVermögen`, also risikoneutral.
- Beim Handelstakt kennt der Agent seine Handelstage nicht; Aktionen an gesperrten Tagen werden verworfen.
- Die PPO-Hyperparameter sind FinRL-Defaults und ungeprüft.
- Nur ganze Stücke, kein Spread, keine Steuern — bewusst, siehe `kickoff_vorbereitung.md`.
