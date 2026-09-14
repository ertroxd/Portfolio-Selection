# Thema 7 — RL zur semi-automatischen Portfolio-Selection (FinRL + Trade Republic)

Referat „Introduction to Reinforcement and Deep Learning", Prof. Dr. Guericke.
Aufgabenstellung und Vorüberlegungen: [`kickoff_vorbereitung.md`](kickoff_vorbereitung.md).

Ein PPO-Agent verteilt 10.000 € auf zwei Xetra-ETFs — iShares Core MSCI World
(`EUNL.DE`) und iShares Core MSCI EM IMI (`IS3N.DE`) — und zahlt dabei die
**feste** Ordergebühr von Trade Republic (1 € je Order) statt der prozentualen
Kosten, die FinRL eingebaut hat. Verglichen wird gegen einfache Benchmarks nach
Rendite und Risiko.

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

# Smoke-Test der Trainingspipeline, ca. 1 Minute, Ergebnis bedeutungslos
.venv\Scripts\python.exe run_training.py --timesteps 2000 --seeds 42 --end 2026-09-09 --tag smoke
```

Beim ersten Start werden die Kursdaten von Yahoo Finance geladen und unter
`data/` zwischengespeichert.

## Dateien

| Datei | Zweck |
|---|---|
| `tr_env.py` | **Kern der Arbeit.** `TradeRepublicEnv` — FinRLs `StockTradingEnv` mit fixer Ordergebühr und optionalem Handelstakt. |
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

- **Depotwert** — Agent gegen MSCI World Buy & Hold und 1/N Buy & Hold, dazu der Vorsprung gegenüber dem Markt.
- **Trades** — Kursverlauf je ETF mit markierten Käufen und Verkäufen, das vollständige Orderbuch und die Zahl der Order-Wünsche, die nicht ausgeführt wurden.
- **Depot** — Zusammensetzung aus Cash und ETFs über die Zeit, in € und in Anteilen.
- **Was hat er gelernt?** — Kurzdiagnose (Buy & Hold, Dauerhandel oder Umschichten; welchem Benchmark die Kurve am ähnlichsten ist), Aktion gegen Indikator je Handelstag und eine Policy-Sonde, die einen Indikator verschiebt und zeigt, wie die Policy reagiert.

## Die bisherigen Läufe

Alle mit `EUNL.DE` + `IS3N.DE`, 10.000 € Start, 60.000 Timesteps, Train
2016–2022, Validierung 2023, Test 2024-01-02 bis 2026-09-08.

| Ordner | Gebühr | Handelstakt | Seeds |
|---|---|---|---|
| `*_fee1_8seeds` | 1 € | täglich | 42–49 |
| `*_fee0_8seeds` | 0 € | täglich | 42–49 |
| `*_daily` | 1 € | täglich | 42–44 |
| `*_monthly` | 1 € | alle 21 Handelstage | 42–44 |
| `*_nofee` | 0 € | täglich | 42–44 |
| `*_smoke`, `*_smoke2` | 1 € | täglich | 42, 2.000 Timesteps — nur Funktionstest |

Die beiden 8-Seed-Läufe reproduzieren:

```bash
.venv\Scripts\python.exe run_training.py --timesteps 60000 --seeds 42 43 44 45 46 47 48 49 --fee 1 --end 2026-09-09 --tag fee1_8seeds
.venv\Scripts\python.exe run_training.py --timesteps 60000 --seeds 42 43 44 45 46 47 48 49 --fee 0 --end 2026-09-09 --tag fee0_8seeds
.venv\Scripts\python.exe vergleich.py --a "runs/*_fee1_8seeds" --b "runs/*_fee0_8seeds"
```

**Befunde aus den 8-Seed-Läufen:**

- PPO schlägt 1/N Buy & Hold nicht: Sharpe 1,22 ± 0,08 mit Gebühr, 1,23 ± 0,07 ohne, 1,32 beim Benchmark. Gepaarter Wilcoxon-Test mit gegen ohne Gebühr: p = 0,84.
- In beiden Varianten werden 3 von 8 Policies zu Vieltradern. Die Gebühr verhindert das nicht, halbiert aber ungefähr die Zahl der Orders.
- Der reine Gebühreneffekt bei identischer Policy reicht von −2 € (2 Orders) bis −842 € (681 Orders). Kostenrelevant wird die Fixgebühr über den Turnover, nicht über den Depotwert.

## Zwei Dinge, die man wissen muss

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

## Benchmarks

- **1/N Buy & Hold** — einmal kaufen, liegen lassen.
- **100 % MSCI World (`EUNL.DE`) Buy & Hold** — der Marktvergleich.
- **1/N periodisch rebalanciert** — gleiche Gebühr wie der Agent, aber keinerlei Intelligenz.

Alle kaufen nur ganze Stücke und zahlen dieselbe Gebühr wie der Agent.

## Reproduzierbarkeit

- FinRL ist auf Commit `2334a5f` gepinnt, alle Pakete auf exakte Versionen, jeder Lauf auf feste Seeds.
- **`--end` immer fest setzen.** Der Default ist „heute", dann wandert das Testfenster mit dem Kalender. `data_split()` schneidet mit `date < end` ab: für einen Test bis einschließlich 08.09.2026 also `--end 2026-09-09`.
- Die Kursdaten werden beim ersten Lauf neu von Yahoo geladen. Yahoo korrigiert Historien gelegentlich nachträglich; Nachrechnungen können deshalb minimal von `runs/*/ergebnisse.csv` abweichen.

## Offene Punkte

- `--end` ist im Default noch „heute".
- `vergleich.py` rechnet mit fest eingetragenen 679 Handelstagen; bei einem anderen Testzeitraum stimmt die Vieltrader-Einstufung nicht mehr.
- Der Docstring von `finrl_shim.py` behauptet, die Platzhalter fielen schon beim Anlegen auf — tatsächlich erst beim Methodenaufruf.
- Nur ein Testfenster, keine Walk-Forward-Validierung.
- Der monatliche Lauf hat nur 3 Seeds und ist damit nicht belastbar.
- Der Reward ist `ΔVermögen`, also risikoneutral.
- Beim Handelstakt kennt der Agent seine Handelstage nicht; Aktionen an gesperrten Tagen werden verworfen.
- `reward_scaling` und die PPO-Hyperparameter sind FinRL-Defaults und ungeprüft.
- Nur ganze Stücke, kein Spread, keine Steuern — bewusst, siehe `kickoff_vorbereitung.md`.
