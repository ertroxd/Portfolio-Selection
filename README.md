# Thema 7 — RL zur semi-automatischen Portfolio-Selection (FinRL + Trade Republic)

Referat "Introduction to Reinforcement and Deep Learning", Prof. Dr. Guericke.
Aufgabenstellung und Vorüberlegungen: siehe [`kickoff_vorbereitung.md`](kickoff_vorbereitung.md).

## Was hier drin ist

| Datei | Zweck |
|---|---|
| `tr_env.py` | **Kern der Arbeit.** `TradeRepublicEnv` — FinRLs `StockTradingEnv` mit *fixer* Ordergebühr statt prozentualer Kosten. |
| `run_training.py` | Pipeline: Daten → Features → Split → PPO über mehrere Seeds → Backtest → Benchmarks → Kennzahlen + Plot. |
| `finrl_shim.py` | Macht `import finrl` ohne Broker-/Scraper-Abhängigkeiten möglich. Siehe unten. |
| `requirements.txt` | Exakte Versionen der lauffähigen Umgebung. |
| `runs/<zeitstempel>_<tag>/` | Ein Ordner je Lauf: `config.json`, `ergebnisse.csv`, `depotwert.png`, Modelle, Orderlisten. |
| `data/` | Kurs-Cache (CSV). Nicht löschen — sonst sind Läufe nicht mehr vergleichbar. |
| `thema7_portfolio_selection_rl.ipynb` | Kommentiertes Notebook für die Präsentation (Colab). |

## Setup

Einmalig, Python **3.12** (nicht 3.13 — pandas 3.x bricht Teile von FinRL):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install --no-deps -e ../FinRL
```

Statt des lokalen Klons geht auch:
`pip install --no-deps git+https://github.com/AI4Finance-Foundation/FinRL.git`

Prüfen, ob es steht:

```bash
.venv/Scripts/python.exe -c "import finrl_shim; from tr_env import TradeRepublicEnv; print('ok')"
```

## Läufe starten

```bash
# Smoke-Test, ca. 1 Minute, Ergebnis ist bedeutungslos
.venv/Scripts/python.exe run_training.py --timesteps 2000 --seeds 42 --tag smoke

# Erster echter Lauf: tägliches Handeln, 3 Seeds
.venv/Scripts/python.exe run_training.py --timesteps 60000 --seeds 42 43 44 --tag daily

# Monatliches Rebalancing ("semi-automatisch")
.venv/Scripts/python.exe run_training.py --timesteps 60000 --seeds 42 43 44 --rebalance 21 --tag monthly

# Gebühren-Sensitivität: was kostet uns die Pauschale wirklich?
.venv/Scripts/python.exe run_training.py --timesteps 60000 --seeds 42 43 44 --fee 0 --tag nofee
```

Wichtige Schalter: `--tickers`, `--start`, `--train-end`, `--valid-end`, `--fee`,
`--rebalance`, `--initial`, `--hmax`, `--timesteps`, `--seeds`.
Alles Weitere: `run_training.py --help`.

## Zwei Dinge, die man wissen muss

**1. FinRLs Import-Kette.** `import finrl` lädt `finrl.train` und `finrl.trade`, und die
ziehen `alpaca_trade_api`, `alpaca`, `wrds`, `selenium` und `webdriver_manager` nach —
Live-Broker-, Datenbank- und Scraper-Anbindung, von der wir nichts benutzen.
`alpaca_trade_api` ist zusätzlich unmaintained und pinnt `pandas < 2`. `finrl_shim.py`
ersetzt diese Paketbäume durch Platzhalter, bevor FinRL importiert wird. Deshalb muss
`import finrl_shim` **vor** jedem `finrl`-Import stehen.

**2. `cost` und `trades` nach dem Backtest.** `DRL_prediction()` verpackt die Umgebung in
einen `DummyVecEnv`, und der setzt sie nach dem letzten Schritt automatisch zurück —
`StockTradingEnv.reset()` nullt dabei `cost` und `trades`. Wer die Gebühren hinterher
ausliest, bekommt still und leise **0**. `TradeRepublicEnv.reset()` sichert die Werte
deshalb vorher nach `last_episode_cost` / `last_episode_trades`.

## Benchmarks

- **1/N Buy & Hold** — einmal kaufen, liegen lassen. `N` Gebühren insgesamt.
- **100 % erster Ticker B&H** — der übliche „gegen den Index"-Vergleich.
- **1/N periodisch rebalanciert** — gleiche Handelsfrequenz und gleiche Gebühr wie der
  Agent, aber keinerlei Intelligenz. Der ehrlichste Gegner.

## Offene Punkte

- Nur ein Testfenster, keine Walk-Forward-Validierung.
- Reward ist `ΔVermögen`, also risikoneutral. Risiko taucht erst in der Auswertung auf.
- Spread und Slippage sind nicht modelliert — bewusst, siehe `kickoff_vorbereitung.md`.
- `reward_scaling=1e-4` stammt aus FinRLs Default für 1 Mio. € Startkapital. Bei 10.000 €
  ist der Reward entsprechend winzig; ob das PPO stört, ist noch nicht geprüft
  (`--reward-scaling` ist deshalb ein Schalter).
