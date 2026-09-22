"""Ein bereits trainiertes Modell unter geaenderten Kosten auswerten.

Warum das noetig ist
--------------------
Ein zweiter Trainingslauf mit `--fee 0` vermischt zwei Effekte:
  (a) die Gebuehr kostet Geld, und
  (b) die Gebuehr veraendert waehrend des Trainings die gelernte Policy.
Beides zusammen ist nicht interpretierbar - der Unterschied zwischen zwei Laeufen
kann genauso gut reine Seed-Varianz sein.

Dieses Skript misst nur (a): es laedt die Policies eines abgeschlossenen Laufs und
laesst sie unveraendert unter einer anderen Gebuehr durch denselben Testzeitraum
laufen. Alles andere - Daten, Splits, Seeds, Gewichte - bleibt identisch.

Beispiel
--------
    python eval_saved.py --run runs/20260910-120216_daily --fee 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import finrl_shim  # noqa: F401  - muss vor jedem finrl-Import stehen
import pandas as pd
from stable_baselines3 import PPO

import run_training as rt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Ordner eines abgeschlossenen Laufs")
    p.add_argument("--fee", type=float, required=True, help="Gebuehr fuer die Auswertung")
    p.add_argument("--rebalance", type=int, default=None, help="optional ueberschreiben")
    opts = p.parse_args()

    run = Path(opts.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    rebalance = opts.rebalance if opts.rebalance is not None else args.rebalance

    print(f"[eval] Lauf     : {run.name}")
    print(f"[eval] trainiert: fee={args.fee:.2f}  rebalance={args.rebalance}")
    print(f"[eval] jetzt    : fee={opts.fee:.2f}  rebalance={rebalance}")

    raw = rt.load_prices(args.tickers, args.start, args.end)
    processed = rt.add_features(raw, args.indicators)

    from finrl.meta.preprocessor.preprocessors import data_split
    from finrl.agents.stablebaselines3.models import DRLAgent

    test = data_split(processed, args.valid_end, args.end)

    rows = []
    for zip_path in sorted(run.glob("ppo_seed*.zip")):
        seed = zip_path.stem.replace("ppo_seed", "")
        env = rt.build_env(test, args, args.indicators, opts.fee, rebalance)
        model = PPO.load(str(zip_path), device="cpu")
        df_acc, _ = DRLAgent.DRL_prediction(model=model, environment=env)

        curve = df_acc.set_index("date")["account_value"]
        curve.index = pd.to_datetime(curve.index)
        # aeltere Laeufe haben kein "risk_free" in config.json -> Modul-Default nutzen
        m = rt.metrics(curve, risk_free_annual=getattr(args, "risk_free", rt.RISK_FREE_RATE))
        m.update(Seed=seed,
                 Orders=env.last_episode_trades or env.trades,
                 Gebuehren=env.last_episode_cost or env.cost)
        rows.append(m)

    res = pd.DataFrame(rows)[["Seed", "Endwert", "CAGR", "Sharpe", "MaxDD",
                              "Orders", "Gebuehren"]]
    print("\n=== Testzeitraum, Policy unveraendert ===")
    print(res.to_string(index=False,
                        formatters={"Endwert": "{:,.0f}".format,
                                    "CAGR": "{:.2%}".format,
                                    "Sharpe": "{:.2f}".format,
                                    "MaxDD": "{:.2%}".format,
                                    "Gebuehren": "{:,.0f}".format}))


if __name__ == "__main__":
    main()
