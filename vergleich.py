"""Zwei Laeufe gegeneinander auswerten - Schwerpunkt Handelsaktivitaet.

Der Vergleich ist ueber die Seeds gepaart: Beide Laeufe benutzen dieselben Seeds,
dieselben Daten und dieselben Splits. Was sich unterscheidet, steht im Kopf der
Ausgabe (Startkapital, Gebuehr, Handelstakt).
Bei n = 8 ist das eine Indikation und kein Beweis; die Teststatistik steht deshalb
bewusst neben der Streuung und nicht an ihrer Stelle.

Beispiel
--------
    python vergleich.py --a "runs/*_k10000_daily" --b "runs/*_k10000_monthly"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

#: Ab wie vielen Orders je Handelstag und Titel gilt eine Policy als "Churning"?
#: 0.5 heisst: der Agent handelt im Schnitt jeden zweiten Tag jeden Titel.
CHURN_SCHWELLE = 0.5


def lade(muster: str) -> tuple[pd.DataFrame, dict, Path]:
    treffer = sorted(glob.glob(muster))
    if not treffer:
        raise SystemExit(f"Kein Lauf gefunden fuer: {muster}")
    run = Path(treffer[-1])
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    d = pd.read_csv(run / "ergebnisse.csv")
    return d[d.Split == "test"].copy(), cfg, run


def handelstage(run: Path) -> int:
    """Zahl der Handelsschritte im Testzeitraum, aus einer Orderliste abgelesen."""
    listen = sorted(run.glob("actions_seed*.csv"))
    if not listen:
        raise SystemExit(f"Keine Orderliste in {run}")
    return len(pd.read_csv(listen[0]))


def kennzahlen(d: pd.DataFrame, cfg: dict, tage: int) -> pd.DataFrame:
    ppo = d[d.Strategie.str.startswith("PPO")].copy()
    ppo["Orders/Tag"] = ppo.Orders / (tage * len(cfg["tickers"]))
    ppo["Churning"] = ppo["Orders/Tag"] >= CHURN_SCHWELLE
    ppo["Gebuehren %"] = ppo.Gebuehren / cfg["initial"]
    return ppo


def kopf(cfg: dict) -> str:
    return (f"Start {cfg['initial']:,.0f} EUR, Gebuehr {cfg['fee']:.2f} EUR, "
            f"Handeln alle {cfg['rebalance']} T, {len(cfg['tickers'])} Titel").replace(",", ".")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Lauf A (Glob)")
    p.add_argument("--b", required=True, help="Lauf B (Glob)")
    args = p.parse_args()

    da, cfga, runa = lade(args.a)
    db, cfgb, runb = lade(args.b)
    a = kennzahlen(da, cfga, handelstage(runa))
    b = kennzahlen(db, cfgb, handelstage(runb))

    print(f"A = {runa.name}  ({kopf(cfga)})")
    print(f"B = {runb.name}  ({kopf(cfgb)})\n")

    for name, d, roh in (("A", a, da), ("B", b, db)):
        print(f"--- Lauf {name} ---")
        t = d[["Strategie", "Endwert", "Sharpe", "MaxDD", "Orders", "Gebuehren %",
               "Orders/Tag", "Churning"]]
        print(t.to_string(index=False, formatters={
            "Endwert": "{:,.0f}".format, "Sharpe": "{:.2f}".format,
            "MaxDD": "{:.2%}".format, "Gebuehren %": "{:.2%}".format,
            "Orders/Tag": "{:.2f}".format}))
        print(f"  Sharpe  {d.Sharpe.mean():.3f} +- {d.Sharpe.std():.3f}"
              f"   Endwert {d.Endwert.mean():,.0f} +- {d.Endwert.std():,.0f}")
        print(f"  Churning-Policies: {int(d.Churning.sum())} von {len(d)}")
        bm = roh[~roh.Strategie.str.startswith("PPO")]
        print("  Benchmarks:")
        print(bm[["Strategie", "Endwert", "Sharpe", "Orders", "Gebuehren"]].to_string(
            index=False, formatters={"Endwert": "{:,.0f}".format, "Sharpe": "{:.2f}".format}))
        print()

    # Gepaarter Test auf dem Sharpe. Nur als Indikation - n = 8.
    try:
        from scipy import stats

        merged = a.merge(b, on="Strategie", suffixes=("_a", "_b"))
        if len(merged) >= 3:
            _, pv = stats.wilcoxon(merged.Sharpe_a, merged.Sharpe_b)
            print(f"Wilcoxon (gepaart, Sharpe A vs B, n={len(merged)}): p = {pv:.3f}")
            print("  Bei n = 8 ist das eine Indikation, kein Nachweis.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
