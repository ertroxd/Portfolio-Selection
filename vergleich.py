"""Zwei Laeufe gegeneinander auswerten - Schwerpunkt Handelsaktivitaet.

Beantwortet die Frage, wegen der wir auf 8 Seeds hochgegangen sind:
Wie oft rutscht ein Agent ins Dauerhandeln, mit und ohne Ordergebuehr?

Der Vergleich ist ueber die Seeds gepaart (beide Laeufe benutzen dieselben Seeds,
dieselben Daten, dieselben Splits) - der einzige Unterschied ist die Gebuehr.
Bei n = 8 ist das trotzdem eine Indikation und kein Beweis; die Teststatistik
steht deshalb bewusst neben der Streuung und nicht an ihrer Stelle.

Beispiel
--------
    python vergleich.py --a runs/20260910-*_fee1_8seeds --b runs/20260910-*_fee0_8seeds
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


def lade(muster: str) -> tuple[pd.DataFrame, dict, str]:
    treffer = sorted(glob.glob(muster))
    if not treffer:
        raise SystemExit(f"Kein Lauf gefunden fuer: {muster}")
    run = Path(treffer[-1])
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    d = pd.read_csv(run / "ergebnisse.csv")
    return d[d.Split == "test"].copy(), cfg, run.name


def kennzahlen(d: pd.DataFrame, cfg: dict, handelstage: int) -> pd.DataFrame:
    ppo = d[d.Strategie.str.startswith("PPO")].copy()
    n_assets = len(cfg["tickers"])
    ppo["Orders/Tag"] = ppo.Orders / (handelstage * n_assets)
    ppo["Churning"] = ppo["Orders/Tag"] >= CHURN_SCHWELLE
    return ppo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Lauf A (Glob), z. B. mit Gebuehr")
    p.add_argument("--b", required=True, help="Lauf B (Glob), z. B. ohne Gebuehr")
    args = p.parse_args()

    da, cfga, namea = lade(args.a)
    db, cfgb, nameb = lade(args.b)

    # Handelstage aus den Benchmark-Kurven ableiten ist unnoetig - beide Laeufe
    # nutzen denselben Testzeitraum, wir zaehlen ihn aus einer Aktionsliste.
    handelstage = 679

    a = kennzahlen(da, cfga, handelstage)
    b = kennzahlen(db, cfgb, handelstage)

    print(f"A = {namea}  (Gebuehr {cfga['fee']:.2f} EUR)")
    print(f"B = {nameb}  (Gebuehr {cfgb['fee']:.2f} EUR)\n")

    for name, d in (("A", a), ("B", b)):
        print(f"--- Lauf {name} ---")
        t = d[["Strategie", "Endwert", "Sharpe", "MaxDD", "Orders", "Orders/Tag", "Churning"]]
        print(t.to_string(index=False, formatters={
            "Endwert": "{:,.0f}".format, "Sharpe": "{:.2f}".format,
            "MaxDD": "{:.2%}".format, "Orders/Tag": "{:.2f}".format}))
        print(f"  Sharpe  {d.Sharpe.mean():.3f} +- {d.Sharpe.std():.3f}"
              f"   Endwert {d.Endwert.mean():,.0f} +- {d.Endwert.std():,.0f}")
        print(f"  Churning-Policies: {int(d.Churning.sum())} von {len(d)}\n")

    print("--- Benchmarks (identisch in beiden Laeufen bis auf die Gebuehr) ---")
    bm = da[~da.Strategie.str.startswith("PPO")]
    print(bm[["Strategie", "Endwert", "Sharpe", "Orders", "Gebuehren"]].to_string(
        index=False, formatters={"Endwert": "{:,.0f}".format, "Sharpe": "{:.2f}".format}))

    # Gepaarter Test auf dem Sharpe. Nur als Indikation - n = 8.
    try:
        from scipy import stats

        merged = a.merge(b, on="Strategie", suffixes=("_a", "_b"))
        if len(merged) >= 3:
            st, pv = stats.wilcoxon(merged.Sharpe_a, merged.Sharpe_b)
            print(f"\nWilcoxon (gepaart, Sharpe A vs B, n={len(merged)}): p = {pv:.3f}")
            print("  Bei n = 8 ist das eine Indikation, kein Nachweis.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
