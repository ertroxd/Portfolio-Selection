"""Thema 7 - Trainings- und Backtest-Pipeline.

Ein Lauf = Daten laden -> Features -> Split -> PPO trainieren (ueber mehrere Seeds)
-> auf Validierung und Test auswerten -> gegen einfache Benchmarks vergleichen
-> Kennzahlen und Plot in einen Ergebnisordner schreiben.

Beispiele
---------
    # Smoke-Test: laeuft in 1-2 Minuten durch, Ergebnis ist bedeutungslos
    python run_training.py --timesteps 2000 --seeds 42 --end 2026-09-09 --tag smoke

    # 8 Seeds, 10.000 EUR Startkapital, taegliches Handeln
    python run_training.py --seeds 42 43 44 45 46 47 48 49 --initial 10000 --end 2026-09-09 --tag k10000_daily

    # dasselbe, Handeln nur alle 21 Handelstage ("semi-automatisch")
    python run_training.py --seeds 42 43 44 45 46 47 48 49 --initial 10000 --rebalance 21 --end 2026-09-09 --tag k10000_monthly
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import finrl_shim  # noqa: F401  - muss vor jedem finrl-Import stehen
import matplotlib

matplotlib.use("Agg")  # keine GUI noetig, wir speichern nur PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"

TRADING_DAYS = 252
#: Grobe Annahme fuer den risikofreien Zins (Sharpe/Sortino) - wir binden keine
#: historische EUR-Zinsreihe (z. B. EURIBOR/€STR) ein, sondern nehmen einen festen
#: Mittelwert, der die Spanne 2016-2026 ueberschlaegig abdeckt (negative
#: EZB-Einlagenzinsen bis 2021, danach steigend). Ueberschreibbar per --risk-free.
RISK_FREE_RATE = 0.02

#: Handelbares Universum: neun Xetra-ETFs/ETCs ueber die grossen Anlageklassen.
UNIVERSUM = {
    "SXR8.DE": "Aktien USA",
    "XSX6.DE": "Aktien Europa",
    "IQQJ.DE": "Aktien Japan",
    "IQQE.DE": "Aktien Schwellenlaender",
    "EUNH.DE": "Euro-Staatsanleihen",
    "D5BG.DE": "Euro-Unternehmensanleihen",
    "4GLD.DE": "Gold (ETC)",
    "IQQ6.DE": "Immobilien global",
    "EXXY.DE": "Rohstoffe breit",
}

#: Markt-Benchmark B4 "Aktien Welt". Wird nicht gehandelt, nur verglichen.
MARKT = "EUNL.DE"


# ----------------------------------------------------------------------
# 1. Daten
# ----------------------------------------------------------------------
def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Kursdaten von Yahoo laden, im Langformat, mit lokalem Cache.

    Der Cache ist wichtig: Yahoo drosselt und liefert gelegentlich andere
    Historien zurueck. Ohne Cache waeren zwei Laeufe nicht vergleichbar.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"prices_{'-'.join(tickers)}_{start}_{end}.csv"
    if cache.exists():
        print(f"[data] Cache: {cache.name}")
        return pd.read_csv(cache)

    import yfinance as yf

    print(f"[data] Download {tickers} {start} .. {end}")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit("yfinance hat nichts geliefert - Ticker/Zeitraum pruefen.")

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=1, future_stack=True).reset_index()
    else:  # einzelner Ticker: Ticker-Ebene fehlt
        df = raw.reset_index()
        df["Ticker"] = tickers[0]

    df = df.rename(
        columns={
            "Date": "date",
            "Ticker": "tic",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume", "tic"]]
    df = df.dropna().sort_values(["date", "tic"]).reset_index(drop=True)

    # Nur Tage behalten, an denen ALLE Titel gehandelt wurden. Sonst rutscht
    # das State-Layout von StockTradingEnv durcheinander.
    complete = df.groupby("date")["tic"].nunique() == len(tickers)
    df = df[df.date.isin(complete[complete].index)].reset_index(drop=True)

    df.to_csv(cache, index=False)
    print(f"[data] {len(df)} Zeilen, {df.date.nunique()} Handelstage -> {cache.name}")
    return df


def add_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    out = fe.preprocess_data(df)
    return out.sort_values(["date", "tic"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Kennzahlen
# ----------------------------------------------------------------------
def metrics(curve: pd.Series, risk_free_annual: float = RISK_FREE_RATE) -> dict:
    """Rendite- und Risikokennzahlen einer Depotwert-Zeitreihe.

    `risk_free_annual` wird in eine taegliche Rate umgerechnet und von den
    Tagesrenditen abgezogen, bevor Sharpe/Sortino berechnet werden - vorher
    wurde hier faelschlich mit 0 % risikofreiem Zins gerechnet (Sharpe = Rendite
    / Vola statt Ueberschussrendite / Vola).
    """
    curve = curve.dropna()
    if len(curve) < 3:
        return {k: float("nan") for k in
                ("CAGR", "Vola", "Sharpe", "Sortino", "MaxDD", "Endwert")}

    ret = curve.pct_change().dropna()
    years = len(curve) / TRADING_DAYS
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vola = ret.std() * np.sqrt(TRADING_DAYS)

    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = ret - rf_daily
    sharpe = (excess.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan
    maxdd = (curve / curve.cummax() - 1).min()

    return {
        "CAGR": cagr,
        "Vola": vola,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Endwert": curve.iloc[-1],
    }


# ----------------------------------------------------------------------
# 3. Benchmarks
# ----------------------------------------------------------------------
def _prices_wide(df: pd.DataFrame) -> pd.DataFrame:
    w = df.pivot(index="date", columns="tic", values="close").sort_index()
    w.index = pd.to_datetime(w.index)
    return w


def bh_equal(df: pd.DataFrame, initial: float, fee: float) -> tuple[pd.Series, dict]:
    """1/N Buy & Hold: einmal kaufen, liegen lassen.

    Mit kleinem Kapital und ganzen Stuecken bleiben teure Titel bei 0 Stueck.
    Dann findet fuer sie keine Order statt, und es wird auch keine Gebuehr gezaehlt.
    """
    px = _prices_wide(df)
    n = px.shape[1]
    budget = (initial - n * fee) / n
    shares = (budget // px.iloc[0]).clip(lower=0).astype(int)
    orders = int((shares > 0).sum())
    cash = initial - float((shares * px.iloc[0]).sum()) - orders * fee
    curve = px.mul(shares, axis=1).sum(axis=1) + cash
    return curve.rename("1/N Buy & Hold"), {"Gebuehren": orders * fee, "Orders": orders}


def bh_single(df: pd.DataFrame, initial: float, fee: float, tic: str) -> tuple[pd.Series, dict]:
    """100 % in einen Titel, Buy & Hold."""
    px = _prices_wide(df)[[tic]]
    shares = int((initial - fee) // px.iloc[0, 0])
    orders = 1 if shares > 0 else 0
    cash = initial - shares * px.iloc[0, 0] - orders * fee
    curve = px.iloc[:, 0] * shares + cash
    return curve.rename(f"100% {tic} B&H"), {"Gebuehren": orders * fee, "Orders": orders}


def rebalance_periodic(
    df: pd.DataFrame, initial: float, fee: float, every: int
) -> tuple[pd.Series, dict]:
    """1/N, alle `every` Handelstage auf Gleichgewicht zurueckgesetzt.

    Der ehrlichste Gegner fuer den Agenten: gleiche Handelsfrequenz,
    gleiche Gebuehr, aber keinerlei Intelligenz.
    """
    px = _prices_wide(df)
    n = px.shape[1]
    shares = pd.Series(0, index=px.columns, dtype=int)
    cash = float(initial)
    values, orders, fees = [], 0, 0.0

    for i, (_, row) in enumerate(px.iterrows()):
        if i % every == 0:
            total = cash + float((shares * row).sum())
            target_val = total / n
            new_shares = pd.Series(
                {t: int(max(0.0, target_val - fee) // row[t]) for t in px.columns}
            )
            traded = (new_shares != shares).sum()
            if traded:
                cash = total - float((new_shares * row).sum()) - traded * fee
                if cash >= 0:  # sonst Rebalancing verwerfen
                    shares, orders, fees = new_shares, orders + traded, fees + traded * fee
                else:
                    cash = total - float((shares * row).sum())
        values.append(cash + float((shares * row).sum()))

    label = f"1/N rebal. alle {every}T"
    return pd.Series(values, index=px.index, name=label), {"Gebuehren": fees, "Orders": orders}


# ----------------------------------------------------------------------
# 4. Hauptlauf
# ----------------------------------------------------------------------
def build_env(split_df, args, indicators, fee, rebalance):
    from tr_env import TradeRepublicEnv

    stock_dim = len(split_df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(indicators) * stock_dim
    kwargs = dict(
        hmax=args.hmax,
        initial_amount=args.initial,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.0] * stock_dim,   # prozentuale Kosten aus - wir nutzen die Fixgebuehr
        sell_cost_pct=[0.0] * stock_dim,
        state_space=state_space,
        stock_dim=stock_dim,
        action_space=stock_dim,
        tech_indicator_list=indicators,
        reward_scaling=args.reward_scaling,
        print_verbosity=10**9,            # FinRLs Episoden-Prints unterdruecken
    )
    # Alte Laeufe haben kein normalize_obs in der config -> roher State wie damals.
    return TradeRepublicEnv(df=split_df, fixed_fee=fee, rebalance_every=rebalance,
                            normalize_obs=getattr(args, "normalize_obs", False), **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=list(UNIVERSUM))
    p.add_argument("--markt", default=MARKT, help="Markt-Benchmark, wird nicht gehandelt")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--train-end", default="2023-01-01", help="exklusiv")
    p.add_argument("--valid-end", default="2024-01-01", help="exklusiv")
    p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"),
                   help="exklusiv - fuer vergleichbare Laeufe immer fest setzen")
    p.add_argument("--indicators", nargs="+", default=["rsi_30", "macd"])
    p.add_argument("--timesteps", type=int, default=60_000)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--fee", type=float, default=1.0, help="EUR je Order")
    p.add_argument("--rebalance", type=int, default=1, help="1=taeglich, 21=monatlich")
    p.add_argument("--initial", type=float, default=10_000.0)
    p.add_argument("--risk-free", type=float, default=RISK_FREE_RATE, dest="risk_free",
                   help="Jaehrlicher risikofreier Zins fuer Sharpe/Sortino (Annahme, siehe RISK_FREE_RATE)")
    p.add_argument("--hmax", type=int, default=None,
                   help="max. Stueck je Order. Standard: eine volle Aktion im billigsten "
                        "Titel bewegt 1/N des Startkapitals")
    p.add_argument("--reward-scaling", type=float, default=None,
                   help="Standard: 100 / Startkapital")
    p.add_argument("--rohe-beobachtung", action="store_true",
                   help="Agent sieht rohe Euro-Betraege statt Anteilen (wie die ersten Laeufe)")
    p.add_argument("--tag", default="run")
    args = p.parse_args()
    args.normalize_obs = not args.rohe_beobachtung
    if args.reward_scaling is None:
        # Reward = Vermoegensaenderung * Skalierung. So ist er relativ zum Startkapital
        # immer gleich gross: 1 % Gewinn gibt bei jedem Kapital denselben Reward.
        args.reward_scaling = 100.0 / args.initial

    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.meta.preprocessor.preprocessors import data_split

    # --- Daten ---
    raw = load_prices(args.tickers, args.start, args.end)
    processed = add_features(raw, args.indicators)

    train = data_split(processed, args.start, args.train_end)
    valid = data_split(processed, args.train_end, args.valid_end)
    test = data_split(processed, args.valid_end, args.end)
    for name, part in (("Train", train), ("Valid", valid), ("Test ", test)):
        if part.empty:
            raise SystemExit(f"{name}-Split ist leer - Zeitraeume pruefen.")
        print(f"[split] {name}: {part.date.min()} .. {part.date.max()}  "
              f"({part.date.nunique()} Handelstage)")

    if args.hmax is None:
        erster_tag = train[train.date == train.date.min()]
        args.hmax = max(1, math.ceil(args.initial / len(args.tickers) / erster_tag.close.min()))

    out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[run ] {out}")
    print(f"[run ] Startkapital {args.initial:,.0f} EUR | {len(args.tickers)} Titel | "
          f"hmax {args.hmax} | reward_scaling {args.reward_scaling:g} | "
          f"Beobachtung {'Anteile' if args.normalize_obs else 'roh'}")

    # --- Benchmarks (auf dem Testzeitraum) ---
    test_tage = test.date.unique()
    test_raw = raw[raw.date.isin(test_tage)]
    markt_raw = load_prices([args.markt], args.start, args.end)
    markt_test = markt_raw[markt_raw.date.isin(test_tage)]
    benchmarks, bench_info = {}, {}
    for curve, info in (
        bh_equal(test_raw, args.initial, args.fee),
        bh_single(markt_test, args.initial, args.fee, args.markt),
        rebalance_periodic(test_raw, args.initial, args.fee, max(args.rebalance, 21)),
    ):
        benchmarks[curve.name] = curve
        bench_info[curve.name] = info

    # --- Training je Seed ---
    agent_curves, rows = {}, []
    for seed in args.seeds:
        print(f"\n[seed {seed}] Training: {args.timesteps} Timesteps ...")
        np.random.seed(seed)
        e_train = build_env(train, args, args.indicators, args.fee, args.rebalance)
        env_train, _ = e_train.get_sb_env()

        agent = DRLAgent(env=env_train)
        model = agent.get_model(
            "ppo",
            model_kwargs=dict(n_steps=2048, ent_coef=0.01, learning_rate=2.5e-4,
                              batch_size=128, device="cpu"),
            seed=seed,
            verbose=0,
        )
        trained = agent.train_model(model=model, tb_log_name=f"ppo_s{seed}",
                                   total_timesteps=args.timesteps)
        trained.save(str(out / f"ppo_seed{seed}.zip"))

        for split_name, split_df in (("valid", valid), ("test", test)):
            e_eval = build_env(split_df, args, args.indicators, args.fee, args.rebalance)
            df_acc, df_act = DRLAgent.DRL_prediction(model=trained, environment=e_eval)
            curve = df_acc.set_index("date")["account_value"]
            curve.index = pd.to_datetime(curve.index)
            # nach dem Auto-Reset von DummyVecEnv stehen die Episodenwerte in
            # last_episode_* (siehe TradeRepublicEnv.reset)
            fees = e_eval.last_episode_cost or e_eval.cost
            orders = e_eval.last_episode_trades or e_eval.trades
            m = metrics(curve, risk_free_annual=args.risk_free)
            m.update(Split=split_name, Strategie=f"PPO seed {seed}",
                     Gebuehren=fees, Orders=orders)
            rows.append(m)
            if split_name == "test":
                agent_curves[f"PPO seed {seed}"] = curve
                df_act.to_csv(out / f"actions_seed{seed}.csv", index=False)
            print(f"[seed {seed}] {split_name}: Endwert {m['Endwert']:>12,.0f} EUR | "
                  f"Sharpe {m['Sharpe']:>6.2f} | MaxDD {m['MaxDD']:>7.1%} | "
                  f"{orders} Orders, {fees:.0f} EUR Gebuehren")

    for name, curve in benchmarks.items():
        m = metrics(curve, risk_free_annual=args.risk_free)
        m.update(Split="test", Strategie=name, **bench_info[name])
        rows.append(m)

    # --- Ergebnisse ---
    res = pd.DataFrame(rows)[
        ["Split", "Strategie", "Endwert", "CAGR", "Vola", "Sharpe", "Sortino",
         "MaxDD", "Orders", "Gebuehren"]
    ].sort_values(["Split", "Strategie"])
    res.to_csv(out / "ergebnisse.csv", index=False)

    show = res[res.Split == "test"].copy()
    for c in ("CAGR", "Vola", "MaxDD"):
        show[c] = show[c].map(lambda v: f"{v:.2%}")
    for c in ("Sharpe", "Sortino"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    for c in ("Endwert", "Gebuehren"):
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    print("\n=== Testzeitraum ===")
    print(show.drop(columns="Split").to_string(index=False))

    plt.figure(figsize=(11, 5.5))
    for name, curve in agent_curves.items():
        plt.plot(curve.index, curve.values, lw=1.8, label=name)
    for name, curve in benchmarks.items():
        plt.plot(curve.index, curve.values, lw=1.2, ls="--", alpha=0.8, label=name)
    plt.title(f"Depotwert im Testzeitraum ({args.valid_end} .. {args.end}), "
              f"Start {args.initial:,.0f} EUR, {args.fee:.2f} EUR/Order, "
              f"Handeln alle {args.rebalance} Handelstage")
    plt.ylabel("Depotwert in EUR")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "depotwert.png", dpi=140)
    print(f"\n[run ] Ergebnisse in {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
