"""Thema 7 - Demo-Seite: den trainierten Agenten Tag fuer Tag nachvollziehen.

Start (im Projektordner, nach setup.ps1 / setup.sh):
    Windows:      .venv\\Scripts\\python.exe -m streamlit run app.py
    macOS/Linux:  .venv/bin/python -m streamlit run app.py

Die Seite laedt eine gespeicherte PPO-Policy aus runs/, laesst sie ueber den
Validierungs- oder Testzeitraum laufen und zeichnet jeden Schritt auf: was der
Agent wollte, was tatsaechlich ausgefuehrt wurde, was es gekostet hat und wie
sich das Depot entwickelt - verglichen mit MSCI World Buy & Hold als Markt.

Funktioniert mit beiden Laufgenerationen: den ersten Laeufen (2 ETFs, roher State)
und den Laeufen mit 9 ETFs, separatem Markt-Benchmark und normierter Beobachtung.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import finrl_shim  # noqa: F401  - muss vor jedem finrl-Import stehen
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import run_training as rt

MARKT_STANDARD = "EUNL.DE"  # iShares Core MSCI World

#: Anlageklasse je Ticker (neue Laeufe aus rt.UNIVERSUM, dazu die der ersten Laeufe).
ANLAGEKLASSE = {**rt.UNIVERSUM, "EUNL.DE": "Aktien Welt", "IS3N.DE": "Aktien Schwellenlaender"}

#: Ab so vielen Titeln wird das Depot nach Gruppen statt je Titel gezeigt.
MAX_EINZELN = 5

# Farben nach Rolle, feste Reihenfolge der Referenzpalette (nie nach Rang vergeben).
FARBE_AGENT = "#2a78d6"   # Slot 1
FARBE_MARKT = "#eb6834"   # Slot 2
FARBE_GLEICH = "#1baf7a"  # Slot 3
SLOTS_REST = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # Slots 4-8
FARBE_GRAU = "#898781"    # gedaempfte Tinte: Cash, Kurslinien, Nebenlinien
FARBE_KAUF = "#2a78d6"    # Kauf/Verkauf: blau/rot plus Dreiecksform (nie nur Farbe)
FARBE_VERKAUF = "#e34948"
GRUPPEN_REIHENFOLGE = ["Aktien", "Anleihen", "Gold", "Immobilien", "Rohstoffe"]


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def eur(v: float) -> str:
    return f"{v:,.0f} €".replace(",", ".")


def pct(v: float) -> str:
    return f"{v:.1%}".replace(".", ",")


def zahl(v: float, stellen: int = 2) -> str:
    return f"{v:.{stellen}f}".replace(".", ",")


def gruppe(ticker: str) -> str:
    klasse = ANLAGEKLASSE.get(ticker, ticker)
    if klasse.startswith("Aktien"):
        return "Aktien"
    if klasse.startswith("Euro-"):
        return "Anleihen"
    return klasse.split(" ")[0]


def depot_spalten(tickers: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Welche Titel zu welcher Depot-Linie gehoeren, und welche Farbe die Linie hat.

    Wenige Titel: eine Linie je Titel. Viele Titel: eine Linie je Anlageklasse,
    damit nie mehr Farben gebraucht werden, als die Palette sauber trennt.
    """
    if len(tickers) <= MAX_EINZELN:
        spalten = {t: [t] for t in tickers}
    else:
        spalten = {}
        for g in GRUPPEN_REIHENFOLGE + sorted({gruppe(t) for t in tickers} - set(GRUPPEN_REIHENFOLGE)):
            mitglieder = [t for t in tickers if gruppe(t) == g]
            if mitglieder:
                spalten[g] = mitglieder
    farben = {name: SLOTS_REST[i % len(SLOTS_REST)] for i, name in enumerate(spalten)}
    return spalten, farben


def layout(fig: go.Figure, y_titel: str, hoehe: int = 380, prozent: bool = False) -> go.Figure:
    fig.update_layout(
        height=hoehe,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=8, r=8, t=36, b=8),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text=y_titel, zeroline=False,
                     tickformat=".0%" if prozent else ",.0f")
    return fig


def zeige(fig: go.Figure) -> None:
    fig.update_layout(separators=",.")  # deutsches Dezimalkomma, Tausenderpunkt
    st.plotly_chart(fig, theme="streamlit", width="stretch")


# ----------------------------------------------------------------------
# Daten, Modell, Rollout (gecacht)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Lade Kursdaten und Indikatoren ...")
def daten(tickers: tuple[str, ...], start: str, end: str, indikatoren: tuple[str, ...]):
    raw = rt.load_prices(list(tickers), start, end)
    processed = rt.add_features(raw, list(indikatoren))
    return raw, processed


@st.cache_data(show_spinner="Lade Marktdaten ...")
def markt_daten(ticker: str, start: str, end: str) -> pd.DataFrame:
    return rt.load_prices([ticker], start, end)


@st.cache_resource(show_spinner="Lade Modell ...")
def modell(pfad: str):
    from stable_baselines3 import PPO

    return PPO.load(pfad, device="cpu")


def split_df(cfg: dict, zeitraum: str) -> pd.DataFrame:
    from finrl.meta.preprocessor.preprocessors import data_split

    _, processed = daten(tuple(cfg["tickers"]), cfg["start"], cfg["end"], tuple(cfg["indicators"]))
    if zeitraum == "Test":
        return data_split(processed, cfg["valid_end"], cfg["end"])
    return data_split(processed, cfg["train_end"], cfg["valid_end"])


@st.cache_data(show_spinner="Agent handelt ...")
def rollout(run_name: str, seed: int, zeitraum: str, fee: float, rebalance: int) -> dict:
    """Policy Tag fuer Tag laufen lassen und jeden Schritt protokollieren.

    Bewusst ohne DummyVecEnv: so gibt es keinen Auto-Reset, und wir koennen vor und
    nach jedem Schritt direkt in den State schauen.
    """
    run = rt.RUNS_DIR / run_name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    df = split_df(cfg, zeitraum)
    model = modell(str(run / f"ppo_seed{seed}.zip"))

    env = rt.build_env(df, args, args.indicators, fee, rebalance)
    tickers = list(df.loc[0].tic.values)  # Reihenfolge im State = Reihenfolge in df
    n = len(tickers)

    obs, _ = env.reset()
    letzter_tag = len(df.index.unique()) - 1
    zeilen, tage, beobachtungen = [], [], []

    while env.day < letzter_tag:
        o = np.asarray(obs, dtype=np.float32)
        aktion, _ = model.predict(o, deterministic=True)
        tag, datum = env.day, env._get_date()
        preise = np.asarray(env.state[1 : 1 + n], dtype=float)

        obs, *_ = env.step(aktion)

        ausgefuehrt = np.asarray(env.actions_memory[-1], dtype=int)
        cash = float(env.state[0])
        bestand = np.asarray(env.state[n + 1 : 2 * n + 1], dtype=int)
        beobachtungen.append(o)
        tage.append({"Datum": datum, "Cash": cash,
                     **{t: bestand[i] * preise[i] for i, t in enumerate(tickers)}})
        for i, t in enumerate(tickers):
            gewollt = int(aktion[i] * args.hmax)  # wie StockTradingEnv: astype(int)
            zeilen.append({
                "Datum": datum, "Tag": tag, "Titel": t, "Kurs": preise[i],
                "Aktion": float(aktion[i]), "Gewollt": gewollt,
                "Ausgefuehrt": int(ausgefuehrt[i]), "Bestand": int(bestand[i]),
                "Gebuehr": fee if ausgefuehrt[i] != 0 else 0.0,
            })

    kurve = pd.Series(env.asset_memory, index=pd.to_datetime(env.date_memory), name="Agent")
    schritte = pd.DataFrame(zeilen)
    schritte["Datum"] = pd.to_datetime(schritte["Datum"])
    depot = pd.DataFrame(tage)
    depot["Datum"] = pd.to_datetime(depot["Datum"])
    return {
        "tickers": tickers, "kurve": kurve, "schritte": schritte,
        "depot": depot.set_index("Datum"), "obs": np.vstack(beobachtungen),
        "orders": int(env.trades), "gebuehren": float(env.cost),
    }


def benchmarks(cfg: dict, zeitraum: str, fee: float) -> dict[str, tuple[pd.Series, dict]]:
    raw, _ = daten(tuple(cfg["tickers"]), cfg["start"], cfg["end"], tuple(cfg["indicators"]))
    tage = split_df(cfg, zeitraum).date.unique()
    teil = raw[raw.date.isin(tage)]
    out = {}
    markt = cfg.get("markt")
    if markt and markt not in cfg["tickers"]:
        m = markt_daten(markt, cfg["start"], cfg["end"])
        out[f"100 % {markt}"] = rt.bh_single(m[m.date.isin(tage)], cfg["initial"], fee, markt)
    for t in cfg["tickers"]:
        out[f"100 % {t}"] = rt.bh_single(teil, cfg["initial"], fee, t)
    out["1/N Buy & Hold"] = rt.bh_equal(teil, cfg["initial"], fee)
    return out


def obs_spalte(cfg: dict, n: int, indikator: str, titel_idx: int) -> int:
    """Position eines Indikators in der Beobachtung des Agenten."""
    davor = 1 + n if cfg.get("normalize_obs") else 1 + 2 * n
    return davor + cfg["indicators"].index(indikator) * n + titel_idx


def obs_name(cfg: dict, indikator: str) -> str:
    if not cfg.get("normalize_obs"):
        return indikator
    if indikator.startswith("rsi"):
        return f"{indikator} / 100"
    if indikator.startswith("macd"):
        return f"{indikator} / Kurs"
    return indikator


# ----------------------------------------------------------------------
# Seite
# ----------------------------------------------------------------------
st.set_page_config(page_title="Thema 7 · Agent beim Handeln", page_icon="📈", layout="wide")

laeufe = [p for p in sorted(rt.RUNS_DIR.glob("*"), reverse=True) if any(p.glob("ppo_seed*.zip"))]
if not laeufe:
    st.error("Keine trainierten Modelle in runs/ gefunden. Erst `run_training.py` laufen lassen.")
    st.stop()

with st.sidebar:
    st.header("Einstellungen")
    start_idx = next((i for i, p in enumerate(laeufe) if p.name.endswith("k10000_daily")),
                     next((i for i, p in enumerate(laeufe) if "smoke" not in p.name), 0))
    run = st.selectbox("Trainingslauf", laeufe, index=start_idx, format_func=lambda p: p.name)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    seeds = sorted(int(z.stem.removeprefix("ppo_seed")) for z in run.glob("ppo_seed*.zip"))
    seed = st.selectbox("Seed", seeds)
    zeitraum = st.radio("Zeitraum", ["Test", "Validierung"], horizontal=True,
                        help="Beide Zeiträume hat der Agent im Training nie gesehen.")
    fee = st.number_input("Gebühr je Order (€)", min_value=0.0, max_value=10.0,
                          value=float(cfg["fee"]), step=0.5,
                          help="Weicht der Wert vom Training ab, handelt dieselbe Policy "
                               "unter anderen Kosten – wie eval_saved.py.")
    rebalance = st.number_input("Handeln alle n Handelstage", min_value=1, max_value=63,
                                value=int(cfg["rebalance"]), step=1)
    markt_optionen = list(dict.fromkeys(
        ([cfg["markt"]] if cfg.get("markt") else []) + cfg["tickers"]))
    standard_markt = cfg.get("markt") or MARKT_STANDARD
    markt = st.selectbox("Markt-Vergleich", markt_optionen,
                         index=markt_optionen.index(standard_markt) if standard_markt in markt_optionen else 0)
    st.divider()
    st.caption(
        f"**Training:** {cfg['timesteps']:,} Timesteps".replace(",", ".")
        + f" · Start {eur(cfg['initial'])} · Gebühr {zahl(cfg['fee'])} € · "
        f"Handeln alle {cfg['rebalance']} T · {len(cfg['tickers'])} Titel · "
        f"hmax {cfg['hmax']:,}".replace(",", ".")
        + f" · Beobachtung {'Anteile' if cfg.get('normalize_obs') else 'roh'}"
        + f"\n\n**Train** {cfg['start']} – {cfg['train_end']} · **Valid** bis {cfg['valid_end']} · "
        f"**Test** bis {cfg['end']} (Enddaten jeweils exklusiv)"
    )

if fee != cfg["fee"] or rebalance != cfg["rebalance"]:
    st.warning("Gebühr oder Handelstakt weichen vom Training ab. Die Policy ist dieselbe, "
               "sie läuft nur unter anderen Bedingungen – das misst den Kosteneffekt, nicht "
               "was ein Agent unter diesen Bedingungen gelernt hätte.")

r = rollout(run.name, seed, zeitraum, fee, rebalance)
tickers, kurve, schritte, depot = r["tickers"], r["kurve"], r["schritte"], r["depot"]
n = len(tickers)
bm = benchmarks(cfg, zeitraum, fee)
markt_kurve, markt_info = bm[f"100 % {markt}"]
gleich_kurve, gleich_info = bm["1/N Buy & Hold"]
markt_name = f"Markt {ANLAGEKLASSE.get(markt, markt)} ({markt})"

st.title("Der Agent beim Handeln")
st.caption(f"{run.name} · Seed {seed} · {zeitraum} {kurve.index[0]:%d.%m.%Y} – "
           f"{kurve.index[-1]:%d.%m.%Y} · Startkapital {eur(cfg['initial'])} · "
           f"{zahl(fee)} € je Order · {n} Titel")

m_agent, m_markt, m_gleich = rt.metrics(kurve), rt.metrics(markt_kurve), rt.metrics(gleich_kurve)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Endwert Agent", eur(m_agent["Endwert"]),
          delta=f"{eur(m_agent['Endwert'] - m_markt['Endwert'])} ggü. Markt")
k2.metric("Sharpe Agent", zahl(m_agent["Sharpe"]),
          delta=f"{zahl(m_agent['Sharpe'] - m_markt['Sharpe'])} ggü. Markt")
k3.metric("Max. Drawdown Agent", pct(m_agent["MaxDD"]),
          delta=f"{(m_agent['MaxDD'] - m_markt['MaxDD']) * 100:+.1f}".replace(".", ",") + " Pp. ggü. Markt")
k4.metric("Orders / Gebühren", f"{r['orders']} / {eur(r['gebuehren'])}",
          delta=f"{pct(r['gebuehren'] / cfg['initial'])} des Startkapitals",
          delta_color="off")

kennz = pd.DataFrame([
    {"Strategie": f"PPO Seed {seed}", **m_agent, "Orders": r["orders"], "Gebühren": r["gebuehren"]},
    {"Strategie": markt_name, **m_markt, "Orders": markt_info["Orders"], "Gebühren": markt_info["Gebuehren"]},
    {"Strategie": "1/N Buy & Hold", **m_gleich, "Orders": gleich_info["Orders"], "Gebühren": gleich_info["Gebuehren"]},
])
st.dataframe(
    kennz[["Strategie", "Endwert", "CAGR", "Vola", "Sharpe", "Sortino", "MaxDD", "Orders", "Gebühren"]],
    hide_index=True, width="stretch",
    column_config={
        "Endwert": st.column_config.NumberColumn(format="%.0f €"),
        "CAGR": st.column_config.NumberColumn(format="percent"),
        "Vola": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Sortino": st.column_config.NumberColumn(format="%.2f"),
        "MaxDD": st.column_config.NumberColumn("Max. Drawdown", format="percent"),
        "Gebühren": st.column_config.NumberColumn(format="%.0f €"),
    },
)

tab_wert, tab_trades, tab_depot, tab_gelernt = st.tabs(
    ["Depotwert", "Trades", "Depot", "Was hat er gelernt?"])

# --- Depotwert -----------------------------------------------------------
with tab_wert:
    st.subheader("Depotwert im Vergleich")
    fig = go.Figure()
    fig.add_scatter(x=kurve.index, y=kurve, name=f"PPO Seed {seed}",
                    line=dict(color=FARBE_AGENT, width=2))
    fig.add_scatter(x=markt_kurve.index, y=markt_kurve, name=markt_name,
                    line=dict(color=FARBE_MARKT, width=2))
    fig.add_scatter(x=gleich_kurve.index, y=gleich_kurve, name="1/N Buy & Hold",
                    line=dict(color=FARBE_GLEICH, width=2))
    fig.update_traces(hovertemplate="%{y:,.0f} €")
    zeige(layout(fig, "Depotwert in €"))

    st.subheader("Vorsprung des Agenten gegenüber dem Markt")
    st.caption("Depotwert Agent ÷ Depotwert Markt − 1. Über null liegt der Agent vorne.")
    vorsprung = kurve / markt_kurve.reindex(kurve.index) - 1
    fig = go.Figure()
    fig.add_scatter(x=vorsprung.index, y=vorsprung, name="Vorsprung", connectgaps=True,
                    line=dict(color=FARBE_AGENT, width=2), hovertemplate="%{y:+.1%}")
    fig.add_hline(y=0, line_width=1, line_color=FARBE_GRAU)
    fig.update_layout(showlegend=False)
    zeige(layout(fig, "Vorsprung", hoehe=260, prozent=True))

    with st.expander("Tabelle"):
        st.dataframe(pd.DataFrame({f"PPO Seed {seed}": kurve, markt_name: markt_kurve,
                                   "1/N Buy & Hold": gleich_kurve}).round(0), width="stretch")

# --- Trades --------------------------------------------------------------
with tab_trades:
    orders = schritte[schritte.Ausgefuehrt != 0].copy()
    orders["Richtung"] = np.where(orders.Ausgefuehrt > 0, "Kauf", "Verkauf")
    orders["Stück"] = orders.Ausgefuehrt.abs()
    orders["Volumen"] = orders["Stück"] * orders.Kurs

    gewollt = schritte[schritte.Gewollt != 0]
    blockiert = gewollt[gewollt.Ausgefuehrt == 0]
    gesperrt = int((blockiert.Tag % rebalance != 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Käufe", int((orders.Richtung == "Kauf").sum()))
    c2.metric("Verkäufe", int((orders.Richtung == "Verkauf").sum()))
    c3.metric("Handelstage", f"{orders.Datum.nunique()} von {schritte.Datum.nunique()}")
    c4.metric("Nicht ausgeführte Wünsche", len(blockiert),
              help=f"Der Agent wollte handeln, es kam aber keine Order zustande. "
                   f"{gesperrt} davon an Tagen ohne Handelserlaubnis (Handelstakt), "
                   f"der Rest an fehlendem Cash, fehlendem Bestand oder einer Order kleiner "
                   f"als die Gebühr.")

    st.subheader("Orders je Titel")
    je_titel = (
        schritte.groupby("Titel")
        .agg(Käufe=("Ausgefuehrt", lambda s: int((s > 0).sum())),
             Verkäufe=("Ausgefuehrt", lambda s: int((s < 0).sum())),
             Gebühren=("Gebuehr", "sum"),
             Bestand_Ende=("Bestand", "last"))
        .reindex(tickers)
        .reset_index()
    )
    je_titel.insert(1, "Anlageklasse", je_titel.Titel.map(lambda t: ANLAGEKLASSE.get(t, "")))
    st.dataframe(je_titel.rename(columns={"Bestand_Ende": "Stück am Ende"}),
                 hide_index=True, width="stretch",
                 column_config={"Gebühren": st.column_config.NumberColumn(format="%.0f €")})

    meist = orders.Titel.value_counts()
    titel_wahl = st.selectbox("Kursverlauf mit Orders für", tickers,
                              index=tickers.index(meist.index[0]) if len(meist) else 0,
                              format_func=lambda t: f"{t} · {ANLAGEKLASSE.get(t, '')}")
    kurs = schritte[schritte.Titel == titel_wahl]
    o = orders[orders.Titel == titel_wahl]
    fig = go.Figure()
    fig.add_scatter(x=kurs.Datum, y=kurs.Kurs, name="Kurs",
                    line=dict(color=FARBE_GRAU, width=2), hovertemplate="%{y:.2f} €")
    for richtung, symbol, col in (("Kauf", "triangle-up", FARBE_KAUF),
                                  ("Verkauf", "triangle-down", FARBE_VERKAUF)):
        teil = o[o.Richtung == richtung]
        fig.add_scatter(
            x=teil.Datum, y=teil.Kurs, mode="markers", name=richtung,
            marker=dict(symbol=symbol, size=11, color=col,
                        line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=(np.stack([teil["Stück"], teil.Volumen, teil.Gebuehr], axis=-1)
                        if len(teil) else None),
            hovertemplate=f"{richtung}: %{{customdata[0]}} Stück · %{{customdata[1]:,.0f}} € "
                          f"· Gebühr %{{customdata[2]:.2f}} €<extra></extra>",
        )
    zeige(layout(fig, "Kurs in €", hoehe=340))

    st.subheader("Orderbuch")
    if orders.empty:
        st.info("Der Agent hat in diesem Zeitraum keine einzige Order ausgeführt.")
    else:
        st.dataframe(
            orders[["Datum", "Titel", "Richtung", "Stück", "Kurs", "Volumen", "Gebuehr", "Bestand"]]
            .rename(columns={"Gebuehr": "Gebühr", "Bestand": "Bestand danach"}),
            hide_index=True, width="stretch",
            column_config={
                "Datum": st.column_config.DateColumn(format="DD.MM.YYYY"),
                "Kurs": st.column_config.NumberColumn(format="%.2f €"),
                "Volumen": st.column_config.NumberColumn(format="%.0f €"),
                "Gebühr": st.column_config.NumberColumn(format="%.2f €"),
            },
        )

# --- Depot ---------------------------------------------------------------
with tab_depot:
    spalten, farben = depot_spalten(tickers)
    linien = pd.DataFrame({"Cash": depot.Cash,
                           **{name: depot[mitglieder].sum(axis=1) for name, mitglieder in spalten.items()}})
    teile = [("Cash", FARBE_GRAU)] + [(name, farben[name]) for name in spalten]
    einheit = "je Titel" if len(tickers) <= MAX_EINZELN else "nach Anlageklasse"

    st.subheader(f"Zusammensetzung in € {einheit}")
    st.caption("Nach dem Handel des jeweiligen Tages, bewertet zum Tageskurs.")
    fig = go.Figure()
    for spalte, col in teile:
        fig.add_scatter(x=linien.index, y=linien[spalte], name=spalte, stackgroup="wert",
                        line=dict(width=0.5, color=col), fillcolor=col,
                        hovertemplate="%{y:,.0f} €")
    zeige(layout(fig, "Wert in €"))

    st.subheader(f"Anteile am Depot {einheit}")
    anteile = linien.div(linien.sum(axis=1), axis=0)
    fig = go.Figure()
    for spalte, col in teile:
        fig.add_scatter(x=anteile.index, y=anteile[spalte], name=spalte, stackgroup="anteil",
                        line=dict(width=0.5, color=col), fillcolor=col,
                        hovertemplate="%{y:.0%}")
    fig = layout(fig, "Anteil", prozent=True)
    fig.update_yaxes(range=[0, 1])
    zeige(fig)

    st.subheader("Durchschnittlicher Anteil je Titel")
    mittel = depot.div(depot.sum(axis=1), axis=0).mean().rename("Anteil im Mittel").reset_index()
    mittel.columns = ["Position", "Anteil im Mittel"]
    mittel.insert(1, "Anlageklasse", mittel.Position.map(lambda t: ANLAGEKLASSE.get(t, "")))
    st.dataframe(mittel.sort_values("Anteil im Mittel", ascending=False), hide_index=True,
                 width="stretch",
                 column_config={"Anteil im Mittel": st.column_config.ProgressColumn(
                     format="percent", min_value=0.0, max_value=1.0)})

    with st.expander("Tabelle (Werte in € je Tag)"):
        st.dataframe(depot.round(0), width="stretch")

# --- Was hat er gelernt? -------------------------------------------------
with tab_gelernt:
    st.subheader("Kurzdiagnose")
    n_tage = schritte.Datum.nunique()
    rate = r["orders"] / (n_tage * n)
    anteil_cash = depot.Cash / depot.sum(axis=1)
    voll = anteil_cash[anteil_cash < 0.05]
    groesster = depot.drop(columns="Cash").div(depot.sum(axis=1), axis=0).mean()

    # Welchem Buy-&-Hold-Portfolio sieht die Kurve am aehnlichsten?
    abstand = {name: float(np.sqrt(((kurve / k.reindex(kurve.index) - 1) ** 2).mean()))
               for name, (k, _) in bm.items()}
    naechster = min(abstand, key=abstand.get)

    if r["orders"] == 0:
        verhalten = ("**Kein Handel.** Der Agent hat keine einzige Order ausgeführt und das "
                     "Startkapital als Bargeld gehalten.")
    elif r["orders"] <= 2 * n:
        verhalten = (f"**Praktisch Buy & Hold.** {r['orders']} Orders im gesamten Zeitraum – "
                     f"einmal kaufen, danach Stillstand.")
    elif rate >= 0.5:
        verhalten = (f"**Dauerhandel.** {r['orders']} Orders, im Schnitt "
                     f"{zahl(rate)} je Titel und Tag. Genau das Verhalten, bei dem die "
                     f"Fixgebühr teuer wird.")
    else:
        verhalten = (f"**Gelegentliches Umschichten.** {r['orders']} Orders an "
                     f"{orders.Datum.nunique()} von {n_tage} Handelstagen.")

    st.markdown(verhalten)
    zeilen_md = [f"- Durchschnittlicher Cash-Anteil: **{pct(anteil_cash.mean())}**"
                 + (f", voll investiert (unter 5 % Cash) ab **{voll.index[0]:%d.%m.%Y}**"
                    if len(voll) else ", nie voll investiert")]
    if r["orders"] > 0:
        zeilen_md.append(f"- Größte Position im Mittel: **{groesster.idxmax()}** "
                         f"({ANLAGEKLASSE.get(groesster.idxmax(), '')}) mit {pct(groesster.max())}")
    zeilen_md.append(f"- Am ähnlichsten zu **{naechster}** – mittlere relative Abweichung "
                     f"{pct(abstand[naechster])}")
    st.markdown("\n".join(zeilen_md))
    if r["orders"] > 0 and abstand[naechster] < 0.01 and rate >= 0.5:
        st.info(f"Unter 1 % Abweichung von **{naechster}** – trotz {r['orders']} Orders. "
                f"Der viele Handel bewegt das Ergebnis kaum, er kostet nur Gebühren.")
    elif r["orders"] > 0 and abstand[naechster] < 0.01:
        st.info(f"Unter 1 % Abweichung: Dieser Agent hat im Kern gelernt, **{naechster}** "
                f"nachzubauen. Das ist keine Handelsstrategie, sondern eine feste Allokation.")
    if cfg.get("normalize_obs"):
        schwelle = 1 / cfg["hmax"]
        st.caption(f"Totzone: Eine Aktion wird erst ab ±{zahl(schwelle, 3)} zu mindestens einem "
                   f"Anteil (hmax = {cfg['hmax']:,}).".replace(",", ".")
                   + " Bei kleinem Startkapital ist diese Zone breit.")

    st.divider()
    st.subheader("Was der Agent sieht – und was er daraus macht")
    st.caption("Jeder Punkt ist ein Handelstag. Waagrecht ein Indikator, senkrecht die "
               "Rohaktion der Policy (−1 = maximal verkaufen, +1 = maximal kaufen). "
               "Form und Farbe zeigen, was tatsächlich ausgeführt wurde.")
    s1, s2 = st.columns(2)
    titel = s1.selectbox("Titel", tickers, key="scatter_titel",
                         format_func=lambda t: f"{t} · {ANLAGEKLASSE.get(t, '')}")
    indikator = s2.selectbox("Indikator", cfg["indicators"], key="scatter_ind")

    spalte = obs_spalte(cfg, n, indikator, tickers.index(titel))
    s = schritte[schritte.Titel == titel].reset_index(drop=True)
    s["Wert"] = r["obs"][:, spalte]
    fig = go.Figure()
    for name, maske, symbol, col in (
        ("Kein Handel", s.Ausgefuehrt == 0, "circle", FARBE_GRAU),
        ("Kauf", s.Ausgefuehrt > 0, "triangle-up", FARBE_KAUF),
        ("Verkauf", s.Ausgefuehrt < 0, "triangle-down", FARBE_VERKAUF),
    ):
        teil = s[maske]
        fig.add_scatter(x=teil.Wert, y=teil.Aktion, mode="markers", name=name,
                        marker=dict(symbol=symbol, size=8, color=col, opacity=0.75),
                        customdata=teil.Datum.dt.strftime("%d.%m.%Y"),
                        hovertemplate="%{customdata}<br>" + obs_name(cfg, indikator)
                                      + " %{x:.3f} · Aktion %{y:+.2f}<extra>" + name + "</extra>")
    fig.update_xaxes(title_text=f"{obs_name(cfg, indikator)} ({titel})")
    fig.update_yaxes(title_text="Rohaktion", range=[-1.05, 1.05], tickformat=".1f")
    fig.update_layout(height=380, hovermode="closest", margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    zeige(fig)

    st.subheader("Policy-Sonde")
    st.caption("Alle Eingaben bleiben wie an einem gewählten Tag – nur ein Indikator eines "
               "Titels wird über seinen beobachteten Wertebereich verschoben. Die farbige "
               "Linie ist die Aktion für genau diesen Titel, die grauen Linien die Aktionen "
               "für alle anderen. Wechselwirkungen zwischen Eingaben zeigt diese Ansicht nicht.")
    daten_liste = list(s.Datum.dt.strftime("%d.%m.%Y"))
    p1, p2, p3 = st.columns([2, 1, 1])
    tag_wahl = p1.select_slider("Ausgangstag", options=daten_liste,
                                value=daten_liste[len(daten_liste) // 2])
    sonde_titel = p2.selectbox("Titel", tickers, key="sonde_titel")
    sonde_ind = p3.selectbox("Indikator", cfg["indicators"], key="sonde_ind")

    t_idx = daten_liste.index(tag_wahl)
    ti = tickers.index(sonde_titel)
    sp = obs_spalte(cfg, n, sonde_ind, ti)
    beobachtet = r["obs"][:, sp]
    werte = np.linspace(beobachtet.min(), beobachtet.max(), 80)
    X = np.repeat(r["obs"][t_idx][None, :], len(werte), axis=0)
    X[:, sp] = werte
    policy = modell(str(run / f"ppo_seed{seed}.zip"))
    aktionen, _ = policy.predict(X.astype(np.float32), deterministic=True)

    fig = go.Figure()
    for i, t in enumerate(tickers):
        if i != ti:
            fig.add_scatter(x=werte, y=aktionen[:, i], name="andere Titel",
                            legendgroup="andere", showlegend=(i == (1 if ti == 0 else 0)),
                            line=dict(color=FARBE_GRAU, width=1), opacity=0.6,
                            hovertemplate=f"{t} %{{y:+.2f}}<extra></extra>")
    fig.add_scatter(x=werte, y=aktionen[:, ti], name=f"Aktion {sonde_titel}",
                    line=dict(color=FARBE_AGENT, width=2.5), hovertemplate="%{y:+.2f}")
    fig.add_vline(x=float(beobachtet[t_idx]), line_width=1, line_color=FARBE_GRAU,
                  annotation_text="tatsächlicher Wert", annotation_position="top")
    fig.add_hline(y=0, line_width=1, line_color=FARBE_GRAU)
    fig.update_xaxes(title_text=f"{obs_name(cfg, sonde_ind)} ({sonde_titel})")
    fig.update_yaxes(title_text="Rohaktion", range=[-1.05, 1.05], tickformat=".1f")
    fig.update_layout(height=340, hovermode="x unified", margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    zeige(fig)
