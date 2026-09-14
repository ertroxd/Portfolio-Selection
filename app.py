"""Thema 7 - Demo-Seite: den trainierten Agenten Tag fuer Tag nachvollziehen.

Start (im Projektordner, nach setup.ps1 / setup.sh):
    Windows:      .venv\\Scripts\\python.exe -m streamlit run app.py
    macOS/Linux:  .venv/bin/python -m streamlit run app.py

Die Seite laedt eine gespeicherte PPO-Policy aus runs/, laesst sie ueber den
Validierungs- oder Testzeitraum laufen und zeichnet jeden Schritt auf: was der
Agent wollte, was tatsaechlich ausgefuehrt wurde, was es gekostet hat und wie
sich das Depot entwickelt - verglichen mit MSCI World Buy & Hold als Markt.
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

MARKT_TIC = "EUNL.DE"  # iShares Core MSCI World - unser "Markt"

# Farben nach Rolle, feste Reihenfolge der Referenzpalette (nie nach Rang vergeben).
FARBE_AGENT = "#2a78d6"   # Slot 1
FARBE_MARKT = "#eb6834"   # Slot 2 - der Markt-ETF, ueberall dieselbe Farbe
FARBE_GLEICH = "#1baf7a"  # Slot 3 - 1/N
WEITERE_SLOTS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7"]  # Slots 4-7 fuer weitere ETFs
FARBE_CASH = "#898781"    # gedaempfte Tinte - Cash ist kein "Titel"
FARBE_KAUF = "#2a78d6"    # Kauf/Verkauf: blau/rot plus Dreiecksform (nie nur Farbe)
FARBE_VERKAUF = "#e34948"


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def eur(v: float) -> str:
    return f"{v:,.0f} €".replace(",", ".")


def pct(v: float) -> str:
    return f"{v:.1%}".replace(".", ",")


def zahl(v: float, stellen: int = 2) -> str:
    return f"{v:.{stellen}f}".replace(".", ",")


def farben_titel(tickers: list[str], markt: str) -> dict[str, str]:
    """Jeder ETF behaelt seine Farbe, egal was gerade ausgewaehlt ist."""
    out, frei = {}, iter(WEITERE_SLOTS)
    for t in tickers:
        out[t] = FARBE_MARKT if t == markt else next(frei)
    return out


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
        "cfg": cfg, "tickers": tickers, "kurve": kurve, "schritte": schritte,
        "depot": depot.set_index("Datum"), "obs": np.vstack(beobachtungen),
        "orders": int(env.trades), "gebuehren": float(env.cost),
    }


def benchmarks(cfg: dict, zeitraum: str, fee: float) -> dict[str, tuple[pd.Series, dict]]:
    raw, _ = daten(tuple(cfg["tickers"]), cfg["start"], cfg["end"], tuple(cfg["indicators"]))
    teil = raw[raw.date.isin(split_df(cfg, zeitraum).date.unique())]
    out = {}
    for t in cfg["tickers"]:
        out[f"100 % {t}"] = rt.bh_single(teil, cfg["initial"], fee, t)
    out["1/N Buy & Hold"] = rt.bh_equal(teil, cfg["initial"], fee)
    return out


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
    start_idx = next((i for i, p in enumerate(laeufe) if p.name.endswith("daily_fee1")), 0)
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
    markt = st.selectbox("Markt-Vergleich", cfg["tickers"],
                         index=cfg["tickers"].index(MARKT_TIC) if MARKT_TIC in cfg["tickers"] else 0)
    st.divider()
    st.caption(
        f"**Training:** {cfg['timesteps']:,} Timesteps".replace(",", ".")
        + f" · Gebühr {zahl(cfg['fee'])} € · Handeln alle {cfg['rebalance']} T · Start {eur(cfg['initial'])}"
        + f"\n\n**Train** {cfg['start']} – {cfg['train_end']} · **Valid** bis {cfg['valid_end']} · "
        f"**Test** bis {cfg['end']} (Enddaten jeweils exklusiv)"
    )

if fee != cfg["fee"] or rebalance != cfg["rebalance"]:
    st.warning("Gebühr oder Handelstakt weichen vom Training ab. Die Policy ist dieselbe, "
               "sie läuft nur unter anderen Bedingungen – das misst den Kosteneffekt, nicht "
               "was ein Agent unter diesen Bedingungen gelernt hätte.")

r = rollout(run.name, seed, zeitraum, fee, rebalance)
tickers, kurve, schritte, depot = r["tickers"], r["kurve"], r["schritte"], r["depot"]
farbe = farben_titel(tickers, markt)
bm = benchmarks(cfg, zeitraum, fee)
markt_kurve, markt_info = bm[f"100 % {markt}"]
gleich_kurve, gleich_info = bm["1/N Buy & Hold"]
markt_name = "MSCI World (EUNL.DE)" if markt == MARKT_TIC else f"100 % {markt}"

st.title("Der Agent beim Handeln")
st.caption(f"{run.name} · Seed {seed} · {zeitraum} {kurve.index[0]:%d.%m.%Y} – "
           f"{kurve.index[-1]:%d.%m.%Y} · Startkapital {eur(cfg['initial'])} · "
           f"{zahl(fee)} € je Order")

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
    fig.add_scatter(x=vorsprung.index, y=vorsprung, name="Vorsprung",
                    line=dict(color=FARBE_AGENT, width=2), hovertemplate="%{y:+.1%}")
    fig.add_hline(y=0, line_width=1, line_color=FARBE_CASH)
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

    for t in tickers:
        st.subheader(f"{t}{' · Markt' if t == markt else ''}")
        kurs = schritte[schritte.Titel == t]
        o = orders[orders.Titel == t]
        fig = go.Figure()
        fig.add_scatter(x=kurs.Datum, y=kurs.Kurs, name="Kurs",
                        line=dict(color=farbe[t], width=2), hovertemplate="%{y:.2f} €")
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
        zeige(layout(fig, "Kurs in €", hoehe=320))

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
    teile = [("Cash", FARBE_CASH)] + [(t, farbe[t]) for t in tickers]

    st.subheader("Zusammensetzung in €")
    st.caption("Nach dem Handel des jeweiligen Tages, bewertet zum Tageskurs.")
    fig = go.Figure()
    for spalte, col in teile:
        fig.add_scatter(x=depot.index, y=depot[spalte], name=spalte, stackgroup="wert",
                        line=dict(width=0.5, color=col), fillcolor=col,
                        hovertemplate="%{y:,.0f} €")
    zeige(layout(fig, "Wert in €"))

    st.subheader("Anteile am Depot")
    anteile = depot.div(depot.sum(axis=1), axis=0)
    fig = go.Figure()
    for spalte, col in teile:
        fig.add_scatter(x=anteile.index, y=anteile[spalte], name=spalte, stackgroup="anteil",
                        line=dict(width=0.5, color=col), fillcolor=col,
                        hovertemplate="%{y:.0%}")
    fig = layout(fig, "Anteil", prozent=True)
    fig.update_yaxes(range=[0, 1])
    zeige(fig)

    with st.expander("Tabelle"):
        st.dataframe(depot.round(0), width="stretch")

# --- Was hat er gelernt? -------------------------------------------------
with tab_gelernt:
    st.subheader("Kurzdiagnose")
    n_tage, n = schritte.Datum.nunique(), len(tickers)
    rate = r["orders"] / (n_tage * n)
    anteil_cash = depot.Cash / depot.sum(axis=1)
    voll = anteil_cash[anteil_cash < 0.05]

    # Welchem Buy-&-Hold-Portfolio sieht die Kurve am aehnlichsten?
    abstand = {name: float(np.sqrt(((kurve / k.reindex(kurve.index) - 1) ** 2).mean()))
               for name, (k, _) in bm.items()}
    naechster = min(abstand, key=abstand.get)

    if r["orders"] <= 2 * n:
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
                    if len(voll) else ", nie voll investiert"),
                 f"- Am ähnlichsten zu **{naechster}** – mittlere relative Abweichung "
                 f"{pct(abstand[naechster])}"]
    st.markdown("\n".join(zeilen_md))
    if abstand[naechster] < 0.01 and rate >= 0.5:
        st.info(f"Unter 1 % Abweichung von **{naechster}** – trotz {r['orders']} Orders. "
                f"Der viele Handel bewegt das Ergebnis kaum, er kostet nur Gebühren.")
    elif abstand[naechster] < 0.01:
        st.info(f"Unter 1 % Abweichung: Dieser Agent hat im Kern gelernt, **{naechster}** "
                f"nachzubauen. Das ist keine Handelsstrategie, sondern eine feste Allokation.")

    st.divider()
    st.subheader("Was der Agent sieht – und was er daraus macht")
    st.caption("Jeder Punkt ist ein Handelstag. Waagrecht ein Indikator, senkrecht die "
               "Rohaktion der Policy (−1 = maximal verkaufen, +1 = maximal kaufen). "
               "Form und Farbe zeigen, was tatsächlich ausgeführt wurde.")
    s1, s2 = st.columns(2)
    titel = s1.selectbox("Titel", tickers, key="scatter_titel")
    indikator = s2.selectbox("Indikator", cfg["indicators"], key="scatter_ind")

    spalte = 1 + 2 * n + cfg["indicators"].index(indikator) * n + tickers.index(titel)
    s = schritte[schritte.Titel == titel].reset_index(drop=True)
    s["Wert"] = r["obs"][:, spalte]  # State-Layout von StockTradingEnv
    fig = go.Figure()
    for name, maske, symbol, col in (
        ("Kein Handel", s.Ausgefuehrt == 0, "circle", FARBE_CASH),
        ("Kauf", s.Ausgefuehrt > 0, "triangle-up", FARBE_KAUF),
        ("Verkauf", s.Ausgefuehrt < 0, "triangle-down", FARBE_VERKAUF),
    ):
        teil = s[maske]
        fig.add_scatter(x=teil.Wert, y=teil.Aktion, mode="markers", name=name,
                        marker=dict(symbol=symbol, size=8, color=col, opacity=0.75),
                        customdata=teil.Datum.dt.strftime("%d.%m.%Y"),
                        hovertemplate="%{customdata}<br>" + indikator
                                      + " %{x:.2f} · Aktion %{y:+.2f}<extra>" + name + "</extra>")
    fig.update_xaxes(title_text=f"{indikator} ({titel})")
    fig.update_yaxes(title_text="Rohaktion", range=[-1.05, 1.05], tickformat=".1f")
    fig.update_layout(height=380, hovermode="closest", margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    zeige(fig)

    st.subheader("Policy-Sonde")
    st.caption("Alle Eingaben bleiben wie an einem gewählten Tag – nur ein Indikator wird "
               "über seinen beobachteten Wertebereich verschoben. So sieht man, worauf die "
               "Policy reagiert. Wechselwirkungen zwischen Eingaben zeigt diese Ansicht nicht.")
    daten_liste = list(s.Datum.dt.strftime("%d.%m.%Y"))
    p1, p2, p3 = st.columns([2, 1, 1])
    tag_wahl = p1.select_slider("Ausgangstag", options=daten_liste,
                                value=daten_liste[len(daten_liste) // 2])
    sonde_titel = p2.selectbox("Titel", tickers, key="sonde_titel")
    sonde_ind = p3.selectbox("Indikator", cfg["indicators"], key="sonde_ind")

    t_idx = daten_liste.index(tag_wahl)
    sp = 1 + 2 * n + cfg["indicators"].index(sonde_ind) * n + tickers.index(sonde_titel)
    beobachtet = r["obs"][:, sp]
    werte = np.linspace(beobachtet.min(), beobachtet.max(), 80)
    X = np.repeat(r["obs"][t_idx][None, :], len(werte), axis=0)
    X[:, sp] = werte
    policy = modell(str(run / f"ppo_seed{seed}.zip"))
    aktionen, _ = policy.predict(X.astype(np.float32), deterministic=True)

    fig = go.Figure()
    for i, t in enumerate(tickers):
        fig.add_scatter(x=werte, y=aktionen[:, i], name=f"Aktion {t}",
                        line=dict(color=farbe[t], width=2), hovertemplate="%{y:+.2f}")
    fig.add_vline(x=float(beobachtet[t_idx]), line_width=1, line_color=FARBE_CASH,
                  annotation_text="tatsächlicher Wert", annotation_position="top")
    fig.add_hline(y=0, line_width=1, line_color=FARBE_CASH)
    fig.update_xaxes(title_text=f"{sonde_ind} ({sonde_titel})")
    fig.update_yaxes(title_text="Rohaktion", range=[-1.05, 1.05], tickformat=".1f")
    fig.update_layout(height=340, hovermode="x unified", margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    zeige(fig)
