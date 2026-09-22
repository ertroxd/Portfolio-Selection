"""Trade-Republic-Environment: StockTradingEnv mit fixer Ordergebuehr.

FinRLs StockTradingEnv rechnet Transaktionskosten rein prozentual ab
(`preis * stueck * (1 +- cost_pct)`), siehe
`finrl/meta/env_stock_trading/env_stocktrading.py`, `_buy_stock()` / `_sell_stock()`.
Trade Republic verlangt stattdessen eine *feste* Pauschale je ausgefuehrter Order
(aktuell 1,00 EUR Abwicklungskostenpauschale, 0,00 EUR bei Sparplanausfuehrung).

Diese Klasse ersetzt deshalb genau die beiden Methoden. State-Aufbau, Reward und
Reporting bleiben vom Eltern-Environment.

State-Layout von StockTradingEnv (intern, immer roh in Euro):
    state[0]                            = Cash
    state[1 .. stock_dim]               = aktuelle Kurse je Titel
    state[stock_dim+1 .. 2*stock_dim]   = gehaltene Stueckzahlen je Titel
    danach                              = technische Indikatoren, je Indikator ein Block
                                          ueber alle Titel (erst alle rsi_30, dann alle macd)

Beobachtung des Agenten
-----------------------
Mit ``normalize_obs=False`` sieht der Agent genau diesen rohen State - so liefen die
ersten Laeufe. Das hat zwei Nachteile: Bei 1.000.000 EUR Startkapital bekommt das
neuronale Netz Eingaben in Millionenhoehe, bei 1.000 EUR in Tausenderhoehe, Laeufe
mit verschiedenem Kapital sind also nicht vergleichbar. Und die Kurse im Testzeitraum
liegen weit ausserhalb dessen, was der Agent im Training gesehen hat.

Mit ``normalize_obs=True`` sieht der Agent nur massstabsfreie Groessen:
    [Cash-Anteil, Depotanteil je Titel, RSI/100 je Titel, MACD/Kurs je Titel]
Der interne State bleibt roh - damit wird weiter gehandelt und abgerechnet.
"""
from __future__ import annotations

import finrl_shim  # noqa: F401  - muss vor dem finrl-Import stehen
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from gymnasium import spaces


class TradeRepublicEnv(StockTradingEnv):
    """StockTradingEnv mit fixer Ordergebuehr statt prozentualer Kosten.

    Parameters
    ----------
    fixed_fee : float
        Pauschale in EUR je ausgefuehrter Order (Kauf wie Verkauf).
        1.00 = Abwicklungskostenpauschale, 2.00 = "Direktpreis an der Boerse",
        0.00 = Sparplanausfuehrung.
    rebalance_every : int
        Handeln nur an jedem n-ten Handelstag. 1 = taeglich (FinRL-Default),
        21 = ungefaehr monatlich. Dient der Abbildung von "semi-automatisch":
        wenige Entscheidungen pro Jahr, die ein Mensch pruefen koennte.
    normalize_obs : bool
        Agent sieht Anteile statt Euro-Betraege (siehe Moduldocstring).
    """

    def __init__(self, *args, fixed_fee: float = 1.0, rebalance_every: int = 1,
                 normalize_obs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_fee = float(fixed_fee)
        self.rebalance_every = max(1, int(rebalance_every))
        self.normalize_obs = bool(normalize_obs)
        self.last_episode_cost = 0.0
        self.last_episode_trades = 0
        if self.normalize_obs:
            dim = 1 + self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))

    # ------------------------------------------------------------------

    def beobachtung(self) -> np.ndarray:
        """Massstabsfreie Sicht auf den aktuellen State (siehe Moduldocstring)."""
        n = self.stock_dim
        s = np.asarray(self.state, dtype=np.float64)
        cash, preise, bestand = s[0], s[1 : 1 + n], s[1 + n : 1 + 2 * n]
        werte = preise * bestand
        gesamt = cash + werte.sum()
        teile = [np.array([cash / gesamt]), werte / gesamt]

        indikatoren = s[1 + 2 * n :]
        for j, name in enumerate(self.tech_indicator_list):
            block = indikatoren[j * n : (j + 1) * n]
            if name.startswith("rsi"):
                block = block / 100.0
            elif name.startswith("macd"):
                block = block / preise
            teile.append(block)
        return np.concatenate(teile).astype(np.float32)

    def _obs(self, state):
        return self.beobachtung() if self.normalize_obs else state

    def reset(self, *, seed=None, options=None):
        """Bilanz der abgeschlossenen Episode sichern, dann zuruecksetzen.

        Noetig, weil `DummyVecEnv` die Umgebung nach dem letzten Schritt automatisch
        zuruecksetzt und `StockTradingEnv.reset()` dabei `cost` und `trades` auf 0
        setzt. Wer nach `DRL_prediction()` die Gebuehren auslesen will, sieht sonst
        immer 0 - und merkt es nicht.
        """
        self.last_episode_cost = float(getattr(self, "cost", 0.0))
        self.last_episode_trades = int(getattr(self, "trades", 0))
        state, info = super().reset(seed=seed, options=options)
        return self._obs(state), info

    def episode_kosten(self) -> tuple[float, int]:
        """Gebuehren und Orders der letzten abgeschlossenen Episode (siehe reset())."""
        return self.last_episode_cost or self.cost, self.last_episode_trades or self.trades

    def step(self, actions):
        state, reward, terminal, truncated, info = super().step(actions)
        return self._obs(state), reward, terminal, truncated, info

    # ------------------------------------------------------------------

    def _may_trade(self) -> bool:
        """An Nicht-Rebalancing-Tagen wird jede Order verworfen."""
        return self.day % self.rebalance_every == 0

    def _buy_stock(self, index, action):
        if not self._may_trade():
            return 0

        price = self.state[index + 1]
        # Die Gebuehr muss VOR der Stueckzahlberechnung abgezogen werden, sonst
        # kann das Cash-Konto nach dem Kauf negativ werden. Genau hier weicht die
        # Fixgebuehr strukturell vom multiplikativen Original ab:
        #   FinRL:  available = cash // (price * (1 + pct))
        #   hier:   available = (cash - fee) // price
        cash_for_shares = self.state[0] - self.fixed_fee
        if cash_for_shares <= 0 or price <= 0:
            return 0

        shares = min(int(action), int(cash_for_shares // price))
        if shares <= 0:
            # Keine Order -> keine Gebuehr. Nichtstun muss kostenlos bleiben,
            # sonst lernt der Agent Unsinn.
            return 0

        self.state[0] -= price * shares + self.fixed_fee
        self.state[index + self.stock_dim + 1] += shares
        self.cost += self.fixed_fee
        self.trades += 1
        return shares

    def _sell_stock(self, index, action):
        if not self._may_trade():
            return 0

        held = self.state[index + self.stock_dim + 1]
        shares = min(int(abs(action)), int(held))
        if shares <= 0:
            return 0

        price = self.state[index + 1]
        proceeds = price * shares - self.fixed_fee
        if proceeds <= 0:
            # Order kleiner als die Gebuehr -> waere ein Verlustgeschaeft.
            return 0

        self.state[0] += proceeds
        self.state[index + self.stock_dim + 1] -= shares
        self.cost += self.fixed_fee
        self.trades += 1
        return shares
