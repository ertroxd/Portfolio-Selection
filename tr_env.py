"""Trade-Republic-Environment: StockTradingEnv mit fixer Ordergebuehr.

FinRLs StockTradingEnv rechnet Transaktionskosten rein prozentual ab
(`preis * stueck * (1 +- cost_pct)`), siehe
`finrl/meta/env_stock_trading/env_stocktrading.py`, `_buy_stock()` / `_sell_stock()`.
Trade Republic verlangt stattdessen eine *feste* Pauschale je ausgefuehrter Order
(aktuell 1,00 EUR Abwicklungskostenpauschale, 0,00 EUR bei Sparplanausfuehrung).

Diese Klasse ersetzt deshalb genau die beiden Methoden. Alles andere - State-Aufbau,
Reward, Reset, Reporting - bleibt unveraendert vom Eltern-Environment.

State-Layout von StockTradingEnv (zum Mitlesen):
    state[0]                            = Cash
    state[1 .. stock_dim]               = aktuelle Kurse je Titel
    state[stock_dim+1 .. 2*stock_dim]   = gehaltene Stueckzahlen je Titel
    danach                              = technische Indikatoren
"""
from __future__ import annotations

import finrl_shim  # noqa: F401  - muss vor dem finrl-Import stehen
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


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
    """

    def __init__(self, *args, fixed_fee: float = 1.0, rebalance_every: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_fee = float(fixed_fee)
        self.rebalance_every = max(1, int(rebalance_every))
        self.last_episode_cost = 0.0
        self.last_episode_trades = 0

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Bilanz der abgeschlossenen Episode sichern, dann zuruecksetzen.

        Noetig, weil `DummyVecEnv` die Umgebung nach dem letzten Schritt automatisch
        zuruecksetzt und `StockTradingEnv.reset()` dabei `cost` und `trades` auf 0
        setzt. Wer nach `DRL_prediction()` die Gebuehren auslesen will, sieht sonst
        immer 0 - und merkt es nicht.
        """
        self.last_episode_cost = float(getattr(self, "cost", 0.0))
        self.last_episode_trades = int(getattr(self, "trades", 0))
        return super().reset(seed=seed, options=options)

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
