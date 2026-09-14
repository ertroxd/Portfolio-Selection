"""Macht `import finrl` ohne Broker-Abhaengigkeiten moeglich.

Problem
-------
`finrl/__init__.py` importiert `finrl.train` / `finrl.trade`, und diese ziehen ueber
`finrl/meta/data_processor.py` und `finrl/meta/env_stock_trading/env_stock_papertrading.py`
die Pakete `alpaca_trade_api`, `alpaca` (alpaca-py), `wrds` sowie `selenium` /
`webdriver_manager` nach - also Live-Broker-, WRDS-Datenbank- und Scraper-Anbindung.
Wir benutzen davon nichts: unsere Kursdaten laden wir selbst ueber yfinance,
gehandelt wird ausschliesslich im Backtest.

`alpaca_trade_api` ist zusaetzlich unmaintained und pinnt pandas < 2, was den Rest der
Umgebung kaputt machen wuerde. Deshalb werden diese Paketbaeume hier durch leere
Platzhalter ersetzt, bevor FinRL importiert wird.

Sobald irgendetwas tatsaechlich auf die Platzhalter zugreift, ist das ein Fehler in
unserem Code - nicht ein stiller Fallback. Die Attribute liefern deshalb Dummy-Klassen,
die beim Instanziieren sofort auffallen wuerden.

Verwendung: als allererste Zeile importieren, vor jedem `finrl`-Import.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

#: Paketbaeume, die FinRL importiert, wir aber nie benutzen.
BLOCKED = ("alpaca_trade_api", "alpaca", "wrds", "selenium", "webdriver_manager")


class _StubModule(types.ModuleType):
    """Modul, das jedes angefragte Attribut als Dummy-Klasse zurueckgibt."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__doc__": f"Platzhalter fuer {self.__name__}.{name}"})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def install() -> None:
    """Shim aktivieren (idempotent)."""
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


install()
