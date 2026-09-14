# Kickoff-Vorbereitung — Thema 7 (FinRL + Trade Republic)

> Teil A + Fokus 1 = meine eine Seite. Fokus 2 und 3 habe ich zusätzlich mitgemacht,
> damit wir im Termin zu jedem Punkt Material haben — nicht um jemandem die Arbeit
> wegzunehmen. Wer 2 oder 3 übernommen hat: bitte unabhängig davon eigene Notizen
> mitbringen, sonst haben wir statt drei Meinungen nur eine.

---

## Teil A.1 — Aufgabenstellung in eigenen Worten

**Originaltext (Folie 4 + 6, „rl_referate v2.pdf"):**
Titel: *„RL zur semi-automatischen Portfolio-Selection auf Basis von FinRL und Trade-Republic Konditionen"*
Detail: *„Einarbeitung in FinRL und Formulierung der Portfolio-Auswahl als Reinforcement-Learning-Problem. Berücksichtigung realistischer Restriktionen und Transaktionskosten auf Basis der Trade-Republic-Konditionen. Backtesting der resultierenden Strategie und Vergleich mit einfachen Benchmarks unter Berücksichtigung von Rendite und Risiko."*

**Was verlangt wird — inhaltlich (4 Bausteine, alle Pflicht):**
1. FinRL verstehen und Portfolio-Auswahl sauber als MDP formulieren — State, Action, Reward, Transition explizit benennen. Nicht „wir haben FinRL laufen lassen".
2. Realistische Restriktionen **und** Transaktionskosten nach TR-Konditionen einbauen.
3. Backtesting der resultierenden Strategie.
4. Vergleich mit **einfachen** Benchmarks, und zwar nach **Rendite und Risiko** — beides, nicht nur Rendite.

**Was verlangt wird — formal (Folien 2, 3, 10):**
- Gruppe 3–4 Personen, **5 Min pro Person** + Diskussion → bei 3 Personen ca. 15 Min Vortrag. Das ist wenig. Scope entsprechend klein halten.
- „Keine Wissensinseln" — jeder muss zu jedem Teil Fragen beantworten können. Aufgabenteilung ≠ Wissensteilung.
- Abgabe: Präsentation **und** Quellcode (GitHub-URL bevorzugt), in den ILIAS-Abgabeordner.
- **AI-Disclaimer**: was wurde wie wozu genutzt. Ausdrücklich: AI-Lösungen und Quellen müssen verstanden und geprüft sein, wir haften für das Ergebnis.
- Präsentation muss **ohne Tonspur nachvollziehbar** sein → Kommentare + Anhang einplanen, das ist Arbeit und keine Formalie.
- Bewertet wird u. a.: Abgrenzung der Lösung, Verplausibilisierung des Forschungsstands, Innovation, Erkenntnisgewinn für die Kommilitonen, Selbstständigkeit, Diskussion.

**Was Interpretationsspielraum ist:**
- **„semi-automatisch"** ist nirgends definiert. Das ist unser wichtigster Freiheitsgrad und gleichzeitig der Ort, an dem der Punkt „Innovation Lösungsansatz" verdient wird. Mögliche Lesarten: (a) Agent schlägt Rebalancing vor, Mensch bestätigt; (b) niedrige Entscheidungsfrequenz (monatlich statt täglich), damit ein Mensch überhaupt mitkommt; (c) Agent verteilt nur eine feste Sparrate, das Grundportfolio ist gesetzt. Wir müssen uns im Termin auf **eine** Lesart festlegen und sie auf einer Folie begründen.
- **Asset-Universum**: Anzahl, Assetklasse, Währung — komplett offen.
- **„einfache Benchmarks"**: nicht spezifiziert. Wir wählen.
- **Algorithmus**: offen (FinRL liefert A2C, DDPG, TD3, SAC, PPO).
- **„realistische Restriktionen"**: offen — ganze Stücke vs. Bruchstücke, kein Leerverkauf, kein Hebel, Mindestbeträge, Handelszeiten.
- **Zeiträume und Splits**: offen.
- **Kennzahlen**: offen (Sharpe, MaxDD, Sortino, CAGR, Vola).
- „**Berücksichtigung** der TR-Konditionen" heißt nicht „vollständige Nachbildung". Vereinfachen ist erlaubt — aber jede Vereinfachung muss benannt und begründet werden. Folie 2 sagt sogar explizit: *„Teil der Aufgabe wird es sein, Themen selbstständig abzugrenzen."* Die Weglass-Liste ist damit **Teil der Leistung**, nicht Faulheit. Ich würde sie als eigene Folie bringen.

**Was erkennbar nicht verlangt ist** (mein Vorschlag zur Abgrenzung): Live- oder Paper-Trading, Steuermodellierung, eigene Kursprognose, Multi-Agent, systematische Hyperparameter-Studie.

---

## Teil A.2 — FinRL-Repo, Überblick (nur gelesen, nicht ausgeführt)

Lokaler Stand: `C:\Users\Erik\Desktop\FinRL`, Version **1.66.32**.

| Pfad | Inhalt / Relevanz |
|---|---|
| `finrl/meta/preprocessor/` | `yahoodownloader.py`, `preprocessors.py` → `FeatureEngineer`, `data_split()` |
| `finrl/meta/env_stock_trading/env_stocktrading.py` | **`StockTradingEnv`** — Cash + ganzzahlige Stückzahlen, prozentuale Kosten |
| `finrl/meta/env_portfolio_allocation/env_portfolio.py` | `StockPortfolioEnv` — Gewichte via Softmax, Kovarianzmatrix im State. **Kosten werden ignoriert** (s. Fokus 1) |
| `finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py` | `PortfolioOptimizationEnv` — neuer, EIIE/PVM, zwei Gebührenmodelle `wvm` / `trf` |
| `finrl/agents/stablebaselines3/models.py` | `MODELS = {a2c, ddpg, td3, sac, ppo}` über Stable-Baselines3 |
| `finrl/agents/portfolio_optimization/` | EIIE-Architektur + eigener Policy-Gradient-Trainer |
| `finrl/plot.py` | `backtest_stats()` über pyfolio, `backtest_plot()` gegen `baseline_ticker` (Default `^DJI`) |
| `finrl/config.py` | Default-Zeiträume, `INDICATORS`-Liste, Hyperparameter je Algorithmus |

Gelesenes Notebook: `examples/FinRL_PortfolioOptimizationEnv_Demo.ipynb` — `PortfolioOptimizationEnv` mit `initial_amount=100000`, `comission_fee_pct=0.0025`, `time_window=50`, `features=["close","high","low"]`, Modell `"pg"` mit EIIE-Policy, Training 2011–2019, Test getrennt für 2020 / 2021 / 2022, einziger Benchmark **Uniform Buy & Hold**.

**Wichtigster Befund:** die drei Envs behandeln Transaktionskosten grundverschieden (von „gar nicht" bis „iterativer Fixpunkt"). Die Wahl der Env ist bei unserem Thema keine Geschmacksfrage, sondern die zentrale Designentscheidung. Details in Fokus 1.

---

## Teil A.3 — Mein Vorschlag: Forschungsfrage und Scope

**Forschungsfrage (ein Satz):**
> *Kann ein PPO-Agent, der ein kleines EUR-ETF-Portfolio unter Trade Republics Fixgebühr von 1 € pro Order monatlich rebalanciert, nach Kosten eine bessere risikoadjustierte Rendite (Sharpe, Max Drawdown) erzielen als ein 1/N-Buy-and-Hold — und ab welchem Portfoliowert kippt das Ergebnis?*

Der zweite Halbsatz ist mir wichtig. Eine **Fixgebühr** wirkt relativ zum Portfoliowert: 1 € auf 1.000 € sind 0,1 % pro Order, auf 100.000 € sind es 0,001 %. Genau das kann ein prozentuales Kostenmodell — also das, was FinRL und die gesamte von mir gesichtete Literatur verwenden — strukturell nicht abbilden. Das ist unser Alleinstellungsmerkmal gegenüber „noch ein DRL-Trading-Referat" und liefert den geforderten Erkenntnisgewinn.

**Scope:**
- **Assets:** 3 EUR-notierte Xetra-ETFs + Cash. Vorschlag: `EUNL.DE` (MSCI World), `IS3N.DE` (EM IMI), `EUNA.DE` (Euro-Staatsanleihen). Wenige Assets, weil die Fixgebühr genau dann relevant wird und weil 15 Min Vortrag keine 30 Titel tragen.
- **Zeiträume:** Training 2015-01-01 – 2021-12-31 · Validierung 2022-01-01 – 2022-12-31 (Zins-/Crashjahr, guter Robustheitstest) · Test 2023-01-01 – 2025-12-31. Test wird bis zur Abgabe nicht angefasst.
- **Algorithmus:** **PPO**. On-policy, stabil, wenig Hyperparameter-Gefrickel, Standard in der Literatur. A2C nur als Sanity-Check, falls Zeit bleibt.
- **Environment:** eigene Subklasse `TradeRepublicEnv(StockTradingEnv)` mit überschriebenem `_buy_stock` / `_sell_stock`. Begründung siehe Fokus 1.
- **Entscheidungsfrequenz:** monatlich — das ist zugleich unsere Definition von „semi-automatisch": eine Handvoll Entscheidungen pro Jahr, die ein Mensch prüfen und freigeben könnte.
- **Benchmarks:** (1) 1/N Buy & Hold, (2) 100 % MSCI World Buy & Hold, (3) monatlicher 1/N-**Sparplan** — bei TR gebührenfrei und damit der ehrlichste Gegner.
- **Kennzahlen:** CAGR, annualisierte Vola, Sharpe, Sortino, Max Drawdown, Turnover, **absolute Gebühren in €** und Gebühren als % des Endvermögens.
- **Bewusst weggelassen:** Steuern (KapESt, Soli, Vorabpauschale, Teilfreistellung) · Spread / Slippage / Market Impact · Fremdwährung (entfällt durch EUR-Assets) · Leerverkäufe und Hebel · Intraday · Live-/Paper-Trading · Hyperparameter-Sweep · Crypto und Einzelaktien · Bruchstückhandel.

**Konkret zum Sanity-Check:** 10.000 € Startkapital, 3 Assets, monatliches Rebalancing → max. 36 Orders/Jahr = 36 € = **0,36 % p. a.** Das ist die Größenordnung einer ETF-TER, also klar performance-relevant. Bei 1.000 € Startkapital wären es 3,6 % p. a. und die Strategie ist tot. Genau diese Kippstelle wollen wir zeigen.

---

## Teil A.4 — Drei Fragen, die ich allein nicht klären konnte

1. **Was heißt „semi-automatisch" für Guericke?** Reicht das Framing „Agent schlägt vor, Mensch bestätigt, deshalb monatliche Frequenz", oder erwartet er ein tatsächlich implementiertes Human-in-the-Loop-Element? Davon hängt ab, ob wir eine Interaktionskomponente bauen müssen oder nicht.
2. **Wie viel Statistik erwartet er?** Ein Trainingslauf pro Algorithmus, oder mehrere Random Seeds mit Streuungsangabe? Die Literatur verlangt eindeutig mehrere Seeds (siehe Fokus 3, Schwäche 3), das kostet aber Rechenzeit — und der Bewertungspunkt „Verplausibilisierung Stand der Forschung" deutet darauf hin, dass er es sehen will.
3. **Wie tief muss der Benchmark-Vergleich gehen?** Plot + Kennzahlentabelle, oder Robustheits-/Signifikanzaussagen (Walk-Forward, Deflated Sharpe Ratio)? Das entscheidet über den Umfang von Phase 1 erheblich.

*(Nebenfrage an die Gruppe, keine an den Prof: Geben wir einen FinRL-Fork ab oder ein eigenes Repo, das FinRL als Dependency zieht? Ich bin für Letzteres — sauberer für den AI-Disclaimer und für die Frage „was ist eure Leistung".)*

---
---

## Fokus 1 — Code: Wo FinRL Transaktionskosten verrechnet

### Die Hauptstelle

**Datei:** `finrl/meta/env_stock_trading/env_stocktrading.py`
**Klasse:** `StockTradingEnv`
**Funktionen:** `_sell_stock()` mit innerem `_do_sell_normal()` (ca. Z. 113–177) und `_buy_stock()` mit innerem `_do_buy()` (ca. Z. 179–219).

Die Kostenformel ist **rein proportional**, ohne Fixanteil und ohne Mindestgebühr:

```python
# Verkauf (Z. ~126-140)
sell_amount = price_i * n_sell * (1 - self.sell_cost_pct[i])
self.state[0]  += sell_amount                       # Cash steigt um den Netto-Erlös
self.cost      += price_i * n_sell * self.sell_cost_pct[i]
self.trades    += 1

# Kauf (Z. ~186-207)
available = self.state[0] // (price_i * (1 + self.buy_cost_pct[i]))   # Budgetprüfung
n_buy      = min(available, action)
buy_amount = price_i * n_buy * (1 + self.buy_cost_pct[i])
self.state[0]  -= buy_amount
self.cost      += price_i * n_buy * self.buy_cost_pct[i]
self.trades    += 1
```

Drei Dinge, die man dabei wissen muss:
- `self.cost` ist **reines Reporting**. Der Reward ist `end_total_asset - begin_total_asset` (Z. ~360); Kosten wirken also nur indirekt über den Cash-Bestand. Es gibt keinen expliziten Kostenterm im Reward.
- `step()` skaliert `actions * hmax` und castet auf `int` → **nur ganze Stücke**, keine Bruchstücke.
- Verkäufe werden vor Käufen ausgeführt (`argsort_actions`, Z. ~327–341) — das Budget für Käufe enthält also bereits die Verkaufserlöse desselben Tages.

### Zwei weitere Stellen, die für uns entscheidend sind

**(a) `finrl/meta/env_portfolio_allocation/env_portfolio.py`, `StockPortfolioEnv`:**
`transaction_cost_pct` wird im Docstring dokumentiert (Z. 28), als Konstruktorparameter entgegengenommen (Z. 72) und als Attribut gesetzt (Z. 89) — und danach **nie wieder verwendet**. Ein Grep über das gesamte Repo findet genau diese drei Vorkommen; in `step()` taucht es nicht auf. Wer diese Env benutzt, handelt faktisch **kostenlos**, ohne dass irgendetwas warnt. Für uns ist das ein K.-o.-Kriterium — und gleichzeitig ein sehr vorzeigbarer Befund für das Referat, weil es exakt die Schwäche belegt, die Fokus 3 in der Literatur findet.

**(b) `finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py`, `PortfolioOptimizationEnv.step()` (Z. ~305–335)** — zwei wählbare Modelle:
- `comission_fee_model="wvm"` (weights vector modifier): `fees = Σ |Δw_i| · V`, wird vom Cash-Gewicht abgezogen; reicht das Cash nicht, wird das Rebalancing komplett verworfen. **Auffällig:** in diesem Zweig taucht `comission_fee_pct` überhaupt nicht auf, die Gebühr entspricht damit 100 % des Umsatzes. Sieht nach Bug aus — vor dem Zitieren gegen den aktuellen Upstream prüfen.
- `comission_fee_model="trf"` (transaction remainder factor, Default): iterative Fixpunktberechnung von μ nach Jiang et al. 2017, dann `V_neu = μ · V`. Sauber, aber strikt proportional.

### Aufwand für eine Fixgebühr pro Order

**Gering — geschätzt 30–60 Zeilen, ein Nachmittag inklusive Test.** Nicht der Code ist die Arbeit, sondern die Fallstricke.

Weg: Subklasse `TradeRepublicEnv(StockTradingEnv)`, nur `_buy_stock` und `_sell_stock` überschreiben:

```
fee = FIXED_FEE  wenn n_shares > 0, sonst 0
Verkauf: cash += price*n - fee ;  cost += fee
Kauf:    cash -= price*n + fee ;  cost += fee
Budget:  available = (cash - FIXED_FEE) // price      # additiv statt multiplikativ!
```

Fallstricke, in der Reihenfolge, in der sie beißen:
1. Die Gebühr darf **nur bei tatsächlich ausgeführter Order** anfallen (`n_shares > 0`) — sonst zahlt der Agent fürs Nichtstun und lernt Unsinn.
2. Die Budgetformel muss von multiplikativ auf additiv umgestellt werden. Übernimmt man `cash // (price*(1+pct))`, kann das Cash nach Abzug der Fixgebühr **negativ** werden.
3. TR berechnet die Pauschale bei Teilausführungen nur **einmal je Handelstag**. Bei Tagesdaten und einer Order pro Titel und Tag ist das identisch — muss aber erwähnt werden, sonst ist es eine unbenannte Vereinfachung.
4. `self.cost` fließt nicht in den Reward. Wenn wir wollen, dass der Agent Kosten *lernt* statt sie nur zu *erleiden*, brauchen wir entweder einen expliziten Malus im Reward oder das Argument, dass der Vermögenseffekt reicht. Ich bin für Letzteres, aber wir sollten es bewusst entscheiden.

**Vorarbeit existiert bereits:** `Desktop\-Portfolio-Selection\thema7_portfolio_selection_rl.ipynb` enthält eine solche `TradeRepublicEnv` mit 1-€-Fixgebühr. Ein bereits gefundener Stolperstein: der Train/Test-Split muss über FinRLs `data_split()` laufen (setzt den Index auf `date.factorize()[0]`), **nicht** über `reset_index(drop=True)` — sonst crasht `_get_date()` mit `'str' object has no attribute 'unique'`.

**Warum nicht die anderen Envs:** Bei `StockPortfolioEnv` gibt es weder Stückzahlen noch ein Cash-Konto, nur Gewichte — man müsste erst eine Orderlogik nachbauen, deutlich mehr Aufwand. Bei `PortfolioOptimizationEnv` ginge es (Gewichte und Portfoliowert sind da): ein dritter `comission_fee_model="fixed"` mit `n_orders = Σ 1[|Δw_i|·V > ε]` und `V -= n_orders · 1 €`. Das wäre die elegantere Lösung, kostet aber einen Eingriff in Fremdcode statt einer Subklasse.

---

## Fokus 2 — Kostenmodell Trade Republic

**Quellenlage vorab, weil es eine Falle ist:** Die vollständige, aktuelle Preis- und Leistungsverzeichnis-PDF ist über das Web **nicht mehr abrufbar** (`assets.traderepublic.com/.../Preis_und_Leistungsverzeichnis.pdf` liefert HTTP 403); die Website sagt selbst: *„The full pricing list is available in the app."* Ich habe deshalb zwei offizielle Quellen kombiniert: die **Preisübersicht auf traderepublic.com** (aktuell) und die noch erreichbare **PLV-PDF v04.2023**. Wer die App hat, sollte vor der Abgabe die dortige PLV gegenprüfen und den Stand als Screenshot in den Anhang legen.

### Was TR berechnet

**Preisübersicht traderepublic.com (Stand 09/2026):**
- Orderprovision: **kostenfrei**
- **Abwicklungskostenpauschale: 1,00 €** (früher „Fremdkostenpauschale")
- bei Auswahl „Direktpreis an der Börse": **2,00 €**
- Ausführung von Sparplänen (Aktien, ETFs, Crypto): **kostenfrei**
- Dividenden und Kapitalmaßnahmen: kostenfrei
- Hinweis im Kleingedruckten: *„Other product fees, including spread and third party costs, may apply."*

**PLV v04.2023 ergänzt:**
- Fremdkostenpauschale 1,00 € **je Handelsgeschäft, ausgenommen Sparpläne**
- *„Im Falle von Teilausführungen wird nur einmalig je Handelstag die Fremdkostenpauschale berechnet."*
- Sparplankauf ETF/Aktie: kostenfrei, **Anlagevolumen pro Ausführung 10 € bis 10.000 €**
  ⚠️ Die aktuellen Support-Seiten nennen inzwischen **1 €** Mindestbetrag. Widerspruch zwischen 2023er PDF und heutigem Stand — im Anhang offenlegen.
- Depotführung, Verrechnungskonto, Depotübertrag, Überweisung auf Referenzkonto: kostenfrei
- Postalische Auftragserteilung: 25,00 €

**Fremdwährung** (`support.traderepublic.com/de-de/88`): Umrechnung zum mehrmals täglich ermittelten Interbanken-Geld-/Briefkurs plus währungsabhängiger Marge — USD, GBP, CHF ca. **0,11–0,14 %**, exotischere Währungen deutlich mehr. Belastungen zum Devisenverkaufs-, Gutschriften zum Devisenankaufskurs.

### Unser Kostenmodell als Formel

Vollständig, wie TR real abrechnet:

```
C_t = Σ_i  1[|Δn_i,t| > 0] · f_fix                (Orderpauschale)
    + Σ_i  s_i,t · |Δn_i,t| · p_i,t               (Spread / Handelsplatzkosten)
    + FX_t                                        (Devisenmarge)
    + laufende Produktkosten (TER)                (bereits im Kurs enthalten)

mit f_fix = 1,00 €  (2,00 € bei Direktpreis; 0,00 € bei Sparplanausführung)
und: bei Teilausführungen wird f_fix nur einmal je Handelstag und Wertpapier berechnet.
```

**Was wir abbilden:**
```
C_t = Σ_i  1[|Δn_i,t| > 0] · 1,00 €
```
- 1 € je tatsächlich ausgeführter Order, Kauf **und** Verkauf
- keine Gebühr, wenn nicht gehandelt wird — der Agent darf kostenlos nichts tun
- der Sparplan-Benchmark läuft gebührenfrei (`f_fix = 0`)
- nur EUR-Handelsplätze, damit `FX_t = 0` per Konstruktion

**Was wir bewusst weglassen — und warum:**

| Weggelassen | Begründung |
|---|---|
| Spread / Handelsplatzkosten | In Tages-Schlusskursen nicht abbildbar, keine öffentliche historische Datenquelle. Eine Zahl zu erfinden wäre Scheingenauigkeit. Kommt auf die Limitationen-Folie. |
| Direktpreis-Variante (2,00 €) | Wir nehmen durchgängig 1,00 €. Eine Variable weniger; die Richtung des Ergebnisses ändert sich dadurch nicht. |
| Fremdwährung | Entfällt durch die Asset-Auswahl (EUR-notierte Xetra-ETFs). Das ist eine Konstruktionsentscheidung, keine Vereinfachung. |
| TER der ETFs | Steckt bereits im Kursverlauf der ETF-Zeitreihe. Ein separater Abzug wäre Doppelerfassung. |
| Steuern (KapESt 25 % + Soli, Vorabpauschale, Teilfreistellung 30 %) | Eigenes Referat wert. Verzerrt den Vergleich außerdem für alle Strategien in dieselbe Richtung. |
| Mindest-/Höchstbetrag Sparplan (10 € / 10.000 €) | Nur für den Sparplan-Benchmark relevant, bei unserer Größenordnung nie bindend. |
| Teilausführungsregel | Bei Tagesdaten und einer Order je Titel und Tag ohnehin identisch. |
| Bruchstückhandel | TR kann Bruchstücke, FinRL castet auf `int`. Wir bleiben bei ganzen Stücken — bei ETF-Kursen um 100 € tolerierbar, gehört aber in die Restriktionsliste. |

**Kalibrierung:** 10.000 € Startkapital, 3 Assets, monatliches Rebalancing → max. 36 €/Jahr = 0,36 % p. a., also TER-Größenordnung. Das ist die Zahl, die unsere Forschungsfrage überhaupt erst interessant macht.

---

## Fokus 3 — Forschungsstand

### Quelle 1 — Jiang, Xu & Liang (2017), *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*, arXiv:1706.10059

Die Autoren formulieren Portfolio-Management als modellfreies RL-Problem, bei dem die Aktion direkt der Gewichtsvektor über alle Assets plus Cash ist. Kernbeitrag ist die **EIIE**-Topologie (Ensemble of Identical Independent Evaluators): dasselbe kleine Netz bewertet jedes Asset separat, wodurch das Modell unabhängig von der Assetanzahl und -reihenfolge wird — dazu ein Portfolio-Vector-Memory für den vorherigen Gewichtsvektor und Online Stochastic Batch Learning. Getestet wird auf Kryptowährungen in 30-Minuten-Intervallen mit einer Kommission von 0,25 %, verglichen gegen mehrere publizierte Portfolio-Selection-Strategien; die drei Netzvarianten (CNN, RNN, LSTM) belegen in allen Experimenten die vorderen Plätze, mit vierfachem Ertrag in 50 Tagen. Für uns wichtig: die Kostenbehandlung über den **Transaction Remainder Factor μ** — genau das, was FinRL als `comission_fee_model="trf"` implementiert. Und genau hier liegt unsere Abgrenzung: μ ist strikt proportional und kann eine Fixgebühr prinzipiell nicht darstellen.

### Quelle 2 — Liu et al. (2020, rev. 2022), *FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance*, arXiv:2011.09607

FinRL ist als dreischichtige Bibliothek angelegt (Datenschicht, Environment-Schicht, Agenten-Schicht) und zielt explizit auf Einsteiger und auf Reproduzierbarkeit. Unterstützt werden Märkte von NASDAQ-100 über DJIA und S&P 500 bis CSI 300 sowie die DRL-Algorithmen DQN, DDPG, PPO, SAC, A2C und TD3. Als Anwendungsfälle nennt das Paper Single-Stock-Trading, Multi-Stock-Trading und Portfolio-Allokation. Es wirbt damit, *„important trading constraints such as transaction cost, market liquidity and the investor's degree of risk-aversion"* zu integrieren — was wir im Code (Fokus 1) differenzierter sehen: in `StockTradingEnv` stimmt es, in `StockPortfolioEnv` ist der Kostenparameter wirkungslos. Diese Diskrepanz zwischen Anspruch und Implementierung ist selbst ein zitierfähiger Befund für unser Referat.

### Quelle 3 — Gort, Liu, Sun, Gao, Chen & Wang (2022, rev. 2023), *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting*, arXiv:2209.05559

Das Paper greift das Kernproblem der gesamten Literatur an: optimistisch berichtete Backtest-Ergebnisse sind häufig False Positives durch Overfitting. Vorgeschlagen wird, die Erkennung von Backtest-Overfitting als **Hypothesentest** zu formulieren: viele DRL-Agenten trainieren, für jeden die Overfitting-Wahrscheinlichkeit schätzen und überangepasste Agenten verwerfen, statt den besten Backtest zu berichten. Evaluiert wird auf zehn Kryptowährungen im Zeitraum Mai bis Juni 2022, also über zwei Marktcrashs hinweg. Ergebnis: die weniger überangepassten Agenten erzielen höhere Renditen als die stärker überangepassten, als eine Equal-Weight-Strategie und als der S&P DBM Index. Für uns die methodische Leitplanke — und ein guter Beleg dafür, warum wir Validierung und Test strikt trennen müssen.

*(Ergänzend, falls wir einen Survey brauchen: Hambly, Xu & Yang (2021, rev. 2023), „Recent Advances in Reinforcement Learning in Finance", arXiv:2112.04553 — 60 Seiten über optimale Ausführung, Portfoliooptimierung, Option Pricing/Hedging, Market Making, Order Routing und Robo-Advisory.)*

### Übliche Benchmarks in der Literatur

- **Buy & Hold auf einen Index** (DJIA, S&P 500, CSI 300) — der häufigste, aber auch schwächste Vergleich.
- **Uniform Buy & Hold / 1/N (Equal Weight)** — theoretisch überraschend stark und der eigentlich harte Gegner.
- **Mean-Variance / Markowitz** und **Minimum Variance**.
- Klassische Online-Portfolio-Selection-Verfahren: UCRP, OLMAR, PAMR, Best Stock.
- Die DRL-Algorithmen untereinander (A2C, PPO, DDPG, SAC, TD3) sowie Ensemble-Strategien.

### Übliche Kennzahlen

Kumulierte Rendite bzw. Terminal Wealth und CAGR · annualisierte Volatilität · **Sharpe Ratio** (praktisch immer) · Sortino · Calmar · **Maximum Drawdown**. Seltener, aber methodisch besser: Turnover, absolute Kosten, Deflated bzw. Probabilistic Sharpe Ratio.

### Wiederkehrende methodische Schwächen

1. **Backtest-Overfitting und Selektionsbias** — viele Läufe, berichtet wird der beste. Direkt adressiert in Quelle 3.
2. **Ein einziger Testzeitraum**, oft ein Bullenmarkt. Kein Walk-Forward, keine Trennung nach Marktregimen.
3. **Seed-Varianz wird nicht berichtet.** DRL-Ergebnisse streuen über Random Seeds häufig stärker als der behauptete Vorsprung gegenüber dem Benchmark.
4. **Transaktionskosten zu optimistisch oder gar nicht modelliert** — bei FinRLs `StockPortfolioEnv` sogar stillschweigend null. Spread, Slippage und Market Impact fehlen fast durchgängig.
5. **Survivorship Bias** im Asset-Universum: heutige Indexmitglieder werden rückwirkend gehandelt.
6. **Zu schwache Benchmarks** — verglichen wird gegen einen Index, nicht gegen 1/N, das notorisch schwer zu schlagen ist.
7. **Risikoneutraler Reward**: `reward = ΔVermögen` optimiert nur Rendite; Risiko taucht erst in der Auswertung auf. Gegenbewegung in der Literatur: Differential Sharpe Ratio als Rewardsignal.
8. **Data Leakage** durch Skalierung und Feature-Engineering über den gesamten Zeitraum vor dem Split.

**Konsequenz für uns:** mindestens 3 Seeds mit Streuungsangabe, Validierung strikt vom Test getrennt, Vergleich gegen 1/N **und** Sparplan, Kosten in € explizit ausweisen, eine ehrliche Limitationen-Folie. Punkte 4 und 6 sind gleichzeitig unsere Chance: wenn die Literatur Kosten notorisch unterschätzt und gegen schwache Benchmarks testet, ist „Fixkosten korrekt modelliert, gegen den stärksten einfachen Gegner getestet" ein echter Beitrag.

---

## Quellen

- Aufgabenstellung: `rl_referate v2.pdf`, Prof. Dr. Stefan Guericke, 27.08.2026, Folien 2–4, 6, 10
- FinRL v1.66.32, lokal gelesen (nicht ausgeführt): `C:\Users\Erik\Desktop\FinRL`
- [Trade Republic — Preisübersicht (offiziell, Stand 09/2026)](https://traderepublic.com/de-de?openModal=pricing-scheme)
- [Trade Republic — Preis- und Leistungsverzeichnis, v04.2023 (PDF)](https://assets.traderepublic.com/documents/DE/FURTHER_INFORMATION_PRICE_LIST_20230426144143.pdf)
- [Trade Republic Support — Devisenkonvertierung](https://support.traderepublic.com/de-de/88)
- [Trade Republic Support — Ex-Post-Kosteninformation](https://support.traderepublic.com/de-de/809-Was-ist-die-Ex_Post-Kosteninformation)
- [Jiang, Xu & Liang (2017), arXiv:1706.10059](https://arxiv.org/abs/1706.10059)
- [Liu et al. (2020/2022), FinRL, arXiv:2011.09607](https://arxiv.org/abs/2011.09607)
- [Gort et al. (2022/2023), Backtest Overfitting, arXiv:2209.05559](https://arxiv.org/abs/2209.05559)
- [Hambly, Xu & Yang (2021/2023), arXiv:2112.04553](https://arxiv.org/abs/2112.04553)
