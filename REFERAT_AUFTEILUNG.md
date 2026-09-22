# Referat-Aufteilung — wer macht was

**Nicht Teil des Git-Repos, absichtlich.** Liegt wie `ARCHITEKTUR_BEGRUENDUNG.md`
eine Ebene über `Portfolio-Selection/` — internes Organisationsdokument für
die Gruppe, kein Abgabebestandteil.

**Entschieden am 2026-09-17.** 15 Minuten gesamt, 5 Minuten pro Person +
Diskussion, entlang der vier Pflichtbausteine der Aufgabenstellung
(`Vorlesungsunterlagen/rl_referate v2.pdf`, Folie 6).

---

## Person 1 — **Lars** (übernommen)

**Thema:** Problem, FinRL, MDP-Formulierung
**Deckt ab:** Baustein 1 ("Portfolio-Auswahl als RL-Problem formulieren")

1. **Hook/Motivation:** Warum Portfolio-Selection als RL-Problem, warum interessant (kurz, 1 Folie).
2. **FinRL kurz vorstellen:** Drei-Schichten-Architektur, welche vier Bausteine genutzt werden (`StockTradingEnv`, `FeatureEngineer`, `data_split`, `DRLAgent`) — und die eine Kernaussage, warum FinRL nicht geklont, sondern als gepinnte Dependency eingebunden wird.
3. **MDP-Formulierung:** State (28 Zahlen: Cash-Anteil, Depotanteile, RSI, MACD), Action (9 Zahlen −1..+1), Reward (ΔVermögen × reward_scaling), Episode. Kern von Baustein 1 — hier zählt Präzision, nicht Tempo.

**Vorbereitungsmaterial:**
- `FINRL_TRAINING_BACKTEST.md`, Abschnitte 1–3 (Gesamtablauf, FinRL-Bausteine im Detail, "FinRL wird installiert, nicht verändert") und 6.3 (State/Action/Reward als MDP).
- Für Nachfragen zu Stable-Baselines3/PPO/ActorCriticPolicy: die Erklärung dazu steht im Chat-Verlauf, nicht in einer Datei — bei Bedarf sagen, dann wird sie noch ergänzt.
- `ARCHITEKTUR_BEGRUENDUNG.md` als Backup, falls in der Diskussion nach der Netzarchitektur (nicht nur dem Environment) gefragt wird.

---

## Person 2 — *offen, wer übernimmt?*

**Thema:** Trade-Republic-Kostenmodell und Environment — der eigene Beitrag
**Deckt ab:** Baustein 2 ("realistische Restriktionen und Transaktionskosten") — vermutlich die stärkste Folie für "Innovation Lösungsansatz"

1. **Das Kostenproblem:** FinRL kennt nur prozentuale Kosten, TR verlangt eine Fixgebühr — die Tabelle mit 50€/1.000€/5.000€ zeigt den Effekt sofort visuell.
2. **`TradeRepublicEnv`:** `_buy_stock`/`_sell_stock` kurz erklärt (additiv statt multiplikativ, Gebühr zuerst abziehen, Nichtstun bleibt kostenlos) — mit der 700€-Beispielrechnung.
3. **"Semi-automatisch" und Restriktionen:** die gewählte Interpretation (monatlicher Handelstakt), plus kurz die anderen Restriktionen (ganze Stücke, Totzone) als Übergang zu den Limitationen.

**Vorbereitungsmaterial:**
- `FINRL_TRAINING_BACKTEST.md`, Abschnitte 4 (TradeRepublicEnv Zeile für Zeile) und 5 (Wo genau die Kosten implementiert sind, inkl. Vergleichstabelle Original vs. eigene Formel).
- `kickoff_vorbereitung.md`, Fokus 1 (FinRL-Kostencode) und Fokus 2 (Trade-Republic-Kostenmodell als Formel).
- `ERKLAERUNG.md`, Abschnitt 4 (dieselbe Erklärung ohne Vorwissen, gut zum Gegenlesen).

---

## Person 3 — *offen, wer übernimmt?*

**Thema:** Backtesting, Ergebnisse, Benchmarks, Abgrenzung
**Deckt ab:** Baustein 3 + 4 ("Backtesting", "Vergleich mit Benchmarks nach Rendite und Risiko") — und "Abgrenzung der Lösung"

1. **Backtest-Setup:** Train/Valid/Test-Split, warum getrennt (Overfitting-Vermeidung), 8 Seeds wegen Seed-Varianz.
2. **Benchmarks & Kennzahlen:** 1/N, Markt, periodisch rebalanciert — Sharpe/Sortino/MaxDD, kurz warum diese Auswahl.
3. **Kernergebnis:** die Forschungsfrage beantworten. **Wichtig, siehe Warnhinweis unten** — hier hat sich der Stand seit der ursprünglichen Fragestellung präzisiert.
4. **Limitationen/Abgrenzung:** die "bewusst weggelassen"-Liste (Steuern, Spread, ein Testfenster, risikoneutraler Reward …) — kurz und selbstbewusst, nicht entschuldigend.

> ⚠️ **Wichtig für Person 3 — nicht die alte Formulierung übernehmen:**
> Die ursprüngliche Forschungsfrage war "ab welchem Portfoliowert *kippt*
> das Ergebnis" (`kickoff_vorbereitung.md`, Teil A.3). Mit den echten
> Ergebnissen aus allen 6 Konfigurationen (fertig trainiert, 8 Seeds je
> Konfiguration, Stand 2026-09-16/17) zeigt sich: **PPO schlägt den besten
> einfachen Benchmark in keiner der drei Kapitalstufen im Mittel** — es gibt
> also **keinen beobachteten Kipppunkt**, nur einen klaren, statistisch
> abgesicherten Trend (Wilcoxon-Test zwischen 1.000€- und 1.000.000€-Sharpe,
> n=8 gepaart: p = 0,016). Formulierung im Vortrag: *"klarer Trend, kein
> Crossover in unserem getesteten Bereich"* statt *"wir haben die Kippstelle
> gefunden"*.
>
> Zusätzlich eine **Konfundierung**, die man selbst benennen sollte, bevor
> sie als Frage kommt: der große Sharpe-Sprung zwischen 1.000€ (Ø 0,85) und
> 10.000€ (Ø 1,14) hat vermutlich zwei vermischte Ursachen — nicht nur die
> relativ kleinere Gebühr, sondern auch, dass der Agent bei 1.000€ nur 3 von
> 9 Titeln überhaupt handelt (Ganzstück-/Totzonen-Problem), bei 10.000€
> dagegen 5 von 9. Beide Effekte laufen in dieselbe Richtung und lassen sich
> mit den aktuellen Daten nicht sauber trennen.
>
> Alle Zahlen (Ergebnistabelle über alle 6 Konfigurationen, Wilcoxon-Test,
> Asset-Zähl-Analyse) stehen im Chat-Verlauf vom 2026-09-16/17 — noch nicht
> in eine Datei übertragen. Wer diesen Teil übernimmt, sollte das nachfragen,
> damit die genauen Zahlen nicht neu ausgegraben werden müssen.

**Vorbereitungsmaterial:**
- `FINRL_TRAINING_BACKTEST.md`, Abschnitte 7 (Trainingsablauf), 8 (Backtest, Kennzahlen, Benchmarks) und 10 (offene Diskussionspunkte).
- Ergebnisordner: `runs/20260916-*` (alle 6 Hauptkonfigurationen, fertig).

---

## Gemeinsam / am Ende

- **AI-Disclaimer:** eigene, kurze Folie, gemeinsam getragen — nicht einer Person aufbürden.
- **Diskussion:** alle drei bleiben vorne, niemand "übergibt und setzt sich".

## Zur "keine Wissensinseln"-Anforderung

Jeder muss auch die *anderen* Teile erklären können, nicht nur den eigenen.
Am besten testet ihr euch gegenseitig anhand der neun Diskussionspunkte in
`FINRL_TRAINING_BACKTEST.md`, Abschnitt 10 — über Kreuz, nicht jeder nur zu
seinem eigenen Teil.

## Noch zu klären

- Wer übernimmt Person 2 und Person 3?
- Die beiden ⚠️-Punkte oben (Kippstelle-Framing, Konfundierung) sollten in
  `FINRL_TRAINING_BACKTEST.md` Abschnitt 9 nachgetragen werden, sobald Zeit
  ist — aktuell nur hier und im Chat-Verlauf festgehalten.
