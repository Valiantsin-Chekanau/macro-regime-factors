# Macro-Regime Conditioning of Equity Factors

Conditioning U.S. equity-factor performance (value, momentum, size, profitability,
investment) on a growth × inflation macro-regime layer — built with point-in-time
data to avoid lookahead.

**Status:** Week 1 done (point-in-time data spine + aligned panel). Week 2 done
(growth x inflation regime classifier). Week 3 done (conditional factor performance
+ block-bootstrap Sharpe CIs). Week 4 (non-stationarity + robustness) next. See
`WRITEUP.md` (coming Week 4) for results.

## Thesis
Not a return predictor. Macro is used as a *conditioning variable*: how the equity
factor cross-section reorganizes across macro states, reported with honest error bars.

## Key design choices
- **Point-in-time data (ALFRED vintages)** — only information known as of each date.
- **Interpretable 2×2 classifier** — chosen over latent-state models to avoid silent overfit.
  Growth: YoY INDPRO vs. trailing 5yr mean. Inflation: YoY CPIAUCSL vs. trailing 5yr mean.
  Rationale: `docs/regime_classifier_rationale.md`.
- **Block-bootstrap CIs** — honest uncertainty given few macro cycles (small effective N).
  Block length (6mo) set from the panel's own empirical regime run-length, not tuned
  against the resulting CIs. Rationale: `src/analysis.py` module docstring.

## Data
- Macro: FRED — `INDPRO` (growth), `CPIAUCSL` (inflation); vintages via ALFRED.
- Factors: Ken French Data Library (5-factor + momentum, monthly).

## Results
_Full writeup coming Week 4._ Week 3 headline: of 24 regime x factor Sharpe ratios
(6 factors x 4 regimes), 13 have block-bootstrap 95% CIs that straddle zero —
reported as the finding, not hidden. Standouts that don't straddle zero: HML, RMW,
CMA all positive in Goldilocks and Overheating; HML sharply negative in Stagflation
(point Sharpe -0.55, CI entirely below +0.18); RMW and Mom positive in Stagflation.
Small-N regimes (Stagflation, 108 months) get visibly wider CIs than large ones
(Deflationary slowdown, 174 months) — the effective-N problem showing up exactly
where it should.

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env   # add your FRED_API_KEY
python src/fetch_data.py     # pulls point-in-time macro + factor data -> data/
python src/build_panel.py    # merges into one aligned monthly panel
python src/regimes.py        # labels each month with a growth x inflation regime
```

## Limitations
Small effective sample per regime; regime known only with publication lag; U.S. only.

## Roadmap (v2)
Conditional factor allocation; macro factor attribution.
