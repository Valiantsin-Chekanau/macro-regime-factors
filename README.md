# Macro-Regime Conditioning of Equity Factors

Conditioning U.S. equity-factor performance (value, momentum, size, profitability,
investment) on a growth × inflation macro-regime layer — built with point-in-time
data to avoid lookahead.

**Status:** Week 1 done (point-in-time data spine + aligned panel). Week 2 done
(growth x inflation regime classifier). Week 3 (conditional analysis + error bars)
next. See `WRITEUP.md` (coming Week 4) for results.

## Thesis
Not a return predictor. Macro is used as a *conditioning variable*: how the equity
factor cross-section reorganizes across macro states, reported with honest error bars.

## Key design choices
- **Point-in-time data (ALFRED vintages)** — only information known as of each date.
- **Interpretable 2×2 classifier** — chosen over latent-state models to avoid silent overfit.
  Growth: YoY INDPRO vs. trailing 5yr mean. Inflation: YoY CPIAUCSL vs. trailing 5yr mean.
  Rationale: `docs/regime_classifier_rationale.md`.
- **Block-bootstrap CIs** — honest uncertainty given few macro cycles (small effective N).

## Data
- Macro: FRED — `INDPRO` (growth), `CPIAUCSL` (inflation); vintages via ALFRED.
- Factors: Ken French Data Library (5-factor + momentum, monthly).

## Results
_Coming Week 3-4._

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
