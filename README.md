# Macro-Regime Conditioning of Equity Factors

Conditioning U.S. equity-factor performance (value, momentum, size, profitability,
investment) on a growth × inflation macro-regime layer — built with point-in-time
data to avoid lookahead.

**Status:** Weeks 1-4 done — point-in-time data spine + aligned panel; growth x
inflation regime classifier; conditional factor performance + block-bootstrap
Sharpe CIs; non-stationarity split + one robustness check. Remaining: figures +
final polish. Full results and limitations: `WRITEUP.md`.

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
_Full tables and caveats: `WRITEUP.md`._ Week 3 headline: of 24 regime x factor
Sharpe ratios (6 factors x 4 regimes), 13 have block-bootstrap 95% CIs that
straddle zero — reported as the finding, not hidden. Standouts that don't straddle
zero: HML, RMW, CMA all positive in Goldilocks and Overheating; RMW and Mom
positive in Stagflation; Mkt-RF and Mom positive in Goldilocks, Mkt-RF in
Deflationary slowdown (11 cells in all). HML's sharply negative point Sharpe in
Stagflation (-0.55) does NOT clear the bar: its CI [-1.28, +0.17] still crosses
zero — even the sharpest conditional point estimate in the table stays inside
small-N uncertainty, which is the thesis working as intended. Week 4: only 2 of
the 11 standouts independently reconfirm in both halves of a pre/post-2000 split,
and in 3 cells a premium whose pre-2000 CI cleared zero flips sign after 2000;
10 of 11 survive swapping the classifier's 60mo trailing window for 36mo (77.6%
label agreement).

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env   # add your FRED_API_KEY
python src/fetch_data.py     # pulls point-in-time macro + factor data -> data/
python src/build_panel.py    # merges into one aligned monthly panel
python src/regimes.py        # labels each month with a growth x inflation regime
python -m src.analysis       # conditional stats + block-bootstrap Sharpe CIs
python -m src.stability      # pre/post-2000 non-stationarity split
python -m src.robustness     # one robustness check: 60mo -> 36mo classifier window
python -m pytest tests/ -q   # sanity + regression suite
```

## Limitations
Small effective sample per regime; regime known only with publication lag; U.S. only.

## Roadmap (v2)
Conditional factor allocation; macro factor attribution.
