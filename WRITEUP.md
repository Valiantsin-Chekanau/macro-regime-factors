# Macro-Regime Conditioning of Equity Factors — Writeup

*Status: Weeks 1-4 complete (data spine, classifier, conditional analysis, non-stationarity
+ robustness). Buffer week (figures + final polish) not yet done — see Roadmap below.*

## 1. Methodology

The question is not "does macro predict equity returns" (weak, unstable out-of-sample —
Goyal-Welch 2008) but a narrower, more defensible one: **does the equity factor
cross-section reorganize across growth x inflation states, and how much of any apparent
reorganization survives honest treatment of a tiny effective sample?**

Pipeline:
1. **Data** (`src/fetch_data.py`, `src/build_panel.py`): `INDPRO` (growth) and `CPIAUCSL`
   (inflation) from FRED/ALFRED; Ken French 5-factor + momentum monthly returns.
2. **Regime classifier** (`src/regimes.py`): deterministic 2x2 — YoY growth vs. its own
   trailing 60-month mean gives Up/Down; YoY inflation vs. its trailing 60-month mean gives
   Rising/Falling. Crossed into Goldilocks / Overheating / Stagflation / Deflationary
   slowdown. Chosen over a latent-state model (HMM) specifically to stay auditable —
   rationale in `docs/regime_classifier_rationale.md`.
3. **Conditional stats + bootstrap** (`src/analysis.py`): per regime x factor, annualized
   mean/vol/Sharpe/max drawdown/hit rate, plus a 6-month block-bootstrap 95% CI on Sharpe
   (5,000 replicates, seed fixed for reproducibility).
4. **Non-stationarity + robustness** (`src/stability.py`, `src/robustness.py`, Week 4):
   pre/post-2000 split, and a classifier trailing-window perturbation (60mo vs 36mo).

## 2. Point-in-time treatment (lead with this)

Every macro value used is what was *actually published and not-yet-revised* as of the
decision date, pulled via ALFRED vintages (`get_series_as_of_date`) rather than the
latest-revised series most student projects silently use. This matters concretely: FRED
re-references its index base periods over time (e.g. CPI's 1988 rebasing, INDPRO's 1985/
2003/2010 rebasings), and naively differencing the level panel with `pct_change(12)`
mixes vintages ~14 months apart, producing nonsensical YoY spikes (caught in a Jul-2026
audit: CPI's 1988 re-reference read as -65% "inflation" and silently corrupted ~84% of
regime labels downstream). The fix — computing YoY *within* each vintage before splicing
vintages into the panel — is the single most important correctness gate in this project;
everything below assumes it holds, and `tests/test_sanity.py` regression-tests it directly
against the known rebasing dates.

Labeled sample after the 60-month classifier warm-up: **586 months, Aug 1977 - May 2026.**

## 3. Conditional results, with error bars

Of 24 regime x factor Sharpe ratios (6 factors x 4 regimes), **11 have block-bootstrap 95%
CIs that clear zero** — reported as the finding, not the other 13 hidden:

| Regime | Factor | Sharpe | 95% CI | N |
|---|---|---|---|---|
| Deflationary slowdown | Mkt-RF | 0.91 | [0.44, 1.50] | 174 |
| Goldilocks | Mkt-RF | 0.91 | [0.39, 1.49] | 156 |
| Goldilocks | Mom | 0.82 | [0.22, 1.43] | 156 |
| Goldilocks | RMW | 0.61 | [0.06, 1.23] | 156 |
| Goldilocks | HML | 0.59 | [0.09, 1.15] | 156 |
| Goldilocks | CMA | 0.59 | [0.01, 1.16] | 156 |
| Overheating | CMA | 0.63 | [0.10, 1.19] | 148 |
| Overheating | HML | 0.61 | [0.14, 1.10] | 148 |
| Overheating | RMW | 0.53 | [0.09, 1.05] | 148 |
| Stagflation | RMW | 0.88 | [0.25, 1.76] | 108 |
| Stagflation | Mom | 0.80 | [0.19, 1.42] | 108 |

Reading it straight: value (HML), profitability (RMW), and investment (CMA) all clear
zero together in both Goldilocks and Overheating — i.e. when growth is running above
trend, these three "quality-adjacent" factors earn a premium regardless of inflation
direction. Momentum and quality (RMW) are the two factors that clear zero in Stagflation,
where nothing else does. The equity risk premium itself (Mkt-RF) only clears zero in the
two growth-up-adjacent regimes (Goldilocks, Deflationary slowdown) — not Overheating or
Stagflation, which is a sensible pattern (equities like growth without an inflation scare,
or like being past the trough) but is exactly the kind of thing worth flagging as an
*inference*, not a proven mechanism, given N.

**Least intuitive result, and arguably the most interesting one nobody had named until this
writeup:** Deflationary slowdown carries the highest and tightest-lower-bound Mkt-RF Sharpe
in the table (0.91, CI floor 0.44) despite the regime name sounding unambiguously bad. The
likely reason: "Deflationary slowdown" (growth down, inflation down) is not synonymous with
recession — it also captures disinflationary recoveries (falling inflation off a high base
while growth is still below its own trailing trend), which historically have been
equity-friendly (falling discount rates, multiple expansion). The regime label describes
the growth/inflation *state*, not the business-cycle *phase* within it — worth a sentence
in any presentation of this table so a reader doesn't over-read the name.

**What does NOT clear the bar, despite looking dramatic:** HML in Stagflation has the most
negative point Sharpe in the entire table (-0.55) — the "value gets crushed in stagflation"
story a reader would expect — but its CI is [-1.28, +0.17], still crossing zero. Even the
single sharpest conditional point estimate on offer stays inside small-N uncertainty. This
is the thesis working as intended, not a failure to find a clean result.

**A caveat on the 11/24 count itself:** with 24 independent 95%-CI tests, roughly 24 x 0.05
= 1.2 would be expected to clear zero by chance alone if every true Sharpe were exactly
zero. Eleven clearing is well above that chance rate, which is some comfort that the count
isn't pure multiple-comparisons noise — but no formal multiple-comparisons correction
(Bonferroni, Benjamini-Hochberg) has been applied here, and the 24 tests are not
independent of each other (same underlying return series sliced four overlapping ways), so
this is a directional sanity check, not a p-value adjustment.

## 4. Non-stationarity findings (Week 4)

Split the labeled sample at 2000 (pre: 269 months, 1977-1999; post: 317 months, 2000-2026)
and reran the identical conditional + bootstrap pipeline on each half independently.

**Regime frequency shifted between halves** — Overheating went from the smallest regime
pre-2000 (44 months) to the largest post-2000 (104 months); Goldilocks went the other way
(92 -> 64). That alone is a real, if unsurprising, non-stationarity: the growth/inflation
mix of the post-2000 sample (two long expansions, a low-and-falling inflation regime for
most of it, then the 2021-23 inflation spike) looks different from 1977-1999.

**Of the 11 full-sample standouts, only 2 independently reconfirm in both halves:**
Deflationary slowdown/Mkt-RF and Overheating/RMW clear zero on both sides of the 2000
split. Every other full-sample standout — including the Goldilocks HML/RMW/CMA/Mom cluster
that reads as the cleanest story in section 3 — clears zero on the full sample and on *at
most one* of the two halves, not both. Concretely: Goldilocks/HML, RMW, CMA all clear
zero pre-2000 (Sharpes 0.73-1.28) but do NOT clear post-2000 — HML and RMW flip outright
negative (-0.11, -0.12) while CMA merely fades (+0.37, CI straddling); Stagflation/RMW
runs the other way, clearing only post-2000 (+0.37 pre -> +1.31 post). This is presented
as the honest limitation it is: halving N roughly halves the
months per regime-half-cell (Stagflation drops to ~54 months per half), which widens CIs
enough that most individual-period estimates simply can't independently clear the bar even
when the full-sample estimate does. **We cannot distinguish "the relationship is genuinely
non-stationary" from "we no longer have the power to see the same relationship" with this
sample size** — which is itself the finding this project is built to report honestly rather
than paper over. The sign-flip analysis below sharpens this in three cells.

**Point-Sharpe sign flips:** 7 of the 24 comparable regime x factor cells flip sign between
halves, and they come in two kinds. In 3 of the 7 — Goldilocks/HML (+1.02 -> -0.11),
Goldilocks/RMW (+1.28 -> -0.12), and Deflationary slowdown/Mom (+1.26 -> -0.16) — the
pre-2000 CI *clears* zero: a premium that was statistically distinguishable from zero in
the first half flipped sign in the second. The post-2000 CIs straddle zero, so the precise
claim is "a once-clear premium vanished and points the other way," not "a negative premium
is now established" — but these three cells are the most direct evidence of
non-stationarity this project produces. The other 4 flips (e.g. Stagflation/Mkt-RF: +0.54
pre -> -0.18 post) have no clearing CI in either half: reversals indistinguishable from
small-N estimation noise compounding when the sample is halved. (An earlier draft claimed
no flip had a zero-clearing half; that traced to a comparison bug in the analysis code —
`is` vs `==` on a NumPy boolean — caught in a Jul-2026 audit and regression-tested since.)

## 5. Robustness check (Week 4, ONE alternative only)

Guide-sanctioned choice: the classifier's trailing window was locked at 60 months (5yr);
the guide's own text names "trailing 3-5yr mean" as the range considered, so the other end
of that explicitly-sanctioned range — 36 months (3yr) — is the one alternative tested here,
via `src/robustness.py`.

*(Why the window rather than the guide's other listed option, the proxy swap
INDPRO<->PAYEMS or CPI<->core PCE: the trailing window is the classifier's only free
parameter, so perturbing it stress-tests the one design choice actually made in Week 2 —
and it holds the input data fixed, so any movement in the conclusions is attributable to
the classifier itself rather than to a different macro series with its own revision and
coverage quirks. The proxy swap is the natural next check, deliberately not run to keep to
the guide's one-alternative mandate; it is first in line for v2.)*

**Label agreement:** 77.6% of months get the same regime under both windows. Not identical
(the window choice matters) but not scrambled either — a classifier that flipped on every
other choice inside its own sanctioned range would be a real problem; this one isn't that.
One composition note: the 36mo window's shorter warm-up labels 610 months vs. the base's
586 (24 extra months in the mid-1970s), so each window's conditional tables run on their
own full labeled sample; the agreement rate is computed on the 586 months labeled under
both, and the survival comparison below inherits that small sample difference.

**Conclusion survival:** 11/24 cells clear zero under the 60mo (base) window; 10/24 clear
under the 36mo (alt) window. **10 of the 11 base standouts survive under the alt window
unchanged** — the sole casualty is Goldilocks/RMW, which clears under 60mo (Sharpe 0.61,
CI [0.06, 1.23]) but not 36mo. No new cell clears zero under the alt window that didn't
already clear under the base window (no false-positive standouts introduced by the window
choice). This is a reassuring result: the Section 3 findings are not an artifact of picking
5yr specifically over 3yr.

## 6. Honest limitations

- **Effective N is genuinely small.** The 586 labeled months arrive in only 80 contiguous
  regime runs (median length ~5.5 months, 17-23 runs per regime), and runs of the same
  regime cluster into far fewer distinct macro episodes: merging same-regime runs
  separated by less than a year gives ~10-12 episodes per regime; a coarser two-year gap
  rule gives 6-9, with Stagflation at ~6 (essentially the late-70s/early-80s, ~1990,
  2008, and 2021-23 clusters). The unit of independent evidence is closer to the episode
  than the month. This is why the block bootstrap — not iid
  resampling — is the whole point of Week 3, and why Week 4's pre/post split predictably
  produces wide, mostly-inconclusive per-half CIs (Section 4).
- **Point Sharpe uses standard iid annualization** (`mean x 12 / (vol x sqrt(12))`). Lo
  (2002) shows this multiplier is biased when returns are autocorrelated, which they are
  here (regimes persist in runs). The block bootstrap correctly captures autocorrelation in
  the *interval width*, but the reported point Sharpe itself does not apply Lo's
  autocorrelation-adjusted annualization — the point estimates in Sections 3-5 should be
  read as standard-convention Sharpes, not bias-corrected ones.
- **Resampling scheme.** A stationary/episode bootstrap (blocks keyed to actual regime-run
  boundaries rather than a fixed 6-month length) was considered and declined for scope —
  the fixed-length moving block bootstrap is simpler, matches the guide's one-bootstrap-
  scheme mandate, and the fixed length was itself chosen from the panel's own empirical
  median run length (not tuned against resulting CIs). Worth naming as the natural v1.1
  refinement if wide CIs need tightening without adding false precision.
- **The regime label is real-time computable, but economically stale.** Because the data
  spine is point-in-time, month T's label uses only values published before T begins — the
  signal itself is knowable ex ante (that is exactly what the ALFRED plumbing buys). The
  honest caveats are different ones: the newest print available at the start of T describes
  reference month T-2, so each label reflects the growth/inflation state of roughly two
  months earlier; and nothing here adds portfolio construction, transaction costs, or
  turnover. Section 3's numbers describe realized association with an ex-ante-knowable
  label — not a strategy backtest.
- **U.S. only, single classifier design.** No claim about other markets or about whether a
  materially different classifier (more macro variables, a rolling window with decay,
  hysteresis to reduce quadrant-flipping) would tell a different story — v2 territory.

## 7. The v2 path (fall, post-move)

Built directly on this regime layer, not a rewrite:
- **Conditional factor allocation** — tilt factor weights by current regime; backtest
  honestly (transaction costs, and point-in-time regime detection — the regime is only
  known with the same publication lag noted above, so any backtest must lag the regime
  signal by however long it actually takes to know it).
- **Macro factor attribution** — decompose a portfolio's returns into inflation-beta /
  real-rate-beta / growth-beta; "what is my book's inflation sensitivity" is a real
  buy-side question this regime layer is already halfway built to answer.
- Candidates surfaced by this writeup specifically: a proxy-swap robustness check
  (PAYEMS/core PCE) once FRED access is available; an episode/cluster bootstrap as a
  tighter-CI alternative to the fixed-block scheme; richer macro conditioning (the growth
  x inflation 2-variable set was deliberately scoped down from a larger candidate list in
  Week 2 — see project notes — with the richer set deferred here).
