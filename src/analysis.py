"""
Week 3: conditional factor performance across macro regimes + honest error bars.

This is the actual finding of the project, not plumbing. Intersect the regime labels
(src/regimes.py) with Ken French factor returns and ask: does the factor
cross-section reorganize across growth x inflation states, and how much of that is
signal vs noise given a tiny effective sample of macro cycles?

Two things a careless implementation gets wrong here (both are landmines from the
build guide, made explicit so they can't sneak back in during Week 4 edits):

1. RF handling. French factors ship in PERCENT, divided by 100 on load. Of the six
   return series analyzed (Mkt-RF, SMB, HML, RMW, CMA, Mom), NONE need RF subtracted
   again before computing Sharpe: Mkt-RF is already market-minus-risk-free, and
   SMB/HML/RMW/CMA/Mom are long-short, self-financing zero-cost portfolios (no
   capital tied up, so there's no "cash alternative" to net out). RF itself is only
   the risk-free level and is deliberately excluded from FACTOR_COLS -- it is not a
   factor to condition on.

2. Drawdown on a regime-conditional series is not a real backtest. "Max drawdown of
   HML during Stagflation months" means: take only the calendar months labeled
   Stagflation, splice their HML returns together IN CHRONOLOGICAL ORDER (skipping
   the interleaved non-Stagflation months), and compound that spliced series. This
   creates artificial seams at regime-switch boundaries -- there's no real
   portfolio that experiences this exact path -- but it's the standard
   simplification for conditional performance tables and is flagged here rather
   than presented as a real equity curve.

Block bootstrap, not iid: monthly factor returns are autocorrelated and regimes
persist in runs (median run length in this panel is ~5-6 months, see
docs/regime_classifier_rationale.md-adjacent analysis in this module's tests) --
resampling single months independently would destroy exactly the run structure
that makes "which regime you're in" a meaningful conditioning variable. Block
length is set to 6 months: the guide's own rough suggestion was ~12, but the
empirical median regime run-length in this panel is 5-6 months (mean 7.3,
17-23 runs per regime) -- 12 would span multiple regime switches on average, so 6
is used instead. This is picked once from the data's own persistence structure,
not tuned by trial and error against the resulting CIs.

Effective N is genuinely small (Stagflation = 108 months = maybe 3-4 independent
stretches). Expect wide CIs. A CI straddling zero is the honest finding, not a
failure to find one -- report it, don't chase a cleaner result that isn't there.

Run:
    python src/analysis.py
Requires data/processed/panel_regimes.parquet (produced by src/regimes.py).
Writes:
    data/processed/conditional_stats.parquet      -- point estimates
    data/processed/conditional_bootstrap.parquet  -- Sharpe block-bootstrap CIs
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PROCESSED = Path("data/processed")

# The six French return series worth conditioning on. RF (risk-free level) is
# deliberately excluded -- see module docstring, landmine #1.
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

MONTHS_PER_YEAR = 12

BLOCK_LENGTH = 6  # months; empirical median regime run-length, see module docstring
N_BOOTSTRAP = 5000
CI_PERCENTILES = (2.5, 97.5)  # 95% percentile CI
SEED = 20260704  # fixed for reproducibility -- not re-rolled to get a nicer CI


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_labeled_panel() -> pd.DataFrame:
    """Labeled months only (dropna on regime, same sample Week 2 reported), factor
    columns rescaled from percent to decimal."""
    df = pd.read_parquet(DATA_PROCESSED / "panel_regimes.parquet")
    labeled = df.dropna(subset=["regime"]).copy()
    for col in FACTOR_COLS:
        labeled[col] = labeled[col] / 100.0
    return labeled


# ---------------------------------------------------------------------------
# Point-estimate stats
# ---------------------------------------------------------------------------

def _max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the compounded equity curve from a (possibly regime-spliced,
    see landmine #2) monthly return series. Returns a value in (-1, 0]; NaN if the
    series is empty."""
    if returns.empty:
        return np.nan
    curve = (1.0 + returns).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return drawdown.min()


def sharpe_of(values: np.ndarray) -> float:
    """Annualized Sharpe from monthly decimal returns. No RF subtraction -- see
    module docstring, landmine #1: all six FACTOR_COLS are already zero-cost."""
    n = len(values)
    if n < 2:
        return np.nan
    vol_m = values.std(ddof=1)
    if vol_m == 0 or np.isnan(vol_m):
        return np.nan
    mean_m = values.mean()
    return (mean_m * MONTHS_PER_YEAR) / (vol_m * np.sqrt(MONTHS_PER_YEAR))


def factor_stats(returns: pd.Series) -> dict:
    """Point estimates for one regime x factor return series (monthly, decimal)."""
    n = len(returns)
    if n == 0:
        return {
            "n_months": 0, "ann_mean": np.nan, "ann_vol": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan,
        }
    values = returns.to_numpy()
    vol_m = returns.std(ddof=1) if n > 1 else np.nan
    return {
        "n_months": n,
        "ann_mean": returns.mean() * MONTHS_PER_YEAR,
        "ann_vol": vol_m * np.sqrt(MONTHS_PER_YEAR) if pd.notna(vol_m) else np.nan,
        "sharpe": sharpe_of(values),
        "max_drawdown": _max_drawdown(returns),
        "hit_rate": (returns > 0).mean(),
    }


def conditional_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per regime x factor: N months, annualized mean/vol, Sharpe, max drawdown,
    hit rate. Adds an 'All (unconditional)' pseudo-regime row per factor so the
    conditional numbers have an unconditional baseline to compare against."""
    rows = []
    for regime in sorted(panel["regime"].unique()):
        sub = panel.loc[panel["regime"] == regime].sort_index()
        for factor in FACTOR_COLS:
            rows.append({"regime": regime, "factor": factor, **factor_stats(sub[factor])})
    full = panel.sort_index()
    for factor in FACTOR_COLS:
        rows.append({"regime": "All (unconditional)", "factor": factor, **factor_stats(full[factor])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block bootstrap CI on Sharpe
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    rng: np.random.Generator,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = N_BOOTSTRAP,
):
    """
    Moving block bootstrap: overlapping blocks of `block_length` months, resampled
    with replacement and concatenated back to length n, replicated `n_boot` times.
    Sharpe is recomputed on each replicate; the CI is the empirical
    [2.5, 97.5] percentile of the replicate Sharpes.

    Returns (point_sharpe, ci_lo, ci_hi, n_months). CI is NaN if n < 2.
    """
    values = returns.to_numpy()
    n = len(values)
    if n < 2:
        return sharpe_of(values), np.nan, np.nan, n

    block_len = min(block_length, n)
    n_blocks_needed = int(np.ceil(n / block_len))
    max_start = n - block_len  # last valid block start index (inclusive)

    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        rep = np.concatenate([values[s:s + block_len] for s in starts])[:n]
        boot_sharpes[b] = sharpe_of(rep)

    ci_lo, ci_hi = np.nanpercentile(boot_sharpes, CI_PERCENTILES)
    return sharpe_of(values), ci_lo, ci_hi, n


def bootstrap_table(panel: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Same regime x factor grid as conditional_table, but Sharpe + block-bootstrap
    CI instead of the full stat set. One shared RNG (seeded) across the whole
    table so results are reproducible run-to-run."""
    rng = np.random.default_rng(seed)
    rows = []
    for regime in sorted(panel["regime"].unique()):
        sub = panel.loc[panel["regime"] == regime].sort_index()
        for factor in FACTOR_COLS:
            sharpe, lo, hi, n = block_bootstrap_sharpe_ci(sub[factor], rng)
            rows.append({
                "regime": regime,
                "factor": factor,
                "n_months": n,
                "sharpe": sharpe,
                "ci_lo_95": lo,
                "ci_hi_95": hi,
                "ci_width": (hi - lo) if pd.notna(hi) and pd.notna(lo) else np.nan,
                "straddles_zero": (lo < 0 < hi) if pd.notna(lo) and pd.notna(hi) else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    panel = load_labeled_panel()
    print(f"labeled panel: {panel.shape[0]} months, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"\nregime counts:\n{panel['regime'].value_counts()}")

    stats = conditional_table(panel)
    stats_path = DATA_PROCESSED / "conditional_stats.parquet"
    stats.to_parquet(stats_path)
    print(f"\nconditional performance table -> {stats_path}")
    print(stats.round(4).to_string(index=False))

    print(f"\nblock bootstrap (block={BLOCK_LENGTH}mo, {N_BOOTSTRAP} reps, seed={SEED})...")
    boot = bootstrap_table(panel)
    boot_path = DATA_PROCESSED / "conditional_bootstrap.parquet"
    boot.to_parquet(boot_path)
    print(f"bootstrap Sharpe CIs -> {boot_path}")
    print(boot.round(3).to_string(index=False))

    conditional_rows = boot[boot["regime"] != "All (unconditional)"]
    n_straddle = conditional_rows["straddles_zero"].sum()
    n_total = conditional_rows["straddles_zero"].notna().sum()
    print(
        f"\n{n_straddle}/{n_total} conditional regime x factor Sharpe CIs straddle "
        f"zero -- reported as-is, not chased away, per project thesis."
    )


if __name__ == "__main__":
    main()
