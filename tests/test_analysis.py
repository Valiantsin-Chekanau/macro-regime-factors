"""
Week 3 test coverage: the two RF/drawdown landmines (see src/analysis.py docstring)
plus the bootstrap's two make-or-break properties -- reproducibility (same seed,
same answer) and honesty (small-N regimes get wider CIs, not narrower).

Run:
    python tests/test_analysis.py     # or: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo root on sys.path so `import src.analysis` works whether this file is run
# directly (python tests/test_analysis.py) or via pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    FACTOR_COLS,
    block_bootstrap_sharpe_ci,
    conditional_table,
    factor_stats,
    load_labeled_panel,
    sharpe_of,
)

DATA_PROCESSED = Path("data/processed")


def test_rf_excluded_from_factor_cols():
    """RF is the risk-free level, not a factor to condition on -- see landmine #1.
    Including it would let a future edit compute a nonsensical 'Sharpe of RF'."""
    assert "RF" not in FACTOR_COLS
    assert set(FACTOR_COLS) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"}


def test_factor_scale_is_decimal_not_percent():
    """French factors ship in percent; load_labeled_panel must rescale. A monthly
    decimal factor return of +/-50% would already be a historic outlier, so this
    also catches an accidental double-rescale (dividing by 100 twice)."""
    panel = load_labeled_panel()
    for col in FACTOR_COLS:
        assert panel[col].abs().max() < 0.5, f"{col} looks like it's still in percent"


def test_sharpe_matches_hand_calc():
    """No RF subtraction, no ddof surprises: Sharpe = (mean*12) / (std*sqrt(12))
    with sample (ddof=1) std, computed by hand on a small fixed series."""
    values = np.array([0.01, -0.02, 0.03, 0.00, 0.015, -0.01])
    expected = (values.mean() * 12) / (values.std(ddof=1) * np.sqrt(12))
    assert abs(sharpe_of(values) - expected) < 1e-12


def test_sharpe_undefined_below_two_points():
    assert np.isnan(sharpe_of(np.array([0.01])))
    assert np.isnan(sharpe_of(np.array([])))


def test_drawdown_bounds():
    """Max drawdown of a compounded return series is always in [-1, 0]."""
    panel = load_labeled_panel()
    stats = conditional_table(panel)
    dd = stats["max_drawdown"].dropna()
    assert (dd <= 0).all() and (dd >= -1).all()


def test_conditional_table_n_matches_regime_counts():
    """N months per regime x factor must equal the regime's raw month count --
    silently dropping rows here would understate the effective-N problem, not fix it."""
    panel = load_labeled_panel()
    counts = panel["regime"].value_counts()
    stats = conditional_table(panel)
    for regime, n in counts.items():
        rows = stats.loc[stats["regime"] == regime, "n_months"]
        assert (rows == n).all(), f"{regime}: expected N={n}, got {rows.unique()}"


def test_bootstrap_reproducible_with_seed():
    """Same seed -> identical CI bounds. If this ever fails, someone introduced
    unseeded randomness (e.g. a fresh default_rng() call inside the loop)."""
    panel = load_labeled_panel()
    sub = panel.loc[panel["regime"] == "Stagflation", "HML"].sort_index()

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    r1 = block_bootstrap_sharpe_ci(sub, rng1, n_boot=500)
    r2 = block_bootstrap_sharpe_ci(sub, rng2, n_boot=500)
    assert r1 == r2


def test_bootstrap_ci_widens_for_smaller_effective_n():
    """Stagflation (108 months, the smallest regime) should get a wider Sharpe CI
    than Deflationary slowdown (174 months, the largest) for the same factor --
    the honest-uncertainty point of the whole exercise. Checked on Mkt-RF, the
    least noisy series, so the comparison isn't swamped by factor-specific vol."""
    panel = load_labeled_panel()
    rng = np.random.default_rng(7)

    small = panel.loc[panel["regime"] == "Stagflation", "Mkt-RF"].sort_index()
    large = panel.loc[panel["regime"] == "Deflationary slowdown", "Mkt-RF"].sort_index()

    _, lo_small, hi_small, _ = block_bootstrap_sharpe_ci(small, rng, n_boot=2000)
    _, lo_large, hi_large, _ = block_bootstrap_sharpe_ci(large, rng, n_boot=2000)

    assert (hi_small - lo_small) > (hi_large - lo_large)


def test_labeled_panel_matches_regimes_output():
    """No extra filtering snuck into analysis.py beyond dropping unlabeled months --
    the labeled count here must match what regimes.py itself reports."""
    raw = pd.read_parquet(DATA_PROCESSED / "panel_regimes.parquet")
    expected_n = raw["regime"].notna().sum()
    assert len(load_labeled_panel()) == expected_n


if __name__ == "__main__":
    for fn in [
        test_rf_excluded_from_factor_cols,
        test_factor_scale_is_decimal_not_percent,
        test_sharpe_matches_hand_calc,
        test_sharpe_undefined_below_two_points,
        test_drawdown_bounds,
        test_conditional_table_n_matches_regime_counts,
        test_bootstrap_reproducible_with_seed,
        test_bootstrap_ci_widens_for_smaller_effective_n,
        test_labeled_panel_matches_regimes_output,
    ]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\nall Week 3 tests passed")
