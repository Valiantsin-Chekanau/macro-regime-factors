"""
Sanity harness — the checks that would have caught the cross-vintage YoY bug.

The Week 1 sanity check validated the vintage PLUMBING (as-of != latest-revised) but
never the DERIVED series; CPI's 1988 re-reference then sailed through as -65%
"inflation." Lesson encoded here: bound-check every transformed input before a stage
locks.

Run after the pipeline (fetch_data -> build_panel -> regimes):
    python tests/test_sanity.py     # or: pytest tests/
"""

from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path("data/processed")


def _macro():
    return pd.read_parquet(DATA_PROCESSED / "macro_pit.parquet")


def _regimes():
    return pd.read_parquet(DATA_PROCESSED / "panel_regimes.parquet")


def test_no_calendar_gaps():
    """Positional ops (rolling windows) assume every month is present."""
    for df in (_macro(), _regimes()):
        idx = df.index
        gaps = pd.date_range(idx.min(), idx.max(), freq="MS").difference(idx)
        assert len(gaps) == 0, f"missing months: {list(gaps[:5])}"


def test_yoy_within_economic_bounds():
    """Postwar US: |CPI YoY| < 25% (max ~14.8%, 1980), |INDPRO YoY| < 40%
    (extremes ~ -16% Apr-2020, ~ +17% base-effect 2021). The cross-vintage bug
    produced -65% / -25% — either bound catches a recurrence instantly."""
    m = _macro()
    assert m["CPIAUCSL_yoy"].dropna().abs().max() < 0.25
    assert m["INDPRO_yoy"].dropna().abs().max() < 0.40


def test_no_rebase_fingerprints_in_yoy():
    """Regression on the specific bug: at known rebasing dates the PIT *level*
    jumps 8-66%, but same-vintage YoY must stay smooth month-over-month."""
    m = _macro()
    checks = {
        "CPIAUCSL_yoy": ["1988-03-01"],                     # 1967=100 -> 1982-84=100
        "INDPRO_yoy": ["1985-08-01", "2003-01-01", "2010-07-01"],  # base changes
    }
    for col, dates in checks.items():
        for d in dates:
            d = pd.Timestamp(d)
            prev = d - pd.DateOffset(months=1)
            jump = abs(m.loc[d, col] - m.loc[prev, col])
            assert jump < 0.05, f"{col} jumps {jump:.1%} at {d.date()} — vintage mixing?"


def test_regime_mapping_exhaustive_and_consistent():
    """Every labeled month maps to exactly the quadrant its states imply."""
    r = _regimes().dropna(subset=["regime"])
    names = {
        ("Up", "Falling"): "Goldilocks",
        ("Up", "Rising"): "Overheating",
        ("Down", "Rising"): "Stagflation",
        ("Down", "Falling"): "Deflationary slowdown",
    }
    assert set(r["regime"].unique()) <= set(names.values())
    recomputed = r.apply(lambda x: names[(x["growth_state"], x["inflation_state"])], axis=1)
    assert (recomputed == r["regime"]).all()


def test_labeled_block_has_no_nans():
    """No silent NaNs inside the sample Week 3 will condition on."""
    r = _regimes().dropna(subset=["regime"])
    core = r[["INDPRO", "CPIAUCSL", "growth_yoy", "inflation_yoy",
              "growth_trend", "inflation_trend"]]
    assert core.notna().all().all()


if __name__ == "__main__":
    for fn in [
        test_no_calendar_gaps,
        test_yoy_within_economic_bounds,
        test_no_rebase_fingerprints_in_yoy,
        test_regime_mapping_exhaustive_and_consistent,
        test_labeled_block_has_no_nans,
    ]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\nall sanity checks passed")
