"""
Week 4 test coverage: the fixed unconditional-CI dead code, the stability split's
partition property, and the robustness alt-window classifier not silently matching
(or silently diverging from) the base window.

Run:
    python tests/test_week4.py     # or: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    FACTOR_COLS,
    bootstrap_table,
    load_labeled_panel,
)
from src.regimes import TRAILING_WINDOW_MONTHS, add_regime, add_states  # noqa: E402
from src.robustness import ALT_WINDOW_MONTHS, build_regimes, label_agreement  # noqa: E402
from src.stability import SPLIT_YEAR, sign_flip_table, split_panel  # noqa: E402

DATA_PROCESSED = Path("data/processed")


def test_bootstrap_table_includes_unconditional_row():
    """Regression: main()'s straddle-rate filter (`regime != 'All (unconditional)'`)
    was previously a no-op because bootstrap_table never emitted that row -- it only
    existed in conditional_table. Fixed by adding the same pseudo-regime row here."""
    panel = load_labeled_panel()
    boot = bootstrap_table(panel, seed=1)
    uncond = boot.loc[boot["regime"] == "All (unconditional)"]
    assert set(uncond["factor"]) == set(FACTOR_COLS)
    assert len(uncond) == len(FACTOR_COLS)


def test_add_states_default_window_unchanged():
    """The window= parameter added for the robustness check must not change
    default behavior -- add_states(panel) must equal add_states(panel, window=60)."""
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
    default = add_states(panel)
    explicit = add_states(panel, window=TRAILING_WINDOW_MONTHS)
    pd.testing.assert_series_equal(default["growth_trend"], explicit["growth_trend"])
    pd.testing.assert_series_equal(default["inflation_trend"], explicit["inflation_trend"])


def test_stability_split_is_a_true_partition():
    """Every labeled month goes to exactly one of pre/post, cutoff at SPLIT_YEAR,
    and nothing is dropped or duplicated across the split."""
    panel = load_labeled_panel()
    pre, post = split_panel(panel)
    assert len(pre) + len(post) == len(panel)
    assert pre.index.year.max() < SPLIT_YEAR
    assert post.index.year.min() >= SPLIT_YEAR
    assert set(pre.index).isdisjoint(set(post.index))


def test_robustness_alt_window_neither_identical_nor_scrambled():
    """The 36mo alt-window classifier must actually classify differently from the
    60mo base (otherwise the robustness check tests nothing) but agree on a clear
    majority of months (otherwise the classifier is too window-sensitive to trust
    at either setting -- that would be a real finding, not a bug, but worth a
    sanity bound given what Week 4 actually measured: ~78% agreement)."""
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
    base = build_regimes(panel, TRAILING_WINDOW_MONTHS)
    alt = build_regimes(panel, ALT_WINDOW_MONTHS)
    agreement = label_agreement(base, alt)
    rate = agreement["agree"].mean()
    assert 0.5 < rate < 1.0, f"agreement rate {rate:.1%} outside sane [50%, 100%) band"


def test_sign_flip_table_detects_clearing_half():
    """Regression (Jul 5 2026 audit): sign_flip_table compared straddles_zero with
    `is False`; on a bool-dtype column row access yields np.bool_, and
    `np.False_ is False` evaluates False -- so either_period_clears_zero silently
    came out False for EVERY cell, and the writeup wrongly claimed no sign flip
    had a zero-clearing half. Build one cell whose pre-half CI clears zero and
    whose Sharpe flips sign: the flip must be flagged AND the clearing half
    detected; a second cell flips with no clearing half and must stay False."""
    cols = ["regime", "factor", "sharpe", "ci_lo_95", "ci_hi_95", "n_months",
            "straddles_zero"]
    pre = pd.DataFrame([
        ["Goldilocks", "HML", 1.02, 0.40, 1.70, 92, False],
        ["Goldilocks", "SMB", -0.18, -0.90, 0.50, 92, True],
    ], columns=cols)
    post = pd.DataFrame([
        ["Goldilocks", "HML", -0.11, -0.70, 0.50, 64, True],
        ["Goldilocks", "SMB", 0.38, -0.20, 1.00, 64, True],
    ], columns=cols)
    assert pre["straddles_zero"].dtype == bool  # the dtype that triggered the bug

    flips = sign_flip_table(pre, post).set_index(["regime", "factor"])
    hml, smb = flips.loc[("Goldilocks", "HML")], flips.loc[("Goldilocks", "SMB")]
    assert bool(hml["sign_flip"]) and bool(hml["either_period_clears_zero"])
    assert bool(smb["sign_flip"]) and not bool(smb["either_period_clears_zero"])


def test_robustness_uses_sanctioned_window_range():
    """Both windows tested must fall inside the guide's own stated 3-5yr range,
    not an arbitrary third value -- this is what makes it a robustness check on the
    locked design choice rather than a new tuned parameter."""
    assert ALT_WINDOW_MONTHS == 36
    assert TRAILING_WINDOW_MONTHS == 60


if __name__ == "__main__":
    for fn in [
        test_bootstrap_table_includes_unconditional_row,
        test_add_states_default_window_unchanged,
        test_stability_split_is_a_true_partition,
        test_sign_flip_table_detects_clearing_half,
        test_robustness_alt_window_neither_identical_nor_scrambled,
        test_robustness_uses_sanctioned_window_range,
    ]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\nall Week 4 tests passed")
