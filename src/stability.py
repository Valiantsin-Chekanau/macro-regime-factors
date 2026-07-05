"""
Week 4: non-stationarity check -- does each regime x factor relationship hold or
flip sign pre- vs post-2000?

Why this matters (see macro_regime_conditioning_guide.md sec 0 and Week 4): the
inflation-equity relationship is documented to have flipped sign across eras (e.g.
Ilmanen and others on the pre-1998 vs post-1998 stock-bond/inflation correlation
regime change). If a Week 3 "standout" (a CI that clears zero on the full sample)
is actually just one era's relationship dragging the full-sample average, that's a
more honest and more interesting finding than the pooled number -- report the split,
don't paper over it.

Split: calendar-year cutoff at 2000 (not tuned -- it's the guide's own suggested
cutoff and lands close to the sample midpoint). This is NOT a rolling-window
analysis -- guide allows either "pre/post split OR rolling windows"; a single split
is picked here to stay within the Week 4 timebox (one robustness idea, not a grid
of window lengths).

Each sub-period gets its own conditional_table + bootstrap_table, computed with the
exact same functions as Week 3 (src/analysis.py) -- no new methodology, just a
narrower sample. Small-N regimes in a sub-period will get very wide (or undefined,
if a regime has <2 months in a sub-period) CIs; this is reported, not hidden, same
as the Week 3 thesis.

Run (from the repo root -- this module imports src.analysis, so it must run as a
module, not as a script path):
    python -m src.stability
Requires data/processed/panel_regimes.parquet (produced by src/regimes.py).
Writes:
    data/processed/stability_pre2000_conditional.parquet
    data/processed/stability_pre2000_bootstrap.parquet
    data/processed/stability_post2000_conditional.parquet
    data/processed/stability_post2000_bootstrap.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    FACTOR_COLS,
    bootstrap_table,
    conditional_table,
    load_labeled_panel,
)

DATA_PROCESSED = Path("data/processed")
SPLIT_YEAR = 2000  # guide's suggested cutoff; not tuned


def split_panel(panel: pd.DataFrame, split_year: int = SPLIT_YEAR):
    pre = panel.loc[panel.index.year < split_year].sort_index()
    post = panel.loc[panel.index.year >= split_year].sort_index()
    return pre, post


def sign_flip_table(boot_pre: pd.DataFrame, boot_post: pd.DataFrame) -> pd.DataFrame:
    """For each regime x factor cell present in both sub-periods: does the point
    Sharpe change sign, and does either period's CI clear zero (i.e. is the flip
    backed by more than sampling noise in at least one half)?"""
    pre = boot_pre[boot_pre["regime"] != "All (unconditional)"].set_index(["regime", "factor"])
    post = boot_post[boot_post["regime"] != "All (unconditional)"].set_index(["regime", "factor"])
    common = pre.index.intersection(post.index)

    rows = []
    for key in common:
        regime, factor = key
        p, q = pre.loc[key], post.loc[key]
        if pd.isna(p["sharpe"]) or pd.isna(q["sharpe"]):
            continue
        flips = np.sign(p["sharpe"]) != np.sign(q["sharpe"]) and p["sharpe"] != 0 and q["sharpe"] != 0
        # `== False`, NOT `is False`: a bool-dtype column yields np.bool_ on row
        # access, and `np.False_ is False` evaluates False -- the `is` version
        # silently zeroed this column for every cell (caught in the Jul 5 2026
        # audit; regression-tested in tests/test_week4.py).
        either_clears = (p["straddles_zero"] == False) or (q["straddles_zero"] == False)  # noqa: E712
        rows.append({
            "regime": regime,
            "factor": factor,
            "sharpe_pre": p["sharpe"],
            "sharpe_post": q["sharpe"],
            "n_pre": p["n_months"],
            "n_post": q["n_months"],
            "sign_flip": flips,
            "either_period_clears_zero": either_clears,
        })
    return pd.DataFrame(rows).sort_values(["sign_flip", "either_period_clears_zero"], ascending=False)


def main():
    panel = load_labeled_panel()
    pre, post = split_panel(panel)
    print(
        f"full labeled sample: {panel.shape[0]} months "
        f"({panel.index.min().date()} -> {panel.index.max().date()})"
    )
    print(
        f"pre-{SPLIT_YEAR}:  {pre.shape[0]} months "
        f"({pre.index.min().date()} -> {pre.index.max().date()})"
    )
    print(
        f"post-{SPLIT_YEAR}: {post.shape[0]} months "
        f"({post.index.min().date()} -> {post.index.max().date()})"
    )

    print(f"\nregime counts, pre-{SPLIT_YEAR}:\n{pre['regime'].value_counts()}")
    print(f"\nregime counts, post-{SPLIT_YEAR}:\n{post['regime'].value_counts()}")

    for label, sub in [("pre2000", pre), ("post2000", post)]:
        cond = conditional_table(sub)
        cond_path = DATA_PROCESSED / f"stability_{label}_conditional.parquet"
        cond.to_parquet(cond_path)

        boot = bootstrap_table(sub)
        boot_path = DATA_PROCESSED / f"stability_{label}_bootstrap.parquet"
        boot.to_parquet(boot_path)
        print(f"\n[{label}] conditional -> {cond_path}, bootstrap -> {boot_path}")

    boot_pre = pd.read_parquet(DATA_PROCESSED / "stability_pre2000_bootstrap.parquet")
    boot_post = pd.read_parquet(DATA_PROCESSED / "stability_post2000_bootstrap.parquet")
    flips = sign_flip_table(boot_pre, boot_post)

    n_flip = flips["sign_flip"].sum()
    n_common = len(flips)
    n_flip_and_clears = ((flips["sign_flip"]) & (flips["either_period_clears_zero"])).sum()
    print(
        f"\n{n_flip}/{n_common} regime x factor cells with a point-Sharpe sign flip "
        f"pre- vs post-{SPLIT_YEAR}; {n_flip_and_clears} of those flips have at "
        f"least one sub-period CI clearing zero (i.e. not just noise on both sides)."
    )
    print("\ntop flips (sign flip first, zero-clearing flips first within that):")
    print(flips.head(10).round(3).to_string(index=False))

    flips_path = DATA_PROCESSED / "stability_sign_flips.parquet"
    flips.to_parquet(flips_path)
    print(f"\nsaved -> {flips_path}")


if __name__ == "__main__":
    main()
