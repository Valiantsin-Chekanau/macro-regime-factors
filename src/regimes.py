"""
Week 2: growth x inflation regime classifier.

Design: interpretable 2x2, not a latent-state model (HMM/regime-switching). See the
"Why not an HMM" note in this module for the rationale — short version: with ~50 years
of monthly data and maybe 6-8 full macro cycles, a latent-state model has enough
freedom to fit noise and call it a "regime," and the fit is not falsifiable by eye.
A deterministic rule you can compute on a napkin is falsifiable and auditable.

Definitions (locked, not tuned):
- Growth state: YoY % change in INDPRO, compared to its own trailing 60-month mean.
  Up if YoY > trailing mean, else Down.
- Inflation state: YoY % change in CPIAUCSL, compared to its own trailing 60-month mean.
  Rising if YoY > trailing mean, else Falling.
- Both YoY and CPIAUCSL/INDPRO inputs come from data/processed/panel.parquet, which is
  already point-in-time (Week 1) -- so nothing computed here reintroduces lookahead.

Why 60-month (5yr) trailing mean, not an expanding (full-history) mean:
An expanding mean anchors "normal" growth/inflation to the entire sample average,
including the 1970s-80s stagflation era. That drags the baseline up for inflation and
makes recent decades look artificially "cool" relative to a stale 1970s-heavy average.
A rolling 5yr window instead asks "is growth/inflation running hot or cold relative to
what's been normal *recently*" -- closer to how a regime is actually experienced, and
the guide's brief explicitly allows either; this is the pick, not a tuned choice.

Cost: the first 60 months of any YoY series lack a full trailing window and are
dropped (reported below, not silently NaN-filled).

Run:
    python src/regimes.py
Requires data/processed/panel.parquet (produced by src/build_panel.py).
"""

from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path("data/processed")

TRAILING_WINDOW_MONTHS = 60  # 5yr, see module docstring for why not expanding-mean


def add_states(panel: pd.DataFrame) -> pd.DataFrame:
    """Add YoY growth/inflation and Up/Down, Rising/Falling state columns."""
    out = panel.copy()

    out["growth_yoy"] = out["INDPRO"].pct_change(12)
    out["inflation_yoy"] = out["CPIAUCSL"].pct_change(12)

    out["growth_trend"] = out["growth_yoy"].rolling(
        TRAILING_WINDOW_MONTHS, min_periods=TRAILING_WINDOW_MONTHS
    ).mean()
    out["inflation_trend"] = out["inflation_yoy"].rolling(
        TRAILING_WINDOW_MONTHS, min_periods=TRAILING_WINDOW_MONTHS
    ).mean()

    out["growth_state"] = None
    out["inflation_state"] = None
    has_trend = out["growth_trend"].notna() & out["inflation_trend"].notna()

    out.loc[has_trend, "growth_state"] = out.loc[has_trend].apply(
        lambda r: "Up" if r["growth_yoy"] > r["growth_trend"] else "Down", axis=1
    )
    out.loc[has_trend, "inflation_state"] = out.loc[has_trend].apply(
        lambda r: "Rising" if r["inflation_yoy"] > r["inflation_trend"] else "Falling",
        axis=1,
    )

    return out


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
    print(f"panel in:  {panel.shape[0]} rows, {panel.index.min().date()} -> {panel.index.max().date()}")

    panel = add_states(panel)

    n_unlabeled = panel["growth_state"].isna().sum()
    print(
        f"\nfirst {n_unlabeled} months lack a full {TRAILING_WINDOW_MONTHS}-month "
        f"trailing window (YoY eats 12mo, trend eats {TRAILING_WINDOW_MONTHS}mo more) "
        f"-> state is unlabeled there, not silently zero-filled."
    )

    labeled = panel.dropna(subset=["growth_state", "inflation_state"])
    print(
        f"\nlabeled sample: {labeled.shape[0]} rows, "
        f"{labeled.index.min().date()} -> {labeled.index.max().date()}"
    )
    print("\ngrowth_state counts:")
    print(labeled["growth_state"].value_counts())
    print("\ninflation_state counts:")
    print(labeled["inflation_state"].value_counts())

    out_path = DATA_PROCESSED / "panel.parquet"
    panel.to_parquet(out_path)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
