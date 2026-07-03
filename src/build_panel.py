"""
Week 1 DoD: align point-in-time macro data with factor returns into one monthly panel.

Alignment logic (why this isn't just a naive join):
- macro_pit[T] = the macro state KNOWN as of the start of month T (see fetch_data.py),
  including within-vintage YoY columns computed inside each date's vintage.
- french_factors[T] = the factor return REALIZED DURING month T.
Pairing macro_pit[T] with french_factors[T] on the same index date is exactly the
causal ordering you want: only information available BEFORE month T's returns begin
is used to describe month T. No lookahead is introduced by this join itself — that
work already happened in fetch_data.py.

Run:
    python src/build_panel.py
Requires data/processed/macro_pit.parquet and data/raw/french_factors.parquet
(produced by fetch_data.py).
"""

from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path("data/processed")
DATA_RAW = Path("data/raw")

MACRO_COLS = ["INDPRO", "INDPRO_yoy", "CPIAUCSL", "CPIAUCSL_yoy"]


def main():
    macro = pd.read_parquet(DATA_PROCESSED / "macro_pit.parquet")
    factors = pd.read_parquet(DATA_RAW / "french_factors.parquet")

    print(f"macro_pit:       {macro.shape[0]} rows, {macro.index.min().date()} -> {macro.index.max().date()}")
    print(f"french_factors:  {factors.shape[0]} rows, {factors.index.min().date()} -> {factors.index.max().date()}")

    panel = macro.join(factors, how="inner")
    print(f"\nafter inner join on date index: {panel.shape[0]} rows, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")

    # CPIAUCSL's point-in-time coverage doesn't start until ALFRED's first CPI
    # vintage (~1972-08) — drop rows missing any macro column rather than silently
    # carrying NaNs into Week 2's classifier.
    before = len(panel)
    panel = panel.dropna(subset=MACRO_COLS)
    dropped = before - len(panel)
    print(f"dropped {dropped} rows with missing macro data (pre-1972 CPI vintage coverage gap)")
    print(f"\nfinal usable panel: {panel.shape[0]} rows, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"columns: {list(panel.columns)}")

    assert panel.isna().sum().sum() == 0, "unexpected NaNs remain in the final panel"

    out_path = DATA_PROCESSED / "panel.parquet"
    panel.to_parquet(out_path)
    print(f"\nsaved -> {out_path}")

    print("\nhead:")
    print(panel.head(3))
    print("\ntail:")
    print(panel.tail(3))


if __name__ == "__main__":
    main()
