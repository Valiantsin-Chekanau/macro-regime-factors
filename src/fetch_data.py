"""
Week 1 data spine: point-in-time macro series + Ken French factor returns.

Core idea (the credibility anchor of this project): a regime label for month M is
only valid if it uses macro values that were ACTUALLY PUBLISHED and not-yet-revised
as of the decision date. Pulling "latest revised" INDPRO/CPI and pretending you knew
it in real time is lookahead bias, full stop.

Two paths, in priority order:
  1. ALFRED vintages (preferred) — for each monthly decision date T, ask FRED
     "what was the most recently published value as of T?" via get_series_as_of_date.
  2. Lag-floor fallback (safety net) — if ALFRED vintage history doesn't reach far
     enough back for a series, shift the latest-revised series by its known
     publication delay instead. Never let vintage-fetching block progress.

Run:
    python src/fetch_data.py
Requires FRED_API_KEY in .env (see .env.example).
"""

import os
import time
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import pandas_datareader.data as web

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

# Growth / inflation proxies. Guide's primary + robustness alternates.
MACRO_SERIES = {
    "INDPRO": "growth",       # industrial production, monthly — primary growth proxy
    "CPIAUCSL": "inflation",  # CPI-U, SA, monthly — primary inflation proxy
}

# Rough publication delays (guide's numbers) — fallback path ONLY.
# Confirm against actual ALFRED vintage dates before trusting these blindly.
PUBLICATION_LAG_DAYS = {
    "INDPRO": 17,     # ~2-3 wks after month-end reference period
    "CPIAUCSL": 17,   # ~2-3 wks after month-end reference period
}

FRED_HISTORY_START = "1960-01-01"  # French factors + FRED both go back this far

FF_5FACTOR = "F-F_Research_Data_5_Factors_2x3"
FF_MOMENTUM = "F-F_Momentum_Factor"


# ---------------------------------------------------------------------------
# Point-in-time macro pull
# ---------------------------------------------------------------------------

def monthly_decision_dates(start=FRED_HISTORY_START, end=None):
    """First-of-month decision dates: 'what did we know as of the start of month M?'"""
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(start=start, end=end, freq="MS")


def get_asof_last_known(fred: Fred, series_id: str, as_of_date: pd.Timestamp):
    """
    Most recent value of `series_id` that was PUBLISHED as of `as_of_date`,
    using ALFRED vintages. Returns (reference_period, value) or (None, None)
    if nothing had been published yet.
    """
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    vintage_series = fred.get_series_as_of_date(series_id, as_of_str)
    if vintage_series is None or vintage_series.empty:
        return None, None
    vintage_series = vintage_series.dropna()
    if vintage_series.empty:
        return None, None
    return vintage_series.index[-1], vintage_series.iloc[-1]


def fetch_pit_series_alfred(fred: Fred, series_id: str, decision_dates) -> pd.Series:
    """Point-in-time panel for one series across all decision dates, via ALFRED."""
    values = {}
    for d in decision_dates:
        try:
            ref_period, val = get_asof_last_known(fred, series_id, d)
        except Exception as e:  # noqa: BLE001 — ALFRED can be flaky per-date; log and skip
            warnings.warn(f"{series_id} as-of {d.date()} failed: {e}")
            ref_period, val = None, None
        if val is not None:
            values[d] = val
    if not values:
        raise RuntimeError(f"ALFRED vintages returned nothing for {series_id}")
    return pd.Series(values, name=series_id).sort_index()


def fetch_pit_series_lag_floor(fred: Fred, series_id: str, decision_dates) -> pd.Series:
    """
    Fallback: pull latest-revised series once, then lag it by its publication delay.
    Less rigorous than ALFRED (ignores revisions, just delays availability) but
    never blocks — used only if ALFRED vintages error out or don't cover the range.
    """
    latest = fred.get_series(series_id)
    lag_days = PUBLICATION_LAG_DAYS.get(series_id, 21)
    values = {}
    for d in decision_dates:
        available_as_of = latest[latest.index + pd.Timedelta(days=lag_days) <= d]
        if not available_as_of.empty:
            values[d] = available_as_of.iloc[-1]
    return pd.Series(values, name=series_id).sort_index()


def fetch_macro_panel(fred: Fred) -> pd.DataFrame:
    decision_dates = monthly_decision_dates()
    cols = {}
    for series_id in MACRO_SERIES:
        try:
            print(f"  [ALFRED] {series_id} ...")
            cols[series_id] = fetch_pit_series_alfred(fred, series_id, decision_dates)
        except Exception as e:  # noqa: BLE001
            print(f"  [ALFRED FAILED for {series_id}: {e}] falling back to lag-floor")
            cols[series_id] = fetch_pit_series_lag_floor(fred, series_id, decision_dates)
        time.sleep(0.2)  # be polite to the API
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Ken French factor returns
# ---------------------------------------------------------------------------

def fetch_french_factors() -> pd.DataFrame:
    """5-factor (Mkt-RF, SMB, HML, RMW, CMA, RF) + momentum (Mom), monthly, in %."""
    five = web.DataReader(FF_5FACTOR, "famafrench", start=FRED_HISTORY_START)[0]
    mom = web.DataReader(FF_MOMENTUM, "famafrench", start=FRED_HISTORY_START)[0]
    mom.columns = [c.strip() for c in mom.columns]
    five.columns = [c.strip() for c in five.columns]
    factors = five.join(mom, how="inner")
    factors.index = factors.index.to_timestamp()
    return factors


# ---------------------------------------------------------------------------
# Sanity check (Week 1 definition-of-done): prove vintages != latest-revised
# ---------------------------------------------------------------------------

def sanity_check_vintages(fred: Fred, series_id: str, sample_dates):
    print(f"\nSanity check for {series_id}: as-of value vs latest-revised value")
    print(f"{'as_of_date':<12}{'ref_period':<12}{'as_of_value':>14}{'latest_value':>14}{'differs?':>10}")
    latest = fred.get_series(series_id)
    for d in sample_dates:
        ref_period, as_of_val = get_asof_last_known(fred, series_id, d)
        if ref_period is None:
            continue
        latest_val = latest.get(ref_period)
        differs = "YES" if (latest_val is not None and abs(latest_val - as_of_val) > 1e-9) else "no"
        print(f"{d.date()!s:<12}{ref_period.date()!s:<12}{as_of_val:>14.3f}{latest_val:>14.3f}{differs:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not found — check your .env file")

    fred = Fred(api_key=api_key)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Fetching point-in-time macro panel (INDPRO, CPIAUCSL)...")
    macro_pit = fetch_macro_panel(fred)
    macro_pit.to_parquet(DATA_PROCESSED / "macro_pit.parquet")
    print(f"  saved {macro_pit.shape[0]} rows -> data/processed/macro_pit.parquet")

    print("\nFetching Ken French factor returns (5-factor + momentum)...")
    factors = fetch_french_factors()
    factors.to_parquet(DATA_RAW / "french_factors.parquet")
    print(f"  saved {factors.shape[0]} rows -> data/raw/french_factors.parquet")

    # Definition of done: prove the vintage logic actually does something.
    sample_dates = pd.to_datetime(["1990-06-01", "2001-06-01", "2009-06-01", "2020-06-01"])
    for series_id in MACRO_SERIES:
        try:
            sanity_check_vintages(fred, series_id, sample_dates)
        except Exception as e:  # noqa: BLE001
            print(f"  sanity check skipped for {series_id}: {e}")

    print("\nDone. Next: merge macro_pit + factors monthly, build the regime classifier (Week 2).")


if __name__ == "__main__":
    main()
