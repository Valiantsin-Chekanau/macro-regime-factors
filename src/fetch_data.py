"""
Week 1 data spine: point-in-time macro series + Ken French factor returns.

Core idea (the credibility anchor of this project): a regime label for month M is
only valid if it uses macro values that were ACTUALLY PUBLISHED and not-yet-revised
as of the decision date. Pulling "latest revised" INDPRO/CPI and pretending you knew
it in real time is lookahead bias, full stop.

Implementation note: fredapi's get_series_as_of_date() re-downloads a series' ENTIRE
vintage history on every call and doesn't collapse to "latest known reference period" —
calling it once per decision date is both slow (one full-history HTTP call per date)
and subtly wrong (can return a stale period's late revision instead of the newest
known period). Instead we pull get_series_all_releases() ONCE per series (one bulk
call) and do all point-in-time lookups locally in pandas.

Within-vintage YoY (audit fix, Jul 2026): YoY transforms are computed HERE, inside
each decision date's vintage — value(P, as-of T) / value(P-12, as-of T) - 1 — NOT
downstream via pct_change(12) on the PIT level panel. Differencing the PIT panel
compares two vintages ~14 months apart, so every agency rebasing lands inside the
ratio: CPI's Jan-1988 re-reference (1967=100 -> 1982-84=100) read as "-65% inflation"
for 12 straight months and polluted the 5yr trailing mean into 1993; INDPRO rebasings
(1976/1985/1990/1997/2003/2005/2010/2021) did the same to growth. Within a single
vintage the index base is constant and cancels in the ratio. This is also the
real-time-data-literature convention (Croushore & Stark): transform within a vintage,
then step along the vintage axis.

Two paths, in priority order:
  1. ALFRED vintages (preferred) — described above.
  2. Lag-floor fallback (safety net) — if ALFRED vintage history doesn't cover a
     series, shift the latest-revised series by its known publication delay instead.
     Never let vintage-fetching block progress.

Run:
    python src/fetch_data.py
Requires FRED_API_KEY in .env (see .env.example).
"""

import os
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

# Rough publication delays AFTER THE REFERENCE MONTH ENDS — fallback path ONLY.
# FRED indexes a month's value by the month START, so availability is
# period_start + 1 month + lag_days (audit fix: lagging from period start made
# April data "known" on Apr 18, i.e. ~4 weeks before it was actually published).
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


def fetch_all_releases(fred: Fred, series_id: str) -> pd.DataFrame:
    """ONE bulk call: full vintage history (date=reference period, realtime_start, value)."""
    df = fred.get_series_all_releases(series_id)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df = df.dropna(subset=["value"])  # ALFRED can carry deleted-observation rows
    return df.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def pit_panel_from_releases(releases: pd.DataFrame, decision_dates, series_id: str) -> pd.DataFrame:
    """
    For each decision date T, resolve everything WITHIN the vintage known at T:
      level: newest reference period P whose FIRST release happened on/before T,
             taken at its latest revision with realtime_start <= T.
      yoy:   that level vs. the SAME vintage's value for period P-12 (also latest
             revision <= T). Same vintage => same index base => rebasings cancel.
    Returns a DataFrame with columns [series_id, f"{series_id}_yoy"].
    All done locally in pandas — no network calls.
    """
    first_release = releases.groupby("date")["realtime_start"].min().sort_index()
    by_period = {
        p: g.sort_values("realtime_start")[["realtime_start", "value"]]
        for p, g in releases.groupby("date")
    }

    def value_as_of(period, T):
        g = by_period.get(period)
        if g is None:
            return None
        vis = g[g["realtime_start"] <= T]
        return None if vis.empty else vis["value"].iloc[-1]

    levels, yoys = {}, {}
    for T in decision_dates:
        visible = first_release[first_release <= T]
        if visible.empty:
            continue
        latest_period = visible.index.max()
        v = value_as_of(latest_period, T)
        if v is None:
            continue
        levels[T] = v
        v12 = value_as_of(latest_period - pd.DateOffset(months=12), T)
        if v12 is not None and v12 != 0:
            yoys[T] = v / v12 - 1.0

    if not levels:
        raise RuntimeError(f"no point-in-time values resolved for {series_id}")
    return pd.DataFrame(
        {series_id: pd.Series(levels), f"{series_id}_yoy": pd.Series(yoys)}
    ).sort_index()


def fetch_pit_series_lag_floor(fred: Fred, series_id: str, decision_dates) -> pd.DataFrame:
    """
    Fallback: pull latest-revised series once, then lag it by its publication delay.
    Less rigorous than ALFRED (ignores revisions, just delays availability) but
    never blocks — used only if ALFRED vintages error out or don't cover the range.
    YoY here comes from the single latest vintage, which is internally
    base-consistent, so pct_change(12) is safe on THIS path (and only this path).
    """
    latest = fred.get_series(series_id)
    lag_days = PUBLICATION_LAG_DAYS.get(series_id, 21)
    df = pd.DataFrame({"value": latest, "yoy": latest.pct_change(12)})
    # month indexed by its start; published ~lag_days after the month ENDS
    df["available"] = df.index + pd.DateOffset(months=1) + pd.Timedelta(days=lag_days)

    levels, yoys = {}, {}
    for d in decision_dates:
        vis = df[df["available"] <= d]
        if vis.empty:
            continue
        row = vis.iloc[-1]
        levels[d] = row["value"]
        if pd.notna(row["yoy"]):
            yoys[d] = row["yoy"]
    return pd.DataFrame(
        {series_id: pd.Series(levels), f"{series_id}_yoy": pd.Series(yoys)}
    ).sort_index()


def fetch_macro_panel(fred: Fred):
    """Returns (panel_df, releases_cache) — releases_cache reused later for the sanity check."""
    decision_dates = monthly_decision_dates()
    frames = []
    releases_cache = {}
    for series_id in MACRO_SERIES:
        try:
            print(f"  [ALFRED] {series_id}: pulling full vintage history (1 bulk call)...")
            releases = fetch_all_releases(fred, series_id)
            releases_cache[series_id] = releases
            frame = pit_panel_from_releases(releases, decision_dates, series_id)
            print(f"    resolved {frame[series_id].notna().sum()} monthly PIT levels, "
                  f"{frame[f'{series_id}_yoy'].notna().sum()} within-vintage YoY values")
        except Exception as e:  # noqa: BLE001
            print(f"  [ALFRED FAILED for {series_id}: {e}] falling back to lag-floor")
            frame = fetch_pit_series_lag_floor(fred, series_id, decision_dates)
            releases_cache[series_id] = None
        frames.append(frame)
    return pd.concat(frames, axis=1), releases_cache


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

def sanity_check_vintages(fred: Fred, series_id: str, releases: pd.DataFrame, sample_dates):
    if releases is None:
        print(f"\nSanity check skipped for {series_id} (ALFRED unavailable, used lag-floor)")
        return
    print(f"\nSanity check for {series_id}: as-of value vs latest-revised value")
    print(f"{'as_of_date':<12}{'ref_period':<12}{'as_of_value':>14}{'latest_value':>14}{'differs?':>10}")
    latest = fred.get_series(series_id)
    single_date_panel = pit_panel_from_releases(releases, sample_dates, series_id)[series_id]
    first_release = releases.groupby("date")["realtime_start"].min().sort_index()
    for T, as_of_val in single_date_panel.items():
        visible = first_release[first_release <= T]
        ref_period = visible.index.max()
        latest_val = latest.get(ref_period)
        differs = "YES" if (latest_val is not None and abs(latest_val - as_of_val) > 1e-9) else "no"
        print(f"{T.date()!s:<12}{ref_period.date()!s:<12}{as_of_val:>14.3f}{latest_val:>14.3f}{differs:>10}")


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
    macro_pit, releases_cache = fetch_macro_panel(fred)
    macro_pit.to_parquet(DATA_PROCESSED / "macro_pit.parquet")
    print(f"  saved {macro_pit.shape[0]} rows -> data/processed/macro_pit.parquet")

    print("\nFetching Ken French factor returns (5-factor + momentum)...")
    factors = fetch_french_factors()
    factors.to_parquet(DATA_RAW / "french_factors.parquet")
    print(f"  saved {factors.shape[0]} rows -> data/raw/french_factors.parquet")

    # Definition of done: prove the vintage logic actually does something.
    sample_dates = pd.to_datetime(["1990-06-01", "2001-06-01", "2009-06-01", "2020-06-01"])
    for series_id in MACRO_SERIES:
        sanity_check_vintages(fred, series_id, releases_cache.get(series_id), sample_dates)

    print("\nDone. Next: run src/build_panel.py to merge with factor returns, "
          "then src/regimes.py, then python tests/test_sanity.py.")


if __name__ == "__main__":
    main()
