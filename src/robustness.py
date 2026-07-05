"""
Week 4: the ONE allowed robustness check (guide sec 6: "one robustness
alternative, not a grid search").

What's being tested: the regime classifier's trailing-window length. Week 2
locked a 60-month (5yr) trailing mean as the "normal" baseline for YoY growth/
inflation (docs/regime_classifier_rationale.md). The guide's own Week 2 text
explicitly names "trailing 3-5yr mean" as the range of defensible choices --
5yr was picked, not tuned. This script reruns the whole classifier -> conditional
-> bootstrap pipeline at the OTHER end of that explicitly-sanctioned range (36
months / 3yr) and checks whether the Week 3 conclusions survive.

Why the window and not the guide's other listed option (swap INDPRO<->PAYEMS or
CPI<->core PCE)? The trailing window is the classifier's only free parameter, so
perturbing it stress-tests the one design choice actually made in Week 2 -- and it
holds the input data fixed, so any movement in the conclusions is attributable to
the classifier itself rather than to a different macro series with its own
revision and coverage quirks. The proxy swap is the natural next check,
deliberately not run to keep to the guide's one-alternative mandate; it is first
in line for v2.

Two things compared, in order of how much they'd worry a reviewer:
1. Regime label agreement rate: of months labeled under BOTH windows, what % get
   the same quadrant? Low agreement would mean the classifier is unstable to a
   choice inside its own sanctioned range -- a real problem for the whole project.
2. Whether the Week 3 "standouts" (CIs that clear zero) still clear zero under the
   alt window, and whether any new cells clear zero that didn't before.

Run (from the repo root -- this module imports src.analysis, so it must run as a
module, not as a script path):
    python -m src.robustness
Requires data/processed/panel.parquet (produced by src/build_panel.py).
Writes:
    data/processed/robustness_altwindow_conditional.parquet
    data/processed/robustness_altwindow_bootstrap.parquet
    data/processed/robustness_label_agreement.parquet
"""

from pathlib import Path

import pandas as pd

from src.analysis import FACTOR_COLS, bootstrap_table, conditional_table
from src.regimes import TRAILING_WINDOW_MONTHS, add_regime, add_states

DATA_PROCESSED = Path("data/processed")
ALT_WINDOW_MONTHS = 36  # 3yr -- the other end of the guide's sanctioned 3-5yr range


def _labeled_rescaled(panel: pd.DataFrame) -> pd.DataFrame:
    """Same labeled-months-only + percent-to-decimal rescale as
    analysis.load_labeled_panel, but starting from an in-memory panel rather
    than reading panel_regimes.parquet off disk (so it works for both the
    baseline and alt-window relabeling without writing intermediate files)."""
    labeled = panel.dropna(subset=["regime"]).copy()
    for col in FACTOR_COLS:
        labeled[col] = labeled[col] / 100.0
    return labeled


def build_regimes(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    out = add_states(panel, window=window)
    out = add_regime(out)
    return out


def label_agreement(base: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned comparison of regime labels under both windows, restricted to
    months labeled under both (the alt window has a longer or shorter unlabeled
    warm-up period at the start of the sample)."""
    joined = pd.DataFrame({
        "regime_base": base["regime"],
        "regime_alt": alt["regime"],
    }).dropna()
    joined["agree"] = joined["regime_base"] == joined["regime_alt"]
    return joined


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")

    base_labeled = build_regimes(panel, TRAILING_WINDOW_MONTHS)
    alt_labeled = build_regimes(panel, ALT_WINDOW_MONTHS)

    agreement = label_agreement(base_labeled, alt_labeled)
    agree_path = DATA_PROCESSED / "robustness_label_agreement.parquet"
    agreement.to_parquet(agree_path)
    agree_rate = agreement["agree"].mean()
    print(
        f"regime label agreement, {TRAILING_WINDOW_MONTHS}mo vs {ALT_WINDOW_MONTHS}mo "
        f"window: {agree_rate:.1%} of {len(agreement)} months labeled under both "
        f"-> {agree_path}"
    )
    print("\ndisagreement breakdown (base regime -> alt regime, count):")
    disagree = agreement.loc[~agreement["agree"]]
    print(
        disagree.groupby(["regime_base", "regime_alt"]).size()
        .sort_values(ascending=False).head(10)
    )

    base_panel = _labeled_rescaled(base_labeled)
    alt_panel = _labeled_rescaled(alt_labeled)

    print(f"\nbase ({TRAILING_WINDOW_MONTHS}mo) regime counts:\n{base_panel['regime'].value_counts()}")
    print(f"\nalt ({ALT_WINDOW_MONTHS}mo) regime counts:\n{alt_panel['regime'].value_counts()}")

    alt_cond = conditional_table(alt_panel)
    alt_cond_path = DATA_PROCESSED / "robustness_altwindow_conditional.parquet"
    alt_cond.to_parquet(alt_cond_path)

    alt_boot = bootstrap_table(alt_panel)
    alt_boot_path = DATA_PROCESSED / "robustness_altwindow_bootstrap.parquet"
    alt_boot.to_parquet(alt_boot_path)
    print(f"\nalt-window conditional -> {alt_cond_path}, bootstrap -> {alt_boot_path}")

    base_boot = bootstrap_table(base_panel)
    base_rows = base_boot[base_boot["regime"] != "All (unconditional)"].set_index(["regime", "factor"])
    alt_rows = alt_boot[alt_boot["regime"] != "All (unconditional)"].set_index(["regime", "factor"])
    common = base_rows.index.intersection(alt_rows.index)

    base_clears = set(common[base_rows.loc[common, "straddles_zero"] == False])
    alt_clears = set(common[alt_rows.loc[common, "straddles_zero"] == False])

    print(
        f"\n{len(base_clears)}/{len(common)} cells clear zero under the base "
        f"({TRAILING_WINDOW_MONTHS}mo) window; {len(alt_clears)}/{len(common)} clear "
        f"zero under the alt ({ALT_WINDOW_MONTHS}mo) window."
    )
    still_clear = base_clears & alt_clears
    lost = base_clears - alt_clears
    gained = alt_clears - base_clears
    print(f"survive in both: {sorted(still_clear)}")
    print(f"clear under base only (didn't survive the alt window): {sorted(lost)}")
    print(f"clear under alt only (new, wasn't a base-window standout): {sorted(gained)}")


if __name__ == "__main__":
    main()
