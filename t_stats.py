"""
t-statistics for the earnings momentum backtest.
Reads all_events.csv (made by earnings_backtest.py) - no new downloads needed.
t = average excess return / (standard deviation / square root of number of trades)
Rule of thumb: |t| of 2 or more = unlikely to be luck (about 95% confidence)
"""

import numpy as np
import pandas as pd

HOLD_PERIODS = [1, 3, 5, 20, 60]

events = pd.read_csv("all_events.csv")
print(len(events), "events loaded")

layers = {
    "Baseline": events,
    "Layer 1": events[events["layer1"]],
    "Layer 2": events[events["layer2"]],
    "Layer 3": events[events["layer3"]],
}


def t_stat(series):
    """t-statistic for whether the average is different from zero."""
    s = series.dropna()                 # skip trades with no result (too recent)
    n = len(s)
    if n < 2:
        return np.nan, n
    t = s.mean() / (s.std() / np.sqrt(n))
    return t, n


rows = []
for name, df in layers.items():
    for h in HOLD_PERIODS:
        t, n = t_stat(df[f"excess_{h}"])
        rows.append({
            "layer": name,
            "hold": h,
            "trades": n,
            "avg_excess": df[f"excess_{h}"].mean(),
            "t_stat": t,
            "significant": "YES" if abs(t) >= 2 else "no",
        })

table = pd.DataFrame(rows)
print("\nT-STATISTICS: IS THE AVERAGE EXCESS RETURN DIFFERENT FROM ZERO?")
print(table.to_string(index=False, formatters={
    "avg_excess": "{:.2%}".format,
    "t_stat": "{:.2f}".format,
}))


def welch_t(a, b):
    """t-statistic for whether two groups have different averages."""
    a, b = a.dropna(), b.dropna()
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


confirmed = events[events["layer1"]]
others = events[~events["layer1"]]          # ~ means "not": every event that failed Layer 1

print("\nDOES LAYER 1 BEAT EVERYTHING ELSE?")
for h in HOLD_PERIODS:
    gap = confirmed[f"excess_{h}"].mean() - others[f"excess_{h}"].mean()
    t = welch_t(confirmed[f"excess_{h}"], others[f"excess_{h}"])
    verdict = "YES" if abs(t) >= 2 else "no"
    print(f"  Hold {h:>2}: Layer 1 better by {gap:.2%}, t = {t:.2f}, significant: {verdict}")

    # ---------------------------------------------------------------
# ROBUSTNESS 1: SENSITIVITY GRID - do nearby settings give similar results?
# ---------------------------------------------------------------
GAP_LEVELS = [0.01, 0.02, 0.03]
VOLUME_LEVELS = [1.5, 2.0, 3.0]

print("\nSENSITIVITY GRID: PASSED vs FAILED (how much better, t-stat)")
for h in [5, 20]:
    print(f"\n  Hold {h} sessions")
    for g in GAP_LEVELS:
        for v in VOLUME_LEVELS:
            passed = (events["gap"] >= g) & events["held_gap"] & (events["volume_ratio"] >= v)
            a = events.loc[passed, f"excess_{h}"]
            b = events.loc[~passed, f"excess_{h}"]
            diff = a.mean() - b.mean()
            t = welch_t(a, b)
            print(f"    gap >= {g:.0%}, volume >= {v}x: {a.count():>4} trades, "
                  f"better by {diff:.2%}, t = {t:.2f}")

# ---------------------------------------------------------------
# ROBUSTNESS 2: OUT-OF-SAMPLE - does the effect appear in both halves of history?
# ---------------------------------------------------------------
events["reaction_date"] = pd.to_datetime(events["reaction_date"])
SPLIT_DATE = "2022-01-01"

periods = {
    "2015-2021": events[events["reaction_date"] < SPLIT_DATE],
    "2022-2026": events[events["reaction_date"] >= SPLIT_DATE],
}

print("\nTWO PERIODS: DOES LAYER 1 BEAT THE REST IN BOTH?")
for period_name, df in periods.items():
    print(f"\n  {period_name}")
    yes = df[df["layer1"]]
    no = df[~df["layer1"]]
    for h in [1, 5, 20, 60]:
        diff = yes[f"excess_{h}"].mean() - no[f"excess_{h}"].mean()
        t = welch_t(yes[f"excess_{h}"], no[f"excess_{h}"])
        print(f"    Hold {h:>2}: {yes[f'excess_{h}'].count():>3} trades, "
              f"better by {diff:.2%}, t = {t:.2f}")