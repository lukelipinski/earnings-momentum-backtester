"""
Chart for the README: average excess return vs the S&P 500 (SPY) for events that
PASSED the Layer 1 filter vs events that FAILED it, at every holding period.
Reads all_events.csv (made by earnings_backtest.py) - no new downloads needed.
Error bars = 95% confidence interval (1.96 x standard error).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HOLD_PERIODS = [1, 3, 5, 20, 60]

events = pd.read_csv("all_events.csv")
passed = events[events["layer1"]]
failed = events[~events["layer1"]]


def mean_and_ci(series):
    """Average (in %) and half-width of its 95% confidence interval (in %)."""
    s = series.dropna()
    return s.mean() * 100, 1.96 * s.std() / np.sqrt(len(s)) * 100


p_avg, p_ci, f_avg, f_ci = [], [], [], []
for h in HOLD_PERIODS:
    m, c = mean_and_ci(passed[f"excess_{h}"])
    p_avg.append(m)
    p_ci.append(c)
    m, c = mean_and_ci(failed[f"excess_{h}"])
    f_avg.append(m)
    f_ci.append(c)

x = np.arange(len(HOLD_PERIODS))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, p_avg, width, yerr=p_ci, capsize=4,
       label=f"Passed the filter ({len(passed)} events)")
ax.bar(x + width / 2, f_avg, width, yerr=f_ci, capsize=4,
       label=f"Failed the filter ({len(failed)} events)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{h} day" if h == 1 else f"{h} days" for h in HOLD_PERIODS])
ax.set_xlabel("Holding period (trading days, entry at next day's open)")
ax.set_ylabel("Average excess return vs SPY (%)")
ax.set_title("Post-earnings filter: passed vs failed events (2015-2026, after 0.2% costs)")
ax.legend()
fig.text(0.01, 0.01, "Error bars: 95% confidence intervals. Overlapping holding periods "
         "mean these intervals are likely too narrow.", fontsize=8)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig("holding_period_chart.png", dpi=150)
print("Saved chart to holding_period_chart.png")