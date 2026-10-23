"""
Earnings Momentum Backtester - Version 1
Tests my two-stage earnings momentum system layer by layer:
  Baseline : buy after every earnings release
  Layer 1  : + my post-earnings confirmation (gapped up, held the gap, high volume)
  Layer 2  : + beat EPS estimates
  Layer 3  : + Stage 1 proxy: stock has a history of big earnings reactions
Holding periods: my system's 1-5 sessions, plus 20 and 60 to see where any drift lives.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
LARGE_CAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "AMD", "AVGO",
    "CRM", "ADBE", "ORCL", "INTC", "QCOM", "CSCO", "IBM", "NOW", "SHOP", "UBER",
    "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL", "AXP", "WFC", "C",
    "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "DIS", "CMG",
    "JNJ", "PFE", "MRK", "LLY", "ABBV", "UNH", "XOM", "CVX", "CAT", "BA",
]
MID_CAPS = [   # more volatile, faster-moving names - closer to what my Stage 1 favours
    "ROKU", "SNAP", "ETSY", "CROX", "DKNG", "PLTR", "ENPH", "FSLR", "CELH", "DECK",
    "ELF", "LULU", "CHWY", "W", "PINS", "TWLO", "OKTA", "ZS", "DDOG", "NET",
    "MDB", "CRWD", "TTD", "HUBS", "DOCU", "ZM", "U", "AFRM", "SOFI", "HOOD",
    "COIN", "MRNA", "VRTX", "ALGN", "ULTA", "FIVE", "BURL", "YETI", "SMCI", "ANF",
    "DUOL", "APP", "CAVA", "ONON", "TOST", "RBLX", "LYFT", "EXPE", "ABNB", "BKNG",
]

TICKERS = LARGE_CAPS + MID_CAPS
BENCHMARK = "SPY"
START_DATE = "2015-01-01"
EARNINGS_LIMIT = 40          # past earnings dates per stock (~10 years)

# Post-earnings confirmation (Layer 1) - my checklist made mechanical
MIN_GAP = 0.02               # opened at least 2% above the previous close
MIN_VOLUME_RATIO = 2.0       # volume at least 2x the previous 20-day average
# (holding the gap = close at or above the open; no number needed)


# Stage 1 proxy (Layer 3): "earnings movement potential" + "historical earnings reaction"
HIST_LOOKBACK = 8            # look at the stock's previous 8 earnings reactions
MIN_HIST_EVENTS = 4          # need at least 4 previous reactions to judge
MIN_HIST_MOVE = 0.04         # their average absolute move must be at least 4%


HOLD_PERIODS = [1, 3, 5, 20, 60]   # trading sessions held, counting the entry day
MAIN_HOLD = 5                      # my system's stated horizon is 1-5 sessions
COST = 0.002                       # 0.2% round-trip trading cost



# ---------------------------------------------------------------
# STEP 1: DOWNLOAD DATA
# ---------------------------------------------------------------
def get_prices(ticker):
    """Daily Open/High/Low/Close/Volume, adjusted for splits and dividends."""
    df = yf.Ticker(ticker).history(start=START_DATE, auto_adjust=True)
    if df.empty:
        return None
    df.index = df.index.tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_earnings_dates(ticker):
    """List of (announcement time in New York, EPS surprise %), oldest first."""
    ed = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)
    if ed is None or ed.empty:
        return []
    ed = ed[ed["Reported EPS"].notna()]              # only releases that have happened
    ed.index = ed.index.tz_convert("America/New_York")
    ed = ed.sort_index()                              # oldest first (needed for Layer 3)
    return list(zip(ed.index, ed["Surprise(%)"]))


# ---------------------------------------------------------------
# STEP 2: FIND THE REACTION DAY
# ---------------------------------------------------------------
def reaction_day_index(prices, announce_time):
    """Before 12:00 NY time -> reacts same day. 12:00 or later -> reacts next trading day."""
    day = pd.Timestamp(announce_time.date())
    dates = prices.index
    if announce_time.hour < 12:
        i = dates.searchsorted(day, side="left")
    else:
        i = dates.searchsorted(day, side="right")
    return i if i < len(dates) else None


# ---------------------------------------------------------------
# STEP 3: MEASURE EACH EVENT
# ---------------------------------------------------------------
def analyse_event(ticker, prices, bench, announce_time, surprise, past_moves):
    """past_moves = this stock's earlier reaction-day moves (only known BEFORE this release)."""
    r = reaction_day_index(prices, announce_time)
    if r is None or r < 21:
        return None, None

    prev_close = prices["Close"].iloc[r - 1]
    react_open = prices["Open"].iloc[r]
    react_close = prices["Close"].iloc[r]
    avg_volume = prices["Volume"].iloc[r - 20:r].mean()   # 20 days BEFORE reaction day
    if avg_volume == 0 or np.isnan(avg_volume):
        return None, None

    day_move = react_close / prev_close - 1
    gap = react_open / prev_close - 1
    held_gap = react_close >= react_open
    volume_ratio = prices["Volume"].iloc[r] / avg_volume
    beat = (not pd.isna(surprise)) and (surprise > 0)

    # Stage 1 proxy: average size of this stock's PREVIOUS earnings reactions
    recent = past_moves[-HIST_LOOKBACK:]
    hist_move = np.mean(np.abs(recent)) if len(recent) >= MIN_HIST_EVENTS else np.nan

    confirmed = (gap >= MIN_GAP) and held_gap and (volume_ratio >= MIN_VOLUME_RATIO)
    big_reactor = (not np.isnan(hist_move)) and (hist_move >= MIN_HIST_MOVE)

    row = {
        "ticker": ticker,
        "reaction_date": prices.index[r].date(),
        "eps_surprise_pct": surprise,
        "gap": gap,
        "held_gap": held_gap,
        "day_move": day_move,
        "volume_ratio": volume_ratio,
        "hist_avg_move": hist_move,
        "layer1": confirmed,
        "layer2": confirmed and beat,
        "layer3": confirmed and beat and big_reactor,
    }

    # Trade: buy at the open of the day AFTER the reaction day, hold h sessions
    entry_i = r + 1
    if entry_i >= len(prices):
        return None, day_move
    entry_date = prices.index[entry_i]
    entry_price = prices["Open"].iloc[entry_i]
    if entry_date not in bench.index:
        return None, day_move
    row["entry_date"] = entry_date.date()

    for h in HOLD_PERIODS:
        exit_i = entry_i + h - 1
        if exit_i >= len(prices) or prices.index[exit_i] not in bench.index:
            row[f"ret_{h}"] = row[f"mkt_{h}"] = row[f"excess_{h}"] = np.nan
            continue
        exit_date = prices.index[exit_i]
        ret = prices["Close"].iloc[exit_i] / entry_price - 1 - COST
        mkt = bench.loc[exit_date, "Close"] / bench.loc[entry_date, "Open"] - 1
        row[f"ret_{h}"], row[f"mkt_{h}"], row[f"excess_{h}"] = ret, mkt, ret - mkt

    return row, day_move


# ---------------------------------------------------------------
# STEP 4: STATISTICS
# ---------------------------------------------------------------
def summarise(trades, label, h=MAIN_HOLD):
    t = trades.dropna(subset=[f"ret_{h}"]).sort_values("entry_date")
    if t.empty:
        print(f"\n{label}: no trades")
        return
    ret, exc = t[f"ret_{h}"], t[f"excess_{h}"]
    equity = ret.cumsum()
    drawdown = (equity - equity.cummax()).min()
    print(f"\n{label}  (held {h} sessions)")
    print(f"  Number of trades:          {len(t)}")
    print(f"  Average return per trade:  {ret.mean():.2%}")
    print(f"  Median return per trade:   {ret.median():.2%}")
    print(f"  Win rate:                  {(ret > 0).mean():.1%}")
    print(f"  Average market return:     {t[f'mkt_{h}'].mean():.2%}")
    print(f"  Average excess vs market:  {exc.mean():.2%}")
    print(f"  Beat the market:           {(exc > 0).mean():.1%} of trades")
    print(f"  Worst single trade:        {ret.min():.2%}")
    print(f"  Max drawdown (units):      {drawdown:.2f}")


def holding_period_table(layers):
    """Average excess return for every layer at every holding period."""
    table = pd.DataFrame({
        name: {f"{h} sessions": df[f"excess_{h}"].mean() for h in HOLD_PERIODS}
        for name, df in layers.items()
    }).T
    counts = pd.Series({name: len(df) for name, df in layers.items()}, name="trades")
    print("\nAVERAGE EXCESS RETURN VS MARKET, BY LAYER AND HOLDING PERIOD")
    print(pd.concat([counts, table.map(lambda x: f"{x:.2%}")], axis=1).to_string())


def make_charts(signals, all_events, h=MAIN_HOLD):
    s = signals.dropna(subset=[f"ret_{h}"]).sort_values("entry_date")
    a = all_events.dropna(subset=[f"excess_{h}"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(pd.to_datetime(s["entry_date"]), s[f"ret_{h}"].cumsum(), label="Strategy")
    axes[0].plot(pd.to_datetime(s["entry_date"]), s[f"mkt_{h}"].cumsum(), label="Market (same dates)")
    axes[0].set_title(f"Cumulative return, 1 unit per trade ({h}-session hold)")
    axes[0].legend()
    axes[1].hist(a[f"excess_{h}"], bins=50, alpha=0.5, label="All earnings events")
    axes[1].hist(s[f"excess_{h}"], bins=50, alpha=0.7, label="Layer 1 signals")
    axes[1].set_title("Excess return vs market per trade")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved chart to results.png")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("Downloading benchmark...")
    bench = get_prices(BENCHMARK)

    rows = []
    for ticker in TICKERS:
        try:
            prices = get_prices(ticker)
            dates = get_earnings_dates(ticker)
            if prices is None or not dates:
                print(f"{ticker}: no data, skipped")
                continue
            past_moves = []                       # grows as we move forward through time
            for d, surprise in dates:
                row, move = analyse_event(ticker, prices, bench, d, surprise, past_moves)
                if row:
                    row["group"] = "large" if ticker in LARGE_CAPS else "mid"
                    rows.append(row)
                if move is not None:
                    past_moves.append(move)       # added AFTER use: no look-ahead
            print(f"{ticker}: {len(dates)} earnings dates processed")
        except Exception as e:
            print(f"{ticker}: error ({e}), skipped")
        time.sleep(1)

    events = pd.DataFrame(rows)
    if events.empty:
        print("No events collected - check your internet connection and yfinance version.")
        return
    events.to_csv("all_events.csv", index=False)

    layers = {
        "Baseline (every release)": events,
        "Layer 1: + confirmation": events[events["layer1"]],
        "Layer 2: + EPS beat": events[events["layer2"]],
        "Layer 3: + big past reactor": events[events["layer3"]],
    }
    events[events["layer1"]].to_csv("strategy_trades.csv", index=False)

    for name, df in layers.items():
        summarise(df, name.upper())
    holding_period_table(layers)

    print("\nLARGE CAPS VS MID CAPS (Layer 1)")
    for g in ["large", "mid"]:
        summarise(events[events["layer1"] & (events["group"] == g)], f"Layer 1, {g} caps")

    make_charts(layers["Layer 1: + confirmation"], events)


if __name__ == "__main__":
    main()