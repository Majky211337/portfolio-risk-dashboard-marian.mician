"""
Download daily adjusted close prices for the dashboard's ETFs from Yahoo Finance.

Run by the scheduled GitHub Action (.github/workflows/refresh-data.yml) and can
also be run manually:  python download_daily_data.py

Writes to data/:
- daily_adjusted_prices_all.csv     every date on which at least one ETF traded
- daily_adjusted_prices_common.csv  only dates where all ETFs have a price (used by the app)
plus .xlsx copies of both.

If any ticker cannot be downloaded the script exits with an error and leaves
the existing files untouched, so the app keeps serving the last good dataset.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "GLD", "AGG", "DBC"]
START_DATE = "2006-01-01"
END_DATE = None

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "yf_cache"  # project-local timezone cache (avoids a stale global cache)


def download_one_ticker(ticker: str, max_attempts: int = 3) -> pd.Series:
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Downloading {ticker}, attempt {attempt}/{max_attempts}...")
            df = yf.download(
                tickers=ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
                interval="1d",
            )

            if df.empty:
                raise ValueError("Yahoo returned an empty dataset.")
            if "Adj Close" not in df.columns.get_level_values(0):
                raise ValueError("the 'Adj Close' column is missing.")

            series = df["Adj Close"]
            if isinstance(series, pd.DataFrame):  # recent yfinance returns (Price, Ticker) columns
                series = series.iloc[:, 0]
            series = series.dropna().rename(ticker)

            print(f"{ticker}: OK, {len(series)} rows")
            return series

        except Exception as exc:
            print(f"{ticker}: attempt {attempt} failed: {exc}")
            time.sleep(3)

    raise RuntimeError(f"{ticker}: download failed after {max_attempts} attempts.")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR))

    all_series = []
    for ticker in TICKERS:
        all_series.append(download_one_ticker(ticker))
        time.sleep(1)

    adj_close = pd.concat(all_series, axis=1).sort_index().dropna(how="all")
    adj_close.index.name = None
    adj_close_common = adj_close.dropna(how="any")

    if adj_close_common.empty:
        raise RuntimeError("No dates with prices for every ETF. Keeping the existing data.")

    adj_close.to_csv(OUTPUT_DIR / "daily_adjusted_prices_all.csv")
    adj_close_common.to_csv(OUTPUT_DIR / "daily_adjusted_prices_common.csv")
    adj_close.to_excel(OUTPUT_DIR / "daily_adjusted_prices_all.xlsx")
    adj_close_common.to_excel(OUTPUT_DIR / "daily_adjusted_prices_common.xlsx")

    print("\nDone.")
    print("Rows (all dates):   ", len(adj_close))
    print("Rows (common dates):", len(adj_close_common))
    print("Common period:      ", adj_close_common.index.min().date(), "->", adj_close_common.index.max().date())


if __name__ == "__main__":
    main()
