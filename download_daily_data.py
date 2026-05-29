import time
from pathlib import Path

import pandas as pd
import yfinance as yf


# --------------------------------------------------
# 1. Nastavenia
# --------------------------------------------------

TICKERS = ["SPY", "GLD", "AGG", "DBC"]

START_DATE = "2006-01-01"
END_DATE = None

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Vlastná cache priamo v projekte, aby sme obišli zaseknutú Windows cache
CACHE_DIR = Path("yf_cache")
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR.resolve()))


# --------------------------------------------------
# 2. Funkcia na stiahnutie jedného tickeru
# --------------------------------------------------

def download_one_ticker(ticker: str, max_attempts: int = 3) -> pd.Series:
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Sťahujem {ticker}, pokus {attempt}/{max_attempts}...")

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
                raise ValueError(f"{ticker}: Yahoo vrátil prázdny dataset.")

            if "Adj Close" not in df.columns:
                raise ValueError(f"{ticker}: v dátach chýba stĺpec Adj Close.")

            series = df["Adj Close"].copy()
            series.name = ticker

            print(f"{ticker}: OK, počet riadkov: {len(series)}")
            return series

        except Exception as e:
            print(f"{ticker}: chyba pri pokuse {attempt}: {e}")
            time.sleep(3)

    raise RuntimeError(f"{ticker}: nepodarilo sa stiahnuť dáta po {max_attempts} pokusoch.")


# --------------------------------------------------
# 3. Hlavný beh — chránený main guardom, aby sa script
#    neexekutoval pri importe modulu.
# --------------------------------------------------

def main():
    all_series = []

    for ticker in TICKERS:
        series = download_one_ticker(ticker)
        all_series.append(series)
        time.sleep(1)

    adj_close = pd.concat(all_series, axis=1)
    adj_close = adj_close.sort_index()
    adj_close = adj_close.dropna(how="all")

    # Dataset iba pre spoločné obdobie všetkých ETF
    adj_close_common = adj_close.dropna(how="any")

    # 4. Uloženie dát
    adj_close.to_csv(OUTPUT_DIR / "daily_adjusted_prices_all.csv")
    adj_close_common.to_csv(OUTPUT_DIR / "daily_adjusted_prices_common.csv")

    adj_close.to_excel(OUTPUT_DIR / "daily_adjusted_prices_all.xlsx")
    adj_close_common.to_excel(OUTPUT_DIR / "daily_adjusted_prices_common.xlsx")

    # 5. Kontrola
    print()
    print("Hotovo.")
    print()
    print("Všetky dostupné dáta:")
    print(adj_close.head())
    print(adj_close.tail())

    print()
    print("Spoločné obdobie bez chýbajúcich hodnôt:")
    print(adj_close_common.head())
    print(adj_close_common.tail())

    print()
    print("Počet riadkov - všetky dáta:", len(adj_close))
    print("Počet riadkov - spoločné obdobie:", len(adj_close_common))

    print()
    print("Začiatok spoločného obdobia:", adj_close_common.index.min())
    print("Koniec spoločného obdobia:", adj_close_common.index.max())


if __name__ == "__main__":
    main()