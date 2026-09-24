import logging

import pandas as pd
import pytest

import portfolio_engine as engine

DEFAULT_WEIGHTS = {"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10}
FREQUENCIES = ["never", "monthly", "quarterly", "annually"]

# Streamlit logs a harmless warning when cached functions run outside `streamlit run`.
logging.getLogger("streamlit").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """The production dataset exactly as the app loads it."""
    return engine.load_prices()


@pytest.fixture(scope="session")
def raw_common_csv() -> pd.DataFrame:
    """The production CSV without any cleaning, to audit the file itself."""
    return pd.read_csv(engine.DATA_PATH, index_col=0, parse_dates=True)
