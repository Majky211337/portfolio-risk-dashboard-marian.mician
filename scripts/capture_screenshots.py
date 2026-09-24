"""
Capture README screenshots of the running Streamlit app.

Usage (dev-only dependency: pip install playwright && playwright install chromium):
    1. streamlit run app.py
    2. python scripts/capture_screenshots.py [url]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"

# (file name, tab label or None for the landing view, viewport height)
SHOTS = [
    ("01-overview", None, 1000),
    ("02-risk", "Risk & drawdown", 1500),
    ("03-diversification", "Diversification", 1400),
    ("04-calendar", "Calendar returns", 1650),
    ("05-stress-test", "Stress test", 1400),
    ("06-monte-carlo", "Monte Carlo", 1300),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("[data-testid='stMetric']", timeout=60_000)
        page.wait_for_timeout(2000)

        for name, tab, height in SHOTS:
            page.set_viewport_size({"width": 1600, "height": 1000})
            if tab:
                page.get_by_role("tab", name=tab).click()
            page.set_viewport_size({"width": 1600, "height": height})
            page.wait_for_timeout(2500)
            path = OUT_DIR / f"{name}.png"
            page.screenshot(path=str(path))
            print(f"saved {path.relative_to(OUT_DIR.parent.parent)}")

        browser.close()


if __name__ == "__main__":
    main()
