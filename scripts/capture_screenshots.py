"""
Capture dashboard screenshots for README.

Usage:
    1. Spustite Dash app v inom termináli:  python app_fixed.py
    2. Potom:  python scripts/capture_screenshots.py

Skript otvorí appku v headless Chromium-e cez Playwright, prepne tmavý režim,
zachytí 4 obrázky do docs/screenshots/ a ukončí sa.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


URL = "http://localhost:8050/"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1600, "height": 1000}


async def capture(name: str, page, scroll_to: int = 0, wait_after_scroll_ms: int = 600):
    await page.evaluate(f"window.scrollTo(0, {scroll_to})")
    await page.wait_for_timeout(wait_after_scroll_ms)
    out_path = OUT_DIR / f"{name}.png"
    await page.screenshot(path=str(out_path), full_page=False)
    print(f"  saved {out_path.relative_to(OUT_DIR.parent.parent)}")


async def set_theme(page, theme: str):
    await page.evaluate(
        f"""
        document.documentElement.setAttribute('data-theme', '{theme}');
        localStorage.setItem('dashboard-theme', '{theme}');
        const icon = document.getElementById('theme-toggle-icon');
        const label = document.getElementById('theme-toggle-label');
        if (icon)  icon.textContent  = '{('☀️' if theme == 'dark' else '🌙')}';
        if (label) label.textContent = '{('Light mode' if theme == 'dark' else 'Dark mode')}';
        """
    )
    await page.wait_for_timeout(800)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = await context.new_page()

        print(f"Loading {URL} ...")
        await page.goto(URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_selector("#metrics-cards .metric-card", timeout=20000)
        await page.wait_for_timeout(1500)

        # Light mode shots
        print("Light mode:")
        await set_theme(page, "light")
        await capture("01-hero-light", page, scroll_to=0)
        await capture("02-charts-light", page, scroll_to=700)
        await capture("03-heatmaps-light", page, scroll_to=1700)
        await capture("04-monte-carlo-light", page, scroll_to=2400)

        # Dark mode shots
        print("Dark mode:")
        await set_theme(page, "dark")
        await capture("05-hero-dark", page, scroll_to=0)
        await capture("06-charts-dark", page, scroll_to=700)

        await browser.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
