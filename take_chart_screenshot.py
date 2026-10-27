"""
Capture zoomed chart in Tab 1 with mouse placed over the main content area.
"""

import time
from playwright.sync_api import sync_playwright

def capture_vitals_chart():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        time.sleep(3)

        # Move mouse to center of main content area (x=800, y=500)
        page.mouse.move(800, 500)
        # Scroll down to reveal the vitals trajectory chart
        page.mouse.wheel(0, 1100)
        time.sleep(2)
        page.screenshot(path="outputs/tab1_vitals_chart_hour20.png")
        browser.close()
        print("Captured tab1_vitals_chart_hour20.png")

if __name__ == "__main__":
    capture_vitals_chart()
