"""
Robust script to capture high-resolution screenshots of all 3 tabs at Hour 20.
"""

import time
import os
from playwright.sync_api import sync_playwright

def capture_dashboard():
    os.makedirs("outputs", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        time.sleep(3)

        tabs = page.get_by_role("tab").all()
        print(f"Detected {len(tabs)} tabs.")

        # -----------------------------------------------------------
        # 1. Tab 1: Top Overview (Metrics, Alerts, Bed Grid)
        # -----------------------------------------------------------
        print("Capturing Tab 1: Top Overview at Hour 20...")
        page.screenshot(path="outputs/tab1_overview_hour20.png")

        # Tab 1: Scroll to Patient Vitals Trajectory Chart
        print("Capturing Tab 1: Vitals Chart at Hour 20...")
        try:
            page.mouse.wheel(0, 1100)
            time.sleep(2)
            page.screenshot(path="outputs/tab1_vitals_chart_hour20.png")
        except Exception as e:
            print(f"Error scrolling Tab 1: {e}")

        # -----------------------------------------------------------
        # 2. Tab 2: Medical Log Allocation & Search
        # -----------------------------------------------------------
        print("Capturing Tab 2: Medical Log Allocation & Search...")
        try:
            page.mouse.wheel(0, -1500)
            time.sleep(0.5)
            tabs[1].click()
            time.sleep(2)
            
            search_input = page.locator("div[data-testid='stTextInput'] input")
            if search_input.count() > 0:
                search_input.first.fill("sepsis")
                time.sleep(1.5)
            
            # Click expander if available
            expanders = page.locator("div[data-testid='stExpander']")
            if expanders.count() > 0:
                expanders.first.click()
                time.sleep(1)
                
            page.screenshot(path="outputs/tab2_records_search_hour20.png")
        except Exception as e:
            print(f"Error in Tab 2: {e}")

        # -----------------------------------------------------------
        # 3. Tab 3: Model Performance & Diagnostics
        # -----------------------------------------------------------
        print("Capturing Tab 3: Model Performance & Diagnostics...")
        try:
            page.mouse.wheel(0, -1500)
            time.sleep(0.5)
            tabs[2].click()
            time.sleep(2)
            page.screenshot(path="outputs/tab3_model_diagnostics_hour20.png")
        except Exception as e:
            print(f"Error in Tab 3: {e}")

        browser.close()
        print("All tab screenshots saved successfully in outputs/!")

if __name__ == "__main__":
    capture_dashboard()
