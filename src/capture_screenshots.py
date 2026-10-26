"""
Automated screenshot capture tool using Playwright.
Captures genuine, high-resolution screenshots of all 6 core Streamlit pages.
Saves them into docs/images/ for inclusion in the academic reports.
"""

import os
import time
from playwright.sync_api import sync_playwright


def capture_all_screenshots():
    os.makedirs("docs/images", exist_ok=True)
    url = "http://localhost:8501"

    pages_to_capture = [
        ("1. Project Overview", "docs/images/01_overview.png"),
        ("2. Finite MDP Design", "docs/images/02_mdp_design.png"),
        ("3. Live Traffic Simulation", "docs/images/03_live_simulation.png"),
        ("4. RL Training Dashboard", "docs/images/04_rl_training.png"),
        ("5. Bellman Equation Explorer", "docs/images/05_bellman_explorer.png"),
        ("6. Performance Comparison", "docs/images/06_performance_comparison.png")
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Consistent desktop viewport
        context = browser.new_context(viewport={"width": 1366, "height": 850})
        page = context.new_page()

        print(f"Navigating to {url}...")
        page.goto(url, wait_until="networkidle")
        time.sleep(3)  # Allow Streamlit components to stabilize

        for label, out_path in pages_to_capture:
            print(f"Navigating to page: '{label}'...")
            try:
                # Find and click the radio option in the sidebar
                page.locator(f'label:has-text("{label}")').click()
                time.sleep(2.5)  # Wait for page re-render

                # If on live simulation, click step button once to show active cars
                if "Live Traffic" in label:
                    step_btn = page.locator('button:has-text("Step Forward (5 Intervals)")')
                    if step_btn.count() > 0:
                        step_btn.first.click()
                        time.sleep(1.5)

                page.screenshot(path=out_path, full_page=False)
                print(f"  Captured: {out_path}")
            except Exception as e:
                print(f"  Error capturing {label}: {e}")

        browser.close()
    print("All screenshots successfully captured!")


if __name__ == "__main__":
    capture_all_screenshots()
