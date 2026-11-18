"""
Pytest configuration and fixtures for UI web tests.
"""
import os
import sys
import time
import threading
import pytest
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, server


@pytest.fixture(scope="session")
def app_server():
    """
    Start the Dash app server in a separate thread for testing.
    Returns the base URL where the app is running.
    """
    import socket
    
    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    
    # Configure the app for testing
    app.config.suppress_callback_exceptions = True
    
    # Start server in a separate thread
    def run_server():
        app.run_server(port=port, host='127.0.0.1', debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            import urllib.request
            urllib.request.urlopen(base_url, timeout=1)
            break
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(0.5)
            else:
                pytest.fail(f"Server did not start within {max_attempts * 0.5} seconds")
    
    yield base_url
    
    # Cleanup is handled by daemon thread


@pytest.fixture(scope="function")
def page(pytestconfig):
    """
    Create a new browser page for each test.
    This requires playwright to be installed and browsers downloaded.
    """
    from playwright.sync_api import sync_playwright
    
    browser_type = pytestconfig.getoption("--browser", default="chromium")
    
    with sync_playwright() as p:
        browser = p[browser_type].launch(headless=True)
        page = browser.new_page()
        
        yield page
        
        browser.close()


@pytest.fixture(scope="function")
def app_page(app_server, page):
    """
    Navigate to the app URL and wait for it to load.
    Returns a page ready for testing.
    """
    page.goto(app_server, wait_until="networkidle")
    # Wait a bit more for Dash to fully initialize
    page.wait_for_timeout(1000)
    
    yield page


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--browser",
        action="store",
        default="chromium",
        help="Browser to use for tests: chromium, firefox, or webkit"
    )
