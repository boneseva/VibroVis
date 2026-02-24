# VibroVis UI Tests

This directory contains UI web tests for the VibroVis application using pytest and Playwright.

## Setup

### 1. Install Dependencies

First, install the Python testing dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Install Playwright Browsers

After installing the Python packages, you need to install the Playwright browsers:

```powershell
playwright install
```

Or for a specific browser:

```powershell
playwright install chromium
playwright install firefox
playwright install webkit
```

## Running Tests

### Run All Tests

```powershell
pytest
```

### Run Tests Verbosely

```powershell
pytest -v
```

### Run Specific Test Files

```powershell
pytest tests/test_ui_basic.py
pytest tests/test_ui_interactions.py
```

### Run Tests by Marker

Run only basic UI tests:
```powershell
pytest -m ui
```

Run only integration tests:
```powershell
pytest -m integration
```

Run tests excluding slow ones:
```powershell
pytest -m "not slow"
```

### Run Tests in a Specific Browser

By default, tests run in Chromium. To use a different browser:

```powershell
pytest --browser firefox
pytest --browser webkit
```

### Run Tests with Screenshots on Failure

```powershell
pytest --browser chromium --screenshot=only-on-failure
```

### Run Tests Headless or Headed

Tests run headless by default. To see the browser (useful for debugging):

Edit `tests/conftest.py` and change:
```python
browser = p[browser_type].launch(headless=True)
```
to:
```python
browser = p[browser_type].launch(headless=False)
```

## Test Structure

- `test_ui_basic.py` - Basic tests checking that components load and are visible
- `test_ui_interactions.py` - Tests for user interactions like clicking buttons and dropdowns
- `conftest.py` - Pytest fixtures for starting the app server and creating browser pages

## Test Markers

- `@pytest.mark.ui` - UI/web tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer to run

## Troubleshooting

### Server doesn't start

If tests fail because the server doesn't start:
- Make sure port 8050 (or the port found by the test) is not already in use
- Check that all dependencies are installed correctly
- Verify that `app.py` can run independently

### Browser not found

If you get errors about browsers not being found:
```powershell
playwright install --help  # See all options
playwright install chromium  # Install just Chromium
```

### Tests timeout

If tests timeout waiting for elements:
- The app might be loading slowly - try increasing timeout values in tests
- Check that your data cache exists (app loads faster with cache)
- Make sure the server started successfully

## Adding New Tests

1. Create a new test file in `tests/` directory, or add to existing files
2. Use the `app_page` fixture to get a page ready for testing
3. Use Playwright's `expect()` API for assertions
4. Add appropriate markers (`@pytest.mark.ui`, etc.)

Example:
```python
@pytest.mark.ui
def test_my_feature(app_page):
    """Test description."""
    element = app_page.locator('#my-element')
    expect(element).to_be_visible(timeout=10000)
    element.click()
    # ... more test code
```
