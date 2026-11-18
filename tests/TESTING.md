# Running UI Tests for VibroVis

This guide explains how to set up and run the UI web tests for VibroVis.

## Quick Start

### Step 1: Install Testing Dependencies

```powershell
pip install -r requirements.txt
```

This will install pytest, playwright, and other testing tools.

### Step 2: Install Playwright Browsers

After installing Python packages, you need to install the browser binaries:

```powershell
playwright install
```

Or just install Chromium (recommended for faster setup):

```powershell
playwright install chromium
```

### Step 3: Run the Tests

**Option A: Using the test runner script (recommended)**

```powershell
python run_tests.py
```

**Option B: Using pytest directly**

```powershell
pytest
```

**Option C: Run tests with more output**

```powershell
pytest -v
```

## Test Files Created

All test files have been created:

✅ `tests/test_ui_basic.py` - Tests that check if components load correctly  
✅ `tests/test_ui_interactions.py` - Tests for user interactions  
✅ `tests/conftest.py` - Test configuration and fixtures  
✅ `tests/README.md` - Detailed test documentation  
✅ `pytest.ini` - Pytest configuration  
✅ `run_tests.py` - Easy-to-use test runner script

## Common Test Commands

### Run All Tests
```powershell
pytest
# or
python run_tests.py
```

### Run Only Basic Tests
```powershell
python run_tests.py --basic-only
```

### Run Only Interaction Tests
```powershell
python run_tests.py --interactions-only
```

### Run Tests in Different Browser
```powershell
python run_tests.py --browser firefox
```

### Skip Slow Tests
```powershell
python run_tests.py --no-slow
```

### Run Specific Test File
```powershell
pytest tests/test_ui_basic.py
```

### Run with Verbose Output
```powershell
pytest -v -s
```

## What the Tests Check

### Basic Tests (`test_ui_basic.py`)
- ✅ App loads successfully
- ✅ Scatter plot is visible
- ✅ Filter panel is visible
- ✅ All dropdowns exist (location, model, clusters, etc.)
- ✅ Histogram is visible
- ✅ Audio player exists
- ✅ Spectrogram plot exists
- ✅ Preset dropdown exists
- ✅ Autoplay button exists

### Interaction Tests (`test_ui_interactions.py`)
- ✅ Location dropdown can be clicked
- ✅ Model dropdown works
- ✅ Autoplay button toggles
- ✅ Filter sections expand/collapse
- ✅ Toggle filters button works
- ✅ Preset dropdown is clickable
- ✅ Scatter plot is interactive
- ✅ Histogram type dropdown works
- ✅ Resample button works
- ✅ Complete user flow

## Troubleshooting

### Error: "playwright not found"
```powershell
pip install playwright
playwright install chromium
```

### Error: "Server did not start"
- Make sure port 8050 (or the auto-detected port) is not in use
- Check that all app dependencies are installed
- Verify `app.py` can run manually

### Tests timeout waiting for elements
- The app might be loading slowly
- Check that your data cache exists (app loads faster with cache)
- Increase timeout in test files if needed

### Want to see the browser while testing?
Edit `tests/conftest.py` line 76 and change:
```python
browser = p[browser_type].launch(headless=True)
```
to:
```python
browser = p[browser_type].launch(headless=False)
```

## Running Tests Regularly

You can run tests whenever you make changes:

```powershell
# Quick check - basic tests only
python run_tests.py --basic-only

# Full test suite
python run_tests.py

# Just verify app loads
pytest tests/test_ui_basic.py::test_app_loads -v
```

## Adding More Tests

To add new tests:

1. Edit `tests/test_ui_basic.py` or `tests/test_ui_interactions.py`
2. Or create a new file like `tests/test_my_feature.py`
3. Use the `app_page` fixture to get a browser page
4. Use Playwright's `expect()` API for assertions

Example:
```python
@pytest.mark.ui
def test_my_feature(app_page):
    """Test my new feature."""
    element = app_page.locator('#my-element')
    expect(element).to_be_visible(timeout=10000)
```

## Next Steps

- Run the tests now: `python run_tests.py`
- Add more specific tests for your features
- Integrate tests into CI/CD pipeline (optional)
- Add tests for specific user workflows
