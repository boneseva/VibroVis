# Installing Test Dependencies

## Quick Installation Steps

### Step 1: Install Python Packages

```powershell
pip install -r requirements.txt
```

This will install:
- pytest (test framework)
- playwright (browser automation)

### Step 2: Install Playwright Browsers

After installing the Python packages, you **must** install the browser binaries:

```powershell
playwright install chromium
```

Or install all browsers:

```powershell
playwright install
```

### Step 3: Verify Installation

Check that playwright is installed:

```powershell
python -c "import playwright; print('Playwright installed!')"
```

Check that pytest is installed:

```powershell
pytest --version
```

### Step 4: Run Tests

Now you can run the tests:

```powershell
python run_tests.py
```

## Troubleshooting

### Error: "No module named 'playwright'"

You need to install the Python package:
```powershell
pip install playwright
```

Then install the browsers:
```powershell
playwright install chromium
```

### Error: "Executable doesn't exist"

You need to install the browser binaries:
```powershell
playwright install chromium
```

### All-in-One Command

To install everything at once:

```powershell
pip install -r requirements.txt && playwright install chromium
```

## What Gets Installed

- **pytest** - Test framework
- **playwright** - Browser automation library
- **Browser binaries** - Chromium, Firefox, or WebKit (you choose which to install)
