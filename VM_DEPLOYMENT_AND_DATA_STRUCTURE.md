# VibroVis / RanaVis — VM Deployment & Data Architecture Guide

This document defines how VibroVis / RanaVis is structured, how data must be laid out on the Virtual Machine (VM), why the observed errors occur, and how to verify or fix them.

---

## 1. Quick Summary of the Two Issues

### Issue 1: Broken Logo (`/assets/logo.svg`)
* **Root Cause**: The raw logo design assets were stored in `/logo/`, which was listed in `.gitignore`. The Dash frontend serves static assets from `/assets/`, but `/assets/logo.svg` had never been tracked in Git. When pulling the repository onto the VM, `/assets/logo.svg` was missing (404 Not Found).
* **Fix Implemented**: `assets/logo.svg` has been added and committed to the `ranavis` branch. Running `git push` locally and `git pull` on the VM resolves this immediately.

### Issue 2: "No data found for project 'Citimon', location '2026-06-C35', model 'perchv2'"
* **Root Cause**: `.gitignore` intentionally excludes the `/data/` directory because bioacoustic parquets and audio files are tens of gigabytes. Therefore, `git pull` **does not copy data files** to the VM.
* **Why the error appeared**: The dropdown menus and UI initialized properly for project `Citimon`, but when Dash attempted to read the underlying parquet data, the partition file `data/cache/partitions/Citimon/perchv2.parquet` was missing on the VM filesystem.

---

## 2. Required Data Directory Structure on the VM

The application resolves project data through `hierarchy.py` and `cache/reader.py`. For projects like `Citimon` and `Multimon` to load and stream audio, the `data/` directory must be structured on the VM as follows:

```
VibroVis/
├── assets/
│   ├── logo.svg                            # Application logo (tracked in Git)
│   ├── favicon.svg                         # Favicon (tracked in Git)
│   └── custom.css                          # Stylesheets
│
├── data/                                   # Ignored by Git (must be copied or mounted)
│   ├── cache/
│   │   ├── final_data.parquet              # (Optional) Legacy monolithic dataset fallback
│   │   └── partitions/                     # Primary partition cache
│   │       ├── Citimon/
│   │       │   └── perchv2.parquet         # Data for Citimon (contains locations like 2026-06-C35)
│   │       └── Multimon/
│   │           └── perchv2.parquet         # Data for Multimon (contains locations like Banjšice)
│   │
│   └── recordings/                         # Raw audio recordings for player/spectrogram streaming
│       ├── Citimon/
│       │   └── <Location>/                 # e.g., 2026-06-C35/
│       │       └── *.mp3 / *.wav
│       └── Multimon/
│           └── <Location>/                 # e.g., Banjšice/
│               └── *.mp3 / *.wav
│
└── app.py                                  # Application entrypoint
```

### Alternative: Self-Contained Project Directories
Alternatively, if projects are organized as independent project folders under `data/`:
```
data/
└── <ProjectName>/                          # e.g., data/Citimon/
    ├── project.json                        # Optional: {"name": "Citimon"}
    ├── cache/
    │   ├── manifest.json                   # Partition index
    │   └── partitions/
    │       └── <Location>/
    │           └── perchv2.parquet
    └── recordings/                         # Audio files
        └── <Location>/
            └── *.mp3
```

---

## 3. How to Transfer Data to the VM

Since `data/` is not in Git, choose one of the following methods to populate data on the VM:

### Option A: Direct Copy from Local Machine via SCP or RSYNC (Recommended)
From your local terminal, transfer the cached partitions to the VM:
```bash
# 1. Transfer the partitioned parquet caches:
scp -r data/cache/partitions user@<VM_IP>:/path/to/VibroVis/data/cache/

# 2. Transfer audio recordings (if needed for audio playback):
scp -r data/recordings user@<VM_IP>:/path/to/VibroVis/data/
```

Or using `rsync` with progress:
```bash
rsync -avzP data/cache/partitions/ user@<VM_IP>:/path/to/VibroVis/data/cache/partitions/
rsync -avzP data/recordings/ user@<VM_IP>:/path/to/VibroVis/data/recordings/
```

### Option B: Build Partitions on the VM
If the raw overview TSVs or audio files exist on the VM, build the cache partitions directly:
```bash
python build_cache.py --project Citimon --output data/cache/partitions/Citimon
python build_cache.py --project Multimon --output data/cache/partitions/Multimon
```

---

## 4. How to Run on the VM

### Production Server (Linux / Docker / VM)
Follow the production architecture defined in `AGENTS.md`:
```bash
gunicorn -w 1 --threads 20 --worker-class gthread -b 0.0.0.0:8080 --timeout 60 app:server
```
* **`-w 1` (Mandatory)**: VibroVis / RanaVis utilizes in-memory caches (`MODEL_DATA_CACHE`, `server_cache`, `MANUAL_LABELS_CACHE`). Running multiple worker processes causes split-brain state.
* **`--threads 20`**: Serves concurrent audio streaming and Dash callbacks.
* **`--timeout 60`**: Accommodates partition reads and large figure serializations.

### Development Server
```bash
python app.py
```
Default URL: `http://127.0.0.1:8050/` or `http://0.0.0.0:8050/`.

---

## 5. Diagnostic Commands for the VM Agent

When an AI agent runs on the VM to verify or fix the setup, the agent should run these checks in order:

### 1. Verify Git Status and Assets
```bash
git branch
# Must be on 'ranavis'

git pull
# Ensure latest commit with assets/logo.svg is pulled

ls -la assets/logo.svg
# Must exist and be ~6KB
```

### 2. Verify Data Cache Files on Disk
```bash
ls -la data/cache/partitions/Citimon/
# Must show perchv2.parquet (~1.3M rows)

ls -la data/cache/partitions/Multimon/
# Must show perchv2.parquet (~8.6M rows)
```

### 3. Verify Project and Data Resolution via Python
```python
# Run via python -c "..."
import hierarchy, read_data

# Step A: Check project discovery
projects = hierarchy.get_available_projects()
print("Discovered projects:", [p.id for p in projects])
# Expected: ['Citimon', 'Multimon']

# Step B: Check metadata discovery
citimon_meta = read_data.get_project_metadata_summary('Citimon')
print("Citimon locations count:", len(citimon_meta['locations']))
print("First 3 Citimon locations:", citimon_meta['locations'][:3])

# Step C: Load test partition row count
df = read_data.load_project_model_data('Citimon', 'perchv2', '2026-06-C35')
print("Loaded Citimon rows:", len(df))
# Expected: > 0 rows
```

### 4. Run the Regression Test Suite
```bash
python -m unittest tests/test_project_hierarchy_sync.py
# Expected: Ran 8 tests in ~15s -> OK
```
