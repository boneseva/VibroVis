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
If the master TSV (`data/zabe+hyla+citimon.tsv`) and the positions TSVs exist on the VM, build a partition directly.
The project must be `zabe` (it is the recording-ID prefix that Raven links use) and `--data-dir data` points at the
local layout:
```bash
python build_cache.py --partitioned --project zabe --data-dir data --location Citimon
python build_cache.py --partitioned --project zabe --data-dir data --location Multimon
```
The builder keeps the previous partition if anything fails (for Citimon also when a folder is not in
`config/citimon_registry.json`). See section 7 before rebuilding Citimon.

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
# Must show perchv2.parquet (~4.35M rows, ~150 MB after the hierarchy rebuild)

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

# Step C: Load test partition row count (Citimon locations are recorders, e.g. 'C35')
df = read_data.load_project_model_data('Citimon', 'perchv2', ['C35'])
print("Loaded Citimon rows:", len(df))
# Expected: > 0 rows
```

### 4. Run the Regression Test Suite
```bash
python -m unittest tests/test_project_hierarchy_sync.py
# Expected: Ran 8 tests in ~15s -> OK
```

---

## 6. Raven Annotation Database (Citimon imports)

Raven selection tables imported through **Labels → Import Raven annotations…** (shown only for the Citimon
project) are stored in their own SQLite database. It is separate from the manual-label store
(`cache/manual_labels/`), which the importer never reads or writes.

| Item | Value |
|---|---|
| Default file | `cache/annotations/raven_annotations.sqlite3` (plus `-wal` / `-shm` while running) |
| Override | `VIBROVIS_ANNOTATIONS_DIR=/srv/vibrovis/annotations` |
| Git / Docker | `/cache/annotations/` is in `.gitignore`; `cache/annotations/` is in `.dockerignore` |
| Schema version | Stored in table `schema_meta` (currently 1). Newer code migrates older files on first open; older code refuses a newer file instead of corrupting it. |

### Persistent storage (required)
The database must live outside the container image and outside the Git checkout's tracked files:
```bash
sudo mkdir -p /srv/vibrovis/annotations
# Docker:
docker run ... -v /srv/vibrovis/annotations:/app/cache/annotations ... <image>
# or bare gunicorn on the VM:
export VIBROVIS_ANNOTATIONS_DIR=/srv/vibrovis/annotations
```
Keep the single-worker gunicorn setting (`-w 1`); uploads awaiting confirmation are held in that worker's memory.

### Backup
Use the online backup command; it is safe while the app is running (do not copy the `.sqlite3` file alone while
the `-wal` file exists):
```bash
python -m raven_import.cli backup /srv/vibrovis/backups/raven_$(date +%Y%m%d_%H%M%S).sqlite3
# e.g. nightly cron:
# 15 2 * * * cd /path/to/VibroVis && VIBROVIS_ANNOTATIONS_DIR=/srv/vibrovis/annotations \
#   python -m raven_import.cli backup /srv/vibrovis/backups/raven_$(date +\%Y\%m\%d).sqlite3
```
Each import batch also stores the original uploaded file bytes, so a backup is self-contained.

### Restore
1. Stop the app.
2. Move the current `raven_annotations.sqlite3`, `-wal` and `-shm` files aside.
3. Copy the chosen backup to `$VIBROVIS_ANNOTATIONS_DIR/raven_annotations.sqlite3`.
4. Start the app and check with `python -m raven_import.cli list`.

### Other commands
```bash
python -m raven_import.cli dry-run FILE.txt --annotator NAME [--fixed-location LOC]   # no writes
python -m raven_import.cli list
python -m raven_import.cli revert BATCH_ID --reason "..."   # marks inactive; data kept
```

### CITI-MON delivery tables (`<location>-<annotator>[-hitro].txt`)
Location, annotator ID and the quick-pass flag (`-hitro`) come from the table file name. Unpadded dates in
`Begin File` (e.g. `2026223_122000.WAV`) are repaired only when `Begin Date` confirms them; selections running
past the real end of the recording are truncated and flagged, keeping the original end. Aliases are exact.
```bash
python -m raven_import.cli import CITI-MON-anotacije/*.txt --citimon-delivery \
    --alias 2026_03-r26=2026_03-R26 --alias 2026_03-r28=2026_03-R28 --note "delivery YYYY-MM-DD" --yes
```
Re-running the same command is safe: identical tables are skipped.

### After the Citimon partition is rebuilt
The Citimon hierarchy rebuild (section 7) keeps every recording ID and clip key, so **no relink is needed or
wanted** for it. Relink only after a rebuild that changes recording IDs or clip times. Stored selections are re-matched
from the stored original files, so the biologists' tables are not needed again:
```bash
python -m raven_import.cli relink --dry-run   # show before/after status counts
python -m raven_import.cli relink             # replace derived links; logged as 'relinked' events
```

---

## 7. Deploying the Citimon hierarchy, location maps and Raven display changes (2026-11)

What changes on the VM:
* Citimon locations become **recorders** (`C08`) with **microlocations** (`C08-P1`), defined in
  `config/citimon_registry.json`. Duplicate folders `2026_03-R35`, `2026-06-C35` and `2026_02-C49` are excluded.
* The location dropdown is multi-select; clicking the small map opens a map for choosing locations
  (Citimon and Multimon; Multimon positions in `config/multimon_locations.json`).
* Raven colouring needs ≥ 1 s overlap or ≥ 50 % of the selection; spectrogram axes show file time.
* The Raven annotation database is **not** modified, re-imported or relinked: recording IDs stay
  `zabe::Citimon/<original folder>/...`, so all existing links keep working.

The new code also runs with the **old** Citimon partition (locations are then the old folder names); the recorder
hierarchy appears once the new partition is in place.

### Step 1 — Back up the VM state (before anything else)
```bash
cd /path/to/VibroVis
git status                       # note any local changes on the VM; do not discard them
mkdir -p ~/vibrovis_backup_$(date +%Y%m%d)
cp -p data/cache/manifest.json data/cache/partitions/Citimon/perchv2.parquet ~/vibrovis_backup_$(date +%Y%m%d)/
python -m raven_import.cli backup ~/vibrovis_backup_$(date +%Y%m%d)/raven_annotations.sqlite3
```

### Step 2 — Update the code
```bash
git fetch && git checkout ranavis && git pull
```
`config/citimon_registry.json` and `config/multimon_locations.json` come with the code. Requirements now pin
`dash>=2.16` (for the loading indicator); rebuild the Docker image or `pip install -r requirements.txt`.

### Step 3 — Audit (read-only) and send the report
```bash
python3 tools/vm_audit.py > vm_audit_report.txt
```
Check in section 8: no **UNREGISTERED** Citimon folders (add them to the registry first, otherwise a rebuild is
blocked), the master TSV location, free memory/disk and the Dash version.

### Step 4 — Get the new Citimon partition (choose one)
**A. Copy the partition built on the workstation** (fast; valid only if the VM has the same Citimon folders and
recordings). Stage it first and check it against the VM's Raven links before replacing anything:
```bash
# on the workstation
scp data/cache/partitions/Citimon/perchv2.parquet data/cache/manifest.json user@<VM>:/tmp/vibrovis_staging/
# on the VM
python3 tools/vm_audit.py --partition /tmp/vibrovis_staging/perchv2.parquet > vm_audit_staged.txt
```
Proceed only if section 3b reports all Raven link keys and linked recordings present, all recording IDs start with
`zabe::Citimon/`, and no (or only expected) missing audio files. Then stop the app and copy both files into
`data/cache/partitions/Citimon/` and `data/cache/` respectively.

**B. Rebuild on the VM** (needs the master TSV and positions TSVs; about 1–1.5 h and 4–6 GiB RAM):
```bash
python -u build_cache.py --partitioned --project zabe --data-dir data --location Citimon > rebuild_citimon.log 2>&1
```

### Step 5 — Restart and check
Restart the app (Docker container or gunicorn, still `-w 1`). In the browser: Citimon → the location dropdown lists
recorders, microlocations list `Cxx-Pn`, the map highlights the selection and opens the location picker when clicked;
Raven colouring still shows. **Do not run `raven_import.cli relink`** for this change.

### Rollback
Stop the app, copy `manifest.json` and `perchv2.parquet` back from the Step 1 backup, `git checkout` the previous
commit, restart. The Raven database was never changed, so it needs no rollback.
