"""
vm_audit.py - READ-ONLY audit of a VibroVis host before deploying the Citimon hierarchy (v3, 2026-11-11).

    cd /path/to/VibroVis && python3 tools/vm_audit.py > vm_audit_report.txt
    # optionally check a staged copy of a new Citimon partition against this host's Raven links:
    python3 tools/vm_audit.py --partition /tmp/staging/perchv2.parquet > vm_audit_report.txt

Guarantees
  * Writes nothing: no files, no directories, no caches. Output goes to stdout only.
  * Does NOT import the application (importing callbacks would create diskcache directories).
  * SQLite files are opened with mode=ro&immutable=1 (no -wal/-shm created, no locks taken).
    If a database has a non-empty -wal file, the report says so: the immutable read then does not include
    the uncheckpointed pages; take an online copy with `python -m raven_import.cli backup <dest>` and audit it.
  * Respects VIBROVIS_MANUAL_LABELS_DIR, VIBROVIS_CLIP_DURATIONS_DIR, VIBROVIS_ANNOTATIONS_DIR.
  * Skips recordings/, positions/, embeddings_positions/ and partitions/ when scanning for text references.
Optional: pyarrow for parquet checks (reported as unavailable otherwise).
"""
import collections, glob, hashlib, json, os, pickle, re, sqlite3, sys

HERE = os.getcwd()
STAGED = sys.argv[sys.argv.index("--partition") + 1] if "--partition" in sys.argv else None
CITI = re.compile(r"^\d{4}[_-]\d{2}-[A-Za-z]\d+(?:-\d+)?$")
HISTORICAL = ("2026_03-C19-1", "2026_03-C19-2", "2026_03-R35", "2026_03-r35")
CURRENT = ("2026_03-C19", "2026_04-C19", "2026_03-C35")
TEXT_EXT = (".json", ".tsv", ".csv", ".txt", ".md")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "recordings", "embeddings_positions",
             "positions", "partitions"}   # positions/: ~10^5 per-file TSVs; partitions/: parquet only


def walk_files(top=".", pred=lambda fn: True):
    """os.walk that never descends into SKIP_DIRS (recordings/embeddings hold ~10^5 files)."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if pred(fn):
                yield os.path.join(dirpath, fn)


def section(t):
    print(f"\n== {t}")


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro&immutable=1", uri=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wal_note(db):
    wal = db + "-wal"
    if os.path.isfile(wal) and os.path.getsize(wal) > 0:
        print(f"  NOTE: {wal} is non-empty ({os.path.getsize(wal)} bytes); immutable read may miss recent writes.")


def diskcache_report(name, directory):
    db = os.path.join(directory, "cache.db")
    if not os.path.isfile(db):
        print(f"  [{name}] {directory}: no cache.db"); return
    wal_note(db)
    try:
        con = ro(db)
        integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        keys = []
        for (k,) in con.execute("SELECT key FROM Cache"):
            if isinstance(k, bytes):
                try: k = pickle.loads(k)
                except Exception: pass
            keys.append(k)
    except Exception as e:
        print(f"  [{name}] {directory}: unreadable ({e})"); return
    locs = collections.Counter(str(k[0]) for k in keys if isinstance(k, tuple) and k)
    citi = {l: c for l, c in locs.items() if CITI.match(l)}
    print(f"  [{name}] {directory}: integrity={integrity} keys={len(keys)} citimon_keys={sum(citi.values())} citimon_locations={len(citi)}")
    for l in HISTORICAL:
        if l in locs: print(f"    HISTORICAL {l}: {locs[l]} keys")
    for l, c in sorted(citi.items()):
        print(f"    {l}: {c}")


print(f"vm_audit v3  cwd={HERE}  python={sys.version.split()[0]}")
print("env:", {k: v for k, v in os.environ.items() if k.startswith("VIBROVIS_")})

section("1. Citimon source folders (data/recordings/Citimon)")
root = os.path.join("data", "recordings", "Citimon")
if os.path.isdir(root):
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p): continue
        files = os.listdir(p)
        mp3 = sorted(f for f in files if f.lower().endswith(".mp3"))
        posd = os.path.join(p, "positions")
        pos = len(os.listdir(posd)) if os.path.isdir(posd) else 0
        span = f"{mp3[0][:15]}..{mp3[-1][:15]}" if mp3 else "-"
        tag = " [HISTORICAL NAME]" if d in HISTORICAL else (" [CURRENT NAME]" if d in CURRENT else "")
        print(f"  {d}: mp3={len(mp3)} positions_tsv={pos} span={span}{tag}")
else:
    print("  MISSING:", root)

section("2. Cache manifest and Citimon partition")
man = os.path.join("data", "cache", "manifest.json")
part = None
if os.path.isfile(man):
    j = json.load(open(man, encoding="utf-8"))
    print("  manifest project:", j.get("project"), "sha256:", sha256(man))
    for p in j.get("partitions", []):
        if p.get("location") == "Citimon":
            print("  ", {k: p.get(k) for k in ("partition_rel_path", "row_count", "signature", "date_extent")})
            part = os.path.join("data", p.get("partition_rel_path") or "")
part = part if part and os.path.isfile(part) else os.path.join("data", "cache", "partitions", "Citimon", "perchv2.parquet")
if os.path.isfile(part):
    print("  partition:", part, "size:", os.path.getsize(part), "sha256:", sha256(part))
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(part, columns=["location", "microlocation", "recording_id"]).to_pandas()
        print("  rows:", len(t), " recordings:", t["recording_id"].nunique(),
              " location==microlocation for all rows:", bool((t["location"] == t["microlocation"]).all()))
        rc = t.drop_duplicates("recording_id")["location"].value_counts()
        for loc, n in sorted(t["location"].value_counts().items()):
            print(f"    {loc}: rows={n} recordings={rc.get(loc, 0)}" + (" [HISTORICAL]" if loc in HISTORICAL else ""))
        rid = t.drop_duplicates("recording_id")
        rel = rid["recording_id"].str.split("::").str[1]
        miss = [r for r in rel if not os.path.isfile(os.path.join("data", "recordings", *r.split("/")))]
        by = collections.Counter(m.split("/")[1] for m in miss)
        print("  cached recordings whose audio path is missing:", len(miss), dict(by))
    except ImportError:
        print("  pyarrow not available: parquet not inspected")
else:
    print("  MISSING partition:", part)

section("3. Raven annotation database")
adir = os.environ.get("VIBROVIS_ANNOTATIONS_DIR") or os.path.join(HERE, "cache", "annotations")
db = os.path.join(adir, "raven_annotations.sqlite3")
if os.path.isfile(db):
    print("  path:", db, "sha256:", sha256(db))
    wal_note(db)
    c = ro(db)
    q = lambda s, a=(): c.execute(s, a).fetchall()
    print("  integrity:", q("PRAGMA quick_check")[0][0], " schema_version:", q("SELECT value FROM schema_meta WHERE key='schema_version'"))
    print("  import_batches total/active:", q("SELECT COUNT(*), SUM(status='active') FROM import_batches")[0])
    print("  annotators (batches):", q("SELECT a.name, COUNT(*) FROM import_batches b JOIN annotators a USING(annotator_id) GROUP BY 1"))
    print("  selections total:", q("SELECT COUNT(*) FROM selections")[0][0])
    print("  selections by status:", q("SELECT status, COUNT(*) FROM selections GROUP BY 1 ORDER BY 2 DESC"))
    print("  selections with recording_id:", q("SELECT COUNT(*) FROM selections WHERE recording_id IS NOT NULL")[0][0])
    print("  selections with >=1 clip link:", q("SELECT COUNT(DISTINCT selection_pk) FROM selection_clip_links")[0][0])
    print("  clip links total/distinct keys/distinct recordings:",
          q("SELECT COUNT(*), COUNT(DISTINCT semantic_clip_key), COUNT(DISTINCT recording_id) FROM selection_clip_links")[0])
    print("  links per model/signature:", q("SELECT model_name, partition_signature, COUNT(*) FROM selection_clip_links GROUP BY 1,2"))
    folders = collections.Counter(r[0].split("::")[1].split("/")[1] for r in q("SELECT recording_id FROM selection_clip_links"))
    print("  link folders:", dict(sorted(folders.items())))
    for h in HISTORICAL:
        n1 = q("SELECT COUNT(*) FROM selection_clip_links WHERE recording_id LIKE ?", (f"%/{h}/%",))[0][0]
        n2 = q("SELECT COUNT(*) FROM selections WHERE recording_id LIKE ? OR location=?", (f"%/{h}/%", h))[0][0]
        if n1 or n2: print(f"  HISTORICAL {h}: links={n1} selections={n2}")
    print("  original table names (source_filename | annotator | identity_scope | status | selections | links):")
    for r in q("""SELECT b.source_filename, a.name, b.identity_scope, b.status,
                  (SELECT COUNT(*) FROM selections s WHERE s.batch_id=b.batch_id),
                  (SELECT COUNT(*) FROM selection_clip_links l JOIN selections s USING(selection_pk) WHERE s.batch_id=b.batch_id)
                  FROM import_batches b JOIN annotators a USING(annotator_id) ORDER BY 1"""):
        print("    ", " | ".join(str(x) for x in r))
    if os.path.isfile(part):
        try:
            import pyarrow.parquet as pq
            keys = [r[0] for r in q("SELECT DISTINCT semantic_clip_key FROM selection_clip_links")]
            found = pq.read_table(part, columns=["semantic_clip_key"], filters=[("semantic_clip_key", "in", keys)]).column(0).to_pylist()
            print(f"  link keys present in Citimon partition: {len(set(found))}/{len(keys)}")
        except Exception as e:
            print("  link-key check skipped:", e)
else:
    print("  no DB at", db)
for b in sorted(walk_files(HERE, lambda fn: fn.startswith("raven_") and fn.endswith(".sqlite3"))):
    if os.path.abspath(b) != os.path.abspath(db):
        try:
            n = ro(b).execute("SELECT COUNT(*) FROM import_batches").fetchone()[0]
            print("  other Raven DB/backup:", b, "batches:", n, "integrity:", ro(b).execute("PRAGMA quick_check").fetchone()[0])
        except Exception as e:
            print("  other Raven DB/backup:", b, "unreadable:", e)

section("3b. Staged Citimon partition (--partition)")
if STAGED:
    if not os.path.isfile(STAGED):
        print("  MISSING:", STAGED)
    else:
        print("  staged:", STAGED, "size:", os.path.getsize(STAGED), "sha256:", sha256(STAGED))
        try:
            import pyarrow.parquet as pq
            cols = set(pq.read_schema(STAGED).names)
            print("  has hierarchy columns (source_folder):", "source_folder" in cols)
            st = pq.read_table(STAGED, columns=["recording_id", "semantic_clip_key", "location"]).to_pandas()
            print("  rows:", len(st), " recordings:", st["recording_id"].nunique(), " locations (recorders):", st["location"].nunique())
            print("  all recording IDs start with 'zabe::Citimon/':", bool(st["recording_id"].str.startswith("zabe::Citimon/").all()))
            if os.path.isfile(db):
                c2 = ro(db)
                links = c2.execute("SELECT DISTINCT semantic_clip_key, recording_id FROM selection_clip_links").fetchall()
                keys, rids = set(st["semantic_clip_key"]), set(st["recording_id"])
                print(f"  Raven link keys present in STAGED partition: {sum(k in keys for k, _ in links)}/{len(links)}")
                print(f"  Raven linked recordings present in STAGED partition: "
                      f"{len({r for _, r in links if r in rids})}/{len({r for _, r in links})}")
                missing = sorted({r for _, r in links if r not in rids})[:10]
                if missing:
                    print("  first missing recordings:", missing)
            rel = st.drop_duplicates("recording_id")["recording_id"].str.split("::").str[1]
            miss = [r for r in rel if not os.path.isfile(os.path.join("data", "recordings", *r.split("/")))]
            print("  staged recordings whose audio file is missing on this host:", len(miss),
                  dict(collections.Counter(m.split("/")[1] for m in miss)))
        except ImportError:
            print("  pyarrow not available: staged parquet not inspected")
else:
    print("  (not requested)")

section("4. Clip-duration metadata, manual labels (diskcache) and backups")
diskcache_report("clip_durations", os.environ.get("VIBROVIS_CLIP_DURATIONS_DIR") or os.path.join(HERE, "cache", "clip_durations"))
diskcache_report("manual_labels", os.environ.get("VIBROVIS_MANUAL_LABELS_DIR") or os.path.join(HERE, "cache", "manual_labels"))
for b in sorted(glob.glob(os.path.join(HERE, "cache", "backups", "*"))):
    if os.path.isdir(b): diskcache_report("backup", b)

section("5. Presets")
for f in sorted(glob.glob("presets/*.json")):
    try:
        j = json.load(open(f, encoding="utf-8"))
        loc, micro = j.get("location"), j.get("microlocation") or []
        citi = CITI.match(str(loc or "")) or any(CITI.match(str(m)) for m in micro)
        print(f"  {f}: location={loc} microlocation={micro}" + ("  [CITIMON]" if citi else ""))
    except Exception as e:
        print(f"  {f}: unreadable ({e})")

section("6. Legacy saved-label stores")
for f in sorted(walk_files(".", lambda fn: fn.startswith("saved_labels") and ".parquet" in fn)):
    print("  ", f, os.path.getsize(f), "bytes")

section("7. Text files referencing historical names (exports, TSVs, configs)")
pat = re.compile("|".join(re.escape(h) for h in HISTORICAL))
if True:
    for p in sorted(walk_files(".", lambda fn: fn.lower().endswith(TEXT_EXT))):
        try:
            if os.path.getsize(p) > 500 * 1024 * 1024: continue
            n = 0
            with open(p, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if pat.search(line): n += 1
            if n: print(f"  {p}: {n} lines")
        except OSError:
            pass

section("8. Deployment readiness (Citimon hierarchy, Multimon map)")
reg_path = os.environ.get("VIBROVIS_CITIMON_REGISTRY") or os.path.join("config", "citimon_registry.json")
reg = None
if os.path.isfile(reg_path):
    try:
        reg = json.load(open(reg_path, encoding="utf-8"))
        print("  registry:", reg_path, "sha256:", sha256(reg_path), "recorders:", len(reg.get("recorders", {})),
              "microlocations:", len(reg.get("microlocations", {})), "source_folders:", len(reg.get("source_folders", {})))
    except Exception as e:
        print("  registry unreadable:", e)
else:
    print("  MISSING registry:", reg_path, "(git pull the ranavis branch)")
if reg is not None and os.path.isdir(root):
    known = set(reg.get("source_folders", {}))
    for entry in reg.get("source_folders", {}).values():
        known.update(entry.get("aliases", []) or [])
    excluded = set(reg.get("excluded_source_folders", []))
    on_disk = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    norm = lambda x: x.replace("-", "_").casefold()
    known_n = {norm(k) for k in known}
    unregistered = [d for d in on_disk if d not in known and norm(d) not in known_n and d not in excluded]
    print("  Citimon folders on disk:", len(on_disk), " excluded (skipped by the builder):",
          sorted(d for d in on_disk if d in excluded))
    print("  UNREGISTERED folders (would block a Citimon rebuild until added to the registry):", unregistered or "none")
    print("  registry folders not on this host:", sorted(k for k in reg.get("source_folders", {})
                                                         if k not in on_disk and k not in excluded) or "none")
masters = [p for p in ("data/zabe+hyla+citimon.tsv", "data/zabe/zabe+hyla+citimon.tsv") if os.path.isfile(p)]
print("  master TSV (zabe+hyla+citimon.tsv):", masters or "MISSING (needed only for a rebuild on this host)")
mm = os.environ.get("VIBROVIS_MULTIMON_LOCATIONS") or os.path.join("config", "multimon_locations.json")
print("  Multimon map config:", mm, "present" if os.path.isfile(mm) else "MISSING (git pull the ranavis branch)")
try:
    import shutil
    print("  free disk in data/:", round(shutil.disk_usage("data" if os.path.isdir("data") else ".").free / 2**30, 1), "GiB")
except Exception:
    pass
try:
    with open("/proc/meminfo") as fh:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in fh if l.split(":")[0] in ("MemTotal", "MemAvailable")}
    print("  memory total/available:", {k: f"{v / 2**20:.1f} GiB" for k, v in mem.items()},
          "(a Citimon rebuild needs roughly 4-6 GiB)")
except OSError:
    pass
try:
    import dash
    print("  dash version:", dash.__version__, "(>= 2.16 needed)")
except ImportError:
    print("  dash not importable in this Python (fine if the app runs in Docker)")

print("\nBrowser localStorage ('local-labels-store') is per-browser and cannot be audited from the server.")
print("END (no files were written)")
