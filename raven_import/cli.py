"""
Command-line helpers for Raven imports.

    python -m raven_import.cli dry-run FILE [FILE ...] [import options]       # never writes
    python -m raven_import.cli import  FILE [FILE ...] [import options] --yes # writes one transaction
    python -m raven_import.cli relink [--batch ID ...] [--model M] [--dry-run]
    python -m raven_import.cli list
    python -m raven_import.cli backup DEST.sqlite3
    python -m raven_import.cli revert BATCH_ID [--reason TEXT]

Import options: --annotator NAME | --annotator-from-table-name, --location-mode {path,fixed,table_name},
--fixed-location LOC, --alias FROM=TO (repeatable), --truncate-at-end, --no-date-repair,
--primary-label COLUMN, --time-mode, --encoding, --model, --data-dir.
Citimon delivery preset: --citimon-delivery (table-name location + annotator, truncation, primary 'Annotation').
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .clip_index import PartitionClipSource
from .config import ENCODINGS, LOCATION_MODES, TIME_MODES, ImportOptions
from .service import commit_import, preview_import, relink_batches
from .store import AnnotationStore


def _add_import_args(p):
    p.add_argument("files", nargs="+")
    p.add_argument("--annotator", default="")
    p.add_argument("--annotator-from-table-name", action="store_true")
    p.add_argument("--model", default="perchv2")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--time-mode", choices=TIME_MODES, default="auto")
    p.add_argument("--location-mode", choices=LOCATION_MODES, default="path")
    p.add_argument("--fixed-location", default=None)
    p.add_argument("--alias", action="append", default=[], metavar="FROM=TO")
    p.add_argument("--truncate-at-end", action="store_true")
    p.add_argument("--no-date-repair", action="store_true")
    p.add_argument("--primary-label", default=None)
    p.add_argument("--encoding", choices=ENCODINGS, default="auto")
    p.add_argument("--file-from-table-name", action="store_true")
    p.add_argument("--citimon-delivery", action="store_true",
                   help="Preset for '<location>-<annotator>[-hitro].txt' tables with an 'Annotation' column")
    p.add_argument("--note", default=None)


def _options(args) -> ImportOptions:
    aliases = {}
    for item in args.alias:
        src, sep, dst = item.partition("=")
        if not sep:
            raise SystemExit(f"--alias expects FROM=TO, got {item!r}")
        aliases[src.strip()] = dst.strip()
    opts = ImportOptions(model_name=args.model, time_mode=args.time_mode, location_mode=args.location_mode,
                         fixed_location=args.fixed_location, encoding=args.encoding,
                         file_from_table_name=args.file_from_table_name,
                         annotator_from_table_name=args.annotator_from_table_name,
                         location_aliases=aliases, truncate_at_recording_end=args.truncate_at_end,
                         repair_unpadded_file_dates=not args.no_date_repair,
                         primary_label_column=args.primary_label)
    if args.citimon_delivery:
        opts.location_mode = "table_name"
        opts.annotator_from_table_name = True
        opts.truncate_at_recording_end = True
        opts.primary_label_column = opts.primary_label_column or "Annotation"
    return opts


def _read_files(paths):
    files = []
    for path in paths:
        with open(path, "rb") as f:
            files.append((path, f.read()))
    return files


def _summary(res) -> dict:
    total_status, total_flags, total_species = Counter(), Counter(), Counter()
    out = {"errors": res.errors, "partition_signature": res.partition_signature,
           "duplicate_clip_rows_removed": res.duplicate_clip_rows_removed, "files": []}
    for fp in res.files:
        total_status.update(fp.stats["status_counts"])
        total_flags.update(fp.stats.get("flag_counts", {}))
        total_species.update(fp.stats.get("species_counts", {}))
        out["files"].append({
            "file": fp.filename, "annotator": fp.annotator, "error": fp.error, "duplicate_of": fp.duplicate_of,
            "warnings": fp.table.warnings, "stats": fp.stats,
            "blocked_examples": [
                {"selection": s.key, "lines": s.row_numbers, "status": r.status, "message": r.message}
                for s, r in zip(fp.table.selections, fp.results) if r.blocked][:20],
        })
    out["totals"] = {"selections": sum(total_status.values()), "status_counts": dict(total_status.most_common()),
                     "flag_counts": dict(total_flags.most_common()),
                     "species_counts": dict(total_species.most_common())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="raven_import")
    ap.add_argument("--db", default=None, help="Annotation DB path (default: $VIBROVIS_ANNOTATIONS_DIR or cache/annotations)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_import_args(sub.add_parser("dry-run"))
    imp = sub.add_parser("import")
    _add_import_args(imp)
    imp.add_argument("--yes", action="store_true", help="Required: confirm writing to the annotation database")
    rl = sub.add_parser("relink")
    rl.add_argument("--batch", action="append", default=None)
    rl.add_argument("--model", default=None)
    rl.add_argument("--data-dir", default="data")
    rl.add_argument("--dry-run", action="store_true")
    sub.add_parser("list")
    bk = sub.add_parser("backup")
    bk.add_argument("dest")
    rv = sub.add_parser("revert")
    rv.add_argument("batch_id")
    rv.add_argument("--reason", default="")

    args = ap.parse_args(argv)
    store = AnnotationStore(args.db)

    if args.cmd in ("dry-run", "import"):
        opts = _options(args)
        files = _read_files(args.files)
        source = PartitionClipSource(base_dir=args.data_dir)
        res = preview_import(files, opts, args.annotator, source, store=store if args.cmd == "import" else None)
        if args.cmd == "dry-run" or res.errors:
            print(json.dumps(_summary(res), indent=2, ensure_ascii=False))
            return 1 if res.errors else 0
        if not args.yes:
            print(json.dumps(_summary(res)["totals"], indent=2, ensure_ascii=False))
            print("Not written. Re-run with --yes to import.", file=sys.stderr)
            return 1
        outcomes = commit_import(files, opts, args.annotator, source, store, res.token, actor="cli", note=args.note)
        print(json.dumps({"db": store.db_path, "outcomes": outcomes}, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "relink":
        reports = relink_batches(store, PartitionClipSource(base_dir=args.data_dir), batch_ids=args.batch,
                                 model_name=args.model, actor="cli", dry_run=args.dry_run)
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "list":
        print(json.dumps(store.list_batches(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "backup":
        print(store.backup_to(args.dest))
        return 0
    if args.cmd == "revert":
        store.revert_batch(args.batch_id, actor="cli", reason=args.reason)
        print(f"Reverted {args.batch_id}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
