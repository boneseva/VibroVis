"""
Preview / commit orchestration shared by the UI wizard and the CLI.

``commit_import`` recomputes the preview server-side and refuses to write when the
inputs no longer match the token the user confirmed (changed options, files,
annotator or partition data).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import BLOCKING_STATUSES, ImportOptions
from .interpret import interpret_label
from .matcher import MatchResult, locations_needed, match_table
from .parser import ParsedTable, parse_raven_table
from .store import AnnotationStore, BatchPayload


class PreviewMismatch(RuntimeError):
    pass


@dataclass
class FilePreview:
    filename: str
    raw: bytes
    table: ParsedTable
    results: List[MatchResult]
    stats: dict
    annotator: str = ""
    file_error: Optional[str] = None
    duplicate_of: Optional[str] = None
    supersede_candidates: List[dict] = field(default_factory=list)

    @property
    def error(self) -> Optional[str]:
        return self.table.fatal_error or self.file_error

    @property
    def importable(self) -> bool:
        return self.error is None and self.duplicate_of is None


@dataclass
class PreviewResult:
    files: List[FilePreview]
    options: ImportOptions
    annotator: str
    model_name: str
    partition_signature: str
    token: str
    errors: List[str] = field(default_factory=list)
    index_warnings: List[str] = field(default_factory=list)
    duplicate_clip_rows_removed: int = 0


def file_stats(table: ParsedTable, results: List[MatchResult], options: Optional[ImportOptions] = None) -> dict:
    options = options or ImportOptions()
    status_counts = Counter(r.status for r in results)
    flag_counts = Counter(f.split(":")[0] for r in results for f in r.flags)
    label_cols = [table.primary_label_column] if table.primary_label_column else table.annotation_columns
    species, interp_rules, uncertain = Counter(), Counter(), 0
    for sel in table.selections:
        for col in label_cols:
            it = interpret_label(sel.annotations.get(col, ""), options.species_vocabulary, options.species_synonyms)
            interp_rules[it.rule] += 1
            uncertain += int(it.uncertain)
            if it.species:
                species[it.species + ("?" if it.uncertain else "")] += 1
    relation_counts = Counter(l.relation for r in results for l in r.links)
    clips = {l.semantic_clip_key for r in results for l in r.links}
    return {
        "data_rows": table.data_row_count,
        "selections": len(table.selections),
        "duplicate_view_rows": table.duplicate_view_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "blocked": sum(v for k, v in status_counts.items() if k in BLOCKING_STATUSES),
        "missing_label": sum(1 for s in table.selections if "missing_label" in s.flags),
        "links": sum(relation_counts.values()),
        "relation_counts": dict(sorted(relation_counts.items())),
        "distinct_clips": len(clips),
        "encoding": table.encoding,
        "table_meta": table.table_meta,
        "flag_counts": dict(sorted(flag_counts.items())),
        "species_counts": dict(sorted(species.items())),
        "interpretation_rules": dict(sorted(interp_rules.items())),
        "uncertain_labels": uncertain,
    }


def identity_scope(table: ParsedTable, options: ImportOptions) -> str:
    """Extra re-import identity: the location a table's name (or an explicit per-table override) assigns it.

    Byte-identical tables for different locations (e.g. header-only tables) are different imports. The raw
    table-name location is used, matching identities stored by earlier 'table_name' imports.
    """
    loc = (options.location_overrides or {}).get(table.filename)
    if not loc and options.location_mode in ("table_name", "auto") and table.table_meta:
        loc = table.table_meta.get("location")
    return f"location:{loc}" if loc else ""


def _token(options: ImportOptions, annotator: str, files: Sequence[Tuple[str, bytes]], signature: str) -> str:
    h = hashlib.sha256()
    h.update(options.fingerprint().encode())
    h.update(b"\0" + annotator.strip().encode("utf-8"))
    for name, data in files:
        h.update(b"\0" + name.encode("utf-8") + b"\0" + hashlib.sha256(data).hexdigest().encode())
    h.update(b"\0" + signature.encode())
    return h.hexdigest()


def preview_import(files: Sequence[Tuple[str, bytes]], options: ImportOptions, annotator: str,
                   clip_source, store: Optional[AnnotationStore] = None) -> PreviewResult:
    errors = list(options.validate())
    if not options.annotator_from_table_name and not (annotator or "").strip():
        errors.append("Annotator is required.")
    signature = clip_source.partition_signature(options.model_name)
    known = clip_source.known_locations(options.model_name)
    if not known:
        errors.append(f"No {options.project} clip data found for model '{options.model_name}'.")
    if options.fixed_location and options.fixed_location not in known:
        errors.append(f"Location '{options.fixed_location}' is not a known {options.project} location.")

    tables = [(name, data, parse_raven_table(data, name, options)) for name, data in files]
    needed = set()
    for _, _, table in tables:
        if not table.fatal_error:
            needed |= locations_needed(table, options, known)
    index = clip_source.load_index(options.model_name, needed)

    previews = []
    for name, data, table in tables:
        results = [] if table.fatal_error else match_table(table, options, index, known)
        fp = FilePreview(filename=table.filename, raw=data, table=table, results=results,
                         stats=file_stats(table, results, options), annotator=(annotator or "").strip())
        override = (options.annotator_overrides or {}).get(table.filename, "").strip()
        if override:
            fp.annotator = override
        elif options.annotator_from_table_name:
            fp.annotator = ((table.table_meta or {}).get("annotator") or "").strip()
            if not fp.annotator:
                fp.file_error = "Annotator not found: table name does not match the configured pattern."
        if store is not None and fp.annotator and not fp.error:
            fp.duplicate_of = store.find_active_duplicate(options.project, fp.annotator, table.sha256,
                                                          identity_scope(table, options))
            fp.supersede_candidates = store.find_supersede_candidates(
                options.project, fp.annotator, table.filename, table.sha256)
        previews.append(fp)

    return PreviewResult(
        files=previews, options=options, annotator=annotator.strip(), model_name=options.model_name,
        partition_signature=signature, token=_token(options, annotator, files, signature), errors=errors,
        index_warnings=index.warnings, duplicate_clip_rows_removed=index.duplicate_rows_removed)


def commit_import(files: Sequence[Tuple[str, bytes]], options: ImportOptions, annotator: str, clip_source,
                  store: AnnotationStore, expected_token: str,
                  supersede: Optional[Dict[str, List[str]]] = None, actor: Optional[str] = None,
                  note: Optional[str] = None) -> List[dict]:
    preview = preview_import(files, options, annotator, clip_source, store)
    if preview.errors:
        raise ValueError("; ".join(preview.errors))
    if preview.token != expected_token:
        raise PreviewMismatch("Inputs changed since the preview. Run the preview again before importing.")

    supersede = supersede or {}
    outcomes, payloads = [], []
    for fp in preview.files:
        if fp.error:
            outcomes.append({"filename": fp.filename, "outcome": "not_imported", "reason": fp.error})
            continue
        if fp.duplicate_of:
            outcomes.append({"filename": fp.filename, "outcome": "duplicate", "batch_id": fp.duplicate_of})
            continue
        allowed = {c["batch_id"] for c in fp.supersede_candidates}
        chosen = [b for b in supersede.get(fp.filename, []) if b]
        if set(chosen) - allowed:
            raise ValueError(f"{fp.filename}: can only supersede earlier imports of the same file by the same annotator.")
        payloads.append(BatchPayload(
            project=options.project, annotator=fp.annotator, raw_content=fp.raw, table=fp.table,
            results=fp.results, options=options.to_dict(), model_name=preview.model_name,
            partition_signature=preview.partition_signature, stats=fp.stats, supersede_batch_ids=chosen, note=note,
            identity_scope=identity_scope(fp.table, options)))
    if payloads:
        outcomes.extend(store.commit_batches(payloads, actor=actor))
    return outcomes


def relink_batches(store: AnnotationStore, clip_source, batch_ids: Optional[Sequence[str]] = None,
                   model_name: Optional[str] = None, actor: Optional[str] = None, dry_run: bool = False) -> List[dict]:
    """Re-resolve stored selections against the current partition and replace their clip links.

    Each batch is re-parsed from its stored raw bytes with its stored options, so the biologists'
    files are not needed again. A batch whose re-parse does not reproduce the stored selections
    is skipped and reported, never partially updated.
    """
    if batch_ids is None:
        batch_ids = [b["batch_id"] for b in store.list_batches() if b["status"] == "active"]
    reports = []
    for batch_id in batch_ids:
        info = store.batch_for_relink(batch_id)
        report = {"batch_id": batch_id, "filename": info["source_filename"], "annotator": info["annotator"],
                  "before": info["status_counts"]}
        if info["status"] != "active":
            reports.append({**report, "outcome": "skipped", "reason": f"batch is {info['status']}"})
            continue
        options = ImportOptions.from_dict(json.loads(info["options_json"]))
        if model_name:
            options.model_name = model_name
        table = parse_raven_table(info["raw_content"], info["source_filename"], options)
        if table.fatal_error or [s.key for s in table.selections] != info["selection_keys"]:
            reports.append({**report, "outcome": "skipped",
                            "reason": table.fatal_error or "re-parse does not reproduce the stored selections"})
            continue
        known = clip_source.known_locations(options.model_name)
        index = clip_source.load_index(options.model_name, locations_needed(table, options, known))
        results = match_table(table, options, index, known)
        signature = clip_source.partition_signature(options.model_name)
        after = dict(sorted(Counter(r.status for r in results).items()))
        report.update(after=after, model_name=options.model_name, partition_signature=signature,
                      links=sum(len(r.links) for r in results))
        if not dry_run:
            store.replace_resolution(batch_id, table, results, options.model_name, signature, actor,
                                     {"before": info["status_counts"], "after": after, "links": report["links"],
                                      "model_name": options.model_name, "partition_signature": signature})
        reports.append({**report, "outcome": "dry_run" if dry_run else "relinked"})
    return reports


# ---- Classification used by the simplified wizard (no change to stored statuses) -----------------------

LINKED_STATUSES = {"matched_single", "matched_multiple"}
# Valid selections that cannot be linked until recordings / partition clips exist. They are imported and
# kept unlinked for `relink`; they are not import errors.
AWAITING_DATA_STATUSES = {
    "no_clip_overlap": "no clip covers the interval in the current partition",
    "recording_not_found": "recording not in the current partition",
    "unknown_location": "location not in the recording catalogue",
    "unknown_location_folder": "Begin Path folder not in the recording catalogue",
}
WARNING_FLAGS = {"truncated_at_recording_end", "missing_label"}


def outcome_class(status: str) -> str:
    if status in LINKED_STATUSES:
        return "linked"
    if status in AWAITING_DATA_STATUSES:
        return "awaiting_data"
    return "unresolved"


def validated_aliases(aliases: Dict[str, str], known: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Keep only aliases whose target is a known location and whose source is not itself a known location."""
    known_set, valid, problems = set(known), {}, []
    for src, dst in (aliases or {}).items():
        if not src or not dst or src == dst:
            problems.append(f"Alias {src!r} -> {dst!r} is incomplete and was ignored.")
        elif dst not in known_set:
            problems.append(f"Alias {src!r} -> {dst!r} ignored: {dst!r} is not a known location.")
        elif src in known_set:
            problems.append(f"Alias {src!r} -> {dst!r} ignored: {src!r} is itself a known location.")
        else:
            valid[src] = dst
    return valid, problems


def suggest_aliases(names: Sequence[str], known: Sequence[str]) -> Dict[str, str]:
    """Case-only differences with exactly one known match, e.g. '2026_03-r26' -> '2026_03-R26'. Never applied
    automatically."""
    by_fold: Dict[str, List[str]] = {}
    for k in known:
        by_fold.setdefault(k.casefold(), []).append(k)
    out = {}
    for name in names:
        if not name or name in set(known):
            continue
        matches = by_fold.get(name.casefold(), [])
        if len(matches) == 1:
            out[name] = matches[0]
    return out


def summarize_preview(prev: PreviewResult, known: Sequence[str]) -> dict:
    """Counts for the review step; selections are counted for tables that would be imported."""
    s = {"files": len(prev.files), "selections_found": sum(len(fp.table.selections) for fp in prev.files),
         "importable_files": [], "file_errors": [], "duplicates": [], "conflicts": [],
         "missing_annotator": [], "missing_location": [], "annotators": set(), "locations": Counter(),
         "selections_to_import": 0, "linked": 0, "awaiting": Counter(), "unresolved": Counter(),
         "warnings": Counter(), "unknown_location_names": Counter(), "quick_pass_files": 0,
         "encoding_fallback": [], "species": Counter()}
    for fp in prev.files:
        meta = fp.table.table_meta or {}
        if fp.error:
            if fp.file_error and fp.file_error.startswith("Annotator"):
                s["missing_annotator"].append(fp.filename)
                # Offer the location fix in the same round when that is missing too.
                if not meta.get("location") and any(r.status == "missing_location" for r in fp.results):
                    s["missing_location"].append(fp.filename)
            else:
                s["file_errors"].append((fp.filename, fp.error))
            continue
        if fp.duplicate_of:
            s["duplicates"].append((fp.filename, fp.duplicate_of))
            continue
        s["importable_files"].append(fp.filename)
        if fp.supersede_candidates:
            s["conflicts"].append((fp.filename, fp.supersede_candidates))
        s["annotators"].add(fp.annotator)
        s["quick_pass_files"] += int(bool(meta.get("quick_pass")))
        if any("not valid UTF-8" in w for w in fp.table.warnings):
            s["encoding_fallback"].append(fp.filename)
        s["species"].update(fp.stats.get("species_counts", {}))
        missing_loc = False
        for sel, res in zip(fp.table.selections, fp.results):
            s["selections_to_import"] += 1
            cls = outcome_class(res.status)
            if cls == "linked":
                s["linked"] += 1
            elif cls == "awaiting_data":
                s["awaiting"][res.status] += 1
            else:
                s["unresolved"][res.status] += 1
            if res.status == "missing_location":
                missing_loc = True
            if res.location:
                s["locations"][res.location] += 1
            elif res.status == "unknown_location" and meta.get("location"):
                s["unknown_location_names"][meta["location"]] += 1
            for flag in list(sel.flags) + list(res.flags):
                if flag.split(":")[0] in WARNING_FLAGS:
                    s["warnings"][flag.split(":")[0]] += 1
        if missing_loc and not meta.get("location"):
            s["missing_location"].append(fp.filename)
    s["annotators"] = sorted(a for a in s["annotators"] if a)
    s["awaiting_total"] = sum(s["awaiting"].values())
    s["unresolved_total"] = sum(s["unresolved"].values())
    s["alias_suggestions"] = suggest_aliases(list(s["unknown_location_names"]), known)
    return s
