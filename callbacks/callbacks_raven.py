"""
Callbacks for the Citimon Raven annotation import wizard.

Flow: upload -> automatic analysis -> review exceptions -> import. Settings that can be inferred come from the
files, the recording catalogue and the configured aliases; overrides live under "Advanced".

Uploaded files are held server-side (single gunicorn worker) under a random token; only the token goes to the
browser. Import re-runs the analysis server-side with the exact options that were reviewed and commits only if
the result matches the reviewed preview token. Nothing here touches MANUAL_LABELS_CACHE.
"""
import base64
import binascii
import json
import secrets
import threading
from collections import Counter, OrderedDict

import dash
from dash import ALL, Input, Output, State, dcc, html
from flask import request

import hierarchy
from layout_raven import MODAL_HIDDEN, MODAL_VISIBLE, UNMAPPED
from raven_import.clip_index import PartitionClipSource
from raven_import.config import BLOCKING_STATUSES, ROLES, ImportOptions, RavenFormatConfig
from raven_import.service import (
    AWAITING_DATA_STATUSES, PreviewMismatch, commit_import, outcome_class, preview_import, relink_batches,
    summarize_preview, validated_aliases,
)
from raven_import.store import AnnotationStore

PROJECT = "Citimon"
MAX_FILES = 100
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_SESSIONS = 8
MAX_ISSUE_ROWS = 60

_uploads = OrderedDict()
_uploads_lock = threading.Lock()

# Module-level so tests can swap them.
CLIP_SOURCE = PartitionClipSource(project=PROJECT, base_dir=hierarchy.DEFAULT_DATA_DIR)
STORE = None

_ERR = {'color': '#b00020'}
_WARN = {'color': '#8a6d3b'}
_OK = {'color': '#28a745'}
_HINT = {'fontSize': '0.8em', 'color': '#666'}

# Human-readable reasons for the review step.
STATUS_TEXT = {
    "malformed_row": "row has a different number of fields than the header",
    "view_conflict": "Raven views of the same selection disagree",
    "invalid_interval": "begin/end time missing, invalid or reversed",
    "missing_source_file": "no source file named",
    "source_file_conflict": "Begin File and Begin Path disagree",
    "source_file_date_mismatch": "file-name date disagrees with Begin Date",
    "source_file_date_unverified": "unpadded file-name date without Begin Date",
    "missing_location": "location not detected",
    "location_conflict": "detected locations disagree",
    "missing_channel": "no channel",
    "channel_not_found": "recording has no such channel",
    "ambiguous_recording": "several recordings match",
    "missing_file_offset": "File Offset (s) missing",
    "ambiguous_time_origin": "times are sequence-relative without File Offset (s)",
    "time_origin_conflict": "File Offset disagrees with Begin Time",
    "spans_file_boundary": "selection crosses into the next file",
    "exceeds_recording_duration": "selection starts after the recording ends",
    **AWAITING_DATA_STATUSES,
}


def get_store():
    global STORE
    if STORE is None:
        STORE = AnnotationStore()
    return STORE


def _put_upload(files):
    token = secrets.token_hex(16)
    with _uploads_lock:
        _uploads[token] = files
        while len(_uploads) > MAX_UPLOAD_SESSIONS:
            _uploads.popitem(last=False)
    return token


def _get_upload(token):
    with _uploads_lock:
        return _uploads.get(token)


def decode_uploads(contents, filenames):
    """Return (files, errors) from dcc.Upload payloads."""
    files, errors = [], []
    contents = contents or []
    filenames = filenames or []
    if len(contents) > MAX_FILES:
        errors.append(f"At most {MAX_FILES} files per import; {len(contents)} given.")
        contents, filenames = contents[:MAX_FILES], filenames[:MAX_FILES]
    for content, name in zip(contents, filenames):
        try:
            data = base64.b64decode(content.split(",", 1)[1], validate=True)
        except (IndexError, ValueError, binascii.Error):
            errors.append(f"{name}: could not read upload.")
            continue
        if len(data) > MAX_FILE_BYTES:
            errors.append(f"{name}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
            continue
        if not name.lower().endswith((".txt", ".tsv")):
            errors.append(f"{name}: expected a .txt or .tsv Raven selection table.")
            continue
        files.append((name, data))
    return files, errors


def parse_aliases(text):
    """'FROM=TO' per line. Malformed lines map to '' so alias validation reports them."""
    aliases = {}
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        src, sep, dst = line.partition("=")
        aliases[src.strip() if sep else line.strip()] = dst.strip() if sep else ""
    return aliases


def build_options(form, fixes, known):
    """ImportOptions from the Advanced settings plus the per-table fixes chosen in the review step.

    form = (model, annotator_source, location_mode, fixed_location, time_mode, file_from_name, encoding, decimal,
            default_channel, duration, role_values, annotation_cols, primary_col, import_flags, aliases_text)
    Returns (options, alias_problems).
    """
    (model, annotator_source, location_mode, fixed_location, time_mode, file_from_name, encoding, decimal,
     default_channel, duration, role_values, annotation_cols, primary_col, import_flags, aliases_text) = form
    fixes = fixes or {}
    aliases = parse_aliases(aliases_text)
    aliases.update(fixes.get("extra_aliases") or {})
    valid_aliases, problems = validated_aliases(aliases, known)
    mapping = {}
    for role, value in zip(ROLES, role_values or []):
        if value == UNMAPPED:
            mapping[role] = ""
        elif value:
            mapping[role] = value
    flags = set(import_flags or [])
    opts = ImportOptions(
        project=PROJECT, model_name=model or "perchv2", encoding=encoding or "auto", time_mode=time_mode or "auto",
        location_mode=location_mode or "auto", fixed_location=fixed_location or None, column_mapping=mapping,
        annotation_columns=list(annotation_cols) if annotation_cols else None,
        primary_label_column=primary_col or None, primary_label_auto=True,
        default_channel=int(default_channel) if default_channel not in (None, "") else None,
        file_from_table_name="on" in (file_from_name or []),
        annotator_from_table_name=(annotator_source or "auto") != "fixed",
        repair_unpadded_file_dates="repair_dates" in flags,
        truncate_at_recording_end="truncate" in flags,
        location_aliases=valid_aliases,
        annotator_overrides=dict(fixes.get("annotator_overrides") or {}),
        location_overrides=dict(fixes.get("location_overrides") or {}),
        recording_duration_s=float(duration) if duration not in (None, "") else 0.0,
        format=RavenFormatConfig(decimal_separator=decimal or "."))
    return opts, problems


def analyse(files, form, fixes, fallback_annotator, store=None):
    """Run the automatic analysis. Returns a dict used by the review renderer and the analysis store."""
    model = form[0] or (CLIP_SOURCE.models() or ["perchv2"])[0]
    form = (model,) + tuple(form[1:])
    known = CLIP_SOURCE.known_locations(model)
    opts, alias_problems = build_options(form, fixes, known)
    prev = preview_import(files, opts, fallback_annotator or "", CLIP_SOURCE, store or get_store())
    summary = summarize_preview(prev, known)
    per_file = {}
    for fp in prev.files:
        c = Counter(outcome_class(r.status) for r in fp.results)
        per_file[fp.filename] = {"linked": c["linked"], "awaiting": c["awaiting_data"], "unresolved": c["unresolved"]}
    importable = not prev.errors and bool(summary["importable_files"])
    needs_ack = bool(summary["unresolved_total"] or summary["file_errors"] or summary["missing_annotator"])
    state = {
        "preview_token": prev.token if importable else None,
        "fingerprint": opts.fingerprint(),
        "fixes": fixes or {},
        "fallback_annotator": fallback_annotator or "",
        "model": model,
        "needs_ack": needs_ack,
        "conflicts": {fn: [c["batch_id"] for c in cands] for fn, cands in summary["conflicts"]},
        "per_file": per_file,
    }
    return {"prev": prev, "summary": summary, "options": opts, "alias_problems": alias_problems,
            "known": known, "state": state, "importable": importable}


def _table(header, rows):
    th = {'textAlign': 'left', 'padding': '3px 6px', 'borderBottom': '1px solid #ccc', 'whiteSpace': 'nowrap'}
    td = {'padding': '3px 6px', 'borderBottom': '1px solid #eee', 'verticalAlign': 'top'}
    return html.Table([html.Thead(html.Tr([html.Th(h, style=th) for h in header])),
                       html.Tbody([html.Tr([html.Td(c, style=td) for c in r]) for r in rows])],
                      style={'borderCollapse': 'collapse', 'fontSize': '0.82em', 'width': '100%'})


def _fmt_interval(res):
    if res.file_t0_s is None:
        return ""
    return f"{res.file_t0_s:g}–{res.file_t1_s:g} s"


def _reasons(counter):
    return "; ".join(f"{n} {STATUS_TEXT.get(k, k)}" for k, n in counter.most_common())


def render_review(result):
    """Concise summary + only the exceptions that need a decision; row-level details collapsed."""
    prev, s = result["prev"], result["summary"]
    if prev.errors:
        return html.Div([html.Div(e, style=_ERR) for e in prev.errors])

    locs = s["locations"]
    loc_text = f"{len(locs)}" + (f" ({', '.join(sorted(locs)[:8])}{', …' if len(locs) > 8 else ''})" if locs else "")
    to_import = len(s["importable_files"])
    rows = [
        ["Tables analysed", f"{s['files']} — {to_import} to import"
         + (f", {len(s['duplicates'])} already imported" if s["duplicates"] else "")
         + (f", {len(s['file_errors']) + len(s['missing_annotator'])} need attention"
            if s["file_errors"] or s["missing_annotator"] else "")],
        ["Selections found", f"{s['selections_found']} ({s['selections_to_import']} in tables to import)"],
        ["Annotators", ", ".join(s["annotators"]) or "–"],
        ["Locations", loc_text or "–"],
        ["Quick-pass tables", s["quick_pass_files"]],
        [html.Span("Linked to clips", style=_OK), s["linked"]],
        ["Valid, awaiting recordings or clips", f"{s['awaiting_total']}"
         + (f" — {_reasons(s['awaiting'])}" if s["awaiting_total"] else "")],
        [html.Span("Warnings", style=_WARN if sum(s["warnings"].values()) else {}),
         ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in s["warnings"].most_common()) or "0"],
        [html.Span("Unresolved", style=_ERR if s["unresolved_total"] else {}),
         f"{s['unresolved_total']}" + (f" — {_reasons(s['unresolved'])}" if s["unresolved_total"] else "")],
    ]
    parts = [_table(["", ""], rows)]
    if s["awaiting_total"]:
        parts.append(html.Div("Awaiting selections are valid and will be imported unlinked; use Re-link (Advanced) "
                              "once the recordings or partition clips are available.", style=_HINT))
    if s["unknown_location_names"]:
        parts.append(html.Div("Locations not in the recording catalogue: "
                              + ", ".join(f"{k} ({v})" for k, v in sorted(s["unknown_location_names"].items())),
                              style=_HINT))
    if s["duplicates"]:
        parts.append(html.Div(f"Skipped automatically — identical to an earlier import by the same annotator: "
                              + ", ".join(fn for fn, _ in s["duplicates"]), style=_HINT))
    for w in result["alias_problems"]:
        parts.append(html.Div(w, style={**_WARN, 'fontSize': '0.8em'}))

    attention, fixable = [], False
    if s["missing_annotator"]:
        fixable = True
        attention.append(html.Div([
            html.Div("Annotator not found in these table names — enter it, or the table is skipped:"),
            *[html.Div([html.Span(fn, style={'flex': 1, 'fontSize': '0.85em'}),
                        dcc.Input(id={'type': 'raven-fix-annotator', 'index': fn}, type='text',
                                  placeholder='Annotator ID', className='input-field', style={'width': '160px'})],
                       style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '3px 0'})
              for fn in s["missing_annotator"]]]))
    if s["missing_location"]:
        fixable = True
        overrides = result["state"]["fixes"].get("location_overrides", {})
        attention.append(html.Div([
            html.Div("Location not detected for these tables — choose it, or their selections stay unresolved:"),
            *[html.Div([html.Span(fn, style={'flex': 1, 'fontSize': '0.85em'}),
                        dcc.Dropdown(id={'type': 'raven-fix-location', 'index': fn}, value=overrides.get(fn),
                                     options=result["known"], placeholder='Location…', style={'width': '220px'})],
                       style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '3px 0'})
              for fn in s["missing_location"]]]))
    if s["alias_suggestions"]:
        fixable = True
        attention.append(html.Div([
            html.Div("These table-name locations differ from a known location only in letter case. Map them?"),
            dcc.Checklist(id={'type': 'raven-fix-alias', 'index': 0}, value=[], labelStyle={'display': 'block'},
                          options=[{'label': f" {src} → {dst}", 'value': json.dumps([src, dst])}
                                   for src, dst in sorted(s["alias_suggestions"].items())]),
        ]))
    if fixable:
        attention.append(html.Button("Apply and re-analyse", id={'type': 'raven-apply-fixes', 'index': 0},
                                     className='app-button'))
    for fn, cands in s["conflicts"]:
        attention.append(html.Div([
            html.Div(f"{fn}: an earlier import of this table by the same annotator has different content "
                     f"({', '.join(c['batch_id'][:8] + ' ' + c['created_at'] for c in cands)})."),
            dcc.RadioItems(id={'type': 'raven-conflict', 'index': fn}, value=None, labelStyle={'display': 'block'},
                           options=[{'label': ' Replace the earlier import (it is kept, marked superseded)',
                                     'value': 'replace'},
                                    {'label': ' Keep both imports active', 'value': 'keep'}]),
        ]))
    if s["file_errors"]:
        attention.append(html.Div("Cannot be read and will be skipped: "
                                  + "; ".join(f"{fn} ({err})" for fn, err in s["file_errors"]), style=_ERR))
    if result["state"]["needs_ack"]:
        bits = []
        if s["unresolved_total"]:
            bits.append(f"store {s['unresolved_total']} unresolved selection(s) with their reasons")
        skipped = len(s["file_errors"]) + len(s["missing_annotator"])
        if skipped:
            bits.append(f"skip {skipped} table(s) listed above")
        attention.append(dcc.Checklist(id={'type': 'raven-ack', 'index': 0}, value=[], options=[
            {'label': " Import anyway: " + " and ".join(bits) + ". Unresolved rows stay visible for later review.",
             'value': 'ack'}]))
    if attention:
        parts.append(html.Div([html.Div("Needs your attention", style={'fontWeight': '600', 'marginBottom': '4px'}),
                               *[html.Div(a, style={'marginBottom': '8px'}) for a in attention]],
                              style={'border': '1px solid #f0c36d', 'backgroundColor': '#fff8e1', 'borderRadius': '6px',
                                     'padding': '8px', 'marginTop': '8px', 'fontSize': '0.9em'}))
    elif result["importable"]:
        parts.append(html.Div("Nothing needs your attention. Press Import.", style={**_OK, 'marginTop': '6px'}))
    parts.append(html.Details([html.Summary("Row-level details"), render_preview(prev)],
                              style={'marginTop': '8px', 'fontSize': '0.9em'}))
    return html.Div(parts)


def render_diagnostics(result):
    prev, opts = result["prev"], result["options"]
    rows = [
        ["Model / partition", f"{prev.model_name} / {prev.partition_signature}"],
        ["Duplicate clip rows ignored (cluster settings)", prev.duplicate_clip_rows_removed],
        ["Location source", opts.location_mode],
        ["Annotator source", "table names" if opts.annotator_from_table_name else "one annotator for all"],
        ["Aliases in use", ", ".join(f"{k}→{v}" for k, v in opts.location_aliases.items()) or "–"],
        ["Per-table overrides", ", ".join([f"{k}: {v}" for k, v in opts.annotator_overrides.items()]
                                          + [f"{k}: {v}" for k, v in opts.location_overrides.items()]) or "–"],
        ["Time origin / truncation / date repair",
         f"{opts.time_mode} / {opts.truncate_at_recording_end} / {opts.repair_unpadded_file_dates}"],
    ]
    extra = [html.Div(w, style=_WARN) for w in prev.index_warnings + result["alias_problems"]]
    return html.Div([_table(["", ""], rows), *extra])


def render_preview(prev):
    """Per-table details (collapsed by default in the review)."""
    parts = []
    many = len(prev.files) > 3
    for fp in prev.files:
        t, s = fp.table, fp.stats
        sc = s["status_counts"]
        meta = s.get("table_meta") or {}
        matched = sc.get("matched_single", 0) + sc.get("matched_multiple", 0)
        headline = (f"{fp.filename} — annotator {fp.annotator or '?'}"
                    + (f", location {meta.get('location')}" if meta.get("location") else "")
                    + (", quick pass" if meta.get("quick_pass") else "")
                    + (f" · {s['selections']} selections, {matched} linked, {s['blocked']} not linked"
                       if not fp.error else " · cannot import"))
        card = []
        if fp.error:
            card.append(html.Div(f"Cannot import: {fp.error}", style=_ERR))
            parts.append(html.Details([html.Summary(headline, style=_ERR), *card], open=True,
                                      style={'borderTop': '1px solid #ddd', 'paddingTop': '6px', 'marginTop': '6px'}))
            continue
        if fp.duplicate_of:
            card.append(html.Div(f"Already imported by this annotator (import {fp.duplicate_of[:8]}) — skipped.",
                                 style=_WARN))
        for w in t.warnings:
            card.append(html.Div(w, style={**_WARN, 'fontSize': '0.85em'}))
        summary = [
            ["Encoding", s["encoding"]], ["Data rows", s["data_rows"]], ["Selections", s["selections"]],
            ["Duplicate view rows collapsed", s["duplicate_view_rows"]],
            ["Linked (one clip / several)", f"{sc.get('matched_single', 0)} / {sc.get('matched_multiple', 0)}"],
            ["Clip links", s["links"]],
            ["Corrections / flags", ", ".join(f"{k}: {v}" for k, v in s.get("flag_counts", {}).items()) or "–"],
            ["Interpreted species", ", ".join(f"{k}: {v}" for k, v in s.get("species_counts", {}).items()) or "–"],
        ]
        not_linked = [[STATUS_TEXT.get(k, k), outcome_class(k).replace("_", " "), v]
                      for k, v in sc.items() if k in BLOCKING_STATUSES or k == "no_clip_overlap"]
        card.append(html.Div([
            html.Div(_table(["", ""], summary), style={'flex': 1, 'minWidth': '260px'}),
            html.Div(_table(["Not linked because", "Kind", "Rows"], not_linked) if not_linked else "All linked.",
                     style={'flex': 1, 'minWidth': '220px'}),
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}))
        issues = []
        for sel, res in zip(t.selections, fp.results):
            if res.blocked or res.status == "no_clip_overlap" or "missing_label" in sel.flags or res.flags:
                note = res.message + (" Missing label." if "missing_label" in sel.flags else "")
                issues.append([sel.key, ",".join(str(n) for n in sel.row_numbers), res.status,
                               "; ".join(res.flags), note.strip(), res.location or "", _fmt_interval(res)])
        if issues:
            card.append(html.Details([
                html.Summary(f"Rows that are not linked or carry flags ({len(issues)})"),
                _table(["Sel.", "Lines", "Status", "Flags", "Reason", "Location", "File interval"],
                       issues[:MAX_ISSUE_ROWS]),
                html.Div(f"… {len(issues) - MAX_ISSUE_ROWS} more" if len(issues) > MAX_ISSUE_ROWS else ""),
            ]))
        parts.append(html.Details([html.Summary(headline, style={'fontWeight': '600'}), *card], open=not many,
                                  style={'borderTop': '1px solid #ddd', 'paddingTop': '6px', 'marginTop': '6px'}))
    return html.Div(parts)


def render_batches(batches):
    if not batches:
        return html.Div("No Raven imports yet.", style={'color': '#666'})
    rows = []
    for b in batches:
        st = b.get("stats") or {}
        rows.append([b["batch_id"][:8], b["annotator"], b["source_filename"], b["created_at"], b["status"],
                     (b.get("superseded_by") or "")[:8], st.get("selections", ""), st.get("links", "")])
    return _table(["Import", "Annotator", "File", "Imported (UTC)", "Status", "Superseded by", "Selections",
                   "Links"], rows)


def render_relink(reports, dry_run):
    changed = [r for r in reports if r.get("after") is not None and r.get("before") != r.get("after")]
    skipped = [r for r in reports if r.get("outcome") == "skipped"]
    head = (f"{'Dry run: ' if dry_run else ''}{len(reports)} import(s) checked, {len(changed)} "
            f"{'would change' if dry_run else 'changed'}, {sum(r.get('links', 0) for r in reports)} clip links"
            + (f", {len(skipped)} skipped" if skipped else "") + ".")
    rows = [[r["filename"], r["annotator"], json.dumps(r["before"]), json.dumps(r.get("after"))] for r in changed[:30]]
    return html.Div([html.Div(head, style=_OK if not dry_run else {}),
                     _table(["File", "Annotator", "Before", "After"], rows) if rows else ""])


def _actor():
    try:
        return f"ui:{request.remote_addr}"
    except RuntimeError:
        return "ui"


FORM_STATES = [
    State('raven-model-dropdown', 'value'), State('raven-annotator-source', 'value'),
    State('raven-location-mode', 'value'), State('raven-fixed-location', 'value'),
    State('raven-time-mode', 'value'), State('raven-file-from-name', 'value'), State('raven-encoding', 'value'),
    State('raven-decimal', 'value'), State('raven-default-channel', 'value'), State('raven-duration', 'value'),
    State({'type': 'raven-col', 'role': ALL}, 'value'), State('raven-annotation-cols', 'value'),
    State('raven-primary-col', 'value'), State('raven-import-flags', 'value'), State('raven-aliases', 'value'),
]


def import_readiness(analysis, conflict_choices, acknowledged, pending_fixes, settings_match):
    """(enabled, reason). Import is enabled only for a current, safe preview with every decision made."""
    if not analysis:
        return False, "Upload Raven selection tables to start."
    if not analysis.get("preview_token"):
        return False, ("Nothing to import: every table is already imported or needs a fix above."
                       if analysis.get("fingerprint") else "Nothing to import.")
    if not settings_match:
        return False, "Advanced settings changed since the analysis — press 'Re-analyse with these settings'."
    if pending_fixes:
        return False, "Press 'Apply and re-analyse' to use the values you entered."
    undecided = [fn for fn in analysis.get("conflicts", {}) if conflict_choices.get(fn) not in ("replace", "keep")]
    if undecided:
        return False, "Choose how to handle the earlier import of: " + ", ".join(undecided) + "."
    if analysis.get("needs_ack") and not acknowledged:
        return False, "Tick 'Import anyway' to confirm how unresolved selections are handled."
    return True, ""


def import_button_style(enabled):
    return {'backgroundColor': '#28a745' if enabled else '#b8c2bb', 'color': 'white',
            'cursor': 'pointer' if enabled else 'not-allowed', 'opacity': 1 if enabled else 0.8}


def _pending_fixes(analysis, ann_values, ann_ids, loc_values, loc_ids, alias_values):
    """True when a fix control holds a value that the current analysis has not applied yet."""
    fixes = (analysis or {}).get("fixes") or {}
    applied_ann = fixes.get("annotator_overrides") or {}
    applied_loc = fixes.get("location_overrides") or {}
    applied_alias = fixes.get("extra_aliases") or {}
    for v, i in zip(ann_values or [], ann_ids or []):
        if v and v.strip() and applied_ann.get(i["index"]) != v.strip():
            return True
    for v, i in zip(loc_values or [], loc_ids or []):
        if v and applied_loc.get(i["index"]) != v:
            return True
    for group in alias_values or []:
        for v in group or []:
            src, dst = json.loads(v)
            if applied_alias.get(src) != dst:
                return True
    return False


def _fallback_annotator(form, dropdown, new_name):
    return ((new_name or "").strip() or (dropdown or "")) if form[1] == "fixed" else ""


def register_raven_callbacks(app):

    @app.callback(Output('raven-launcher', 'style'), Input('project-dropdown', 'value'))
    def toggle_launcher(project):
        return {'display': 'block' if project == PROJECT else 'none', 'marginBottom': '15px'}

    @app.callback(Output('raven-modal', 'style'),
                  Input('raven-open-btn', 'n_clicks'), Input('raven-close-btn', 'n_clicks'),
                  prevent_initial_call=True)
    def toggle_modal(_open, _close):
        return MODAL_VISIBLE if dash.ctx.triggered_id == 'raven-open-btn' else MODAL_HIDDEN

    @app.callback(
        [Output('raven-annotator-dropdown', 'options'), Output('raven-fixed-location', 'options'),
         Output('raven-model-dropdown', 'options'), Output('raven-model-dropdown', 'value'),
         Output('raven-batches-table', 'children'), Output('raven-revert-select', 'options')],
        Input('raven-open-btn', 'n_clicks'), Input('raven-refresh-signal', 'data'),
        State('raven-model-dropdown', 'value'),
        prevent_initial_call=True)
    def refresh_choices(_n, _signal, current_model):
        store = get_store()
        models = CLIP_SOURCE.models()
        model = current_model if current_model in models else (models[0] if models else None)
        locations = CLIP_SOURCE.known_locations(model) if model else []
        batches = store.list_batches(PROJECT)
        revert_opts = [{'label': f"{b['batch_id'][:8]} · {b['annotator']} · {b['source_filename']}",
                        'value': b['batch_id']} for b in batches if b['status'] == 'active']
        return (store.list_annotators(), locations, models, model, render_batches(batches), revert_opts)

    @app.callback(
        [Output('raven-upload-token', 'data'), Output('raven-upload-summary', 'children'),
         Output('raven-review', 'children'), Output('raven-diagnostics', 'children'),
         Output('raven-analysis', 'data'), Output('raven-confirm-btn', 'children'),
         Output({'type': 'raven-col', 'role': ALL}, 'options'), Output('raven-annotation-cols', 'options'),
         Output('raven-primary-col', 'options'), Output('raven-commit-output', 'children', allow_duplicate=True)],
        Input('raven-upload', 'contents'), Input('raven-reanalyse-btn', 'n_clicks'),
        Input({'type': 'raven-apply-fixes', 'index': ALL}, 'n_clicks'),
        State('raven-upload', 'filename'), State('raven-upload-token', 'data'), State('raven-analysis', 'data'),
        State({'type': 'raven-fix-annotator', 'index': ALL}, 'value'),
        State({'type': 'raven-fix-annotator', 'index': ALL}, 'id'),
        State({'type': 'raven-fix-location', 'index': ALL}, 'value'),
        State({'type': 'raven-fix-location', 'index': ALL}, 'id'),
        State({'type': 'raven-fix-alias', 'index': ALL}, 'value'),
        State('raven-annotator-dropdown', 'value'), State('raven-annotator-new', 'value'),
        *FORM_STATES,
        prevent_initial_call=True)
    def on_analyse(contents, _reanalyse, fix_clicks, filenames, token, analysis, ann_values, ann_ids,
                   loc_values, loc_ids, alias_values, ann_dropdown, ann_new, *form):
        n_out = 10
        trigger = dash.ctx.triggered_id
        if isinstance(trigger, dict) and not any(fix_clicks or []):
            return (dash.no_update,) * n_out   # fix button (re)rendered, not clicked
        fixes = dict((analysis or {}).get("fixes") or {})
        upload_summary = dash.no_update
        if trigger == 'raven-upload':
            files, errors = decode_uploads(contents, filenames)
            token = _put_upload(files) if files else None
            fixes = {}
            upload_summary = html.Div(
                [html.Div(f"{len(files)} table(s) received, {sum(len(d) for _, d in files):,} bytes.")]
                + [html.Div(e, style=_ERR) for e in errors])
        else:
            files = _get_upload(token) if token else None
        if isinstance(trigger, dict):
            fixes["annotator_overrides"] = {**fixes.get("annotator_overrides", {}), **{
                i["index"]: v.strip() for v, i in zip(ann_values or [], ann_ids or []) if v and v.strip()}}
            fixes["location_overrides"] = {**fixes.get("location_overrides", {}), **{
                i["index"]: v for v, i in zip(loc_values or [], loc_ids or []) if v}}
            accepted = [json.loads(v) for group in (alias_values or []) for v in (group or [])]
            fixes["extra_aliases"] = {**fixes.get("extra_aliases", {}), **{src: dst for src, dst in accepted}}
        if not files:
            msg = html.Div("Upload one or more Raven selection tables (uploads expire after newer uploads or a "
                           "restart).", style=_ERR if trigger != 'raven-upload' else _HINT)
            return (token, upload_summary, msg, "", None, "Import", [[]] * len(ROLES), [], [], None)

        fallback = _fallback_annotator(form, ann_dropdown, ann_new)
        try:
            result = analyse(files, form, fixes, fallback)
        except Exception as exc:  # surface unexpected errors instead of a silent failure
            return (token, upload_summary, html.Div(f"Analysis failed: {exc}", style=_ERR), "", None,
                    "Import", [[]] * len(ROLES), [], [], None)
        state = {**result["state"], "upload_token": token}
        headers = []
        for fp in result["prev"].files:
            headers.extend(h for h in fp.table.headers if h not in headers)
        col_opts = [{'label': '(not mapped)', 'value': UNMAPPED}] + [{'label': h, 'value': h} for h in headers]
        ann_opts = [{'label': h, 'value': h} for h in headers]
        s = result["summary"]
        label = (f"Import {len(s['importable_files'])} table(s), {s['selections_to_import']} selection(s)"
                 if result["importable"] else "Import")
        return (token, upload_summary, render_review(result), render_diagnostics(result), state,
                label, [col_opts] * len(ROLES), ann_opts, ann_opts, None)

    @app.callback(
        [Output('raven-commit-output', 'children'), Output('raven-refresh-signal', 'data'),
         Output('raven-analysis', 'data', allow_duplicate=True)],
        Input('raven-confirm-btn', 'n_clicks'),
        State('raven-analysis', 'data'),
        State({'type': 'raven-conflict', 'index': ALL}, 'value'), State({'type': 'raven-conflict', 'index': ALL}, 'id'),
        State({'type': 'raven-ack', 'index': ALL}, 'value'),
        State('raven-note', 'value'), State('raven-refresh-signal', 'data'),
        State('raven-annotator-dropdown', 'value'), State('raven-annotator-new', 'value'),
        *FORM_STATES,
        prevent_initial_call=True)
    def on_confirm(_n, analysis, conflict_values, conflict_ids, ack_values, note, signal, ann_dropdown, ann_new,
                   *form):
        def refuse(text):
            return html.Div(text, style=_ERR), dash.no_update, dash.no_update

        if not analysis or not analysis.get("preview_token"):
            return refuse("Nothing to import yet: upload tables first.")
        files = _get_upload(analysis.get("upload_token"))
        if not files:
            return refuse("The uploaded tables are no longer available; upload them again.")
        model = form[0] or analysis.get("model")
        opts, _ = build_options((model,) + tuple(form[1:]), analysis.get("fixes"), CLIP_SOURCE.known_locations(model))
        fallback = _fallback_annotator(form, ann_dropdown, ann_new)
        if opts.fingerprint() != analysis["fingerprint"] or fallback != analysis.get("fallback_annotator", ""):
            return refuse("Settings changed since the analysis. Press 'Re-analyse with these settings' (Advanced) "
                          "and review again before importing.")
        choices = {i["index"]: v for v, i in zip(conflict_values or [], conflict_ids or [])}
        missing = [fn for fn in analysis.get("conflicts", {}) if choices.get(fn) not in ("replace", "keep")]
        if missing:
            return refuse("Choose how to handle the earlier import of: " + ", ".join(missing))
        if analysis.get("needs_ack") and not any("ack" in (v or []) for v in (ack_values or [])):
            return refuse("Confirm how unresolved selections and unreadable tables are handled "
                          "(checkbox under 'Needs your attention').")
        supersede = {fn: ids for fn, ids in analysis.get("conflicts", {}).items() if choices.get(fn) == "replace"}
        try:
            outcomes = commit_import(files, opts, fallback, CLIP_SOURCE, get_store(), analysis["preview_token"],
                                     supersede=supersede, actor=_actor(), note=(note or None))
        except PreviewMismatch as exc:
            return refuse(f"{exc} Press 'Re-analyse with these settings' (Advanced).")
        except Exception as exc:
            return refuse(f"Import failed, nothing was written: {exc}")

        by = Counter(o["outcome"] for o in outcomes)
        imported = [o for o in outcomes if o["outcome"] == "imported"]
        totals = Counter()
        for o in imported:
            totals.update(analysis.get("per_file", {}).get(o["filename"], {}))
        superseded = sum(len(o.get("superseded", [])) for o in imported)
        lines = [
            html.Div(f"Imported {by['imported']} table(s): {totals['linked']} selection(s) linked, "
                     f"{totals['awaiting']} awaiting recordings or clips, {totals['unresolved']} unresolved.",
                     style={**_OK, 'fontWeight': '600'}),
            html.Div(f"Skipped {by['duplicate']} already-imported table(s); "
                     f"{by['not_imported']} table(s) not imported; {superseded} earlier import(s) superseded."),
        ]
        rows = [[o["filename"], o["outcome"], (o.get("batch_id") or "")[:8],
                 ", ".join(b[:8] for b in o.get("superseded", [])) or o.get("reason", "")] for o in outcomes]
        lines.append(html.Details([html.Summary("Per table"),
                                   _table(["Table", "Outcome", "Import ID", "Superseded / reason"], rows)]))
        return html.Div(lines), (signal or 0) + 1, {**analysis, "preview_token": None}

    @app.callback(
        [Output('raven-confirm-btn', 'disabled'), Output('raven-confirm-btn', 'style'),
         Output('raven-import-hint', 'children')],
        Input('raven-analysis', 'data'),
        Input({'type': 'raven-conflict', 'index': ALL}, 'value'), State({'type': 'raven-conflict', 'index': ALL}, 'id'),
        Input({'type': 'raven-ack', 'index': ALL}, 'value'),
        Input({'type': 'raven-fix-annotator', 'index': ALL}, 'value'),
        State({'type': 'raven-fix-annotator', 'index': ALL}, 'id'),
        Input({'type': 'raven-fix-location', 'index': ALL}, 'value'),
        State({'type': 'raven-fix-location', 'index': ALL}, 'id'),
        Input({'type': 'raven-fix-alias', 'index': ALL}, 'value'),
        Input('raven-annotator-dropdown', 'value'), Input('raven-annotator-new', 'value'),
        *[Input(s.component_id, s.component_property) for s in FORM_STATES])
    def update_import_state(analysis, conflict_values, conflict_ids, ack_values, ann_values, ann_ids,
                            loc_values, loc_ids, alias_values, ann_dropdown, ann_new, *form):
        settings_match = True
        if analysis and analysis.get("preview_token"):
            model = form[0] or analysis.get("model")
            opts, _ = build_options((model,) + tuple(form[1:]), analysis.get("fixes"),
                                    CLIP_SOURCE.known_locations(model))
            settings_match = (opts.fingerprint() == analysis.get("fingerprint")
                              and _fallback_annotator(form, ann_dropdown, ann_new) == analysis.get("fallback_annotator", ""))
        enabled, reason = import_readiness(
            analysis, {i["index"]: v for v, i in zip(conflict_values or [], conflict_ids or [])},
            any("ack" in (v or []) for v in (ack_values or [])),
            _pending_fixes(analysis, ann_values, ann_ids, loc_values, loc_ids, alias_values), settings_match)
        return not enabled, import_button_style(enabled), reason

    @app.callback(Output('raven-revert-confirm', 'displayed'), Input('raven-revert-btn', 'n_clicks'),
                  State('raven-revert-select', 'value'), prevent_initial_call=True)
    def ask_revert(_n, batch_id):
        return bool(batch_id)

    @app.callback([Output('raven-revert-output', 'children'),
                   Output('raven-refresh-signal', 'data', allow_duplicate=True)],
                  Input('raven-revert-confirm', 'submit_n_clicks'),
                  State('raven-revert-select', 'value'), State('raven-refresh-signal', 'data'),
                  prevent_initial_call=True)
    def do_revert(_n, batch_id, signal):
        if not batch_id:
            return dash.no_update, dash.no_update
        try:
            get_store().revert_batch(batch_id, actor=_actor(), reason="reverted in wizard")
        except ValueError as exc:
            return html.Span(str(exc), style=_ERR), dash.no_update
        return html.Span(f"Reverted {batch_id[:8]}.", style=_OK), (signal or 0) + 1

    @app.callback(Output('raven-relink-output', 'children'), Input('raven-relink-dry-btn', 'n_clicks'),
                  prevent_initial_call=True)
    def relink_dry_run(_n):
        try:
            return render_relink(relink_batches(get_store(), CLIP_SOURCE, dry_run=True), dry_run=True)
        except Exception as exc:
            return html.Span(f"Re-link check failed: {exc}", style=_ERR)

    @app.callback(Output('raven-relink-confirm', 'displayed'), Input('raven-relink-btn', 'n_clicks'),
                  prevent_initial_call=True)
    def ask_relink(_n):
        return True

    @app.callback([Output('raven-relink-output', 'children', allow_duplicate=True),
                   Output('raven-refresh-signal', 'data', allow_duplicate=True)],
                  Input('raven-relink-confirm', 'submit_n_clicks'), State('raven-refresh-signal', 'data'),
                  prevent_initial_call=True)
    def do_relink(_n, signal):
        try:
            reports = relink_batches(get_store(), CLIP_SOURCE, actor=_actor())
        except Exception as exc:
            return html.Span(f"Re-link failed, nothing was changed for the failing import: {exc}", style=_ERR), \
                dash.no_update
        return render_relink(reports, dry_run=False), (signal or 0) + 1
