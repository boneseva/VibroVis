"""
Layout for the Citimon Raven annotation import wizard (modal) and its launcher button.
Callbacks live in callbacks/callbacks_raven.py.

Default flow: upload -> automatic analysis -> review exceptions -> import. Every setting that can be
inferred lives under "Advanced", collapsed by default.
"""
import json
import os

from dash import dcc, html

from raven_import.config import ROLES

ROLE_LABELS = {
    "selection": "Selection", "view": "View", "channel": "Channel",
    "begin_time": "Begin time", "end_time": "End time",
    "begin_file": "Begin file", "end_file": "End file",
    "begin_path": "Begin path", "end_path": "End path",
    "file_offset": "File offset", "low_freq": "Low freq", "high_freq": "High freq",
    "begin_date": "Begin date", "begin_clock_time": "Begin clock",
}
UNMAPPED = "__none__"
ALIASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raven_import", "citimon_location_aliases.json")

_SECTION = {'border': '1px solid #e3e3e3', 'borderRadius': '6px', 'padding': '10px 12px', 'marginBottom': '12px',
            'backgroundColor': '#fff'}
_HEADING = {'fontWeight': '600', 'marginBottom': '6px'}
_HINT = {'fontSize': '0.8em', 'color': '#666'}
_FIELD = {'marginBottom': '8px'}
_SUB = {'fontWeight': '600', 'fontSize': '0.9em', 'margin': '10px 0 6px'}


def configured_aliases_text():
    """Approved aliases from raven_import/citimon_location_aliases.json as 'FROM=TO' lines."""
    try:
        with open(ALIASES_FILE, encoding="utf-8") as f:
            aliases = json.load(f).get("aliases", {})
    except (OSError, ValueError):
        return ""
    return "\n".join(f"{src}={dst}" for src, dst in aliases.items())


def raven_import_launcher():
    """Button shown in the Labels panel; visible only for the Citimon project."""
    return html.Div([
        html.Button('Import Raven annotations…', id='raven-open-btn', n_clicks=0, className='app-button',
                    style={'width': '100%'}),
        html.Div("Stored separately; never changes manual labels.", style=_HINT),
    ], id='raven-launcher', style={'display': 'none', 'marginBottom': '15px'})


def _field(label, component, hint=None):
    children = [html.Label(label, style={'fontSize': '0.85em'}), component]
    if hint:
        children.append(html.Div(hint, style=_HINT))
    return html.Div(children, style=_FIELD)


def _advanced():
    column_rows = [
        html.Div([
            html.Span(ROLE_LABELS[role], style={'width': '95px', 'fontSize': '0.82em', 'flexShrink': 0}),
            dcc.Dropdown(id={'type': 'raven-col', 'role': role}, options=[], value=None,
                         placeholder='(automatic)', style={'flex': 1, 'minWidth': 0}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'marginBottom': '4px'})
        for role in ROLES
    ]
    return html.Details([
        html.Summary("Advanced", style=_HEADING),
        html.Div("Defaults are detected from the files and the recording catalogue. Change these only to "
                 "override detection, then press Re-analyse.", style={**_HINT, 'marginBottom': '8px'}),
        html.Div([
            html.Div([
                html.Div("Annotator", style=_SUB),
                dcc.RadioItems(id='raven-annotator-source', value='auto', labelStyle={'display': 'block'},
                               style={'fontSize': '0.85em'}, options=[
                    {'label': ' From table names ("<location>-<ID>[-hitro].txt")', 'value': 'auto'},
                    {'label': ' One annotator for all tables (below)', 'value': 'fixed'}]),
                _field("Existing annotator", dcc.Dropdown(id='raven-annotator-dropdown', options=[],
                                                          placeholder='Select…')),
                _field("…or new annotator", dcc.Input(id='raven-annotator-new', type='text', debounce=True,
                                                      placeholder='Name as it should appear',
                                                      className='input-field', style={'width': '100%'})),
                html.Div("Location", style=_SUB),
                dcc.RadioItems(id='raven-location-mode', value='auto', labelStyle={'display': 'block'},
                               style={'fontSize': '0.85em'}, options=[
                    {'label': ' Automatic (table name, else Begin Path folder)', 'value': 'auto'},
                    {'label': ' Begin Path folder only', 'value': 'path'},
                    {'label': ' Table file name only', 'value': 'table_name'},
                    {'label': ' Fixed location for all rows', 'value': 'fixed'}]),
                _field("Fallback / fixed location", dcc.Dropdown(id='raven-fixed-location', options=[],
                                                                 placeholder='(none)'),
                       "A row whose detected location disagrees with this one is blocked."),
                _field("Location aliases", dcc.Textarea(
                    id='raven-aliases', value=configured_aliases_text(),
                    style={'width': '100%', 'height': '52px', 'fontFamily': 'monospace', 'fontSize': '0.85em'}),
                       "One exact FROM=TO per line; each target must be a known location."),
                _field("Model clip grid", dcc.Dropdown(id='raven-model-dropdown', options=[], clearable=False)),
            ], style={'flex': 1, 'minWidth': '240px'}),
            html.Div([
                html.Div("Timing and format", style=_SUB),
                _field("Time origin", dcc.RadioItems(
                    id='raven-time-mode', value='auto', labelStyle={'display': 'block'},
                    options=[
                        {'label': ' Auto: File Offset (s) if present, else single-file table only', 'value': 'auto'},
                        {'label': ' File Offset (s) required (sequence / multi-file tables)', 'value': 'file_offset'},
                        {'label': ' Single-file table: Begin/End are file-relative', 'value': 'single_file'},
                    ])),
                dcc.Checklist(id='raven-import-flags', value=['repair_dates', 'truncate'],
                              labelStyle={'display': 'block'}, style={'fontSize': '0.85em', 'marginBottom': '8px'},
                              options=[
                    {'label': ' Repair unpadded dates in Begin File (only when Begin Date confirms)',
                     'value': 'repair_dates'},
                    {'label': ' Truncate selections at the recording end (original end kept, row flagged)',
                     'value': 'truncate'},
                ]),
                dcc.Checklist(id='raven-file-from-name', value=[], options=[
                    {'label': ' Infer source file from "<file>.Table.N.selections.txt" when no file column',
                     'value': 'on'}], style={'fontSize': '0.85em', 'marginBottom': '8px'}),
                html.Div([
                    _field("Encoding", dcc.Dropdown(id='raven-encoding', value='auto', clearable=False, options=[
                        {'label': 'Auto (UTF-8, else CP1250)', 'value': 'auto'},
                        {'label': 'UTF-8', 'value': 'utf-8'},
                        {'label': 'Windows-1250 (Central European)', 'value': 'cp1250'},
                        {'label': 'Windows-1252 (Western)', 'value': 'cp1252'}])),
                    _field("Decimal separator", dcc.RadioItems(id='raven-decimal', value='.', inline=True,
                                                               options=[{'label': ' . ', 'value': '.'},
                                                                        {'label': ' , ', 'value': ','}])),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}),
                html.Div([
                    _field("Channel if not in table", dcc.Input(id='raven-default-channel', type='number',
                                                                value=1, min=1, step=1, style={'width': '100%'})),
                    _field("Recording length (s)", dcc.Input(id='raven-duration', type='number', value=60,
                                                             min=1, style={'width': '100%'}),
                           "Used only when the audio file's length can't be read."),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}),
                _field("Note stored with the import (optional)",
                       dcc.Input(id='raven-note', type='text', className='input-field', style={'width': '100%'})),
            ], style={'flex': 1, 'minWidth': '240px'}),
            html.Div([
                html.Div("Columns", style=_SUB),
                *column_rows,
                _field("Annotation columns (empty = all non-measurement columns)", dcc.Checklist(
                    id='raven-annotation-cols', options=[], value=[], inline=True,
                    inputStyle={'marginRight': '3px'}, labelStyle={'marginRight': '12px'})),
                _field("Primary label column", dcc.Dropdown(id='raven-primary-col', options=[],
                                                            placeholder='(automatic: "Annotation")')),
            ], style={'flex': 1, 'minWidth': '260px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'}),
        html.Button('Re-analyse with these settings', id='raven-reanalyse-btn', n_clicks=0, className='app-button'),

        html.Div("Diagnostics", style=_SUB),
        html.Div(id='raven-diagnostics', style={'fontSize': '0.85em'}),

        html.Div("Import history", style=_SUB),
        html.Div(id='raven-batches-table', style={'fontSize': '0.82em', 'overflowX': 'auto'}),
        html.Div([
            dcc.Dropdown(id='raven-revert-select', options=[], placeholder='Active import to revert…',
                         style={'flex': 1}),
            html.Button('Revert import', id='raven-revert-btn', n_clicks=0, className='app-button',
                        style={'backgroundColor': '#dc3545', 'color': 'white'}),
        ], style={'display': 'flex', 'gap': '6px', 'marginTop': '6px'}),
        dcc.ConfirmDialog(id='raven-revert-confirm',
                          message='Revert this import? Its data stays in the database for traceability, '
                                  'but it will no longer be active.'),
        html.Div(id='raven-revert-output', style={'fontSize': '0.85em', 'marginTop': '4px'}),

        html.Div("Re-link after the partition is rebuilt", style=_SUB),
        html.Div("Re-matches stored annotations against the current clip data. Original Raven data is not "
                 "changed; each re-link is logged.", style=_HINT),
        html.Div([
            html.Button('Check (dry run)', id='raven-relink-dry-btn', n_clicks=0, className='app-button'),
            html.Button('Re-link now', id='raven-relink-btn', n_clicks=0, className='app-button'),
        ], style={'display': 'flex', 'gap': '6px', 'marginTop': '6px'}),
        dcc.ConfirmDialog(id='raven-relink-confirm',
                          message='Replace the clip links of all active Raven imports using the current partition?'),
        dcc.Loading(html.Div(id='raven-relink-output', style={'fontSize': '0.85em', 'marginTop': '4px'}), type='dot'),
    ], style=_SECTION)


def raven_import_modal():
    body = html.Div([
        html.Div([
            html.H3("Import Raven annotations — Citimon", style={'margin': 0, 'fontSize': '1.15em'}),
            html.Button('×', id='raven-close-btn', n_clicks=0, title='Close',
                        style={'border': 'none', 'background': 'none', 'fontSize': '1.5em', 'cursor': 'pointer'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '8px'}),
        html.Div("Upload Raven selection tables; annotator, location, timing and species are detected "
                 "automatically. Annotations are kept exactly as exported in a separate store and never change "
                 "manual labels.", style={**_HINT, 'marginBottom': '10px'}),

        html.Div([
            html.Div("1. Upload Raven selection tables (.txt)", style=_HEADING),
            dcc.Upload(id='raven-upload', multiple=True, accept='.txt,.tsv',
                       children=html.Div(['Drag files here or ', html.A('choose files')]),
                       style={'border': '1px dashed #999', 'borderRadius': '6px', 'padding': '14px',
                              'textAlign': 'center', 'cursor': 'pointer'}),
            html.Div(id='raven-upload-summary', style={'marginTop': '6px', 'fontSize': '0.85em'}),
        ], style=_SECTION),

        html.Div([
            html.Div("2. Review", style=_HEADING),
            dcc.Loading(html.Div(id='raven-review', children=html.Div(
                "Upload files to analyse them.", style=_HINT)), type='dot'),
        ], style=_SECTION),

        html.Div([
            html.Div("3. Import", style=_HEADING),
            html.Button('Import', id='raven-confirm-btn', n_clicks=0, disabled=True, className='app-button',
                        style={'backgroundColor': '#b8c2bb', 'color': 'white', 'cursor': 'not-allowed'}),
            html.Div(id='raven-import-hint', style={**_HINT, 'marginTop': '4px'}),
            dcc.Loading(html.Div(id='raven-commit-output', style={'marginTop': '8px'}), type='dot'),
        ], style=_SECTION),

        _advanced(),

        dcc.Store(id='raven-upload-token'),
        dcc.Store(id='raven-analysis'),
        dcc.Store(id='raven-refresh-signal', data=0),
    ], style={'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'padding': '16px', 'width': 'min(1150px, 94vw)',
              'maxHeight': '92vh', 'overflowY': 'auto', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})

    return html.Div(body, id='raven-modal', style=MODAL_HIDDEN)


MODAL_VISIBLE = {'display': 'flex', 'position': 'fixed', 'inset': 0, 'zIndex': 2000,
                 'backgroundColor': 'rgba(0,0,0,0.35)', 'alignItems': 'center', 'justifyContent': 'center'}
MODAL_HIDDEN = {**MODAL_VISIBLE, 'display': 'none'}


# ---- Raven annotations as a "Color by" mode (read-only) ---------------------------------------------------------

def _info_label(text, tooltip):
    return html.Div([
        html.Label(text),
        html.Div(className="tooltip-container", children=[
            html.Span(" ⓘ", className="info-icon"),
            html.Span(tooltip, className="tooltip-text"),
        ]),
    ], className="label-with-info", style={'marginTop': '1em'})


_NOTE = {'fontSize': '0.85em', 'color': '#666', 'fontStyle': 'italic', 'marginTop': '4px'}


def raven_color_controls():
    """Raven-only filters shown under "Color by" when "Imported labels (Raven)" is selected."""
    return html.Div([
        _info_label("Annotators", "Only labels from these annotators are used for colours, comparisons and point "
                                  "details. Empty means all annotators."),
        dcc.Dropdown(id='raven-annotator-filter', options=[], value=[], multi=True, placeholder='All annotators'),
        _info_label("Compare annotators", "Agreement: at least two selected annotators labelled the clip, with the "
                                          "same species. Disagreement: they labelled it differently. A clip only one "
                                          "annotator labelled is neither. Unannotated points stay grey."),
        dcc.RadioItems(id='raven-compare-mode', value='all',
                       options=[{'label': 'All annotations', 'value': 'all'},
                                {'label': 'Agreement only', 'value': 'agree'},
                                {'label': 'Disagreements only', 'value': 'disagree'}],
                       labelStyle={'display': 'block'}, inputStyle={'marginRight': '4px'}),
        html.Div(id='raven-compare-hint', style=_NOTE),
        html.Div(id='raven-coverage-note', style=_NOTE),
        dcc.Store(id='raven-color-store', data={}),
    ], id='raven-filter-section', style={'display': 'none'})


def raven_point_info():
    """Raven annotations of the clicked point, under the audio info row (Raven colour mode only)."""
    return html.Div(id='raven-point-info')
