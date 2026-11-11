"""
Imported Raven annotations as a "Color by" mode ("Imported labels (Raven)").

The scatter plot, legend, Table View and histogram reuse their existing code paths; this module supplies the
read-only annotation data, the per-point aggregation (raven_import.aggregate), category colours, the Raven-only
filters (annotators, compare), point details and playback of a selection's original interval.

Raven annotations are never written and never mixed into manual labels: they only add raven_* columns to the
plotted frame in Raven mode.
"""
import os
import threading
import time

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

import read_data
import utils
from callbacks.callbacks_constants import CLUSTER_COLORS, server_cache
from raven_import import aggregate, viewer
from raven_import.config import DEFAULT_SPECIES_VOCABULARY
from raven_import.store import default_db_path

RAVEN_MODE = 'raven'
DB_PATH = None          # tests set this; None -> the configured annotation database
RECORDINGS_ROOT = os.path.join(read_data.BASE_DATA_DIR, "recordings")
STATS_PREFIX = '__raven__:'
INTERVAL_COLOR = "#ff8c00"
ACCENT = "#d6007a"

SPECIAL_COLORS = {
    aggregate.NO_ANNOTATION: '#D9D9D9',    # same grey as "Unlabeled"
    aggregate.DISAGREE: ACCENT,
    aggregate.MULTIPLE: '#4d4d4d',
    aggregate.UNIDENTIFIED: '#8c8c8c',
    aggregate.NO_LABEL: '#a6a6a6',
}
_PALETTE = [c for c in CLUSTER_COLORS if c != '#D9D9D9']

_INFO = {'fontSize': '0.85em', 'borderLeft': f'3px solid {ACCENT}', 'paddingLeft': '6px', 'margin': '4px 0'}

_cache = {}
_cache_lock = threading.RLock()   # get_annotations builds on get_selections


def _db_path():
    return DB_PATH or default_db_path()


def _signature(path):
    try:
        st = os.stat(path)
        return (path, st.st_size, st.st_mtime_ns)
    except OSError:
        return (path, None, None)


def _cached(key, build):
    sig = key + _signature(_db_path())
    with _cache_lock:
        if sig not in _cache:
            _cache[sig] = build()
        return _cache[sig]


def get_selections(project):
    return _cached(("sel", project), lambda: viewer.load_selections(_db_path(), project))


def get_annotations(project, model):
    """Clip annotations (one row per selection x linked clip) of active imports for this project and model."""
    def build():
        sel = get_selections(project)
        links = viewer.load_links(_db_path(), model_name=model)
        return aggregate.clip_annotations(sel, links[links["selection_pk"].isin(set(sel["selection_pk"]))])
    return _cached(("ann", project, model), build)


def get_colouring_annotations(project, model):
    """Clip annotations that are large enough to colour a clip (aggregate.MIN_OVERLAP_S rule)."""
    def build():
        ann = get_annotations(project, model)
        return ann[aggregate.counts_for_colouring(ann)].reset_index(drop=True)
    return _cached(("ann_colour", project, model, aggregate.MIN_OVERLAP_S, aggregate.MIN_FRAC_OF_SELECTION), build)


def available(project):
    return bool(project) and not get_selections(project).empty


def project_annotators(project):
    sel = get_selections(project) if project else None
    return sorted(sel["annotator"].dropna().unique()) if sel is not None and not sel.empty else []


def effective_annotators(selected, project):
    """Empty selection means all annotators of the project."""
    return list(selected) if selected else project_annotators(project)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def default_color(category, others=()):
    if category in SPECIAL_COLORS:
        return SPECIAL_COLORS[category]
    if category in DEFAULT_SPECIES_VOCABULARY:
        return _PALETTE[DEFAULT_SPECIES_VOCABULARY.index(category) % len(_PALETTE)]
    others = sorted(others)
    i = others.index(category) if category in others else 0
    return _PALETTE[(len(DEFAULT_SPECIES_VOCABULARY) + i) % len(_PALETTE)]


def category_colors(categories, overrides=None):
    categories = list(categories)
    others = [c for c in categories if c not in SPECIAL_COLORS and c not in DEFAULT_SPECIES_VOCABULARY]
    colors = {}
    for c in categories:
        if c == aggregate.NO_ANNOTATION:
            colors[c] = SPECIAL_COLORS[c]
        elif overrides and c in overrides:
            colors[c] = overrides[c]
        else:
            colors[c] = default_color(c, others)
    return colors


# ---------------------------------------------------------------------------
# Scatter pipeline hooks (called from update_figure)
# ---------------------------------------------------------------------------
def prepare(dff_macro, project, model, annotators=None, compare=None):
    """Map annotations onto the filtered (possibly merged) frame before sampling.

    Returns the per-point aggregation for the selected annotators, the ids of points annotated by anyone (for
    priority sampling) and legend stats (clip counts per category, after the compare filter).
    """
    ann = get_colouring_annotations(project, model or "")
    mapped = aggregate.map_to_frame(ann, dff_macro)
    landed = mapped[mapped["match"].isin(["exact", "merged"])] if not mapped.empty else mapped
    agg = aggregate.aggregate_points(mapped, effective_annotators(annotators, project))
    if compare in ("agree", "disagree") and not agg.empty:
        counted = agg[agg["raven_status"] == compare]
    else:
        counted = agg
    clip_counts = (dff_macro.drop_duplicates("row_idx").set_index("row_idx")["clip_count"]
                   if "clip_count" in dff_macro.columns else None)
    stats = {}
    annotated_total = 0
    for pid, cat in counted["raven_category"].items():
        n = int(clip_counts.get(pid, 1)) if clip_counts is not None else 1
        stats[cat] = stats.get(cat, 0) + n
        annotated_total += n
    total = int(dff_macro["clip_count"].sum()) if "clip_count" in dff_macro.columns else len(dff_macro)
    stats[aggregate.NO_ANNOTATION] = max(0, total - int(sum(int(clip_counts.get(p, 1)) if clip_counts is not None
                                                         else 1 for p in agg.index)))
    counts = aggregate.mapping_counts(mapped)
    return {
        "agg": agg,
        "annotated_ids": set(landed["point_id"]) if not landed.empty else set(),
        "stats": stats,
        "annotated_total": annotated_total,
        "links_landed": counts["exact"] + counts["merged"],
        "links_ambiguous": counts["ambiguous"],
        "links_project": int(len(ann)),
    }


def stats_entries(prep):
    out = {STATS_PREFIX + k: int(v) for k, v in prep["stats"].items()}
    out['__raven_annotated_total__'] = int(prep["annotated_total"])
    out['__raven_links_landed__'] = int(prep["links_landed"])
    out['__raven_links_ambiguous__'] = int(prep["links_ambiguous"])
    out['__raven_links_project__'] = int(prep["links_project"])
    return out


def selected_categories(values, ids):
    """None when the legend is not rendered (show everything); otherwise ticked + not-yet-rendered categories."""
    if not ids:
        return None
    ticked = {str(i['index']) for v, i in zip(values or [], ids) if v and 'on' in v}
    rendered = {str(i['index']) for i in ids}
    return ticked, rendered


def apply_to_sample(dff_sampled, prep, compare=None, category_filter=None):
    out = aggregate.attach(dff_sampled, prep["agg"])
    out = aggregate.compare_filter(out, compare)
    if category_filter is not None:
        ticked, rendered = category_filter
        keep = out["raven_category"].isin(ticked) | ~out["raven_category"].isin(rendered)
        out = out[keep]
    return out


def hover_text(dff):
    status = dff["raven_status"].map(lambda s: f" ({aggregate.STATUS_TEXT[s]})" if s in ("agree", "disagree") else "")
    labels = dff["raven_labels"].where(dff["raven_labels"] != "", aggregate.NO_ANNOTATION)
    return (labels + status).tolist()


# ---------------------------------------------------------------------------
# Legend rows (rendered by callbacks_clusters.render_cluster_controls in Raven mode)
# ---------------------------------------------------------------------------
def legend_rows(cluster_stats, raven_colors, previous_values, previous_ids, reset):
    stats = {k[len(STATS_PREFIX):]: int(v) for k, v in (cluster_stats or {}).items() if k.startswith(STATS_PREFIX)}
    if not stats:
        return [html.Div("No Raven annotations on the points in view.",
                         style={'fontSize': '0.85em', 'color': '#666', 'fontStyle': 'italic'})]
    annotated_total = int((cluster_stats or {}).get('__raven_annotated_total__', 0))
    selected, rendered = set(), set()
    if not reset and previous_ids:
        for v, i in zip(previous_values or [], previous_ids):
            rendered.add(str(i['index']))
            if v and 'on' in v:
                selected.add(str(i['index']))
    categories = aggregate.category_order(stats)
    colors = category_colors(categories, raven_colors)
    children = []
    for cat in categories:
        is_none = cat == aggregate.NO_ANNOTATION
        count = stats.get(cat, 0)
        if is_none:
            suffix = f" ({count:,})" if count else ""
        else:
            pct = (count / annotated_total * 100) if annotated_total else 0
            suffix = f" ({count:,}, {pct:.1f}%)"
        checked = (cat in selected) if cat in rendered else True
        children.append(html.Div([
            dcc.Checklist(id={'type': 'raven-checkbox', 'index': cat}, options=[{'label': '', 'value': 'on'}],
                          value=['on'] if checked else [],
                          style={'margin': '0', 'padding': '0', 'display': 'flex', 'marginRight': '8px'}),
            dcc.Input(id={'type': 'raven-color-picker', 'index': cat}, type='color', value=colors[cat],
                      disabled=is_none,
                      style={'width': '40px', 'height': '24px', 'padding': '0', 'border': 'none',
                             'cursor': 'not-allowed' if is_none else 'pointer', 'marginLeft': '5px',
                             'backgroundColor': 'transparent', 'flexShrink': 0, 'opacity': 0.6 if is_none else 1.0}),
            html.Span(cat, title=cat, style={'flex': '1', 'minWidth': '80px', 'fontSize': '0.85em', 'color': '#333',
                                             'marginLeft': '8px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
            html.Span(suffix, style={'fontSize': '0.8em', 'color': '#666', 'marginLeft': 'auto',
                                     'marginRight': '16px', 'whiteSpace': 'nowrap'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                  'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'padding': '4px 8px', 'border': '1px solid #ccc',
                  'whiteSpace': 'nowrap', 'width': '100%', 'height': '40px', 'minHeight': '40px',
                  'boxSizing': 'border-box', 'overflow': 'hidden'}))
    return children


def coverage_note(cluster_stats):
    s = cluster_stats or {}
    if '__raven_links_project__' not in s:
        return ""
    text = (f"{s['__raven_links_landed__']:,} of the project's {s['__raven_links_project__']:,} Raven-linked clips "
            "are on points in the current view.")
    if s.get('__raven_links_ambiguous__'):
        text += (f" {s['__raven_links_ambiguous__']:,} fall into several merged points and are not shown "
                 "(turn merging off to see them).")
    return text + " Hollow markers: at least one label is marked uncertain (?)."


# ---------------------------------------------------------------------------
# Point details
# ---------------------------------------------------------------------------
def _fmt_interval(a):
    a = viewer.clean_row(a)
    if a.get("t0") is None or a.get("t1") is None:
        return "—"
    text = f"{a['t0']:.2f}–{a['t1']:.2f} s"
    if a.get("t1_original") is not None:
        text += f" (cut from {a['t1_original']:.2f} s)"
    return text


def render_point_details(hits, hidden, row):
    if hits is None or hits.empty:
        if hidden:
            return html.Div(f"No Raven annotations from the selected annotators on this point "
                            f"({hidden} from other annotators).", style=_INFO)
        return html.Div("No Raven annotation on this point.", style={**_INFO, 'borderLeftColor': '#D9D9D9'})
    status = row.get("raven_status", "none") if row is not None else "none"
    head = "Raven annotations (read-only)"
    if status in ("agree", "disagree"):
        head += f" · annotators {aggregate.STATUS_TEXT[status]}"
    if not aggregate.counts_for_colouring(hits).any():
        head += (f" · overlap below the minimum ({aggregate.MIN_OVERLAP_S:g} s, or "
                 f"{aggregate.MIN_FRAC_OF_SELECTION:.0%} of the selection), so this point is not coloured")
    clip_count = int(row.get("clip_count", 1)) if row is not None else 1
    if clip_count > 1:
        head += f" · {int(hits.groupby('clip_time_s').ngroups)} of {clip_count} clips in this merged point annotated"
    items = []
    for pk, g in hits.groupby("selection_pk", sort=False):
        a = viewer.clean_row(g.iloc[0].to_dict())
        label = a["label"] or aggregate.NO_LABEL
        if a["key"].startswith(aggregate.OTHER_PREFIX):
            interp = "not a species"
        elif a["key"] in (aggregate.UNIDENTIFIED, aggregate.NO_LABEL):
            interp = ""
        else:
            interp = a["key"] if a["key"] != label.rstrip("?").strip() else ""
        counted = aggregate.counts_for_colouring(g)
        overlap = "; ".join(f"clip {r['clip_time_s']:.1f} s: {r['overlap_s']:.2f} s ({r['frac_of_clip']:.0%})"
                            + ("" if ok else " (below minimum overlap, not used for colouring)")
                            for (_, r), ok in zip(g.iterrows(), counted))
        items.append(html.Li([
            html.B(label), f" → {interp}" if interp else "",
            " (uncertain)" if a["uncertain"] else "",
            f" · {a['annotator']}{' · quick pass' if a['quick_pass'] else ''}",
            f" · selection {_fmt_interval(a)} in {a['source_stem'] or '—'} ch{a['channel']}",
            f" · overlap {overlap}",
            f" · {a['table_file']} #{a['raven_selection']} ",
            html.Button("▶ interval", id={'type': 'raven-play-interval', 'index': int(pk)}, n_clicks=0,
                        title="Play the original selection interval in the audio player and spectrogram",
                        style={'fontSize': '0.9em', 'padding': '0 6px', 'cursor': 'pointer'}),
        ]))
    if hidden:
        items.append(html.Li(f"{hidden} more from annotators not selected.", style={'color': '#666'}))
    return html.Div([html.Span(head + ":", style={'color': ACCENT, 'fontWeight': '600'}),
                     html.Ul(items, style={'margin': '2px 0 4px 18px', 'padding': 0})], style=_INFO)


def _plot_id_from_click(click):
    try:
        cd = click["points"][0].get("customdata")
        return cd[1] if cd is not None and len(cd) > 1 else None
    except (TypeError, KeyError, IndexError, AttributeError):
        return None


def interval_spectrogram(rel_audio, channel, t0, t1, settings):
    """Spectrogram + audio URL for an original selection interval, using the app's existing helpers."""
    pad = 0.5
    segment, samplerate = utils.load_audio_segment(mp3_file_relative_path=rel_audio, clip_time=float(t0),
                                                   clip_duration=float(t1) - float(t0), channel=int(channel),
                                                   padding_s=pad)
    if segment is None:
        return None, None
    (frequency_scale, fft_window_size, window_overlap, window_type, min_freq, max_freq, num_bins, colormap,
     db_floor) = settings
    f, t, sxx = utils.compute_spectrogram(segment=segment, samplerate=samplerate, scale=frequency_scale,
                                          fft_window_size=fft_window_size, window_overlap=window_overlap,
                                          window_type=window_type, min_freq=min_freq, max_freq=max_freq,
                                          num_bins=num_bins, db_floor=-120)
    floor_val = float(db_floor) if db_floor is not None else -80.0
    if sxx.size:
        sxx[sxx < floor_val] = floor_val
    t = t + utils.segment_start_time(t0, pad)          # time in the recording file, like the clip spectrogram
    fig = go.Figure(data=go.Heatmap(z=np.round(sxx, 2), x=np.round(t, 3), y=np.round(f, 1),
                                    colorscale=colormap or 'Viridis', showscale=False, zmin=floor_val))
    fig.add_vrect(x0=float(t0), x1=float(t1), line_width=2, line_color=INTERVAL_COLOR,
                  fillcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=40, r=10, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=utils.SPECTROGRAM_TIME_AXIS_TITLE, showgrid=False),
                      yaxis=dict(title="Freq (Hz)", showgrid=False, type='log' if frequency_scale == 'log' else 'linear'),
                      dragmode='zoom', autosize=True)
    audio = f"/audio_segment_normalized/{rel_audio}/{int(channel)}/{float(t0):.3f}/{float(t1):.3f}"
    return fig, audio


SPECTRO_STATES = [State('frequency-scale', 'value'), State('fft-window-size', 'value'), State('window-overlap', 'value'),
                  State('window-type', 'value'), State('min-freq', 'value'), State('max-freq', 'value'),
                  State('num-bins', 'value'), State('colormap', 'value'), State('db-floor', 'value')]

BASE_COLOR_OPTIONS = [{'label': 'Cluster', 'value': 'cluster'}, {'label': 'Manual Labels', 'value': 'manual'}]
RAVEN_COLOR_OPTION = {'label': 'Imported labels (Raven)', 'value': RAVEN_MODE}


def register_raven_label_callbacks(app):

    @app.callback(
        [Output('color-mode-radio', 'options'), Output('color-mode-radio', 'value'),
         Output('raven-annotator-filter', 'options')],
        Input('project-dropdown', 'value'), State('color-mode-radio', 'value'))
    def raven_color_option(project, color_mode):
        if not available(project):
            return BASE_COLOR_OPTIONS, ('cluster' if color_mode == RAVEN_MODE else no_update), []
        return (BASE_COLOR_OPTIONS + [RAVEN_COLOR_OPTION], no_update,
                [{'label': a, 'value': a} for a in project_annotators(project)])

    @app.callback(Output('raven-filter-section', 'style'), Input('color-mode-radio', 'value'))
    def toggle_raven_filters(color_mode):
        return {'display': 'block' if color_mode == RAVEN_MODE else 'none'}

    @app.callback(
        [Output('raven-compare-mode', 'options'), Output('raven-compare-mode', 'value'),
         Output('raven-compare-hint', 'children')],
        Input('raven-annotator-filter', 'value'), Input('project-dropdown', 'value'),
        State('raven-compare-mode', 'value'))
    def compare_options(selected, project, compare):
        n = len(effective_annotators(selected, project))
        can_compare = n >= 2
        options = [{'label': 'All annotations', 'value': 'all'},
                   {'label': 'Agreement only', 'value': 'agree', 'disabled': not can_compare},
                   {'label': 'Disagreements only', 'value': 'disagree', 'disabled': not can_compare}]
        hint = ("" if can_compare else
                "Agreement and disagreement need at least two annotators; the plot shows the selected annotator's "
                "labels.")
        return options, (compare if can_compare else 'all'), hint

    @app.callback(
        Output({'type': 'raven-checkbox', 'index': ALL}, 'value'),
        Input('all-or-none-cluster', 'value'),
        State({'type': 'raven-checkbox', 'index': ALL}, 'options'),
        prevent_initial_call=True)
    def raven_select_all(select_all, options):
        """The legend's existing "Select All" box also drives the Raven category checkboxes."""
        return [['on'] if select_all else [] for _ in (options or [])]

    @app.callback(Output('raven-coverage-note', 'children'), Input('cluster-stats-store', 'data'))
    def raven_coverage(cluster_stats):
        return coverage_note(cluster_stats)

    @app.callback(
        Output('raven-color-store', 'data'),
        Input({'type': 'raven-color-picker', 'index': ALL}, 'value'),
        State({'type': 'raven-color-picker', 'index': ALL}, 'id'), State('raven-color-store', 'data'),
        prevent_initial_call=True)
    def sync_raven_colors(colors, ids, current):
        updated = dict(current or {})
        for color, id_dict in zip(colors or [], ids or []):
            cat = str(id_dict['index'])
            if cat != aggregate.NO_ANNOTATION and color:
                updated[cat] = color
        return updated if updated != (current or {}) else no_update

    @app.callback(
        Output('raven-point-info', 'children'),
        Input('scatter', 'clickData'), Input('table-click-data-store', 'data'), Input('color-mode-radio', 'value'),
        Input('raven-annotator-filter', 'value'),
        State('filtered-data', 'data'), State('project-dropdown', 'value'), State('model-dropdown', 'value'),
        prevent_initial_call=True)
    def show_point_details(click, table_click, color_mode, annotators, filtered_data, project, model):
        if color_mode != RAVEN_MODE or not available(project):
            return ""
        source = table_click if ctx.triggered_id == 'table-click-data-store' else (click or table_click)
        pid = _plot_id_from_click(source)
        key = filtered_data.get("cache_key") if isinstance(filtered_data, dict) else filtered_data
        dff = server_cache.get(key)
        if dff is None or pid is None or pid not in dff.index:
            return ""
        hits, hidden = aggregate.point_annotations(pid, dff, get_annotations(project, model or ""),
                                                   effective_annotators(annotators, project))
        return render_point_details(hits, hidden, dff.loc[pid].to_dict())

    @app.callback(
        [Output('spectrogram-raw-data-store', 'data', allow_duplicate=True),
         Output('spectrogram-plot', 'figure', allow_duplicate=True),
         Output('fft-warning', 'children', allow_duplicate=True)],
        Input({'type': 'raven-play-interval', 'index': ALL}, 'n_clicks'), State('project-dropdown', 'value'),
        *SPECTRO_STATES, prevent_initial_call=True)
    def play_interval(_clicks, project, *settings):
        if not isinstance(ctx.triggered_id, dict) or not (ctx.triggered and ctx.triggered[0].get('value')):
            return no_update, no_update, no_update
        pk = ctx.triggered_id['index']
        sel = get_selections(project)
        match = sel[sel["selection_pk"] == pk]
        if match.empty:
            return no_update, no_update, "This Raven selection is no longer in the annotation database."
        row = viewer.clean_row(match.iloc[0].to_dict())
        rel = viewer.selection_audio_path(row, RECORDINGS_ROOT)
        if rel is None:
            return no_update, no_update, "No audio is available for this Raven selection."
        fig, audio = interval_spectrogram(rel, row["channel"], row["t0"], row["t1"], settings)
        if fig is None:
            return no_update, no_update, "The audio file could not be read."
        info = (f"Raven: {row['label'] or aggregate.NO_LABEL} — {row['annotator']} — {row['source_stem']} "
                f"{_fmt_interval(row)} (original selection, outlined in the spectrogram)")
        return {"audio_path": audio, "info": info, "_rev": time.time_ns()}, fig, ""
