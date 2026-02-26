"""
Table View callback — paginated, cached, sortable, with row-click audio integration.

Architecture:
  - Filter hash cache: skip merge on page/sort/column visibility changes
  - Column headers: clickable html.Th with pattern IDs → sort-store callback
  - Row click: writes to spectrogram-raw-data-store, reusing the existing
    clientside callback that drives audio-player and info display
"""
import hashlib
import json
import time
import traceback

import pandas as pd
from dash import Input, Output, State, html, no_update, ALL, ctx

from callbacks.callbacks_constants import MODEL_DATA_CACHE, CLUSTER_COLORS, MANUAL_LABELS_CACHE
from utils import apply_manual_labels_efficiently

import utils  # for load_audio_segment + compute_spectrogram
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_COLUMNS = [
    ("file_name",      "File",             True),
    ("date",           "Date",             True),
    ("recording_time", "Recording Time",   True),
    ("cluster_start",  "Start Time",       True),
    ("cluster_end",    "End Time",         True),
    ("duration",       "Duration",         True),
    ("cluster_id",     "Cluster",          True),
    ("manual_label",   "Label",            False),  # shown in manual mode
    ("channel",        "Channel",          False),
    ("microlocation",  "Microlocation",    False),
    ("recorder_type",  "Recorder",         False),
    ("clip_count",     "Clip Count",       False),
]

DEFAULT_VISIBLE_COLS = [k for k, _label, vis in ALL_COLUMNS if vis]
PAGE_SIZE = 200

# display col key → dataframe column for sorting
SORT_COL_MAP = {
    "file_name":      "file_name",
    "date":           "day_dt",
    "recording_time": "start_dt",
    "cluster_start":  "clip_time",
    "cluster_end":    "clip_time",     # proxy — sort by start
    "duration":       "clip_duration",
    "cluster_id":     "cluster_id",
    "manual_label":   "manual_label",
    "channel":        "channel",
    "microlocation":  "microlocation",
    "recorder_type":  "recorder_type",
    "clip_count":     "clip_count",
}

# Primary display col → ordered list of secondary DataFrame columns
SECONDARY_SORT: dict[str, list[str]] = {
    "date":           ["clip_time"],
    "file_name":      ["clip_time"],
    "cluster_id":     ["day_dt", "clip_time"],
    "manual_label":   ["day_dt", "clip_time"],
    "channel":        ["day_dt", "clip_time"],
    "microlocation":  ["day_dt", "clip_time"],
    "recorder_type":  ["day_dt", "clip_time"],
    "recording_time": ["day_dt"],
    "cluster_start":  ["day_dt"],
    "cluster_end":    ["day_dt"],
    "duration":       [],
    "clip_count":     ["day_dt", "clip_time"],
}

# Default sort when no column header is clicked
DEFAULT_SORT_COLS = ["file_name", "day_dt", "start_dt", "clip_time"]


def _apply_sort(df: pd.DataFrame, sort_col: str | None, sort_asc: bool) -> pd.DataFrame:
    """Sort by primary column + automatic secondary keys. Applies a default sort when sort_col is None."""
    
    def sort_key(col):
        # Case-insensitive sorting for string columns to avoid splitting uppercase/lowercase
        if col.name in ["file_name", "manual_label", "microlocation", "recorder_type"]:
            return col.astype(str).str.lower()
        return col

    if not sort_col:
        # Default: file → date → recording time → start time
        defaults = [c for c in DEFAULT_SORT_COLS if c in df.columns]
        return df.sort_values(defaults, ascending=True, key=sort_key) if defaults else df
    primary_df_col = SORT_COL_MAP.get(sort_col)
    if not primary_df_col or primary_df_col not in df.columns:
        return df
    secondary = [c for c in SECONDARY_SORT.get(sort_col, []) if c != primary_df_col and c in df.columns]
    cols = [primary_df_col] + secondary
    asc  = [sort_asc]       + [True] * len(secondary)
    return df.sort_values(cols, ascending=asc, key=sort_key)


# Module-level caches
_TABLE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_filter_hash(selected_channels, selected_num_clusters, selected_dates,
                      hour_range, selected_microlocations, selected_recorders,
                      cluster_checkbox_values, cluster_checkbox_ids,
                      color_mode: str = "cluster") -> str:
    try:
        # Use a sorted fingerprint of label keys so any change (add/change) invalidates the cache
        label_fingerprint = hashlib.md5(str(sorted(MANUAL_LABELS_CACHE.keys())).encode()).hexdigest()[:8]
        key = json.dumps([
            sorted(selected_channels or []),
            selected_num_clusters,
            sorted(selected_dates or []),
            list(hour_range or []),
            sorted(selected_microlocations or []),
            sorted(selected_recorders or []),
            cluster_checkbox_values,
            [d.get("index") for d in (cluster_checkbox_ids or [])],
            label_fingerprint,  # invalidate when any label changes
            color_mode,
        ], sort_keys=True, default=str)
        return hashlib.md5(key.encode()).hexdigest()
    except Exception:
        return "invalid"


def _fmt_seconds(seconds):
    try:
        s = float(seconds)
        if s < 0 or pd.isna(s):
            return ""
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"
    except Exception:
        return ""


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _label_color(label: str) -> str:
    """Consistent per-label color using MD5-stable palette index."""
    idx = int(hashlib.md5(str(label).encode()).hexdigest()[:4], 16) % len(CLUSTER_COLORS)
    return CLUSTER_COLORS[idx]


# ---------------------------------------------------------------------------
# Core: filter + vectorized merge  (also preserves mp3_file for row click)
# ---------------------------------------------------------------------------
def _build_merged_table(dff_raw, selected_channels, selected_num_clusters,
                        selected_dates, hour_range, selected_microlocations,
                        selected_recorders, cluster_checkbox_values, cluster_checkbox_ids,
                        apply_labels: bool = False):
    t0 = time.time()
    if dff_raw is None or dff_raw.empty:
        return pd.DataFrame()

    valid_k_values = sorted(dff_raw["cluster_num"].dropna().unique()) if "cluster_num" in dff_raw.columns else []
    active_k = None
    if selected_num_clusters:
        try:
            v = int(selected_num_clusters)
            if v in valid_k_values:
                active_k = v
        except Exception:
            pass
    if active_k is None and valid_k_values:
        active_k = int(valid_k_values[0])

    mask = pd.Series(True, index=dff_raw.index)
    if active_k is not None and "cluster_num" in dff_raw.columns:
        mask &= dff_raw["cluster_num"] == active_k
    if selected_microlocations is not None:
        if not selected_microlocations:
            mask &= False
        else:
            mask &= dff_raw["microlocation"].isin(selected_microlocations)
    if selected_recorders is not None:
        if not selected_recorders:
            mask &= False
        else:
            mask &= dff_raw["recorder_type"].isin(selected_recorders)
    if selected_channels:
        mask &= dff_raw["channel"].isin(selected_channels)
    if selected_dates:
        try:
            mask &= dff_raw["day_dt"].astype(str).str[:10].isin(selected_dates)
        except Exception:
            pass
    if hour_range:
        mask &= (dff_raw["start_hour_float"] >= hour_range[0]) & \
                (dff_raw["start_hour_float"] <= hour_range[1])

    dff = dff_raw[mask].copy()

    selected_clusters = set()
    if cluster_checkbox_values and cluster_checkbox_ids:
        for val, id_dict in zip(cluster_checkbox_values, cluster_checkbox_ids):
            if val and "on" in val:
                try:
                    selected_clusters.add(int(id_dict["index"]))
                except Exception:
                    pass
    if selected_clusters and "cluster_id" in dff.columns:
        dff = dff[dff["cluster_id"].isin(selected_clusters)]

    if dff.empty:
        return pd.DataFrame()

    if "clip_duration" not in dff.columns:
        dff["clip_duration"] = 5.0
    else:
        dff["clip_duration"] = pd.to_numeric(dff["clip_duration"], errors="coerce").fillna(5.0)
    if "clip_count" not in dff.columns:
        dff["clip_count"] = 1

    dff = dff.sort_values(["file_name", "channel", "cluster_id", "clip_time"]).reset_index(drop=True)
    dff["_clip_end"] = dff["clip_time"] + dff["clip_duration"]

    s_file    = dff["file_name"].astype(str)
    s_channel = dff["channel"].astype(str)
    s_cluster = dff["cluster_id"].astype(str)
    prev_end  = dff["_clip_end"].shift(1)

    is_new = (
        (s_file    != s_file.shift(1))    |
        (s_channel != s_channel.shift(1)) |
        (s_cluster != s_cluster.shift(1)) |
        (dff["clip_time"] > prev_end)
    ).copy()
    is_new.iat[0] = True
    dff["_merge_group"] = is_new.cumsum()

    agg: dict = {
        "clip_time":  ("clip_time",  "first"),
        "_clip_end":  ("_clip_end",  "max"),
        "file_name":  ("file_name",  "first"),
        "channel":    ("channel",    "first"),
        "cluster_id": ("cluster_id", "first"),
        "clip_count": ("clip_count", "sum"),
        "day_dt":     ("day_dt",     "first"),
    }
    # Preserve mp3_file so row-click can load audio; apply manual labels
    for col in ["mp3_file", "start_dt", "microlocation", "recorder_type"]:
        if col in dff.columns:
            agg[col] = (col, "first")

    # ALWAYS apply manual labels regardless of color mode.
    # The label column should always be populated and never reset to Unlabeled.
    if MANUAL_LABELS_CACHE:
        dff = apply_manual_labels_efficiently(dff)
    else:
        dff["manual_label"] = "Unlabeled"
        
    def get_merged_label(s):
        valid = [l for l in s if l != "Unlabeled"]
        if valid:
            # Return the most frequent valid label
            return pd.Series(valid).mode()[0]
        return "Unlabeled"
        
    agg["manual_label"] = ("manual_label", get_merged_label)

    merged = dff.groupby("_merge_group").agg(**agg).reset_index(drop=True)
    merged["clip_duration"] = merged["_clip_end"] - merged["clip_time"]
    merged = merged.drop(columns=["_clip_end"])
    print(f"[TABLE] merge: {len(dff_raw)}→{len(merged)} rows in {time.time()-t0:.2f}s")
    return merged


# ---------------------------------------------------------------------------
# Render one page of the table with sortable headers
# ---------------------------------------------------------------------------
def _render_table(page_df: pd.DataFrame, visible_cols: list,
                  sort_col: str | None = None, sort_asc: bool = True,
                  page_offset: int = 0,
                  color_mode: str = "cluster") -> html.Div:
    if page_df.empty:
        return html.Div("No data to display.", style={"padding": "1rem", "color": "#888"})

    col_label_map = {k: lbl for k, lbl, _ in ALL_COLUMNS}

    # Build sortable header cells
    header_cells = []
    for c in visible_cols:
        label = col_label_map.get(c, c)
        arrow = ""
        if sort_col == c:
            arrow = " ▲" if sort_asc else " ▼"
        header_cells.append(
            html.Th(
                label + arrow,
                id={"type": "table-header", "index": c},
                n_clicks=0,
                style={"cursor": "pointer", "userSelect": "none",
                       "whiteSpace": "nowrap",
                       "color": "#2a3f5f" if sort_col == c else "inherit",
                       "fontWeight": "700" if sort_col == c else "600"},
            )
        )

    # Pre-compute display series
    display: dict = {}
    if "file_name" in visible_cols:
        display["file_name"] = page_df["file_name"].astype(str).fillna("")
    if "date" in visible_cols:
        try:
            display["date"] = pd.to_datetime(page_df["day_dt"]).dt.strftime("%Y-%m-%d").fillna("")
        except Exception:
            display["date"] = pd.Series("", index=page_df.index)
    if "recording_time" in visible_cols:
        if "start_dt" in page_df.columns:
            try:
                display["recording_time"] = pd.to_datetime(page_df["start_dt"]).dt.strftime("%H:%M:%S").fillna("")
            except Exception:
                display["recording_time"] = pd.Series("", index=page_df.index)
        else:
            display["recording_time"] = pd.Series("", index=page_df.index)
    if "cluster_start" in visible_cols:
        display["cluster_start"] = page_df["clip_time"].apply(_fmt_seconds)
    if "cluster_end" in visible_cols:
        display["cluster_end"] = (page_df["clip_time"] + page_df["clip_duration"]).apply(_fmt_seconds)
    if "duration" in visible_cols:
        display["duration"] = page_df["clip_duration"].apply(_fmt_seconds)
    if "cluster_id" in visible_cols:
        display["cluster_id"] = page_df["cluster_id"].astype(int).astype(str)
    if "manual_label" in visible_cols:
        src = page_df["manual_label"] if "manual_label" in page_df.columns else pd.Series("Unlabeled", index=page_df.index)
        display["manual_label"] = src.astype(str).fillna("Unlabeled")
    for col_key in ("channel", "microlocation", "recorder_type"):
        if col_key in visible_cols:
            src = page_df[col_key] if col_key in page_df.columns else pd.Series("", index=page_df.index)
            display[col_key] = src.astype(str).fillna("")
    if "clip_count" in visible_cols:
        display["clip_count"] = page_df["clip_count"].astype(int).astype(str)

    # Row background colors — by manual_label in manual mode, by cluster_id otherwise
    if color_mode == "manual" and "manual_label" in page_df.columns:
        bg_colors = [_hex_to_rgba(_label_color(lbl)) for lbl in page_df["manual_label"].fillna("Unlabeled")]
    else:
        cluster_ids = page_df["cluster_id"].astype(int)
        bg_colors = [_hex_to_rgba(CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]) for cid in cluster_ids]

    rows = []
    for i in range(len(page_df)):
        # row-level index in the full merged df (page_offset + i) stored as data attr
        row_idx = page_offset + i
        cells = [html.Td(display[c].iat[i] if c in display else "") for c in visible_cols]
        rows.append(
            html.Tr(
                cells,
                id={"type": "table-row", "index": row_idx},
                n_clicks=0,
                style={
                    "backgroundColor": bg_colors[i],
                    "cursor": "pointer",
                },
                className="table-row-clickable",
            )
        )

    return html.Div(
        html.Table(
            [html.Thead(html.Tr(header_cells)), html.Tbody(rows)],
            className="cluster-table"
        ),
        style={"overflowX": "auto"}
    )


# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------
def register_table_callbacks(app):

    # -- Tab switching (clientside, instant) --
    app.clientside_callback(
        """
        function(nScatter, nTable) {
            var triggered = window.dash_clientside.callback_context.triggered;
            var isTable = triggered.length > 0 &&
                          triggered[0].prop_id === 'tab-btn-table.n_clicks';
            var sp = document.getElementById('scatter-pane');
            var tp = document.getElementById('table-pane');
            var bs = document.getElementById('tab-btn-scatter');
            var bt = document.getElementById('tab-btn-table');
            if (sp && tp && bs && bt) {
                if (isTable) {
                    sp.classList.remove('view-pane--active');
                    tp.classList.add('view-pane--active');
                    bs.classList.remove('view-tab-btn--active');
                    bt.classList.add('view-tab-btn--active');
                } else {
                    tp.classList.remove('view-pane--active');
                    sp.classList.add('view-pane--active');
                    bt.classList.remove('view-tab-btn--active');
                    bs.classList.add('view-tab-btn--active');
                }
            }
            return isTable ? 'table-tab' : 'scatter-tab';
        }
        """,
        Output('main-view-tabs', 'data'),
        Input('tab-btn-scatter', 'n_clicks'),
        Input('tab-btn-table', 'n_clicks'),
        prevent_initial_call=True,
    )

    # -- Prev page (clientside, no server round-trip) --
    app.clientside_callback(
        "function(n, page) { return Math.max(0, (page || 0) - 1); }",
        Output('table-page-store', 'data'),
        Input('table-prev-btn', 'n_clicks'),
        State('table-page-store', 'data'),
        prevent_initial_call=True,
    )

    # -- Next page --
    @app.callback(
        Output('table-page-store', 'data', allow_duplicate=True),
        Input('table-next-btn', 'n_clicks'),
        State('table-page-store', 'data'),
        State('table-total-pages-store', 'data'),
        prevent_initial_call=True,
    )
    def next_page(_, page, total):
        return min((total or 1) - 1, (page or 0) + 1)

    # -- Column header click → update sort store, reset page --
    @app.callback(
        Output('table-sort-store', 'data'),
        Output('table-page-store', 'data', allow_duplicate=True),
        Input({'type': 'table-header', 'index': ALL}, 'n_clicks'),
        State({'type': 'table-header', 'index': ALL}, 'id'),
        State('table-sort-store', 'data'),
        prevent_initial_call=True,
    )
    def update_sort(n_clicks_list, ids, current_sort):
        if not ctx.triggered_id:
            return no_update, no_update
        # triggered_id is a dict like {'type': 'table-header', 'index': 'cluster_id'}
        col_key = ctx.triggered_id.get("index") if isinstance(ctx.triggered_id, dict) else None
        if col_key is None:
            return no_update, no_update

        current_sort = current_sort or {"col": None, "asc": True}
        if current_sort.get("col") == col_key:
            new_asc = not current_sort.get("asc", True)
        else:
            new_asc = True

        return {"col": col_key, "asc": new_asc}, 0  # reset to page 0

    # -- Row click → audio + spectrogram + histogram (mirrors scatter click) --
    @app.callback(
        Output('spectrogram-raw-data-store', 'data', allow_duplicate=True),
        Output('spectrogram-plot', 'figure', allow_duplicate=True),
        Output('table-click-data-store', 'data'),
        Output('fft-warning', 'children', allow_duplicate=True),
        Input({'type': 'table-row', 'index': ALL}, 'n_clicks'),
        State({'type': 'table-row', 'index': ALL}, 'id'),
        State('frequency-scale', 'value'),
        State('fft-window-size', 'value'),
        State('window-overlap', 'value'),
        State('window-type', 'value'),
        State('min-freq', 'value'),
        State('max-freq', 'value'),
        State('num-bins', 'value'),
        State('colormap', 'value'),
        State('db-floor', 'value'),
        State('color-mode-radio', 'value'),
        prevent_initial_call=True,
    )
    def table_row_click(n_clicks_list, ids,
                        frequency_scale, fft_window_size, window_overlap,
                        window_type, min_freq, max_freq, num_bins, colormap, db_floor,
                        color_mode):
        import plotly.graph_objects as go
        if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
            return no_update, no_update, no_update, no_update

        # Ignore fires caused by component re-render (n_clicks==0, not a real click)
        if not ctx.triggered or ctx.triggered[0].get('value', 0) == 0:
            return no_update, no_update, no_update, no_update

        row_global_idx = ctx.triggered_id.get("index")
        if row_global_idx is None:
            return no_update, no_update, no_update, no_update

        cached = _TABLE_CACHE.get("entry")
        if cached is None:
            return no_update, no_update, no_update, "Click a point on the scatter first to load data."

        merged = cached.get("df")
        if merged is None or merged.empty or row_global_idx >= len(merged):
            return no_update, no_update, no_update, no_update

        # Apply the same sort that's currently active
        sort_info = _TABLE_CACHE.get("sort") or {}
        merged = _apply_sort(merged, sort_info.get("col"), sort_info.get("asc", True))

        row = merged.iloc[row_global_idx]

        if "mp3_file" not in merged.columns or pd.isna(row.get("mp3_file")):
            return no_update, no_update, no_update, "Audio path not available for this row."

        try:
            start_time   = float(row["clip_time"])
            clip_duration = float(row.get("clip_duration", 5.0))
            channel       = int(row["channel"])
            mp3_file      = str(row["mp3_file"])

            segment, samplerate = utils.load_audio_segment(
                mp3_file_relative_path=mp3_file,
                clip_time=start_time,
                clip_duration=clip_duration,
                channel=channel,
                padding_s=0.5,
            )

            if segment is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis={'visible': False}, yaxis={'visible': False})
                return ({"x": [], "y": [], "z": [], "info": "Error: Audio load failed",
                         "audio_path": "", "_rev": time.time_ns()},
                        empty_fig, None, "Error: Could not load audio segment.")

            f, t, Sxx_db = utils.compute_spectrogram(
                segment=segment,
                samplerate=samplerate,
                scale=frequency_scale,
                fft_window_size=fft_window_size,
                window_overlap=window_overlap,
                window_type=window_type,
                min_freq=min_freq,
                max_freq=max_freq,
                num_bins=num_bins,
                db_floor=-120,
            )

            floor_val = float(db_floor) if db_floor is not None else -80.0
            Sxx_db[Sxx_db < floor_val] = floor_val
            Sxx_db = np.round(Sxx_db, 2)
            f = np.round(f, 1)
            t = np.round(t, 3)

            # Build Plotly figure (same as compute_spectrogram_data)
            fig = go.Figure(data=go.Heatmap(
                z=Sxx_db, x=t, y=f,
                colorscale=colormap if colormap else 'Viridis',
                showscale=False,
                zmin=floor_val,
                zmax=float(np.max(Sxx_db)),
            ))
            fig.update_layout(
                margin=dict(l=40, r=10, t=10, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Time (s)", showgrid=False),
                yaxis=dict(title="Freq (Hz)", showgrid=False,
                           type='log' if frequency_scale == 'log' else 'linear'),
                dragmode='zoom',
                autosize=True,
            )

            audio_path = (
                f"/audio_segment_normalized/{mp3_file}/{channel}"
                f"/{start_time}/{start_time + clip_duration}"
            )
            info = (
                f"{row.get('file_name', mp3_file)}"
                f" at {start_time:.2f}s"
                f" (cluster {int(row['cluster_id'])})"
            )

            # Synthetic click-data for histogram (matches scatter customdata format).
            # In manual mode the histogram filters by manual_label; in cluster mode by cluster_id.
            if color_mode == "manual" and "manual_label" in row.index and not pd.isna(row["manual_label"]):
                hist_key = str(row["manual_label"])
            else:
                hist_key = str(int(row["cluster_id"]))
            table_click_data = {"points": [{"customdata": [hist_key, None, None]}]}

            store_data = {
                "x": t.tolist(), "y": f.tolist(), "z": Sxx_db.tolist(),
                "audio_path": audio_path, "info": info,
                "_rev": time.time_ns(),
            }
            return store_data, fig, table_click_data, ""

        except Exception:
            traceback.print_exc()
            empty_fig = go.Figure()
            empty_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis={'visible': False}, yaxis={'visible': False})
            return ({"x": [], "y": [], "z": [], "info": "Error loading audio",
                     "audio_path": "", "_rev": time.time_ns()},
                    empty_fig, None, "Error loading audio for selected row.")

    # -- Main table render --
    @app.callback(
        Output("table-view-output", "children"),
        Output("table-page-info", "children"),
        Output("table-total-pages-store", "data", allow_duplicate=True),
        Input("main-view-tabs", "data"),
        Input("model-data-ready-signal", "data"),
        Input("channel-checklist", "value"),
        Input("num-cluster-dropdown", "value"),
        Input({"type": "cluster-checkbox", "index": ALL}, "value"),
        Input("date-dropdown", "data"),
        Input("hour-slider", "value"),
        Input("merge-threshold", "value"),
        Input("microlocation-dropdown", "value"),
        Input("recorder-type-dropdown", "value"),
        Input("table-column-selector", "value"),
        Input("table-page-store", "data"),
        Input("table-sort-store", "data"),
        Input("color-mode-radio", "value"),
        Input("manual-labels-store", "data"),  # Re-render when labels are saved
        State({"type": "cluster-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_table(active_tab, _sig, selected_channels, selected_num_clusters,
                     cluster_checkbox_values, selected_dates, hour_range, _merge_threshold,
                     selected_microlocations, selected_recorders, visible_cols, page,
                     sort_state, color_mode, _label_trigger, cluster_checkbox_ids):

        if active_tab != "table-tab":
            return no_update, no_update, no_update

        try:
            dff_raw = MODEL_DATA_CACHE.get("df")
            if dff_raw is None or dff_raw.empty:
                return (html.Div("No data loaded. Select a location and model.",
                                 style={"padding": "2rem", "color": "#888"}),
                        "", 1)

            if not visible_cols:
                visible_cols = DEFAULT_VISIBLE_COLS

            # Cache check — only re-merge when filters change
            filter_hash = _make_filter_hash(
                selected_channels, selected_num_clusters, selected_dates,
                hour_range, selected_microlocations, selected_recorders,
                cluster_checkbox_values, cluster_checkbox_ids,
                color_mode=color_mode or "cluster",
            )
            cached = _TABLE_CACHE.get("entry")
            if cached and cached.get("hash") == filter_hash:
                merged = cached["df"]
            else:
                merged = _build_merged_table(
                    dff_raw, selected_channels, selected_num_clusters,
                    selected_dates, hour_range, selected_microlocations,
                    selected_recorders, cluster_checkbox_values, cluster_checkbox_ids,
                    apply_labels=(color_mode == "manual"),
                )
                _TABLE_CACHE["entry"] = {"hash": filter_hash, "df": merged}

            if merged.empty:
                return (html.Div("No rows match the current filters.",
                                 style={"padding": "2rem", "color": "#888"}),
                        "0 rows", 1)

            # Apply sorting with secondary keys
            sort_col = sort_state.get("col")
            sort_asc = sort_state.get("asc", True)
            _TABLE_CACHE["sort"] = sort_state  # save for row-click callback

            merged = _apply_sort(merged, sort_col, sort_asc)

            total_rows  = len(merged)
            total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
            page        = max(0, min(int(page or 0), total_pages - 1))
            page_offset = page * PAGE_SIZE
            page_df     = merged.iloc[page_offset : page_offset + PAGE_SIZE]
            page_info   = f"Page {page + 1} of {total_pages}  ({total_rows} rows)"

            t0 = time.time()
            result = _render_table(page_df, visible_cols, sort_col, sort_asc, page_offset,
                                   color_mode=color_mode or "cluster")
            print(f"[TABLE] page {page+1}/{total_pages} rendered in {time.time()-t0:.3f}s")

            return result, page_info, total_pages

        except Exception:
            traceback.print_exc()
            return (html.Div("Error building table — check the server console.",
                             style={"padding": "2rem", "color": "red"}),
                    "error", 1)
