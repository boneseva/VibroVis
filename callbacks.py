import time
import dash
from dash import Input, Output, State, callback_context, html, dcc, ALL, MATCH
import plotly.express as px
import plotly.graph_objects as go
import os
import flask
import soundfile as sf
import io
import pandas as pd
import numpy as np
import uuid
import json

import read_data
import utils
from datetime import datetime, timedelta

initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}
server_cache = {}

CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def generate_date_selector(df, selected_dates=None):
    """
    Generates Date Selector.
    If selected_dates is provided (list of strings 'YYYY-MM-DD'),
    only checks those dates. Otherwise checks ALL.
    """
    if df is None or df.empty or 'day_dt' not in df.columns:
        return html.Div("No date data found.")

    dates = pd.to_datetime(df['day_dt']).dt.date.unique()
    dates = sorted(dates)

    # Create a set for O(1) lookups
    # If selected_dates is None, we assume "Select All" mode.
    selected_set = set(selected_dates) if selected_dates is not None else None

    # --- 1. SIMPLE DROPDOWN (< 10 dates) ---
    if len(dates) < 10:
        all_values = [d.strftime('%Y-%m-%d') for d in dates]

        # Determine value: Intersection of available dates and saved dates
        if selected_set is not None:
            current_value = [d for d in all_values if d in selected_set]
        else:
            current_value = all_values  # Default to all

        options = [{'label': d.strftime('%Y-%m-%d'), 'value': d.strftime('%Y-%m-%d')} for d in dates]

        return html.Div([
            dcc.Dropdown(
                id={'type': 'simple-date-dropdown', 'index': 0},
                options=options,
                value=current_value,
                multi=True,
                clearable=True,
                placeholder="Select dates..."
            )
        ], style={'padding': '5px 0'})

    # --- 2. HIERARCHICAL TREE (>= 10 dates) ---
    year_dict = {}
    for date in dates:
        year = date.year
        month = date.month
        if year not in year_dict: year_dict[year] = {}
        if month not in year_dict[year]: year_dict[year][month] = []
        year_dict[year][month].append(date)

    year_elements = []
    for year in sorted(year_dict.keys()):
        month_elements = []
        for month in sorted(year_dict[year].keys()):
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B')

            day_options = []
            day_values = []

            # Build Day Options and determine which are checked
            current_month_dates = year_dict[year][month]
            for d in current_month_dates:
                d_str = d.strftime('%Y-%m-%d')
                day_options.append({'label': f" {d.day:02d}", 'value': d_str})

                # CHECK LOGIC:
                # If selected_set is None -> Check All
                # If selected_set exists -> Check if d_str is in it
                if selected_set is None or d_str in selected_set:
                    day_values.append(d_str)

            # Determine Month "Select All" State
            # If we selected ALL available days in this month, check the "All" box
            month_select_all_val = []
            if len(day_values) == len(current_month_dates) and len(day_values) > 0:
                month_select_all_val = ['all']

            month_elements.append(
                html.Details([
                    html.Summary(
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'month-select-all', 'year': year, 'month': month},
                                options=[{'label': '', 'value': 'all'}],
                                value=month_select_all_val,
                                labelStyle={'display': 'inline-block', 'marginRight': '5px'},
                                style={'display': 'inline-block', 'marginRight': '5px'},
                                persistence=False
                            ),
                            html.Span(month_name, style={'cursor': 'pointer'})
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                        style={'cursor': 'pointer', 'paddingLeft': '30px', 'listStyle': 'none'}
                    ),
                    html.Div([
                        dcc.Checklist(
                            id={'type': 'date-checklist', 'year': year, 'month': month},
                            options=day_options,
                            value=day_values,  # <--- CRITICAL: Now respects the preset
                            labelStyle={'display': 'block'},
                            style={'paddingLeft': '40px'},
                            persistence=False
                        )
                    ])
                ], open=False, style={'marginBottom': '5px'})
            )

        # Year "Select All" Logic is purely visual here, defaults to checked if children exist
        # You could make this smarter, but 'all' is usually fine for the summary level
        year_elements.append(
            html.Details([
                html.Summary(
                    html.Div([
                        dcc.Checklist(
                            id={'type': 'year-select-all', 'year': year},
                            options=[{'label': '', 'value': 'all'}],
                            value=['all'],  # Simplified for now
                            labelStyle={'display': 'inline-block', 'marginRight': '5px'},
                            style={'display': 'inline-block', 'marginRight': '5px'},
                            persistence=False
                        ),
                        html.Span(str(year), style={'cursor': 'pointer', 'fontWeight': 'bold'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    style={'cursor': 'pointer', 'listStyle': 'none'}
                ),
                html.Div(month_elements, style={'paddingLeft': '15px'})
            ], open=True, style={'marginBottom': '10px'})
        )

    return html.Div([
        html.Div(
            year_elements,
            style={'maxHeight': '300px', 'overflow': 'auto', 'border': '1px solid #ddd', 'padding': '10px',
                   'borderRadius': '4px'}
        ),
        dcc.Store(id='date-tree-structure', data=year_dict)
    ])

def set_initial_data(df):
    global initial_df
    initial_df = df


def merge_clips_vectorized(dff, merge_threshold):
    """Vectorized merge function."""
    if dff.empty:
        return dff
    if 'clip_count' not in dff.columns:
        dff['clip_count'] = 1
    dff['file_name'] = dff['file_name'].astype('category')
    dff['channel'] = dff['channel'].astype('category')
    dff = dff.sort_values(['file_name', 'channel', 'clip_time']).copy()
    same_context = (
            (dff['file_name'] == dff['file_name'].shift(1)) &
            (dff['channel'] == dff['channel'].shift(1)) &
            (dff['cluster_id'] == dff['cluster_id'].shift(1))
    )
    prev_points = dff[['x', 'y']].shift(1).to_numpy()
    curr_points = dff[['x', 'y']].to_numpy()
    distances = np.linalg.norm(curr_points - prev_points, axis=1)
    spatial_proximity = distances < merge_threshold
    prev_clip_end = dff['clip_time'].shift(1) + dff['clip_duration'].shift(1)
    temporal_continuity = dff['clip_time'] < prev_clip_end
    merge_mask = same_context & temporal_continuity & spatial_proximity
    merge_groups = (~merge_mask).cumsum()
    dff['clip_end'] = dff['clip_time'] + dff['clip_duration']

    agg_dict = {
        'x': ('x', 'mean'), 'y': ('y', 'mean'),
        'clip_time': ('clip_time', 'first'), 'clip_end': ('clip_end', 'last'),
        'recorder_type': ('recorder_type', 'first'), 'clip_count': ('clip_count', 'sum'),
        'cluster_id': ('cluster_id', 'first'), 'file_name': ('file_name', 'first'),
        'channel': ('channel', 'first'), 'start_dt': ('start_dt', 'first'),
        'time_of_day': ('time_of_day', 'first'), 'model_name': ('model_name', 'first'),
        'mp3_file': ('mp3_file', 'first'), 'row_idx': ('row_idx', 'first'),
        'cluster_num': ('cluster_num', 'first'),
        'microlocation': ('microlocation', 'first'), 'day_dt': ('day_dt', 'first'),
        'start_hour_float': ('start_hour_float', 'first')
    }
    grouped = dff.groupby(merge_groups).agg(**agg_dict).reset_index(drop=True)
    grouped['clip_duration'] = grouped['clip_end'] - grouped['clip_time']
    grouped = grouped.drop(columns=['clip_end'])
    return grouped


def register_callbacks(dash_app):
    global app
    app = dash_app

    @app.server.route("/audio_segment_normalized/<path:filename>/<int:channel>/<float:start>/<float:end>")
    def serve_audio_segment_normalized(filename, channel, start, end):
        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=filename,
            clip_time=start,
            clip_duration=(end - start),
            channel=channel,
            padding_s=0.5
        )
        if segment is None:
            return flask.abort(404)
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment * (0.99 / peak)
        buf = io.BytesIO()
        sf.write(buf, segment, samplerate, format='mp3')
        buf.seek(0)
        return flask.send_file(buf, mimetype="audio/mp3")

    @app.callback(
        Output('model-data-ready-signal', 'data'),
        Input('model-dropdown', 'value'),
        Input('location-dropdown', 'value')
    )
    def load_data_into_server_cache(selected_model, selected_location):
        """Loads data into MODEL_DATA_CACHE."""
        if not selected_model or not selected_location:
            MODEL_DATA_CACHE['df'] = None
        else:
            data_path = read_data.SAVE_PATH
            if not os.path.exists(data_path):
                print(f"FATAL: Source data not found at {data_path}.")
                MODEL_DATA_CACHE['df'] = None
            else:
                print(f"Loading data for model: {selected_model} AND location: {selected_location}...")
                cols_to_load = ['x', 'y', 'location', 'microlocation', 'model_name', 'channel',
                                'cluster_num', 'cluster_id', 'day_dt', 'start_hour_float',
                                'file_name', 'clip_time', 'clip_duration', 'mp3_file', 'row_idx',
                                'start_dt', 'time_of_day', 'recorder_type']
                try:
                    model_df = pd.read_parquet(
                        data_path,
                        filters=[
                            [('model_name', '==', selected_model),
                             ('location', '==', selected_location)]
                        ],
                        columns=cols_to_load
                    )
                    model_df = model_df.reset_index(drop=True)
                    MODEL_DATA_CACHE['df'] = model_df
                    print(f"Cached {len(model_df)} rows.")
                except Exception as e:
                    print(f"Error loading data: {e}")
                    MODEL_DATA_CACHE['df'] = None
        return time.time()

    @app.callback(
        Output('cluster-list-container', 'children'),
        [Input('num-cluster-dropdown', 'value'),
         Input('model-data-ready-signal', 'data'),
         Input('preset-cluster-selection', 'data')],
        [State('cluster-color-store', 'data'),
         State('cluster-name-store', 'data'),
         State('last-preset-load-time', 'data')]
    )
    def render_cluster_controls(num_clusters, model_ready, preset_selected_ids, current_colors, current_names,
                                last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: dff = initial_df
        if not num_clusters or dff.empty: return []

        try:
            k = int(num_clusters)
            # Handle possible string/int mismatch in filtering
            if 'cluster_num' in dff.columns:
                clusters = sorted(dff[dff['cluster_num'] == k]['cluster_id'].dropna().unique())
            else:
                clusters = sorted(dff['cluster_id'].dropna().unique())
        except:
            return []

        children = []

        # FIX: Increased timeout to 6.0s
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        effective_preset_ids = preset_selected_ids if is_preset_active else None

        for c in clusters:
            c_int = int(c)
            c_str = str(c_int)

            default_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
            color_val = current_colors[c_str] if current_colors and c_str in current_colors else default_color
            name_val = current_names[c_str] if current_names and c_str in current_names else c_str

            # CHECKBOX LOGIC
            is_checked = True
            if effective_preset_ids is not None:
                is_checked = c_int in effective_preset_ids

            row = html.Div([
                dcc.Checklist(
                    id={'type': 'cluster-checkbox', 'index': c_int},
                    options=[{'label': '', 'value': 'on'}],
                    value=['on'] if is_checked else [],
                    style={'margin': '0', 'padding': '0', 'display': 'flex'}
                ),
                dcc.Input(
                    id={'type': 'cluster-color-picker', 'index': c_int},
                    type='color',
                    value=color_val,
                    style={'width': '20px', 'height': '20px', 'padding': '0', 'border': 'none', 'cursor': 'pointer',
                           'marginLeft': '5px', 'backgroundColor': 'transparent'}
                ),
                dcc.Input(
                    id={'type': 'cluster-name-input', 'index': c_int},
                    type='text',
                    value=name_val,
                    debounce=True,
                    placeholder=c_str,
                    style={'width': '60px', 'border': 'none', 'backgroundColor': 'transparent', 'fontSize': '0.85em',
                           'color': '#333', 'marginLeft': '5px', 'textOverflow': 'ellipsis'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px',
                      'padding': '2px 8px', 'border': '1px solid #ccc', 'whiteSpace': 'nowrap'})

            children.append(row)

        return children

    @app.callback(
        Output('cluster-name-store', 'data'),
        Input({'type': 'cluster-name-input', 'index': ALL}, 'value'),
        State({'type': 'cluster-name-input', 'index': ALL}, 'id'),
        State('cluster-name-store', 'data'),
        prevent_initial_call=True
    )
    def sync_cluster_names(new_names, ids, current_store):
        if current_store is None:
            current_store = {}

        updated_store = current_store.copy()

        for name_val, id_dict in zip(new_names, ids):
            idx_str = str(id_dict['index'])

            if name_val and name_val.strip() != "" and name_val != idx_str:
                updated_store[idx_str] = name_val
            else:
                if idx_str in updated_store:
                    del updated_store[idx_str]

        return updated_store

    @app.callback(
        Output('cluster-color-store', 'data'),
        Input({'type': 'cluster-color-picker', 'index': ALL}, 'value'),
        State({'type': 'cluster-color-picker', 'index': ALL}, 'id'),
        prevent_initial_call=True
    )
    def sync_cluster_colors(colors, ids):
        """
        Collects all color inputs and updates the store.
        """
        color_map = {}
        for color, id_dict in zip(colors, ids):
            # id_dict is usually {'type': '...', 'index': <cluster_id>}
            idx = id_dict['index']
            color_map[str(idx)] = color
        return color_map

    @app.callback(
        Output({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
        Input('all-or-none-cluster', 'value'),
        State({'type': 'cluster-checkbox', 'index': ALL}, 'options'),
        prevent_initial_call=True
    )
    def handle_cluster_select_all(select_all_value, options_list):
        """
        Updates all cluster checkboxes when 'Select All' is toggled.
        """
        is_selected = (len(select_all_value) > 0)

        # options_list is a list of lists of options (since it's ALL pattern)
        # e.g. [[{'label': '', 'value': 'on'}], ...]

        new_values = []
        for _ in options_list:
            if is_selected:
                new_values.append(['on'])
            else:
                new_values.append([])
        return new_values

    @app.callback(
        [Output('scatter', 'figure'),
         Output('filtered-data', 'data'),
         Output('clip-count-max-store', 'data'),
         Output('merge-max-store', 'data'),
         Output('sampled-indices-store', 'data'),
         Output('anim-ranges-store', 'data'),
         Output('sampling-info-display', 'children')],
        [Input('model-data-ready-signal', 'data'),
         Input('channel-checklist', 'value'),
         Input('num-cluster-dropdown', 'value'),
         Input({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         Input('cluster-color-store', 'data'),
         Input('date-dropdown', 'data'),
         Input('hour-slider', 'value'),
         Input('max-points', 'value'),
         Input('resample-btn', 'n_clicks'),
         Input('merge-switch', 'on'),
         Input('merge-threshold', 'value'),
         Input('clip-count-threshold', 'value'),
         Input('microlocation-dropdown', 'value'),
         Input('recorder-type-dropdown', 'value')],
        [State('scatter', 'figure'),
         State('sampled-indices-store', 'data'),
         State('location-dropdown', 'value'),
         State('last-preset-load-time', 'data')]
    )
    def update_figure(model_ready_signal, selected_channels, selected_num_clusters,
                      cluster_checkbox_values, cluster_colors_data,
                      selected_dates, hour_range, max_points, n_clicks, merge_on, merge_threshold,
                      clip_count_threshold,
                      selected_microlocations, selected_recorders, current_figure_state, stored_indices,
                      selected_location, last_preset_time):

        ctx = callback_context
        all_triggered_ids = [t['prop_id'] for t in ctx.triggered] if ctx.triggered else []

        is_fresh_load = any('model-data-ready-signal' in t_id for t_id in all_triggered_ids)
        is_resample_click = any('resample-btn' in t_id for t_id in all_triggered_ids)
        is_k_change = any('num-cluster-dropdown' in t_id for t_id in all_triggered_ids)
        is_cluster_checkbox_click = any('cluster-checkbox' in t_id for t_id in all_triggered_ids)

        # 2. RETRIEVE DATA
        dff_raw = MODEL_DATA_CACHE.get('df')

        # --- 3. STALE CACHE GUARD ---
        if dff_raw is not None and not dff_raw.empty and selected_location:
            cached_location = dff_raw['location'].iloc[0]
            if cached_location != selected_location:
                fig = go.Figure()
                fig.update_layout(
                    annotations=[{"text": "Loading data...", "xref": "paper", "yref": "paper",
                                  "showarrow": False, "font": {"size": 16}}],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(visible=False);
                fig.update_yaxes(visible=False)
                return fig, None, 1, 100, [], None, ""

        if dff_raw is None:
            fig = go.Figure()
            fig.update_layout(
                annotations=[{"text": "Select a model and location to begin.", "xref": "paper", "yref": "paper",
                              "showarrow": False, "font": {"size": 16}}],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(visible=False);
            fig.update_yaxes(visible=False)
            return fig, None, 1, 100, [], None, ""

        # --- 4. PRESET VS MANUAL LOGIC ---
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)
        should_force_defaults = is_fresh_load and not is_preset_active

        if should_force_defaults:
            # NUCLEAR OVERWRITE: Reset filters to defaults on manual location change
            selected_channels = []
            selected_num_clusters = None
            selected_microlocations = []
            selected_recorders = []
            selected_dates = []
            hour_range = None
            clip_count_threshold = 1
            cluster_checkbox_values = []
            stored_indices = None

        if is_fresh_load:
            stored_indices = None

        # --- 5. ROBUST INPUT PARSING ---
        selected_clusters = []
        has_checkbox_inputs = False

        if not should_force_defaults and ctx.inputs_list and len(ctx.inputs_list) > 3:
            cluster_inputs = ctx.inputs_list[3]
            if isinstance(cluster_inputs, list) and len(cluster_inputs) > 0:
                has_checkbox_inputs = True
                for input_item in cluster_inputs:
                    val = input_item.get('value')
                    id_dict = input_item.get('id')
                    if val and 'on' in val:
                        try:
                            selected_clusters.append(int(id_dict['index']))
                        except:
                            pass

        # --- 6. SERVER-SIDE AUTO-CORRECTION ---
        valid_k_values = sorted(dff_raw['cluster_num'].dropna().unique())
        active_k = None

        if selected_num_clusters:
            try:
                if int(selected_num_clusters) in valid_k_values:
                    active_k = int(selected_num_clusters)
            except:
                pass

        if active_k is None and valid_k_values:
            active_k = int(valid_k_values[0])

        # --- 7. MERGE LOGIC ---
        dff_base = None
        max_distance = 100
        if merge_on:
            current_cache_key = (active_k, merge_threshold)
            stored_cache_key = MERGED_DATA_CACHE.get('key')
            if current_cache_key != stored_cache_key or MERGED_DATA_CACHE.get('df') is None:
                if active_k is None:
                    dff_base = pd.DataFrame()
                else:
                    dff_to_merge = dff_raw[dff_raw['cluster_num'] == active_k].copy()
                    merged_dff = merge_clips_vectorized(dff_to_merge, merge_threshold)
                    MERGED_DATA_CACHE['df'] = merged_dff
                    MERGED_DATA_CACHE['key'] = current_cache_key
                    dff_base = merged_dff
            else:
                dff_base = MERGED_DATA_CACHE.get('df')

            if not dff_base.empty:
                min_v, max_v = dff_base['x'].min(), dff_base['x'].max()
                min_y, max_y = dff_base['y'].min(), dff_base['y'].max()
                max_distance = int(np.sqrt((max_v - min_v) ** 2 + (max_y - min_y) ** 2))
        else:
            dff_base = dff_raw

        # --- 8. FILTERING ---
        dff_macro = dff_base.copy()

        if active_k is not None:
            try:
                dff_macro = dff_macro[dff_macro['cluster_num'] == active_k]
            except:
                pass

        total_clips_at_location = len(dff_macro)

        if selected_microlocations:
            valid_micros = set(dff_macro['microlocation'].unique())
            relevant_micros = [m for m in selected_microlocations if m in valid_micros]
            if relevant_micros:
                dff_macro = dff_macro[dff_macro['microlocation'].isin(relevant_micros)]

        if selected_recorders:
            valid_recs = set(dff_macro['recorder_type'].unique())
            relevant_recs = [r for r in selected_recorders if r in valid_recs]
            if relevant_recs:
                dff_macro = dff_macro[dff_macro['recorder_type'].isin(relevant_recs)]

        if selected_channels:
            valid_chans = set(dff_macro['channel'].unique())
            relevant_chans = [c for c in selected_channels if c in valid_chans]
            if relevant_chans:
                dff_macro = dff_macro[dff_macro['channel'].isin(relevant_chans)]

        if selected_dates:
            valid_dates_in_data = set(pd.to_datetime(dff_macro['day_dt']).dt.strftime('%Y-%m-%d'))
            relevant_dates = [d for d in selected_dates if d in valid_dates_in_data]
            if relevant_dates:
                selected_datetimes = pd.to_datetime(relevant_dates).normalize()
                dff_macro = dff_macro[dff_macro['day_dt'].isin(selected_datetimes)]

        if hour_range:
            dff_macro = dff_macro[
                (dff_macro['start_hour_float'] >= hour_range[0]) & (dff_macro['start_hour_float'] <= hour_range[1])]

        if "clip_count" not in dff_macro.columns:
            dff_macro["clip_count"] = 1
        if clip_count_threshold and clip_count_threshold > 1:
            dff_macro = dff_macro[dff_macro["clip_count"] >= int(clip_count_threshold)]

        total_clips_available = dff_macro['clip_count'].sum() if not dff_macro.empty else 0

        # --- 9. SAMPLING (WITH SAFETY NET) ---
        dff_sampled = dff_macro
        new_indices_to_store = dash.no_update

        # Trigger resampling if: Fresh Load, Click, K Changed, No Indices, or other filters
        should_resample = is_fresh_load or is_resample_click or is_k_change or not stored_indices

        if not should_resample:
            other_filters = ['merge-switch', 'merge-threshold', 'date-dropdown', 'hour-slider',
                             'microlocation-dropdown']
            if any(f in t_id for t_id in all_triggered_ids for f in other_filters):
                should_resample = True

        if max_points and dff_macro['clip_count'].sum() > max_points:
            if should_resample:
                # PRIMARY RESAMPLE PATH
                rng = np.random.default_rng(n_clicks if is_resample_click else 42)
                shuffled_indices = rng.permutation(dff_macro.index)
                if len(shuffled_indices) > 0:
                    shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                    cumulative_counts = np.cumsum(shuffled_counts)
                    cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                    final_indices = shuffled_indices[:cutoff_idx]
                    if final_indices.size == 0 and len(dff_macro) > 0: final_indices = shuffled_indices[:1]
                    dff_sampled = dff_macro.loc[final_indices]
                    new_indices_to_store = dff_sampled['row_idx'].tolist()
                else:
                    dff_sampled = dff_macro
                    new_indices_to_store = []
            else:
                # TRY USING STORED INDICES
                dff_sampled = dff_macro[dff_macro['row_idx'].isin(stored_indices)]

                # --- SAFETY NET (THE FIX) ---
                # If using stored indices resulted in 0 rows, but we actually HAVE data,
                # it means the indices were stale (race condition). Force resample now.
                if dff_sampled.empty and not dff_macro.empty:
                    rng = np.random.default_rng(42)
                    shuffled_indices = rng.permutation(dff_macro.index)
                    if len(shuffled_indices) > 0:
                        shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                        cumulative_counts = np.cumsum(shuffled_counts)
                        cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                        final_indices = shuffled_indices[:cutoff_idx]
                        if final_indices.size == 0 and len(dff_macro) > 0: final_indices = shuffled_indices[:1]
                        dff_sampled = dff_macro.loc[final_indices]
                        new_indices_to_store = dff_sampled['row_idx'].tolist()

        else:
            new_indices_to_store = []
            dff_sampled = dff_macro

        dff_final = dff_sampled.copy()

        # --- 10. VISUAL CLUSTER FILTERING ---
        if has_checkbox_inputs and not should_force_defaults:
            # FIX: Only filter if user clicked checkbox or preset is active.
            # If K changed, ignore old checkboxes.
            should_apply_checkbox_filter = is_cluster_checkbox_click or is_preset_active

            if should_apply_checkbox_filter:
                if not dff_final.empty:
                    temp_cluster_ids = pd.to_numeric(dff_final['cluster_id'], errors='coerce')
                    if selected_clusters:
                        # Intersection Check
                        valid_clusters_in_data = set(temp_cluster_ids.unique())
                        selection_set = set(selected_clusters)
                        if not selection_set.isdisjoint(valid_clusters_in_data):
                            dff_final = dff_final[temp_cluster_ids.isin(selected_clusters)]
                        else:
                            if is_preset_active:
                                dff_final = dff_final[temp_cluster_ids.isin(selected_clusters)]
                    else:
                        dff_final = dff_final[dff_final['cluster_id'] == -9999]
            else:
                pass  # Ignore checkboxes if K changed or during transitions

        dff = dff_final

        clips_visible = dff_final['clip_count'].sum() if not dff_final.empty else 0
        clips_sampled = dff_sampled['clip_count'].sum() if not dff_sampled.empty else 0

        if total_clips_available > 0:
            sampling_text = f"Visible: {int(clips_visible):,} | Filtered: {int(total_clips_available):,} | Total: {int(total_clips_at_location):,}"
        else:
            sampling_text = "No data available"

        # --- 11. PLOT GENERATION ---
        x_range, y_range = None, None
        if current_figure_state and not is_fresh_load and 'initial_load' not in all_triggered_ids and not is_k_change:
            try:
                x_range = current_figure_state['layout']['xaxis']['range']
                y_range = current_figure_state['layout']['yaxis']['range']
            except KeyError:
                pass

        if dff.empty:
            fig = go.Figure()
            fig.update_layout(annotations=[
                {"text": "No data found.", "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 16}}],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            if x_range:
                fig.update_xaxes(visible=False, range=x_range);
                fig.update_yaxes(visible=False, range=y_range)
            else:
                fig.update_xaxes(visible=False);
                fig.update_yaxes(visible=False)
            return fig, None, 1, max_distance, new_indices_to_store, None, ""

        dff = dff.reset_index(drop=True)
        dff['plot_id'] = dff.index
        dff['marker_size'] = 10 + 1 * (dff['clip_count'] - 1)
        dff['cluster_id'] = dff['cluster_id'].astype(str)
        dff['cache_key'] = str(uuid.uuid4())

        if 'start_hour_float' not in dff.columns: dff['start_hour_float'] = 0
        if 'day_dt' in dff.columns:
            dff['day_int'] = dff['day_dt'].dt.dayofyear
        else:
            dff['day_int'] = 1

        final_color_map = {}
        unique_clusters = dff['cluster_id'].unique()
        for cid in unique_clusters:
            if cluster_colors_data and cid in cluster_colors_data:
                final_color_map[cid] = cluster_colors_data[cid]
            else:
                try:
                    c_int = int(float(cid))
                    final_color_map[cid] = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
                except:
                    final_color_map[cid] = '#888888'

        fig = px.scatter(
            dff, x="x", y="y", color="cluster_id", size="marker_size",
            color_discrete_map=final_color_map,
            hover_data=["clip_count", "file_name", "clip_time", "channel"],
            custom_data=["cluster_id", "row_idx", "plot_id", "start_hour_float", "day_int"]
        )
        fig.update_traces(marker={'sizeref': 1, 'sizemode': 'diameter'})
        fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        if x_range:
            fig.update_xaxes(visible=False, range=x_range);
            fig.update_yaxes(visible=False, range=y_range)
        else:
            fig.update_xaxes(visible=False);
            fig.update_yaxes(visible=False)

        max_clip_count = int(dff['clip_count'].max()) if not dff.empty else 1
        server_cache[dff['cache_key'].iloc[0]] = dff

        min_h = dff['start_hour_float'].min()
        max_h = dff['start_hour_float'].max()
        min_d = dff['day_int'].min()
        max_d = dff['day_int'].max()
        ranges_data = {'daily': [min_h, max_h], 'yearly': [min_d, max_d]}

        return fig, dff['cache_key'].iloc[
            0], max_clip_count, max_distance, new_indices_to_store, ranges_data, sampling_text

    @app.callback(
        Output('spectrogram-raw-data-store', 'data'),
        Output('fft-warning', 'children'),
        [Input("scatter", "clickData"),
         Input('filtered-data', 'data'),
         Input('frequency-scale', 'value'),
         Input('fft-window-size', 'value'),
         Input('window-overlap', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value')],
        prevent_initial_call=True)
    def compute_spectrogram_data(clickData, filtered_data_cache_key,
                                 frequency_scale, fft_window_size, window_overlap,
                                 window_type, min_freq, max_freq, num_bins):

        if not clickData or not filtered_data_cache_key:
            return dash.no_update, ""

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
            return dash.no_update, "Error: Filtered data not found in cache."

        point = clickData["points"][0]
        plot_id = point["customdata"][2]

        try:
            row = dff.loc[plot_id]
        except KeyError:
            return dash.no_update, "Error: Clicked point not found. Please re-filter."

        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=row['mp3_file'],
            clip_time=float(row['clip_time']),
            clip_duration=float(row['clip_duration']),
            channel=int(row['channel']),
            padding_s=0.5
        )

        if segment is None:
            return dash.no_update, "Error: Could not load audio segment."

        f, t, Sxx_db = utils.compute_spectrogram(
            segment=segment, samplerate=samplerate, scale=frequency_scale,
            fft_window_size=fft_window_size, window_overlap=window_overlap,
            window_type=window_type, min_freq=min_freq, max_freq=max_freq,
            num_bins=num_bins,
            db_floor=-120
        )

        if Sxx_db.size == 0:
            return dash.no_update, "Warning: Spectrogram computation failed."

        start_time = float(row['clip_time'])
        audio_path = f"/audio_segment_normalized/{row['mp3_file']}/{int(row['channel'])}/{start_time}/{start_time + float(row['clip_duration'])}"
        info = f"{row['file_name']} at {start_time:.2f}s (cluster {row['cluster_id']})"

        return {'x': t.tolist(), 'y': f.tolist(), 'z': Sxx_db.tolist(), 'audio_path': audio_path, 'info': info,
                '_rev': time.time_ns()}, ""

    @app.callback(
        [Output("info", "children", allow_duplicate=True),
         Output("audio-player", "src", allow_duplicate=True),
         Output("spectrogram-plot", "figure"),
         Output('spectrogram-plot-container', 'key')],
        [Input('spectrogram-raw-data-store', 'data'),
         Input('colormap', 'value'),
         Input('db-floor', 'value')],
        prevent_initial_call=True)
    def update_spectrogram_plot_from_cache(data, colormap, db_floor):
        if not data:
            data = {'x': [], 'y': [], 'z': [], 'info': 'No data', 'audio_path': '', '_rev': time.time_ns()}

        z_data = np.array(data['z'])
        if z_data.size > 0:
            z_data[z_data < db_floor] = db_floor

        fig = go.Figure(data=go.Heatmap(
            x=data['x'], y=data['y'], z=z_data.tolist(),
            colorscale=colormap,
            zmin=db_floor,
            zmax=np.max(z_data) if z_data.size > 0 else 0,
            colorbar=dict(title='dB')
        ))

        fig.update_layout(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Frequency (Hz)", type='log'),
            margin=dict(l=40, r=10, t=20, b=80), uirevision=data['_rev']
        )
        return data['info'], data['audio_path'], fig, str(data['_rev'])

    @app.callback(
        [Output('merge-threshold-container', 'style'),
         Output('clip-count-threshold-container', 'style'),
         Output('clip-count-threshold', 'value')],
        Input('merge-switch', 'on'))
    def toggle_merge_sliders(merge_on):
        style = {'display': 'block'} if merge_on else {'display': 'none'}
        return style, style, 1 if merge_on else 1

    @app.callback(
        Output('cluster-histogram', 'figure'),
        [Input("scatter", "clickData"),
         Input('histogram-type-dropdown', 'value'),
         Input('histogram-time-scale-store', 'data'),
         Input('cluster-color-store', 'data'),  # <-- Added dynamic colors
         State('filtered-data', 'data')])
    def show_histogram_for_clicked_cluster(clickData, hist_type, time_scale, cluster_colors, filtered_data_cache_key):
        if not clickData or not filtered_data_cache_key:
            fig = go.Figure()
            title = "Time of Day" if time_scale == 'daily' else "Week of Year"
            rng = [0, 1440] if time_scale == 'daily' else [1, 53]
            fig.update_layout(
                xaxis=dict(title=title, range=rng),
                yaxis_title="Count", showlegend=False,
                margin=dict(l=0, r=0, t=20, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                annotations=[
                    {"text": "Click a point to see its histogram", "xref": "paper", "yref": "paper", "showarrow": False,
                     "font": {"size": 14}}]
            )
            return fig

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None: return go.Figure().update_layout(title_text="Error: Data not found in cache.")

        cluster_id = str(clickData["points"][0]["customdata"][0])
        cluster_df = dff[dff['cluster_id'] == cluster_id].copy()

        # Determine color for this cluster
        bar_color = '#CCCCCC'
        if cluster_colors and cluster_id in cluster_colors:
            bar_color = cluster_colors[cluster_id]
        else:
            try:
                c_int = int(float(cluster_id))
                bar_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
            except:
                pass

        if cluster_df.empty: return go.Figure().update_layout(title_text=f"No data for cluster {cluster_id}")

        if time_scale == 'daily':
            bin_width = 30
            cluster_df['time_of_day_minutes'] = cluster_df['time_of_day'].dt.hour * 60 + cluster_df[
                'time_of_day'].dt.minute
            cluster_df['bin'] = (cluster_df['time_of_day_minutes'] // bin_width) * bin_width

            if hist_type == 'presence':
                group_cols = ['bin']
                if 'location' in cluster_df.columns: group_cols.append('location')
                if 'channel' in cluster_df.columns: group_cols.append('channel')
                presence = cluster_df.drop_duplicates(subset=group_cols).groupby('bin').size().reset_index(
                    name='present')
                all_bins_df = pd.DataFrame({'bin': np.arange(0, 1440, bin_width)})
                presence = all_bins_df.merge(presence, on='bin', how='left').fillna(0)
                fig = px.bar(presence, x='bin', y='present', color_discrete_sequence=[bar_color])
                fig.update_layout(yaxis_title="Presence Count")
            else:
                fig = px.histogram(cluster_df, x='time_of_day_minutes', color_discrete_sequence=[bar_color])
                fig.update_traces(xbins=dict(start=0, end=1440, size=bin_width))
                fig.update_layout(yaxis_title="Clip Count")

            fig.update_layout(xaxis=dict(title="Time of Day", range=[0, 1440], tickvals=np.arange(0, 1440, 60),
                                         ticktext=[f"{h // 60:02d}:00" for h in np.arange(0, 1441, 60)]),
                              showlegend=False, margin=dict(l=0, r=0, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            return fig

        elif time_scale == 'yearly':
            if 'day_dt' not in cluster_df.columns: return go.Figure().update_layout(
                title_text="Error: 'day_dt' column not found.")
            cluster_df['day_dt'] = pd.to_datetime(cluster_df['day_dt'])
            cluster_df['week_of_year'] = cluster_df['day_dt'].dt.isocalendar().week

            if hist_type == 'presence':
                group_cols = ['week_of_year']
                if 'location' in cluster_df.columns: group_cols.append('location')
                if 'channel' in cluster_df.columns: group_cols.append('channel')
                presence = cluster_df.drop_duplicates(subset=group_cols).groupby('week_of_year').size().reset_index(
                    name='present')
                all_bins_df = pd.DataFrame({'week_of_year': np.arange(1, 54)})
                presence = all_bins_df.merge(presence, on='week_of_year', how='left').fillna(0)
                fig = px.bar(presence, x='week_of_year', y='present', color_discrete_sequence=[bar_color])
                fig.update_layout(yaxis_title="Presence Count")
            else:
                fig = px.histogram(cluster_df, x='week_of_year', color_discrete_sequence=[bar_color])
                fig.update_traces(xbins=dict(start=1, end=54, size=1))
                fig.update_layout(yaxis_title="Clip Count")

            fig.update_layout(xaxis=dict(title="Week of Year", range=[0.5, 53.5], tickvals=np.arange(1, 54, 4)),
                              showlegend=False, margin=dict(l=0, r=0, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            return fig

    @app.callback(
        Output('histogram-time-scale-store', 'data'),
        Input('histogram-click-wrapper', 'n_clicks'),
        State('histogram-time-scale-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_histogram_time_scale(n_clicks, current_scale):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return 'yearly' if current_scale == 'daily' else 'daily'

    @app.callback(
        [Output('clip-count-threshold', 'max'), Output('clip-count-threshold', 'marks')],
        Input('clip-count-max-store', 'data'))
    def update_clip_count_slider(max_clip_count):
        max_val = max_clip_count or 1
        marks = {i: str(i) for i in range(1, max_val + 1, max(1, max_val // 10))}
        if max_val > 1: marks[1] = '1'; marks[max_val] = str(max_val)
        return max_val, marks

    @app.callback(
        [Output('merge-threshold', 'max'), Output('merge-threshold', 'marks')],
        Input('merge-max-store', 'data'))
    def update_merge_slider(max_merge):
        max_val = max_merge or 1
        marks = {i: str(i) for i in range(1, max_val + 1, max(1, max_val // 10))}
        if max_val > 1: marks[1] = '1'; marks[max_val] = str(max_val)
        return max_val, marks

    @app.callback(
        [Output('microlocation-dropdown', 'options', allow_duplicate=True),
         Output('microlocation-dropdown', 'value', allow_duplicate=True)],
        Input('model-data-ready-signal', 'data'),
        [State('microlocation-dropdown', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_microlocation_options(model_ready, current_micros, last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: return [], []

        micros = sorted(dff['microlocation'].dropna().unique())
        options = [{'label': m, 'value': m} for m in micros]

        # FIX: Increased timeout to 6.0s
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            # Update options (necessary), but KEEP the value set by the preset
            return options, dash.no_update

        # Default: RESET to Empty
        return options, []

    @app.callback(
        [Output('recorder-type-dropdown', 'options', allow_duplicate=True),
         Output('recorder-type-dropdown', 'value', allow_duplicate=True)],
        [Input('model-data-ready-signal', 'data'),
         Input('microlocation-dropdown', 'value')],
        [State('recorder-type-dropdown', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_recorder_options(model_ready, selected_microlocation, current_recorders, last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: return [], []

        mask = pd.Series(True, index=dff.index)
        if selected_microlocation:
            mask &= dff['microlocation'].isin(selected_microlocation)
        if not mask.any(): return [], []

        recorders = sorted(dff.loc[mask, 'recorder_type'].dropna().unique())
        options = [{'label': m, 'value': m} for m in recorders]

        # FIX: Increased timeout to 6.0s
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            # Update options, but KEEP the value set by the preset
            return options, dash.no_update

        return options, []

    @app.callback(
        [Output('channel-checklist', 'options', allow_duplicate=True),
         Output('channel-checklist', 'value', allow_duplicate=True)],
        Input('model-data-ready-signal', 'data'),
        State("all-or-none-channel", "value"),
        State('channel-checklist', 'value'),
        prevent_initial_call=True
    )
    def update_channel_options_and_values(model_ready, select_all, current_channels):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty:
            dff = initial_df

        if dff.empty:
            return [], []

        channels = sorted(dff['channel'].dropna().unique())
        options = [{'label': str(c), 'value': c} for c in channels]

        if current_channels:
            valid_channels = [c for c in current_channels if c in channels]
            if len(valid_channels) == len(current_channels) and len(valid_channels) > 0:
                return options, dash.no_update
            elif len(valid_channels) > 0:
                return options, valid_channels

        value = channels if select_all else []
        return options, value

    @app.callback(
        Output("channel-checklist", "value", allow_duplicate=True),
        Input("all-or-none-channel", "value"),
        State("channel-checklist", "options"),
        prevent_initial_call=True
    )
    def select_all_none_channel(all_selected, options):
        return [opt["value"] for opt in options] if all_selected else []

    @app.callback(
        [Output('audio-player', 'autoPlay'),
         Output('autoplay-toggle-btn', 'children'),
         Output('autoplay-toggle-btn', 'className')],
        Input('autoplay-toggle-btn', 'n_clicks'),
        State('audio-player', 'autoPlay'))
    def toggle_autoplay(n_clicks, current_autoplay):
        if n_clicks is None or n_clicks == 0: return False, "Autoplay: OFF", 'app-button autoplay-off'
        new_autoplay = not current_autoplay
        return new_autoplay, "Autoplay: ON" if new_autoplay else "Autoplay: OFF", 'app-button autoplay-on' if new_autoplay else 'app-button autoplay-off'

    @app.callback(
        Output('filter-container', 'className'),
        Output('toggle-filters-btn', 'children'),
        Input('toggle-filters-btn', 'n_clicks'),
        State('filter-container', 'className'),
        prevent_initial_call=True
    )
    def toggle_filter_panel(n_clicks, current_class):
        return ('filters-collapsed', '<') if current_class == 'filters-expanded' else ('filters-expanded', '>')

    @app.callback(
        [Output('anim-progress-slider', 'min'),
         Output('anim-progress-slider', 'max'),
         Output('anim-progress-slider', 'marks'),
         Output('anim-progress-slider', 'value'),
         Output('anim-unit-label', 'children')],
        [Input('anim-mode', 'value'),
         Input('anim-ranges-store', 'data')],
        State('anim-progress-slider', 'value')
    )
    def update_animation_slider_config(mode, ranges, current_val):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
        if not ranges: return 0, 24, {0: '0h', 24: '24h'}, 0, " (hours)"

        if mode == 'yearly':
            min_val, max_val = map(int, ranges['yearly'])
            base_date = datetime(2024, 1, 1)
            total_days = max_val - min_val
            step = max(1, total_days // 10)
            marks = {}
            for i in range(min_val, max_val + 1, step):
                marks[i] = {'label': (base_date + timedelta(days=i - 1)).strftime('%b %d'),
                            'style': {'whiteSpace': 'nowrap'}}
            marks[min_val] = {'label': (base_date + timedelta(days=min_val - 1)).strftime('%b %d'),
                              'style': {'whiteSpace': 'nowrap'}}
            marks[max_val] = {'label': (base_date + timedelta(days=max_val - 1)).strftime('%b %d'),
                              'style': {'whiteSpace': 'nowrap'}}
            new_val = min_val if 'anim-mode' in trigger else max(min_val, min(current_val, max_val))
            return min_val, max_val, marks, new_val, " (days)"
        else:
            raw_min, raw_max = ranges['daily']
            min_limit = int(np.floor(raw_min));
            max_limit = int(np.ceil(raw_max))
            if min_limit == max_limit:
                if max_limit < 24:
                    max_limit += 1
                else:
                    min_limit = max(0, min_limit - 1)
            span = max_limit - min_limit
            step = max(1, span // 6)
            marks = {i: f"{i:02d}:00" for i in range(min_limit, max_limit + 1, step)}
            marks[min_limit] = f"{min_limit:02d}:00";
            marks[max_limit] = f"{max_limit:02d}:00"
            new_val = raw_min if 'anim-mode' in trigger else max(raw_min, min(current_val, raw_max))
            return min_limit, max_limit, marks, new_val, " (hours)"


    @app.callback(
        [Output({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'value'),
         Output({'type': 'month-select-all', 'year': MATCH, 'month': MATCH}, 'value')],
        [Input({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'value'),
         Input({'type': 'month-select-all', 'year': MATCH, 'month': MATCH}, 'value')],
        [State({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'options')],
        prevent_initial_call=True
    )
    def sync_month_and_dates(selected_dates_in, month_select_in, options):
        triggered_id = dash.callback_context.triggered_id
        if isinstance(triggered_id, dict):
            triggered_type = triggered_id.get('type')
            if triggered_type == 'month-select-all':
                if options is None: return dash.no_update, dash.no_update
                all_dates = [opt['value'] for opt in options]
                return (all_dates, dash.no_update) if month_select_in == ['all'] else ([], dash.no_update)
            elif triggered_type == 'date-checklist':
                if options is None: return dash.no_update, dash.no_update
                selected_dates_set = set() if selected_dates_in is None else set(selected_dates_in)
                all_options_set = set(opt['value'] for opt in options)
                return (dash.no_update, ['all']) if selected_dates_set == all_options_set else (dash.no_update, [])
        return dash.no_update, dash.no_update

    @app.callback(
        Output({'type': 'year-select-all', 'year': MATCH}, 'value'),
        Input({'type': 'month-select-all', 'year': MATCH, 'month': ALL}, 'value'),
        prevent_initial_call=True
    )
    def update_year_checkbox(all_month_checkboxes):
        if not all_month_checkboxes: return []
        all_months_selected = all(['all' in month_val if month_val else False for month_val in all_month_checkboxes])
        return ['all'] if all_months_selected and len(all_month_checkboxes) > 0 else []


    @app.callback(
        Output({'type': 'month-container-wrapper', 'year': MATCH, 'month': MATCH}, 'style'),
        Input({'type': 'year-toggle', 'year': MATCH, 'month': MATCH}, 'n_clicks'),
        State({'type': 'month-container-wrapper', 'year': MATCH, 'month': MATCH}, 'style'),
        prevent_initial_call=True
    )
    def toggle_year_collapse(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return {'display': 'block'} if current_style and current_style.get('display') == 'none' else {'display': 'none'}

    @app.callback(
        [Output('hour-slider', 'min', allow_duplicate=True),
         Output('hour-slider', 'max', allow_duplicate=True),
         Output('hour-slider', 'value', allow_duplicate=True),
         Output('hour-slider', 'marks', allow_duplicate=True)],
        Input('model-data-ready-signal', 'data'),
        [State('hour-slider', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_hour_slider_configuration(model_ready, current_value, last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty:
            return 0, 24, [0, 24], {h: f"{h:02d}:00" for h in range(0, 25, 4)}

        min_val = dff['start_hour_float'].min()
        max_val = dff['start_hour_float'].max()
        if pd.isna(min_val) or pd.isna(max_val):
            return 0, 24, [0, 24], {h: f"{h:02d}:00" for h in range(0, 25, 4)}

        min_h = int(min_val);
        max_h = int(max_val) + 1
        duration = max_h - min_h
        tick_step = 4
        if duration <= 6:
            tick_step = 1
        elif duration <= 12:
            tick_step = 2
        elif duration <= 18:
            tick_step = 3

        marks = {h: f"{int(h):02d}:00" for h in range(min_h, max_h + 1, tick_step)}

        # FIX: Increased timeout to 6.0s
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            # Don't touch anything if preset is loading
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        return min_h, max_h, [min_h, max_h], marks

    @app.callback(
        Output({'type': 'day-grid-container', 'year': MATCH, 'month': MATCH}, 'style'),
        Input({'type': 'month-toggle', 'year': MATCH, 'month': MATCH}, 'n_clicks'),
        State({'type': 'day-grid-container', 'year': MATCH, 'month': MATCH}, 'style'),
        prevent_initial_call=True
    )
    def toggle_month_collapse(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return {'display': 'block', 'paddingLeft': '20px'} if current_style and current_style.get(
            'display') == 'none' else {'display': 'none', 'paddingLeft': '20px'}

    @app.callback(
        [Output('model-dropdown', 'options', allow_duplicate=True),
         Output('model-dropdown', 'value', allow_duplicate=True)],
        Input('location-dropdown', 'value'),
        State('model-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_model_options(selected_location, current_model):
        if not selected_location:
            return [], None

        dff_loc = initial_df[initial_df['location'] == selected_location]
        models = sorted(dff_loc['model_name'].dropna().unique())
        options = [{'label': m, 'value': m} for m in models]

        if current_model in models:
            return options, dash.no_update

        return options, models[0] if models else None

    @app.callback(
        [Output('num-cluster-dropdown', 'options', allow_duplicate=True),
         Output('num-cluster-dropdown', 'value', allow_duplicate=True)],
        [Input('location-dropdown', 'value'),
         Input('model-dropdown', 'value')],
        State('num-cluster-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_num_cluster_options(selected_location, selected_model, current_k):
        if not selected_location or not selected_model:
            return [], None

        dff_model = initial_df[
            (initial_df['location'] == selected_location) &
            (initial_df['model_name'] == selected_model)
            ]
        num_clusters = sorted(dff_model['cluster_num'].dropna().unique())
        options = [{'label': str(int(c)), 'value': int(c)} for c in num_clusters]

        if current_k and num_clusters:
            try:
                if int(current_k) in [int(x) for x in num_clusters]:
                    return options, dash.no_update
            except:
                pass

        return options, num_clusters[0] if num_clusters else None

    @app.callback(
        [Output('anim-interval', 'disabled'), Output('anim-play-btn', 'children')],
        Input('anim-play-btn', 'n_clicks'), State('anim-interval', 'disabled')
    )
    def toggle_animation(n_clicks, is_disabled):
        if n_clicks == 0: return True, "▶ Play"
        return (False, "⏸ Pause") if is_disabled else (True, "▶ Play")

    @app.callback(Output('anim-interval', 'interval'), Input('anim-speed-slider', 'value'))
    def update_anim_speed(value):
        return value

    app.clientside_callback(
        """
        function(n_intervals, current_val, mode, ranges) {
            if (!ranges) return current_val;
            let step = (mode === 'yearly') ? 1.0 : 0.1; 
            let next_val = current_val + step;
            let max_limit = (mode === 'yearly') ? ranges['yearly'][1] : ranges['daily'][1];
            let min_limit = (mode === 'yearly') ? ranges['yearly'][0] : ranges['daily'][0];
            if (mode === 'daily') { if (next_val > max_limit) { next_val = min_limit; } }
            else { if (next_val > max_limit) { next_val = min_limit; } }
            return next_val;
        }
        """,
        Output('anim-progress-slider', 'value', allow_duplicate=True),
        Input('anim-interval', 'n_intervals'),
        State('anim-progress-slider', 'value'),
        State('anim-mode', 'value'),
        State('anim-ranges-store', 'data'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        """
        function(time_value, window_size, mode, is_enabled, figure) {
            if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
            const new_fig = JSON.parse(JSON.stringify(figure));
            if (!is_enabled) {
                for (let i = 0; i < new_fig.data.length; i++) {
                    const trace = new_fig.data[i];
                    if (trace.marker) trace.marker.opacity = null;
                    trace.selectedpoints = null;
                }
                return [new_fig, ""];
            }
            let time_str = "";
            if (mode === 'yearly') {
                const date = new Date(2024, 0, time_value); 
                const month = date.toLocaleString('default', { month: 'short' });
                const day = date.getDate();
                time_str = `${month} ${day}`;
            } else {
                let hours = Math.floor(time_value);
                let minutes = Math.floor((time_value - hours) * 60);
                time_str = hours.toString().padStart(2, '0') + ":" + minutes.toString().padStart(2, '0');
            }
            const half_window = window_size / 2.0;
            for (let i = 0; i < new_fig.data.length; i++) {
                const trace = new_fig.data[i];
                if (trace.customdata) {
                    const opacity_array = [];
                    const time_idx = (mode === 'yearly') ? 4 : 3;
                    for (let j = 0; j < trace.customdata.length; j++) {
                        const point_time = trace.customdata[j][time_idx];
                        let dist = Math.abs(point_time - time_value);
                        if (mode === 'daily') { if (dist > 12) dist = 24 - dist; }
                        if (dist <= half_window) {
                            let op = 1.0 - (dist / half_window);
                            opacity_array.push(Math.max(0.1, op)); 
                        } else { opacity_array.push(0.05); }
                    }
                    if (!trace.marker) trace.marker = {};
                    trace.marker.opacity = opacity_array;
                    trace.selectedpoints = null; 
                }
            }
            return [new_fig, time_str];
        }
        """,
        [Output('scatter', 'figure', allow_duplicate=True), Output('anim-time-display', 'children')],
        [Input('anim-progress-slider', 'value'), Input('anim-window-size', 'value'),
         Input('anim-mode', 'value'), Input('anim-enabled-switch', 'on')],
        State('scatter', 'figure'),
        prevent_initial_call=True
    )

    @app.callback(
        Output('date-dropdown', 'data'),
        [Input({'type': 'date-checklist', 'year': ALL, 'month': ALL}, 'value'),
         Input({'type': 'simple-date-dropdown', 'index': ALL}, 'value')])
    def collect_selected_dates(tree_values, dropdown_values):
        """
        Aggregates selected dates from EITHER the Tree OR the Simple Dropdown.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        trigger_id = ctx.triggered[0]['prop_id']

        if 'simple-date-dropdown' in trigger_id:
            if dropdown_values and isinstance(dropdown_values[0], list):
                return dropdown_values[0]
            return []

        selected_dates = []
        for dates in tree_values:
            if dates: selected_dates.extend(dates)
        return list(set(selected_dates))

    @app.callback(
        Output('date-tree-container', 'children'),
        [Input('model-data-ready-signal', 'data')],
        [State('date-dropdown', 'data')]
    )
    def update_date_tree_dynamically(model_ready_signal, stored_dates):
        if not model_ready_signal:
            return html.Div("Select a location...")

        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty:
            return html.Div("No data found.")

        dates_to_pass = None

        if stored_dates:
            available_dates = set(pd.to_datetime(dff['day_dt']).dt.strftime('%Y-%m-%d'))
            stored_set = set(stored_dates)

            if not available_dates.isdisjoint(stored_set):
                dates_to_pass = stored_dates
            else:
                dates_to_pass = None

        return generate_date_selector(dff, selected_dates=dates_to_pass)

    @app.callback(
        Output('preset-load-dropdown', 'options'),
        Input('preset-last-action', 'data'),
        prevent_initial_call=False
    )
    def update_preset_dropdown(_):
        if not os.path.exists('presets'):
            os.makedirs('presets')
        files = [f.replace('.json', '') for f in os.listdir('presets') if f.endswith('.json')]
        return [{'label': p, 'value': p} for p in sorted(files)]

    @app.callback(
        [Output('preset-message', 'children'),
         Output('preset-last-action', 'data'),
         Output('preset-save-name', 'value')],
        Input('preset-save-btn', 'n_clicks'),
        State('preset-save-name', 'value'),
        [State('location-dropdown', 'value'),
         State('microlocation-dropdown', 'value'),
         State('recorder-type-dropdown', 'value'),
         State('channel-checklist', 'value'),
         State('model-dropdown', 'value'),
         State('num-cluster-dropdown', 'value'),
         State('max-points', 'value'),
         State('merge-switch', 'on'),
         State('merge-threshold', 'value'),
         State('clip-count-threshold', 'value'),
         State('frequency-scale', 'value'),
         State('fft-window-size', 'value'),
         State('window-overlap', 'value'),
         State('num-bins', 'value'),
         State('min-freq', 'value'),
         State('max-freq', 'value'),
         State('colormap', 'value'),
         State('db-floor', 'value'),
         State('cluster-color-store', 'data'),
         State('cluster-name-store', 'data'),
         State('date-dropdown', 'data'),
         State('hour-slider', 'value'),
         # NEW: Capture Dynamic Checkbox States
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id')],
        prevent_initial_call=True
    )
    def save_preset(n_clicks, name,
                    loc, micro, rec, chans, model, k, max_pts,
                    merge_on, merge_th, clip_th,
                    freq_scale, win_size, win_over, bins, min_f, max_f, cmap, db,
                    colors, names, dates, hours,
                    cluster_vals, cluster_ids):  # <--- New Args

        if not name:
            return "Please enter a name.", dash.no_update, dash.no_update

        # LOGIC: Extract indices of checked clusters
        # cluster_vals is like [['on'], [], ['on']]
        # cluster_ids is like [{'index': 0}, {'index': 1}, {'index': 2}]
        selected_clusters = []
        for val, id_dict in zip(cluster_vals, cluster_ids):
            if val and 'on' in val[0]:  # Check if 'on' is present
                selected_clusters.append(id_dict['index'])

        preset_data = {
            'location': loc,
            'microlocation': micro,
            'recorder': rec,
            'channels': chans,
            'model': model,
            'num_clusters': k,
            'max_points': max_pts,
            'merge_switch': merge_on,
            'merge_threshold': merge_th,
            'clip_count_threshold': clip_th,
            'spectrogram': {
                'scale': freq_scale, 'size': win_size, 'overlap': win_over,
                'bins': bins, 'min': min_f, 'max': max_f, 'cmap': cmap, 'db': db
            },
            'cluster_colors': colors,
            'cluster_names': names,
            'dates': dates,
            'hours': hours,
            'selected_clusters': selected_clusters  # <--- Save this
        }

        try:
            with open(f"presets/{name}.json", 'w') as f:
                json.dump(preset_data, f, indent=4)
            return f"Saved '{name}' successfully!", time.time(), ""
        except Exception as e:
            return f"Error saving: {str(e)}", dash.no_update, dash.no_update

    @app.callback(
        [Output('location-dropdown', 'value'),
         Output('microlocation-dropdown', 'value', allow_duplicate=True),
         Output('recorder-type-dropdown', 'value', allow_duplicate=True),
         Output('channel-checklist', 'value', allow_duplicate=True),
         Output('model-dropdown', 'value', allow_duplicate=True),
         Output('num-cluster-dropdown', 'value', allow_duplicate=True),
         Output('max-points', 'value'),
         Output('merge-switch', 'on'),
         Output('merge-threshold', 'value'),
         Output('clip-count-threshold', 'value', allow_duplicate=True),
         Output('frequency-scale', 'value'),
         Output('fft-window-size', 'value'),
         Output('window-overlap', 'value'),
         Output('num-bins', 'value'),
         Output('min-freq', 'value'),
         Output('max-freq', 'value'),
         Output('colormap', 'value'),
         Output('db-floor', 'value'),
         Output('cluster-color-store', 'data', allow_duplicate=True),
         Output('cluster-name-store', 'data', allow_duplicate=True),
         Output('date-dropdown', 'data', allow_duplicate=True),
         Output('hour-slider', 'value', allow_duplicate=True),
         Output('preset-cluster-selection', 'data'),
         Output('last-preset-load-time', 'data')],  # <--- NEW OUTPUT
        Input('preset-load-btn', 'n_clicks'),
        State('preset-load-dropdown', 'value'),
        prevent_initial_call=True
    )
    def load_preset(n_clicks, preset_name):
        if not preset_name:
            return [dash.no_update] * 24  # Updated count to 24

        path = f"presets/{preset_name}.json"
        if not os.path.exists(path):
            return [dash.no_update] * 24

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            def g(key, default=None):
                return data.get(key, default)

            spec = data.get('spectrogram', {})

            return (
                g('location'),
                g('microlocation', []),
                g('recorder', []),
                g('channels', []),
                g('model'),
                g('num_clusters'),
                g('max_points', 10000),
                g('merge_switch', False),
                g('merge_threshold', 20),
                g('clip_count_threshold', 1),
                spec.get('scale', 'mel'),
                spec.get('size', 4096),
                spec.get('overlap', 0.9),
                spec.get('bins', 512),
                spec.get('min', 50),
                spec.get('max', 5000),
                spec.get('cmap', 'Viridis'),
                spec.get('db', -100),
                g('cluster_colors', {}),
                g('cluster_names', {}),
                g('dates', []),
                g('hours', [0, 24]),
                g('selected_clusters', None),
                time.time()  # <--- Return current timestamp
            )
        except Exception as e:
            print(f"Error loading preset: {e}")
            return [dash.no_update] * 24

    @app.callback(
        Output('date-dropdown', 'data', allow_duplicate=True),
        Input('model-data-ready-signal', 'data'),
        [State('date-dropdown', 'data'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def reset_date_filter_on_load(model_ready, current_dates, last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: return []

        available_dates_set = set(pd.to_datetime(dff['day_dt']).dt.strftime('%Y-%m-%d'))

        # FIX: Increased timeout to 6.0s to account for slow data loading
        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            # If a preset loaded this data, TRUST the preset's choice.
            # Do not overwrite it.
            return dash.no_update

        # Otherwise (Manual Change), RESET to ALL available dates
        return list(available_dates_set)