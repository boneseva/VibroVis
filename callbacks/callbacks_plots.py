"""
Plot-related callbacks: scatter plot, spectrogram, and histogram.
"""
import time
import os
import traceback # Added for debugutils
import math
import hashlib
from collections import Counter
import dash
from dash import Input, Output, State, callback_context, ALL, html, no_update, clientside_callback
import plotly.express as px
import pandas as pd
import uuid

import utils
from callbacks.callbacks_constants import MODEL_DATA_CACHE, MERGED_DATA_CACHE, server_cache, initial_df, CLUSTER_COLORS, MANUAL_LABELS_CACHE, MANUAL_LABELS_LOCK, CLIP_DURATION_CACHE
from utils import apply_manual_labels_efficiently

# Performance profiling
ENABLE_PROFILING = False
from .callbacks_data import merge_clips_vectorized

import numpy as np
import plotly.graph_objects as go


def get_label_for_clip(loc, micro, file_basename, channel, clip_time):
    """
    Get the label for a clip identified by its audio content key.
    Content-based: keyed on round(clip_time * 10) so the same physical clip is
    labelled consistently across model/K switches.
    """
    key = (loc, micro, file_basename, int(channel), round(float(clip_time) * 10))
    label = MANUAL_LABELS_CACHE.get(key)
    return label if label and label != 'Unlabeled' else 'Unlabeled'


def register_plot_callbacks(app):


    @app.callback(
        [Output('scatter', 'figure'),
         Output('filtered-data', 'data'),
         Output('clip-count-max-store', 'data'),
         Output('merge-max-store', 'data'),
         Output('params-store', 'data'),
         Output('anim-ranges-store', 'data'),
         Output('sampling-info-display', 'children'),
         Output('cluster-stats-store', 'data'),
         Output('scatter', 'clickData'),
         Output('merge-switch', 'on', allow_duplicate=True)],
        [Input('model-data-ready-signal', 'data'),
         Input('channel-checklist', 'value'),
         Input('num-cluster-dropdown', 'value'),
         Input({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         Input('cluster-color-store', 'data'),
         Input('cluster-name-store', 'data'),
         Input('date-dropdown', 'data'),
         Input('hour-slider', 'value'),
         Input('max-points', 'value'),
         Input('resample-btn', 'n_clicks'),
         Input('priority-sampling-toggle', 'value'),
         Input('merge-switch', 'on'),
         Input('merge-threshold', 'value'),
         Input('clip-count-threshold', 'value'),
         Input('microlocation-dropdown', 'value'),
         Input('recorder-type-dropdown', 'value'),
         Input('color-mode-radio', 'value'),
         Input('manual-labels-store', 'data'),
         Input('label-color-store', 'data'),
         Input('label-name-store', 'data'),
         Input({'type': 'label-checkbox', 'index': ALL}, 'value')],
        [State('scatter', 'figure'),
         State('params-store', 'data'),
         State('location-dropdown', 'value'),
         State('last-preset-load-time', 'data'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id'),
         State({'type': 'label-checkbox', 'index': ALL}, 'id'),
         State('filtered-data', 'data')],
        prevent_initial_call='initial_duplicate'
    )
    def update_figure(model_ready_signal, selected_channels, selected_num_clusters,
                      cluster_checkbox_values, cluster_colors_data, cluster_names_data,
                      selected_dates, hour_range, max_points, resample_clicks, priority_sampling_on,
                      merge_on, merge_threshold, clip_count_threshold,
                      selected_microlocations, selected_recorders, 
                      color_mode, manual_labels_trigger, label_colors_data, label_names_data,
                      label_checkbox_values,
                      current_figure_state, stored_indices,
                      selected_location, last_preset_time, cluster_checkbox_ids,
                      label_checkbox_ids, current_filtered_key):
        try:
            return _update_figure_impl(model_ready_signal, selected_channels, selected_num_clusters,
                      cluster_checkbox_values, cluster_colors_data, cluster_names_data,
                      selected_dates, hour_range, max_points, resample_clicks, priority_sampling_on,
                      merge_on, merge_threshold, clip_count_threshold,
                      selected_microlocations, selected_recorders, 
                      color_mode, manual_labels_trigger, label_colors_data, label_names_data,
                      label_checkbox_values,
                      current_figure_state, stored_indices,
                      selected_location, last_preset_time, cluster_checkbox_ids,
                      label_checkbox_ids, current_filtered_key)
        except Exception:
            import traceback
            traceback.print_exc()
            return (dash.no_update,) * 10

    def _update_figure_impl(model_ready_signal, selected_channels, selected_num_clusters,
                      cluster_checkbox_values, cluster_colors_data, cluster_names_data,
                      selected_dates, hour_range, max_points, resample_clicks, priority_sampling_on,
                      merge_on, merge_threshold, clip_count_threshold,
                      selected_microlocations, selected_recorders, 
                      color_mode, manual_labels_trigger, label_colors_data, label_names_data,
                      label_checkbox_values,
                      current_figure_state, stored_indices,
                      selected_location, last_preset_time, cluster_checkbox_ids,
                      label_checkbox_ids, current_filtered_key=None):

        t_start = time.time() if ENABLE_PROFILING else None
        t_checkpoint = {}
        t_last = t_start if ENABLE_PROFILING else None
        
        def checkpoint(name):
            if ENABLE_PROFILING and t_last is not None:
                t_now = time.time()
                t_checkpoint[name] = (t_now - t_last) * 1000  # Store in milliseconds
                return t_now
            return t_last

        ctx = callback_context
        all_triggered_ids = [t['prop_id'] for t in ctx.triggered] if ctx.triggered else []
        t_last = checkpoint('init') or t_last

        # ---------------------------------------------------------------
        # FAST PATH: label-only update
        # When ONLY manual-labels-store fires we only need to recolor the
        # already-sampled data.  Skip the full filter/merge/sample pipeline
        # and update server_cache in-place so filtered-data key stays the
        # same → update_table fires only ONCE (from manual-labels-store),
        # not twice.
        # ---------------------------------------------------------------
        is_label_only = (
            len(all_triggered_ids) == 1
            and all_triggered_ids[0] == 'manual-labels-store.data'
        )
        if is_label_only and current_filtered_key and server_cache.get(current_filtered_key) is not None:
            try:
                dff_fast = server_cache[current_filtered_key].copy()
                if 'clip_duration' not in dff_fast.columns:
                    dff_fast['clip_duration'] = 5.0
                dff_fast = apply_manual_labels_efficiently(dff_fast)

                is_manual_now = (color_mode == 'manual')
                color_col_fast = 'manual_label' if is_manual_now else 'cluster_id_str'
                if 'cluster_id_str' not in dff_fast.columns:
                    dff_fast['cluster_id_str'] = dff_fast['cluster_id'].astype(str)

                # Build color map (mirrors full-path logic)
                color_map_fast = {}
                if is_manual_now:
                    unique_labels_fast = sorted(dff_fast['manual_label'].unique())
                    for lbl in unique_labels_fast:
                        lbl_str = str(lbl)
                        if lbl == 'Unlabeled':
                            color_map_fast[lbl_str] = '#D9D9D9'
                        elif label_colors_data and lbl_str in label_colors_data:
                            color_map_fast[lbl_str] = label_colors_data[lbl_str]
                        else:
                            sorted_all = sorted([l for l in unique_labels_fast if l != 'Unlabeled'])
                            if 'Unlabeled' in unique_labels_fast:
                                sorted_all.append('Unlabeled')
                            try:
                                color_map_fast[lbl_str] = CLUSTER_COLORS[sorted_all.index(lbl) % len(CLUSTER_COLORS)]
                            except Exception:
                                color_map_fast[lbl_str] = CLUSTER_COLORS[0]
                else:
                    for c in dff_fast['cluster_id'].unique():
                        c_str = str(c)
                        if cluster_colors_data and c_str in cluster_colors_data:
                            color_map_fast[c_str] = cluster_colors_data[c_str]
                        else:
                            try:
                                color_map_fast[c_str] = CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)]
                            except Exception:
                                color_map_fast[c_str] = CLUSTER_COLORS[0]

                dff_fast['color_col_content'] = dff_fast[color_col_fast] if color_col_fast in dff_fast.columns else 'Unlabeled'
                dff_fast['mapped_color'] = dff_fast[color_col_fast].astype(str).map(color_map_fast).fillna('#888888')

                # Preserve zoom
                x_range_fast, y_range_fast = None, None
                if current_figure_state:
                    try:
                        x_range_fast = current_figure_state['layout']['xaxis']['range']
                        y_range_fast = current_figure_state['layout']['yaxis']['range']
                    except Exception:
                        pass

                label_name_fast = 'Label' if is_manual_now else 'Cluster'
                fig_fast = go.Figure()
                if not dff_fast.empty:
                    unique_groups_fast = sorted(dff_fast[color_col_fast].unique())
                    if 'Unlabeled' in unique_groups_fast:
                        unique_groups_fast.remove('Unlabeled')
                        unique_groups_fast = ['Unlabeled'] + unique_groups_fast
                    _cd_cols = ["color_col_content", "row_idx", "plot_id", "start_hour_float",
                                "day_int", "clip_count", "file_name", "clip_time", "channel", "date_time_str"]
                    _cd_cols_present = [c for c in _cd_cols if c in dff_fast.columns]
                    for group in unique_groups_fast:
                        gp = dff_fast[dff_fast[color_col_fast] == group]
                        if gp.empty:
                            continue
                        fig_fast.add_trace(go.Scattergl(
                            x=gp['x'], y=gp['y'],
                            mode='markers',
                            marker=dict(size=gp['marker_size'], color=gp['mapped_color'],
                                        sizemode='diameter', sizeref=1, opacity=1.0),
                            customdata=gp[_cd_cols_present].to_numpy(),
                            hovertemplate=f'<b>{label_name_fast}:</b> %{{customdata[0]}}<br>'
                                          '<b>File:</b> %{customdata[6]}<br>'
                                          '<b>Clip Count:</b> %{customdata[5]}<br>'
                                          '<b>Time:</b> %{customdata[7]:.2f}s<br>'
                                          '<b>Date & Time:</b> %{customdata[9]}<br>'
                                          '<b>Channel:</b> %{customdata[8]}<extra></extra>',
                            name=str(group), showlegend=True,
                        ))
                fig_fast.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                if x_range_fast:
                    fig_fast.update_xaxes(visible=False, range=x_range_fast)
                    fig_fast.update_yaxes(visible=False, range=y_range_fast)
                else:
                    fig_fast.update_xaxes(visible=False)
                    fig_fast.update_yaxes(visible=False)

                # Update cache in-place — same key keeps filtered-data unchanged
                server_cache[current_filtered_key] = dff_fast
                # Return same filtered-data key so update_table is NOT triggered a second time
                return (fig_fast, current_filtered_key,
                        no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)
            except Exception:
                import traceback as _tb
                _tb.print_exc()
                # Fall through to full pipeline on error
        # ---------------------------------------------------------------

        is_fresh_load = any('model-data-ready-signal' in t_id for t_id in all_triggered_ids)
        is_resample_click = any('resample-btn' in t_id for t_id in all_triggered_ids)
        is_k_change = any('num-cluster-dropdown' in t_id for t_id in all_triggered_ids)
        is_cluster_checkbox_click = any('cluster-checkbox' in t_id for t_id in all_triggered_ids)

        dff_raw = MODEL_DATA_CACHE.get('df')
        t_last = checkpoint('cache_get') or t_last

        # ------------------------------------------------------------------
        # ONE-TIME DISKCACHE SNAPSHOT
        # Build a plain-dict copy of the label cache ONCE per callback call.
        # All downstream apply_manual_labels_efficiently() calls receive this
        # dict, bypassing the lock + iterkeys + per-key disk reads that would
        # otherwise fire 6-7 times on every Resample/filter click.
        # ------------------------------------------------------------------
        _label_snapshot: dict = {}
        with MANUAL_LABELS_LOCK:
            try:
                for _k in MANUAL_LABELS_CACHE.iterkeys():
                    try:
                        _label_snapshot[_k] = MANUAL_LABELS_CACHE[_k]
                    except KeyError:
                        pass
            except Exception:
                pass
        
        # Ensure clip_duration exists (critical for merging)
        if dff_raw is not None and 'clip_duration' not in dff_raw.columns:
            dff_raw['clip_duration'] = 5.0

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
                msg = "Loading data..."
                return fig, None, 1, 100, [], None, msg, {'total': 0}, None, False

        if dff_raw is None:
            fig = go.Figure()
            fig.update_layout(
                annotations=[{"text": "Select a model and location to begin.", "xref": "paper", "yref": "paper",
                              "showarrow": False, "font": {"size": 16}}],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(visible=False);
            fig.update_yaxes(visible=False)
            msg = "Select a model and location to begin."
            return fig, None, 1, 100, [], None, msg, {'total': 0}, None, False

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)
        should_force_defaults = is_fresh_load and not is_preset_active

        if should_force_defaults:
            # When forcing defaults (fresh load), show ALL data initially.
            selected_channels = None
            selected_num_clusters = None
            selected_microlocations = None
            selected_recorders = None
            selected_dates = None
            hour_range = None
            clip_count_threshold = 1
            cluster_checkbox_values = []
            stored_indices = None

        if is_fresh_load:
            stored_indices = None

        selected_clusters = []
        has_checkbox_inputs = False

        # Robust Population of selected_clusters (Strings and Ints)
        selected_clusters = set()
        if cluster_checkbox_values and cluster_checkbox_ids:
            for val, id_dict in zip(cluster_checkbox_values, cluster_checkbox_ids):
                if val and 'on' in val:
                    idx = id_dict['index']
                    selected_clusters.add(str(idx))
                    try:
                        selected_clusters.add(int(idx))
                    except:
                        pass
                        
        is_manual_mode = (color_mode == 'manual')

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

        # 1. Filter dff_raw directly (Filter First, Merge Later)
        dff_base = dff_raw
        mask = pd.Series(True, index=dff_base.index)

        if active_k is not None:
             # If using numeric dropdown, strict K filter
             if 'cluster_num' in dff_base.columns:
                 mask &= (dff_base['cluster_num'] == active_k)

        if selected_microlocations is not None:
            if not selected_microlocations:  # Empty list selected
                 mask &= False
            else:
                 mask &= dff_base['microlocation'].isin(selected_microlocations)

        if selected_recorders is not None:
            if not selected_recorders: # Empty list selected
                 mask &= False
            else:
                 mask &= dff_base['recorder_type'].isin(selected_recorders)

        if selected_channels is not None:
            mask &= dff_base['channel'].isin(selected_channels) if selected_channels else pd.Series(True, index=dff_base.index)

        if selected_dates:
            if 'day_dt_str' in dff_base.columns:
                mask &= dff_base['day_dt_str'].isin(selected_dates)
            else:
                 try:
                    valid_dates_in_data = set(pd.to_datetime(dff_base['day_dt']).dt.strftime('%Y-%m-%d'))
                    relevant_dates = [d for d in selected_dates if d in valid_dates_in_data]
                    if relevant_dates:
                         # Ensure we match the data type in day_dt
                         # If day_dt is datetime64, we need comparable timestamps
                         # Usually day_dt provided by read_data is normalized to midnight
                         selected_datetimes = pd.to_datetime(relevant_dates)
                         mask &= dff_base['day_dt'].isin(selected_datetimes)
                 except:
                    pass

        if hour_range:
             if 'start_hour_float' in dff_base.columns:
                 mask &= (dff_base['start_hour_float'] >= hour_range[0]) & (dff_base['start_hour_float'] <= hour_range[1])
            
        t_last = checkpoint('mask_creation') or t_last
        dff_filtered = dff_base[mask].copy()
        
        # 2. Merge Logic
        if merge_on and not dff_filtered.empty:
            # Dynamic merging on filtered data
            # Cache key must include filters or we skip cache for dynamic accuracy
            # Given performance fix, we can try direct merge first.
            
            merged_dff = merge_clips_vectorized(dff_filtered, merge_threshold)
            dff_macro = merged_dff
            
            # 3. Apply Post-Merge Filters (Clip Count)
            if clip_count_threshold and clip_count_threshold > 1:
                dff_macro = dff_macro[dff_macro['clip_count'] >= int(clip_count_threshold)]
                
            # Update generic stats
            total_clips_available = dff_macro['clip_count'].sum()
            
        else:
            dff_macro = dff_filtered
            if clip_count_threshold and clip_count_threshold > 1 and 'clip_count' in dff_macro.columns:
                 dff_macro = dff_macro[dff_macro['clip_count'] >= int(clip_count_threshold)]
            total_clips_available = len(dff_macro)

        t_last = checkpoint('merge_logic') or t_last
        
        # dff_macro is now ready for sampling
        total_clips_at_location = len(dff_raw) 
        total_clips_at_location = len(dff_raw[dff_raw['cluster_num'] == active_k])
        
        # Calculate max distance for slider
        max_distance = 100
        if not dff_macro.empty and 'x' in dff_macro.columns:
             min_v, max_v = dff_macro['x'].min(), dff_macro['x'].max()
             min_y, max_y = dff_macro['y'].min(), dff_macro['y'].max()
             max_distance = int(np.sqrt((max_v - min_v) ** 2 + (max_y - min_y) ** 2))

        dff_sampled = dff_macro
        new_indices_to_store = dash.no_update

        # Calculate comprehensive statistics for both Cluster and Label lists.
        # Keep a separate labeled-only denominator for manual-label percentages.
        cluster_stats = {
            'total': int(total_clips_available) if not dff_macro.empty else 0,
            '__labeled_total__': 0,
        }
        if not dff_macro.empty:
             # Apply labels to dff_macro ONCE (used for both stats and all downstream work).
             # Pass the pre-built snapshot dict so apply_manual_labels_efficiently skips
             # lock + diskcache I/O entirely.
             if 'clip_duration' not in dff_macro.columns:
                 dff_macro['clip_duration'] = 5.0
             if _label_snapshot and 'manual_label' not in dff_macro.columns:
                 dff_macro = apply_manual_labels_efficiently(dff_macro, _label_snapshot)
             elif 'manual_label' not in dff_macro.columns:
                 dff_macro['manual_label'] = 'Unlabeled'

             # 1. Cluster Stats
             if 'cluster_id' in dff_macro.columns:
                 c_counts = dff_macro.groupby('cluster_id')['clip_count'].sum().to_dict()
                 for cid, count in c_counts.items():
                     cluster_stats[str(cid)] = int(count)

             # 2. Label Stats
             if 'manual_label' in dff_macro.columns:
                 l_counts = dff_macro.groupby('manual_label')['clip_count'].sum().to_dict()
                 for lbl, count in l_counts.items():
                     cluster_stats[str(lbl)] = int(count)
                 labeled_total = sum(count for lbl, count in l_counts.items() if str(lbl) != 'Unlabeled')
                 cluster_stats['__labeled_total__'] = int(labeled_total)

        group_col = 'cluster_id'
        is_manual_mode = (color_mode == 'manual')

        # Always compute manual labels if cache is not empty, for sampling priority
        has_manual_labels = bool(_label_snapshot)

        # CRITICAL FIX: If is_manual_mode is True, we MUST return a df with 'manual_label' column
        # even if the cache is empty. apply_manual_labels_efficiently handles empty cache by
        # setting everything to 'Unlabeled'.
        if has_manual_labels or is_manual_mode:
             # Labels already applied above (stats section). Guard against double-work.
             if 'manual_label' not in dff_macro.columns:
                 dff_macro = dff_macro.copy()  # own the df before mutating
                 if 'clip_duration' not in dff_macro.columns:
                     dff_macro['clip_duration'] = 5.0
                 dff_macro = apply_manual_labels_efficiently(dff_macro, _label_snapshot)

        if is_manual_mode:
            group_col = 'manual_label'
        else:
            group_col = 'cluster_id'
        
        # Ensure cluster_stats keys are strings for JSON compatibility
        if cluster_stats:
            cluster_stats = {str(k): int(v) for k, v in cluster_stats.items()}
        else:
            cluster_stats = {'total': 0}

        # -------------------------------------------------------------------
        # VISIBILITY FILTERING (Clusters / Labels Checkboxes)
        # -------------------------------------------------------------------
        # Prepare selected_clusters set based on checkbox values and IDs
        selected_clusters = set()
        ui_known_labels = set()

        if cluster_checkbox_values and cluster_checkbox_ids:
            for val, id_dict in zip(cluster_checkbox_values, cluster_checkbox_ids):
                idx = id_dict['index']
                idx_str = str(idx)
                ui_known_labels.add(idx_str)
                
                if val and 'on' in val:
                    selected_clusters.add(idx_str)
                    try:
                        selected_clusters.add(int(idx))
                    except:
                        pass
        
        # Prepare selected_labels set
        selected_labels = set()
        ui_known_labels_for_labels = set()
        if label_checkbox_values and label_checkbox_ids:
            for val, id_dict in zip(label_checkbox_values, label_checkbox_ids):
                index_str = str(id_dict['index'])
                ui_known_labels_for_labels.add(index_str)
                if val and 'on' in val:
                    selected_labels.add(index_str)
        
        # FIX RACE CONDITION:
        # Avoid rows "disappearing" from the view before their new label checkbox spawns.
        # Use the already-built snapshot (no extra diskcache I/O).
        all_cached_labels = set(_label_snapshot.values())

        # 1. Labels filter race condition: Ensure ANY label in cache is considered "selected" if its UI checkbox is missing.
        missing_from_ui_labels = all_cached_labels - ui_known_labels_for_labels
        for missing_lbl in missing_from_ui_labels:
            selected_labels.add(str(missing_lbl))
            
        # 2. Clusters filter race condition: In manual mode, we also add these to the cluster filter pool
        missing_from_ui_clusters = all_cached_labels - ui_known_labels
        for missing_lbl in missing_from_ui_clusters:
            selected_clusters.add(str(missing_lbl))

        has_checkbox_inputs = bool(cluster_checkbox_ids) or bool(label_checkbox_ids) or (is_manual_mode and bool(_label_snapshot))
        
        do_filter = has_checkbox_inputs and not should_force_defaults
        if do_filter and not selected_clusters and not selected_labels:
            if not is_cluster_checkbox_click:
                do_filter = False

        if do_filter:
            # Filter dff_macro BEFORE sampling so resamples pull 100% quota from visible groups
            if not dff_macro.empty:
                # 1. Filter by Cluster
                if 'cluster_id' in dff_macro.columns:
                    dff_macro = dff_macro[dff_macro['cluster_id'].astype(str).isin(selected_clusters)]
                
                # 2. Filter by Label
                if label_checkbox_ids:
                    if 'manual_label' not in dff_macro.columns:
                        if 'clip_duration' not in dff_macro.columns: dff_macro['clip_duration'] = 5.0
                        dff_macro = apply_manual_labels_efficiently(dff_macro, _label_snapshot)
                    dff_macro = dff_macro[dff_macro['manual_label'].isin(selected_labels)]
            
            # Update filtered count to reflect the state after visibility filters
            total_clips_available = dff_macro['clip_count'].sum() if 'clip_count' in dff_macro.columns else len(dff_macro)
        else:
            total_clips_available = 0
            
        # -------------------------------------------------------------------

        should_resample = is_fresh_load or is_resample_click or is_k_change or not stored_indices

        if not should_resample:
            other_filters = ['merge-switch', 'merge-threshold', 'date-dropdown', 'hour-slider',
                             'microlocation-dropdown', 'color-mode-radio', 'model-data-ready-signal',
                             'max-points']
            if any(f in t_id for t_id in all_triggered_ids for f in other_filters):
                should_resample = True

        if max_points and dff_macro['clip_count'].sum() > max_points:
            if should_resample:
                # Apply labels BEFORE sampling to enable Priority Sampling of labeled points.
                # dff_macro already has manual_label from the stats/labeling pass above;
                # no need to copy or re-apply here.
                is_manual_mode = (color_mode == 'manual')

                # Sampling logic: We use a SET of chosen indices and then filter dff_macro
                # to ensure the final dff_sampled maintains its stable, intrinsic order (File Name + Time).
                final_indices_set = set()
                rng_seed = None if is_resample_click else 42
                
                # Convert checklist value to boolean
                is_priority_on = bool(priority_sampling_on and 'on' in priority_sampling_on)
                
                # Check for manual labels column existence AND toggle
                if is_priority_on and 'manual_label' in dff_macro.columns:
                     # Prioritize labeled points (everything that is not 'Unlabeled')
                     labeled_mask = dff_macro['manual_label'] != 'Unlabeled'
                     unlabeled_mask = dff_macro['manual_label'] == 'Unlabeled'
                     
                     labeled_indices = dff_macro[labeled_mask].index
                     unlabeled_indices = dff_macro[unlabeled_mask].index
                     
                     # Calculate how much space labeled points take
                     labeled_count_sum = dff_macro.loc[labeled_indices, 'clip_count'].sum()
                     
                     if labeled_count_sum <= max_points:
                         # 1. Take ALL labeled points
                         final_indices_set.update(labeled_indices)
                         
                         # 2. Fill remainder with Unlabeled
                         remaining_quota = max_points - labeled_count_sum
                         
                         if remaining_quota > 0 and len(unlabeled_indices) > 0:
                             rng = np.random.default_rng(rng_seed)
                             shuffled_unlabeled = rng.permutation(unlabeled_indices)
                             
                             shuffled_counts = dff_macro.loc[shuffled_unlabeled, 'clip_count'].values
                             cumulative_counts = np.cumsum(shuffled_counts)
                             cutoff_idx = np.searchsorted(cumulative_counts, remaining_quota, side='right')
                             
                             if cutoff_idx == 0 and remaining_quota > 0: pass 
                             
                             sampled_unlabeled = shuffled_unlabeled[:cutoff_idx]
                             final_indices_set.update(sampled_unlabeled)
                     else:
                         # Labeled points alone exceed max_points. Sample them.
                         rng = np.random.default_rng(rng_seed)
                         shuffled_labeled = rng.permutation(labeled_indices)
                         
                         shuffled_counts = dff_macro.loc[shuffled_labeled, 'clip_count'].values
                         cumulative_counts = np.cumsum(shuffled_counts)
                         cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                         final_indices_set.update(shuffled_labeled[:cutoff_idx])
                
                else:
                    # Standard random sampling
                    rng = np.random.default_rng(rng_seed)
                    shuffled_indices = rng.permutation(dff_macro.index)
                    if len(shuffled_indices) > 0:
                        shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                        cumulative_counts = np.cumsum(shuffled_counts)
                        cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                        final_indices_set.update(shuffled_indices[:cutoff_idx])
                    
                # Fallback: ensure at least one point if macro not empty
                if len(final_indices_set) == 0 and len(dff_macro) > 0: 
                     final_indices_set.update(dff_macro.index[:1])
                
                # CRITICAL: We filter the original dff_macro to return rows in their STABLE, INTRINSIC order.
                # Since dff_macro is sorted by File Name/Time (after merging), this is the order the user expects.
                dff_sampled = dff_macro[dff_macro.index.isin(final_indices_set)]
                new_indices_to_store = dff_sampled['row_idx'].tolist()
            else:
                dff_sampled = dff_macro[dff_macro['row_idx'].isin(stored_indices)]
                # Stable order is preserved by the stored_indices order.
                pass

                if dff_sampled.empty and not dff_macro.empty:
                    rng = np.random.default_rng(None if is_resample_click else 42)
                    shuffled_indices = rng.permutation(dff_macro.index)
                    if len(shuffled_indices) > 0:
                        shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                        cumulative_counts = np.cumsum(shuffled_counts)
                        cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                        final_indices = shuffled_indices[:cutoff_idx]
                        if final_indices.size == 0 and len(dff_macro) > 0: final_indices = shuffled_indices[:1]
                        dff_sampled = dff_macro[dff_macro.index.isin(final_indices)]
                        new_indices_to_store = dff_sampled['row_idx'].tolist()

        else:
            if not dff_macro.empty:
                new_indices_to_store = dff_macro['row_idx'].tolist()
            else:
                new_indices_to_store = []
            
            dff_sampled = dff_macro

        # Calculate Visibility Stats AFTER final filters
        clips_visible = dff_sampled['clip_count'].sum() if not dff_sampled.empty else 0
        if total_clips_available > 0:
            sampling_text = f"Visible: {int(clips_visible):,} | Filtered: {int(total_clips_available):,} | Total: {int(total_clips_at_location):,}"
        else:
            sampling_text = "No data available"

        x_range, y_range = None, None
        if current_figure_state and not is_fresh_load and 'initial_load' not in all_triggered_ids and not is_k_change:
            try:
                x_range = current_figure_state['layout']['xaxis']['range']
                y_range = current_figure_state['layout']['yaxis']['range']
            except KeyError:
                pass

        if dff_sampled.empty:
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
            return fig, None, 1, max_distance, new_indices_to_store, None, sampling_text, cluster_stats, None, False

        t_last = checkpoint('sampling') or t_last
        dff = dff_sampled.reset_index(drop=True).copy()
        # plot_id assignment moved to after sorting
        # Clamp marker size to avoid browser layout errors with massive points
        # Assuming clip_count is at least 1. Fillna just in case.
        dff['clip_count'] = dff['clip_count'].fillna(1)
        raw_size = 10 + 3 * (dff['clip_count'] - 1)
        dff['marker_size'] = raw_size.clip(upper=80)
        
        if 'cluster_id_str' not in dff.columns:
            dff['cluster_id_str'] = dff['cluster_id'].astype(str)
        
        dff['cache_key'] = str(uuid.uuid4())

        if 'start_hour_float' not in dff.columns:
            dff['start_hour_float'] = 0
            # manual_label already present from dff_macro; no need to re-apply

        # Ensure day_int is always available for plotting
        if 'day_dt' in dff.columns:
            dff['day_int'] = pd.to_datetime(dff['day_dt']).dt.dayofyear
        else:
            dff['day_int'] = 1
        
        if 'day_dt' in dff.columns and 'start_hour_float' in dff.columns:
            dff['date_time_str'] = pd.to_datetime(dff['day_dt']).dt.strftime('%d-%m-%Y') + ' ' + \
                                   dff['start_hour_float'].apply(lambda h: f"{int(h):02d}:{int((h % 1) * 60):02d}")
        else:
            dff['date_time_str'] = 'N/A'

        t_last = checkpoint('data_prep') or t_last
        final_color_map = {}
        
        is_manual_mode = (color_mode == 'manual')
        color_col = 'cluster_id_str'
        
        if is_manual_mode:
            # Create keys for mapping
            # Using list comprehension which is generally faster than apply for simple tuple creation
            # Use optimized helper — only if manual_label not already present (it should be)
            if 'clip_duration' not in dff.columns: dff['clip_duration'] = 5.0
            if 'manual_label' not in dff.columns:
                dff = apply_manual_labels_efficiently(dff, _label_snapshot)
            color_col = 'manual_label'
            
        if is_manual_mode:
            # SINGLE SOURCE OF TRUTH: Only use label_colors_data (label-color-store)
            # No more MD5 hash fallbacks - all colors must come from the store
            
            unique_labels = sorted(dff['manual_label'].unique())
            for lbl in unique_labels:
                lbl_str = str(lbl)
                if lbl == 'Unlabeled':
                    # Hardcoded protection for Unlabeled
                    final_color_map[lbl_str] = '#D9D9D9'
                elif label_colors_data and lbl_str in label_colors_data:
                    # Use color from single source of truth
                    final_color_map[lbl_str] = label_colors_data[lbl_str]
                else:
                    # Fallback: this should rarely happen since label-color-store auto-initializes
                    # But just in case, use CLUSTER_COLORS with deterministic index
                    sorted_all_labels = [l for l in unique_labels if l != 'Unlabeled']
                    sorted_all_labels = sorted(sorted_all_labels)
                    if 'Unlabeled' in unique_labels:
                        sorted_all_labels.append('Unlabeled')
                    
                    try:
                        color_index = sorted_all_labels.index(lbl) % len(CLUSTER_COLORS)
                        final_color_map[lbl_str] = CLUSTER_COLORS[color_index]
                    except:
                        final_color_map[lbl_str] = CLUSTER_COLORS[0]
        else:
            # Cluster mode color mapping
            # Priority: 1. cluster_colors_data (Store), 2. Default color logic
            unique_clusters = sorted(dff['cluster_id'].unique())
            for i, c in enumerate(unique_clusters):
                c_str = str(c)
                if cluster_colors_data and c_str in cluster_colors_data:
                    final_color_map[c_str] = cluster_colors_data[c_str]
                else:
                    try:
                        color_idx = int(c)
                    except:
                        color_idx = i
                    final_color_map[c_str] = CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)]

        dff['color_col_content'] = dff[color_col] if color_col in dff else 'Unlabeled'
        t_last = checkpoint('color_map') or t_last
        
        
        # --- APPLY CLUSTER VISIBILITY FILTER HERE (Late Filtering) ---
        # Now that we've computed stats on the full set of applicable data, we filter for display.
        # This allows percentages to remain stable (reflecting total valid data) even when clusters are hidden.
        if has_checkbox_inputs and not should_force_defaults:
             should_apply_checkbox_filter = True 

        # Visibility filtering is now handled at the early dff_sampled stage to ensure accurate 'Visible' stats.
        
        # REORDERING FOR Z-INDEX:
        # We want labeled points to be rendered ON TOP of Unlabeled ones.
        # We achieve this by ordering the TRACES later, NOT by sorting the dataframe here,
        # which would break the visual stability of the data table.
        pass

        # Ensure plot_id is STABLE and matches the lead row index of the clip/merged clip.
        # We NO LONGER re-assign plot_id based on positional index, as that breaks component binding.
        dff['plot_id'] = dff['row_idx']
        
        # We set the index explicitly to plot_id (which is row_idx) so that .loc calls from table-view logic work correctly.
        dff = dff.set_index('plot_id', drop=False)


        # Map colors manually
        dff['mapped_color'] = dff[color_col].astype(str).map(final_color_map).fillna('#888888')

        fig = go.Figure()
        
        label_name = 'Label' if is_manual_mode else 'Cluster'
        
        # Legend support: one trace per color category
        if not dff.empty:
            unique_groups = sorted(dff[color_col].unique())
            
            # Ensure 'Unlabeled' is the FIRST trace so it is rendered at the bottom (Z-order)
            if 'Unlabeled' in unique_groups:
                unique_groups.remove('Unlabeled')
                unique_groups = ['Unlabeled'] + unique_groups
            
            for group in unique_groups:
                gp_df = dff[dff[color_col] == group]
                if gp_df.empty: continue
                
                fig.add_trace(go.Scattergl(
                    x=gp_df['x'], 
                    y=gp_df['y'],
                    mode='markers',
                    marker=dict(
                        size=gp_df['marker_size'],
                        color=gp_df['mapped_color'],
                        sizemode='diameter',
                        sizeref=1,
                        opacity=1.0
                    ),
                    customdata=gp_df[["color_col_content", "row_idx", "plot_id", "start_hour_float", "day_int", "clip_count", "file_name", "clip_time", "channel", "date_time_str"]].to_numpy(),
                    hovertemplate=f'<b>{label_name}:</b> %{{customdata[0]}}<br>' +
                                 '<b>File:</b> %{customdata[6]}<br>' +
                                 '<b>Clip Count:</b> %{customdata[5]}<br>' +
                                 '<b>Time:</b> %{customdata[7]:.2f}s<br>' +
                                 '<b>Date & Time:</b> %{customdata[9]}<br>' +
                                 '<b>Channel:</b> %{customdata[8]}<extra></extra>',
                    name=str(group),
                    showlegend=True
                ))

        fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        t_last = checkpoint('figure_creation') or t_last

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
        t_last = checkpoint('final') or t_last

        if ENABLE_PROFILING and t_start:
            total_time = (time.time() - t_start) * 1000
            timing_str = f"⏱️  Plot Update Timing (total: {total_time:.1f}ms): "
            for name, segment_time in t_checkpoint.items():
                timing_str += f"{name}={segment_time:.1f}ms "

        return fig, dff['cache_key'].iloc[
            0], max_clip_count, max_distance, new_indices_to_store, ranges_data, sampling_text, cluster_stats, None, False if should_force_defaults else no_update


    @app.callback(
        Output('spectrogram-raw-data-store', 'data'),
        Output('spectrogram-plot', 'figure'),
        Output('fft-warning', 'children'),
        [Input("scatter", "clickData"),
         Input('filtered-data', 'data'),
         Input('frequency-scale', 'value'),
         Input('fft-window-size', 'value'),
         Input('window-overlap', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value'),
         # Added these inputs to fix the "Looks Different" issue:
         Input('colormap', 'value'),
         Input('db-floor', 'value')], 
        prevent_initial_call=True)
    def compute_spectrogram_data(clickData, filtered_data_cache_key,
                                 frequency_scale, fft_window_size, window_overlap,
                                 window_type, min_freq, max_freq, num_bins,
                                 colormap, db_floor): # Added arguments

        # 1. Validation
        if not clickData:
            return no_update, no_update, ""
        
        # Initialize empty figure for error states
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )

        if not filtered_data_cache_key:
             return no_update, no_update, ""

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
             return no_update, no_update, "Error: Filtered data not found in cache."

        try:
            point = clickData["points"][0]
            # Depending on your data, plot_id might be point['customdata'][2] or point['id']
            # Preserving your logic:
            plot_id = point["customdata"][2]
        except Exception as e:
            return no_update, no_update, f"Error parsing clickData: {e}"

        try:
             row = dff.loc[plot_id]
        except KeyError:
             return no_update, no_update, "Error: Clicked point not found. Please re-filter."

        # 2. Compute (Standard)
        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=row['mp3_file'],
            clip_time=float(row['clip_time']),
            # Ensure duration exists or default to 5.0
            clip_duration=float(row.get('clip_duration', 5.0)), 
            channel=int(row['channel']),
            padding_s=0.5
        )

        if segment is None:
             err_data = {'info': 'Error: Audio load failed', 'audio_path': '', '_rev': time.time_ns()}
             return err_data, empty_fig, "Error: Could not load audio segment."

        f, t, Sxx_db = utils.compute_spectrogram(
            segment=segment, samplerate=samplerate, scale=frequency_scale,
            fft_window_size=fft_window_size, window_overlap=window_overlap,
            window_type=window_type, min_freq=min_freq, max_freq=max_freq,
            num_bins=num_bins,
            db_floor=-120 # Compute raw first
        )

        if Sxx_db.size == 0:
            return {'info': 'Error', 'audio_path': '', '_rev': time.time_ns()}, go.Figure(), "Warning: Empty Spectrogram"

        # 3. APPLY DB FLOOR (Fixes the "Washed Out" look)
        # This restores the black/solid background for quiet areas
        floor_val = float(db_floor) if db_floor is not None else -80.0
        Sxx_db[Sxx_db < floor_val] = floor_val

        # 4. ROUNDING (Keeps it fast)
        Sxx_db = np.round(Sxx_db, 2)
        f = np.round(f, 1)
        t = np.round(t, 3)

        # 5. Build Figure (With correct Colormap)
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db, x=t, y=f,
            colorscale=colormap if colormap else 'Viridis', # Use selected colormap
            showscale=False,
            zmin=floor_val,     # Lock the scale floor
            zmax=np.max(Sxx_db) # Let max float
        ))

        fig.update_layout(
            margin=dict(l=40, r=10, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Time (s)", showgrid=False),
            yaxis=dict(title="Freq (Hz)", showgrid=False, type='log' if frequency_scale == 'log' else 'linear'),
            dragmode='zoom', # Better interaction than pan
            autosize=True
        )

        # 6. Prepare Store Data
        start_time = float(row['clip_time'])
        audio_path = f"/audio_segment_normalized/{row['mp3_file']}/{int(row['channel'])}/{start_time}/{start_time + float(row.get('clip_duration', 5.0))}"
        info = f"{row['file_name']} at {start_time:.2f}s (cluster {row['cluster_id']})"

        store_data = {
            'audio_path': audio_path,
            'info': info,
            '_rev': time.time_ns()
        }

        return store_data, fig, ""


    app.clientside_callback(
        """
        function(data) {
            // This runs entirely in the browser. 
            // We get the data, pick the strings we need, and update the audio player.
            // No heavy network upload happens!
            
            if (!data) {
                return ["", "", ""];
            }
            
            // Extract just the light-weight strings
            var audio_src = data.audio_path || "";
            var key = data._rev || "";
            var info = data.info || "";
            
            return [audio_src, key, info];
        }
        """,
        [Output("audio-player", "src", allow_duplicate=True),
         Output('spectrogram-plot-container', 'key'),
         Output('info', 'children', allow_duplicate=True)],
        Input('spectrogram-raw-data-store', 'data'),
        prevent_initial_call=True
    )

    @app.callback(
        Output('cluster-histogram', 'figure'),
        [Input("scatter", "clickData"),
         Input('histogram-type-dropdown', 'value'),
         Input('histogram-time-scale-store', 'data'),
         Input('cluster-color-store', 'data'),
         Input('label-color-store', 'data'),
         Input('label-name-store', 'data'),
         Input('color-mode-radio', 'value'),
         Input('table-click-data-store', 'data'),
         State('filtered-data', 'data')])
    def show_histogram_for_clicked_cluster(clickData, hist_type, time_scale, cluster_colors, label_colors, label_names, color_mode,
                                           table_click_data, filtered_data_cache_key):

        # Determine the effective clickData source
        triggered = dash.callback_context.triggered
        triggered_id = triggered[0]['prop_id'].split('.')[0] if triggered else ''
        if triggered_id == 'table-click-data-store' and table_click_data:
            effective_click = table_click_data
        elif clickData:
            effective_click = clickData
        elif table_click_data:
            effective_click = table_click_data
        else:
            effective_click = None

        try:
            if not effective_click or not filtered_data_cache_key:
                fig = go.Figure()
                title = "Time of Day" if time_scale == 'daily' else "Week of Year"
                fig.update_layout(
                    xaxis=dict(title=title),
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

            # Determine the key to filter by
            is_manual = (color_mode == 'manual')
            
            if is_manual:
                 clicked_label = str(effective_click["points"][0]["customdata"][0])
                 
                 if 'manual_label' not in dff.columns:
                     if 'clip_duration' not in dff.columns: dff['clip_duration'] = 5.0
                     dff = apply_manual_labels_efficiently(dff)
                 
                 cluster_df = dff[dff['manual_label'] == clicked_label].copy()
                 target_id_for_color = clicked_label
                 
            else:
                cluster_id = str(effective_click["points"][0]["customdata"][0])
                if 'cluster_id_str' in dff.columns:
                    cluster_df = dff[dff['cluster_id_str'] == cluster_id].copy()
                else:
                    cluster_df = dff[dff['cluster_id'].astype(str) == cluster_id].copy()
                target_id_for_color = cluster_id

            bar_color = '#CCCCCC'
            if is_manual:
                if label_colors and target_id_for_color in label_colors:
                    bar_color = label_colors[target_id_for_color]
                elif cluster_colors and target_id_for_color in cluster_colors:
                    bar_color = cluster_colors[target_id_for_color]
            else:
                if cluster_colors and target_id_for_color in cluster_colors:
                    bar_color = cluster_colors[target_id_for_color]
            
            if bar_color == '#CCCCCC':
                try:
                    if is_manual:
                         all_known_labels = sorted(set(MANUAL_LABELS_CACHE[key] for key in MANUAL_LABELS_CACHE))
                         if 'Unlabeled' not in all_known_labels: all_known_labels.append('Unlabeled')
                         if 'Unlabeled' in all_known_labels:
                             all_known_labels.remove('Unlabeled')
                             all_known_labels.append('Unlabeled')
                         try:
                             c_int = all_known_labels.index(target_id_for_color)
                         except ValueError:
                             c_int = 0
                    else:
                        c_int = int(float(target_id_for_color))
                    
                    bar_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
                except:
                    pass

            if cluster_df.empty: return go.Figure().update_layout(title_text=f"No data for {target_id_for_color}")

            if time_scale == 'daily':
                # Ensure time_of_day is datetime
                if 'time_of_day' not in cluster_df.columns:
                    return go.Figure().update_layout(title_text="Error: 'time_of_day' column missing.")
                
                if not pd.api.types.is_datetime64_any_dtype(cluster_df['time_of_day']):
                    cluster_df['time_of_day'] = pd.to_datetime(cluster_df['time_of_day'], errors='coerce')

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
        except Exception as e:
            return go.Figure().update_layout(title_text=f"Error in histogram: {str(e)}")

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
        [Output('manual-labels-store', 'data'),
         Output('label-saved-msg', 'children'),
         Output('manual-label-input', 'value', allow_duplicate=True)],
        [Input('save-label-btn', 'n_clicks'),
         Input('manual-label-input', 'n_submit'),
         Input({'type': 'table-inline-label', 'index': ALL}, 'value')],
        [State('manual-label-input', 'value'),
         State("scatter", "clickData"),
         State('filtered-data', 'data'),
         State('merge-switch', 'on')],
        prevent_initial_call=True
    )
    def save_manual_label(n_clicks, n_submit, table_label_values, label_text, clickData, filtered_data_cache_key, merge_on):
        import json

        triggered = dash.callback_context.triggered
        if not triggered:
            return dash.no_update, "", dash.no_update

        # Merging is ON → labeling is disabled for instance-based accuracy
        if merge_on:
            return dash.no_update, "Labeling is disabled while Merge is ON.", dash.no_update

        triggered_id_str = triggered[0]['prop_id'].split('.')[0]
        
        if not filtered_data_cache_key:
            return dash.no_update, "", dash.no_update

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
             return dash.no_update, "Error: Data expired.", dash.no_update

        try:
            # Inline Table Edit
            if triggered_id_str.startswith('{'):
                trig_dict = json.loads(triggered_id_str)
                if trig_dict.get('type') == 'table-inline-label':
                    plot_id = trig_dict.get('index')
                    
                    if plot_id not in dff.index:
                        return dash.no_update, "Error: Table point not found.", dash.no_update
                        
                    row = dff.loc[plot_id]
                    
                    # The value is the trigger value
                    val = triggered[0]['value']
                    if val is None:
                        return dash.no_update, dash.no_update, dash.no_update
                    
                    label_val = val.strip() if val else 'Unlabeled'
                    
                    # Extract row data with proper handling
                    loc = row.get('location', 'Unknown')
                    if pd.isna(loc): loc = 'Unknown'
                    micro = row.get('microlocation', 'Unknown')
                    if pd.isna(micro): micro = 'Unknown'
                    channel = int(row['channel'])
                    
                    # CRITICAL: Use cross-platform file path extraction
                    file_basename = str(row['mp3_file']).replace('\\', '/').split('/')[-1]

                    # Content-based key: round(clip_time * 10) gives 0.1 s precision,
                    # stable across model/K switches and safe against float drift.
                    clip_time_key = round(float(row['clip_time']) * 10)
                    instance_key = (loc, micro, file_basename, int(channel), clip_time_key)

                    # Thread-safe cache update — single key per clip instance
                    # Deduplicate: if an existing cache key in the same file/channel overlaps
                    # the current clip by >=50% and already has the same label, skip inserting
                    DEDUPE_THRESHOLD = 0.5
                    should_write = True
                    with MANUAL_LABELS_LOCK:
                        # Scan same-file/channel keys
                        for k in list(MANUAL_LABELS_CACHE):
                            try:
                                k_loc, k_micro, k_f, k_chan, k_sec = k
                            except Exception:
                                continue
                            if (k_loc, k_micro, k_f, int(k_chan)) != (loc, micro, file_basename, int(channel)):
                                continue
                            # Get durations
                            dur_entry = CLIP_DURATION_CACHE.get(k)
                            if isinstance(dur_entry, tuple) and len(dur_entry) == 2:
                                k_dur = int(round(dur_entry[0] * 10))
                            else:
                                k_dur = 0
                            existing_start = int(k_sec)
                            existing_end = existing_start + k_dur
                            new_start = int(clip_time_key)
                            new_end = new_start + int(round(float(row.get('clip_duration', 5.0)) * 10))
                            overlap = max(0, min(existing_end, new_end) - max(existing_start, new_start))
                            shorter = min((existing_end - existing_start) if existing_end > existing_start else 0,
                                          (new_end - new_start) if new_end > new_start else 0)
                            if shorter > 0:
                                frac = overlap / shorter
                                if frac >= DEDUPE_THRESHOLD and MANUAL_LABELS_CACHE.get(k) == label_val:
                                    should_write = False
                                    break
                        if should_write:
                            if label_val == 'Unlabeled':
                                # Interpret Unlabeled as a delete request: remove any existing persistent key
                                if instance_key in MANUAL_LABELS_CACHE:
                                    try:
                                        del MANUAL_LABELS_CACHE[instance_key]
                                    except Exception:
                                        pass
                                    # Remove duration entry as well if present
                                    try:
                                        if instance_key in CLIP_DURATION_CACHE:
                                            del CLIP_DURATION_CACHE[instance_key]
                                    except Exception:
                                        pass
                                    # Log deletion
                                    try:
                                        with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                                            _log.write(f"{time.time()}\tplots_inline_delete\t{instance_key}\tDELETED\n")
                                    except Exception:
                                        pass
                                    unique_trigger = f"{time.time()}_{hash(label_val)}_{len(MANUAL_LABELS_CACHE)}"
                                    return unique_trigger, f"Deleted label", dash.no_update
                                else:
                                    # Nothing to delete — no-op
                                    unique_trigger = f"{time.time()}_{hash(label_val)}_{len(MANUAL_LABELS_CACHE)}"
                                    return unique_trigger, f"No-op: Unlabeled", dash.no_update
                            else:
                                MANUAL_LABELS_CACHE[instance_key] = label_val
                        else:
                            # Already saved by overlapping key — treat as success without new write
                            unique_trigger = f"{time.time()}_{hash(label_val)}_{len(MANUAL_LABELS_CACHE)}"
                            return unique_trigger, f"Saved: {label_val}", dash.no_update
                    # Instrumentation: log manual label writes for debugging intermittent duplication
                    try:
                        with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                            _log.write(f"{time.time()}\tplots_inline_save\t{instance_key}\t{label_val}\n")
                    except Exception:
                        pass
                    # Store (duration, model_name) so apply_manual_labels_efficiently
                    # can use exact match within the same model and overlap match across models.
                    CLIP_DURATION_CACHE[instance_key] = (
                        float(row.get('clip_duration', 5.0)),
                        str(row.get('model_name', ''))
                    )

                    # Verify the update was successful for production reliability
                    with MANUAL_LABELS_LOCK:
                        if MANUAL_LABELS_CACHE.get(instance_key) == label_val:
                            unique_trigger = f"{time.time()}_{hash(label_val)}_{len(MANUAL_LABELS_CACHE)}"
                            return unique_trigger, f"Saved: {label_val}", dash.no_update
                        else:
                            return dash.no_update, f"Error: Failed to save {label_val}", dash.no_update

            # Sidebar button
            if not label_text or not clickData:
                return dash.no_update, "", dash.no_update
                
            point = clickData["points"][0]
            plot_id = point.get("customdata", [])[2]
            
            if plot_id in dff.index:
                row = dff.loc[plot_id]
                
                label_val = label_text.strip()
                loc = row.get('location', 'Unknown')
                if pd.isna(loc): loc = 'Unknown'
                micro = row.get('microlocation', 'Unknown')
                if pd.isna(micro): micro = 'Unknown'
                channel = int(row['channel'])
                
                # CRITICAL: Use cross-platform file path extraction
                file_basename = str(row['mp3_file']).replace('\\', '/').split('/')[-1]

                # Content-based key: round(clip_time * 10) gives 0.1 s precision,
                # stable across model/K switches and safe against float drift.
                clip_time_key = round(float(row['clip_time']) * 10)
                instance_key = (loc, micro, file_basename, int(channel), clip_time_key)


                # Thread-safe write with verification
                # Deduplicate before writing to persistent cache (same logic as inline path)
                DEDUPE_THRESHOLD = 0.5
                should_write = True
                with MANUAL_LABELS_LOCK:
                    for k in list(MANUAL_LABELS_CACHE):
                        try:
                            k_loc, k_micro, k_f, k_chan, k_sec = k
                        except Exception:
                            continue
                        if (k_loc, k_micro, k_f, int(k_chan)) != (loc, micro, file_basename, int(channel)):
                            continue
                        dur_entry = CLIP_DURATION_CACHE.get(k)
                        if isinstance(dur_entry, tuple) and len(dur_entry) == 2:
                            k_dur = int(round(dur_entry[0] * 10))
                        else:
                            k_dur = 0
                        existing_start = int(k_sec)
                        existing_end = existing_start + k_dur
                        new_start = int(clip_time_key)
                        new_end = new_start + int(round(float(row.get('clip_duration', 5.0)) * 10))
                        overlap = max(0, min(existing_end, new_end) - max(existing_start, new_start))
                        shorter = min((existing_end - existing_start) if existing_end > existing_start else 0,
                                      (new_end - new_start) if new_end > new_start else 0)
                        if shorter > 0:
                            frac = overlap / shorter
                            if frac >= DEDUPE_THRESHOLD and MANUAL_LABELS_CACHE.get(k) == label_val:
                                should_write = False
                                break
                    if should_write:
                        if label_val == 'Unlabeled':
                            # Delete existing persistent key instead of writing 'Unlabeled'
                            if instance_key in MANUAL_LABELS_CACHE:
                                try:
                                    del MANUAL_LABELS_CACHE[instance_key]
                                except Exception:
                                    pass
                                try:
                                    if instance_key in CLIP_DURATION_CACHE:
                                        del CLIP_DURATION_CACHE[instance_key]
                                except Exception:
                                    pass
                                # Log deletion
                                try:
                                    with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                                        _log.write(f"{time.time()}\tplots_sidebar_delete\t{instance_key}\tDELETED\n")
                                except Exception:
                                    pass
                                verification_success = True
                                unique_trigger = f"{time.time()}_{hash(label_val)}_1"
                                return unique_trigger, f"Deleted label", label_val
                            else:
                                verification_success = True
                                unique_trigger = f"{time.time()}_{hash(label_val)}_1"
                                return unique_trigger, f"No-op: Unlabeled", label_val
                        else:
                            MANUAL_LABELS_CACHE[instance_key] = label_val
                    else:
                        verification_success = True
                        unique_trigger = f"{time.time()}_{hash(label_val)}_1"
                        return unique_trigger, f"Saved: {label_val}", label_val
                # Log writes for debugging
                try:
                    with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                        _log.write(f"{time.time()}\tplots_sidebar_save\t{instance_key}\t{label_val}\n")
                except Exception:
                    pass
                # Store (duration, model_name) so apply_manual_labels_efficiently
                # can use exact match within the same model and overlap match across models.
                CLIP_DURATION_CACHE[instance_key] = (
                    float(row.get('clip_duration', 5.0)),
                    str(row.get('model_name', ''))
                )

                if verification_success:
                    unique_trigger = f"{time.time()}_{hash(label_val)}_1"
                    return unique_trigger, f"Saved: {label_val}", label_val
                else:
                    return dash.no_update, f"Error: Label verification failed for {label_val}", dash.no_update
            else:
                return dash.no_update, "Error: Point not found.", dash.no_update
                
        except Exception as e:
             return dash.no_update, f"Error: {str(e)}", dash.no_update

    @app.callback(
        Output('manual-label-datalist', 'children'),
        Input('manual-labels-store', 'data')
    )
    def update_manual_label_datalist(store_trigger):
        try:
            unique_labels = sorted(set(MANUAL_LABELS_CACHE[key] for key in MANUAL_LABELS_CACHE))
            return [html.Option(value=label) for label in unique_labels]
        except Exception:
             return []

    @app.callback(
        Output('manual-label-input', 'value'),
        Input("scatter", "clickData"),
        State('filtered-data', 'data'),
        prevent_initial_call=True
    )
    def update_label_input_on_click(clickData, filtered_data_cache_key):
        if not clickData or not filtered_data_cache_key:
            return ""

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None: return ""

        try:
            point = clickData["points"][0]
            plot_id = point.get("customdata", [])[2]
            
            if plot_id in dff.index:
                row = dff.loc[plot_id]
                loc = row.get('location', 'Unknown')
                if pd.isna(loc): loc = 'Unknown'
                micro = row.get('microlocation', 'Unknown') 
                if pd.isna(micro): micro = 'Unknown'
                file_basename = str(row['mp3_file']).replace('\\', '/').split('/')[-1]
                channel = int(row['channel'])
                # Content-based lookup: same clip_time = same label across models
                lbl = get_label_for_clip(loc, micro, file_basename, channel, row['clip_time'])
                return "" if lbl == 'Unlabeled' else lbl
            return ""
        except:
            return ""


