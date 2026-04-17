"""
Cluster-related callbacks: rendering, colors, names, and checkboxes.
"""
import time
import dash
from dash import Input, Output, State, html, dcc, ALL
import pandas as pd
import hashlib # Added for stable coloring

from callbacks.callbacks_constants import MODEL_DATA_CACHE, initial_df, CLUSTER_COLORS, MANUAL_LABELS_CACHE, MANUAL_LABELS_LOCK
from utils import apply_manual_labels_efficiently


def register_cluster_callbacks(app):
    


    @app.callback(
        Output('cluster-list-container', 'children'),
        [Input('num-cluster-dropdown', 'value'),
         Input('model-data-ready-signal', 'data'),
         Input('preset-cluster-selection', 'data'),
         Input('cluster-stats-store', 'data'),
         Input('color-mode-radio', 'value'),
         Input('manual-labels-store', 'data'),
         Input('label-name-store', 'data'),
         Input('label-color-store', 'data')],
        [State('cluster-color-store', 'data'),
         State('cluster-name-store', 'data'),
         State('last-preset-load-time', 'data'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id')]
    )
    def render_cluster_controls(num_clusters, model_ready, preset_selected_ids, cluster_stats,
                                color_mode, manual_labels_trigger, label_names_data, label_colors_data,
                                current_colors, current_names, last_preset_time, 
                                current_checkbox_values, current_checkbox_ids):
        try:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty: dff = initial_df
            if dff.empty: return []

            is_manual = (color_mode == 'manual')
            clusters = []

            # ALWAYS show cluster IDs regardless of color mode.
            # In manual mode we colour each cluster by the most-common label assigned to it.
            if not num_clusters:
                return []
            try:
                k = int(num_clusters)
                if 'cluster_num' in dff.columns:
                    clusters = sorted(dff[dff['cluster_num'] == k]['cluster_id'].dropna().unique())
                else:
                    clusters = sorted(dff['cluster_id'].dropna().unique())
            except Exception:
                return []

            # Build cluster → majority-label map (for coloring in manual mode)
            cluster_to_label: dict = {}
            with MANUAL_LABELS_LOCK:
                manual_cache_has_data = bool(MANUAL_LABELS_CACHE)
            if is_manual and manual_cache_has_data and 'cluster_id' in dff.columns:
                from collections import Counter
                # Build a lookup: (location, microlocation, file_name, channel, clip_time) → label
                # Vectorized: create key tuples for each row, map to label, then groupby cluster_id
                key_cols = ['location', 'microlocation', 'file_name', 'channel', 'clip_time']
                available = [c for c in key_cols if c in dff.columns]
                if available:
                    key_series = dff[available].apply(
                        lambda row: tuple(row[c] if c in row.index else None for c in key_cols), axis=1
                    )
                    label_series = key_series.map(lambda k: MANUAL_LABELS_CACHE.get(k, 'Unlabeled'))
                    # Majority label per cluster_id
                    tmp = pd.DataFrame({'cluster_id': dff['cluster_id'], 'label': label_series})
                    for cid, grp in tmp.groupby('cluster_id'):
                        cluster_to_label[cid] = Counter(grp['label']).most_common(1)[0][0]

            children = []

            is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)
            effective_preset_ids = preset_selected_ids if is_preset_active else None
            
            ctx = dash.callback_context
            triggered_ids = [t['prop_id'] for t in ctx.triggered] if ctx.triggered else []
            
            # Reset logic:
            # If mode changed, we should probably reset/select all.
            is_mode_change = any('color-mode-radio' in t_id for t_id in triggered_ids)
            is_reset_trigger = any('num-cluster-dropdown' in t_id for t_id in triggered_ids) or \
                               any('model-data-ready-signal' in t_id for t_id in triggered_ids) or \
                               is_mode_change

            # Map previous selection state
            previously_selected = set()
            previously_rendered = set()
            should_preserve = (not is_preset_active) and (not is_reset_trigger) and current_checkbox_values and current_checkbox_ids
            
            if should_preserve:
                for val, id_dict in zip(current_checkbox_values, current_checkbox_ids):
                    if val and 'on' in val:
                        # Use stringified index for reliable matching
                        previously_selected.add(str(id_dict['index']))
                
                # Also track what was rendered to distinguish "Unchecked" from "New"
                for id_dict in current_checkbox_ids:
                    previously_rendered.add(str(id_dict['index']))

            total_clips = 0
            if cluster_stats and 'total' in cluster_stats:
                total_clips = cluster_stats['total']

            for i, c in enumerate(clusters):
                # c is always a cluster_id (int or float)
                try:
                    c_idx = int(c) if not hasattr(c, 'item') else c.item()
                except Exception:
                    c_idx = c
                c_label_str = str(c_idx)

                # Determine color
                if is_manual:
                    # Color by majority label of this cluster
                    label_for_color = cluster_to_label.get(c_idx, cluster_to_label.get(c, 'Unlabeled'))
                    lbl_str_for_color = str(label_for_color)
                    
                    if label_for_color == 'Unlabeled':
                        default_color = '#dddddd'
                    elif label_colors_data and lbl_str_for_color in label_colors_data:
                        default_color = label_colors_data[lbl_str_for_color]
                    else:
                        hash_val = int(hashlib.md5(lbl_str_for_color.encode('utf-8')).hexdigest(), 16)
                        default_color = CLUSTER_COLORS[hash_val % len(CLUSTER_COLORS)]
                else:
                    try:
                        default_color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
                    except Exception:
                        default_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

                color_val = current_colors.get(c_label_str, default_color) if current_colors else default_color

                # Display name: show cluster ID, and in manual mode append its majority label
                if is_manual:
                    lbl = cluster_to_label.get(c_idx, cluster_to_label.get(c, 'Unlabeled'))
                    lbl_str = str(lbl)
                    # Use custom label name if available
                    display_lbl = label_names_data.get(lbl_str, lbl_str) if label_names_data else lbl_str
                    name_val = current_names.get(c_label_str, f"{c_label_str} ({display_lbl})") if current_names else f"{c_label_str} ({display_lbl})"
                else:
                    name_val = current_names.get(c_label_str, c_label_str) if current_names else c_label_str

                is_checked = True
                # Ensure reliable comparison key (string)
                c_key_for_check = str(c_idx)
                 
                if is_preset_active:
                     if effective_preset_ids is not None:
                         is_checked = c_idx in effective_preset_ids
                elif should_preserve:
                    # New Logic:
                    # 1. If it was selected before -> True.
                    # 2. If it was NOT selected but WAS rendered -> False (Explicitly unchecked).
                    # 3. If it was NOT rendered -> True (New item).
                    if c_key_for_check in previously_selected:
                        is_checked = True
                    elif c_key_for_check in previously_rendered:
                        is_checked = False
                    else:
                        # New item — default to checked
                        is_checked = True

                # Percentage of clips in this cluster
                percent_str = ""
                if cluster_stats and c_label_str in cluster_stats and total_clips > 0:
                    count = cluster_stats[c_label_str]
                    pct = (count / total_clips) * 100
                    percent_str = f" ({pct:.1f}%)"

                row = html.Div([
                    # Checkbox (scaled for better visibility)
                    dcc.Checklist(
                        id={'type': 'cluster-checkbox', 'index': c_idx},
                        options=[{'label': '', 'value': 'on'}],
                        value=['on'] if is_checked else [],
                        style={'margin': '0', 'padding': '0', 'display': 'flex', 'transform': 'scale(1)', 'marginRight': '8px'}
                    ),
                    # Color picker (small square)
                    dcc.Input(
                        id={'type': 'cluster-color-picker', 'index': c_idx},
                        type='color',
                        value=color_val,
                        style={'width': '40px', 'height': '24px', 'padding': '0', 'border': 'none', 'cursor': 'pointer',
                               'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0}
                    ),
                    # Name / ID input – expands to fill remaining space
                    dcc.Input(
                        id={'type': 'cluster-name-input', 'index': c_idx},
                        type='text',
                        value=name_val,
                        debounce=True,
                        placeholder=c_label_str,
                        style={'flex': '1', 'minWidth': '80px', 'border': 'none', 'backgroundColor': 'transparent',
                               'fontSize': '0.85em', 'color': '#333', 'marginLeft': '8px', 'textAlign': 'left'}
                    ),
                    # Percentage – pushed to far right
                    html.Span(
                        percent_str,
                        style={'fontSize': '0.8em', 'color': '#666', 'marginLeft': 'auto', 'marginRight': '16px', 'whiteSpace': 'nowrap'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px',
                           'padding': '4px 8px', 'border': '1px solid #ccc', 'whiteSpace': 'nowrap',
                           'width': '100%', 'height': '40px', 'minHeight': '40px', 'boxSizing': 'border-box', 'overflow': 'hidden'})

                children.append(row)

            return children
        except Exception:
            try:
                import traceback
                traceback.print_exc()
            except:
                pass
            return []

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
        color_map = {}
        for color, id_dict in zip(colors, ids):
            idx = id_dict['index']
            color_map[str(idx)] = color
        return color_map

    @app.callback(
        Output({'type': 'label-checkbox', 'index': ALL}, 'value'),
        Input('all-or-none-label', 'value'),
        State({'type': 'label-checkbox', 'index': ALL}, 'options'),
        prevent_initial_call=True
    )
    def handle_label_select_all(select_all_value, options_list):
        is_selected = (len(select_all_value) > 0)
        new_values = []
        for _ in options_list:
            if is_selected:
                new_values.append(['on'])
            else:
                new_values.append([])
        return new_values

    @app.callback(
        Output('label-filter-section', 'style'),
        Input('color-mode-radio', 'value')
    )
    def toggle_label_filter_section(color_mode):
        if color_mode == 'manual':
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        Output('label-list-container', 'children'),
        [Input('manual-labels-store', 'data'),
         Input('cluster-color-store', 'data'),
         Input('cluster-name-store', 'data'),
         Input('label-color-store', 'data'),
         Input('label-name-store', 'data'),
         Input('cluster-stats-store', 'data'),
         Input('model-data-ready-signal', 'data')],
        [State({'type': 'label-checkbox', 'index': ALL}, 'value'),
         State({'type': 'label-checkbox', 'index': ALL}, 'id'),
         State('color-mode-radio', 'value')]
    )
    def render_label_controls(manual_labels_trigger, cluster_colors_data, cluster_names_data,
                              label_colors_data, label_names_data, cluster_stats, model_ready,
                              current_values, current_ids, color_mode):
        try:
            import hashlib
            # Get current data to find cluster -> label mapping (for initial color syncing)
            dff = MODEL_DATA_CACHE.get('df')
            label_to_color_seed = {}
            
            if dff is not None and not dff.empty and 'cluster_id' in dff.columns:
                # If we have cluster color overrides, use them to seed label colors
                if cluster_colors_data:
                    tmp = dff[['cluster_id', 'location', 'microlocation', 'file_name', 'channel', 'clip_time']].drop_duplicates('cluster_id')
                    tmp = apply_manual_labels_efficiently(tmp)
                    for _, row in tmp.iterrows():
                        cid_str = str(row['cluster_id'])
                        lbl = row['manual_label']
                        if cid_str in cluster_colors_data and lbl not in label_to_color_seed:
                            label_to_color_seed[lbl] = cluster_colors_data[cid_str]

            # unique_labels from cache
            with MANUAL_LABELS_LOCK:
                unique_labels = sorted(set(MANUAL_LABELS_CACHE[key] for key in list(MANUAL_LABELS_CACHE)))
            if 'Unlabeled' not in unique_labels:
                unique_labels.append('Unlabeled')
            
            # Move Unlabeled to end
            if 'Unlabeled' in unique_labels:
                unique_labels.remove('Unlabeled')
                unique_labels.append('Unlabeled')

            # Map previous selection state
            previously_selected = set()
            previously_rendered = set()
            
            if current_values and current_ids:
                for val, id_dict in zip(current_values, current_ids):
                    if val and 'on' in val:
                        previously_selected.add(str(id_dict['index']))
                    previously_rendered.add(str(id_dict['index']))

            labeled_total = 0
            if cluster_stats and '__labeled_total__' in cluster_stats:
                labeled_total = int(cluster_stats['__labeled_total__'])
            no_labels_applied = (labeled_total == 0)

            children = []
            for lbl in unique_labels:
                lbl_str = str(lbl)
                
                # Determine color
                # Priority: 1. label_colors_data, 2. cluster-seed, 3. hash
                if lbl == 'Unlabeled':
                    default_color = '#D9D9D9'  # Locked to gray
                elif label_colors_data and lbl_str in label_colors_data:
                    default_color = label_colors_data[lbl_str]
                elif lbl in label_to_color_seed:
                    default_color = label_to_color_seed[lbl]
                else:
                    # Use deterministic color assignment from CLUSTER_COLORS
                    sorted_labels = [l for l in unique_labels if l != 'Unlabeled']
                    sorted_labels = sorted(sorted_labels)
                    try:
                        color_index = sorted_labels.index(lbl) % len(CLUSTER_COLORS)
                        default_color = CLUSTER_COLORS[color_index]
                    except:
                        default_color = CLUSTER_COLORS[0]

                color_val = default_color

                # Determine name
                name_val = label_names_data.get(lbl_str, lbl_str) if label_names_data else lbl_str

                is_checked = True
                if lbl_str in previously_rendered:
                    is_checked = lbl_str in previously_selected
                else:
                    is_checked = True

                # Percentage
                percent_str = ""
                count = int(cluster_stats.get(lbl_str, 0)) if cluster_stats else 0
                if lbl_str != 'Unlabeled' and labeled_total > 0:
                    pct = (count / labeled_total) * 100
                    percent_str = f" ({count:,}, {pct:.1f}%)"
                elif lbl_str == 'Unlabeled' and no_labels_applied:
                    percent_str = " (No labels applied yet)"
                elif lbl_str == 'Unlabeled' and count > 0:
                    percent_str = f" ({count:,})"

                row = html.Div([
                    # Checkbox
                    dcc.Checklist(
                        id={'type': 'label-checkbox', 'index': lbl},
                        options=[{'label': '', 'value': 'on'}],
                        value=['on'] if is_checked else [],
                        style={'margin': '0', 'padding': '0', 'display': 'flex', 'transform': 'scale(1)', 'marginRight': '8px'}
                    ),
                    # Color picker
                    dcc.Input(
                        id={'type': 'label-color-picker', 'index': lbl},
                        type='color',
                        value=color_val,
                        disabled=(lbl == 'Unlabeled'),  # Lock Unlabeled color
                        style={'width': '40px', 'height': '24px', 'padding': '0', 'border': 'none', 
                               'cursor': 'pointer' if lbl != 'Unlabeled' else 'not-allowed',
                               'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0,
                               'opacity': 0.6 if lbl == 'Unlabeled' else 1.0}
                    ),
                    # Name Input
                    dcc.Input(
                        id={'type': 'label-name-input', 'index': lbl},
                        type='text',
                        value=name_val,
                        debounce=True,
                        placeholder=lbl_str,
                        style={'flex': '1', 'minWidth': '80px', 'border': 'none', 'backgroundColor': 'transparent',
                               'fontSize': '0.85em', 'color': '#333', 'marginLeft': '8px', 'textAlign': 'left'}
                    ),
                    # Percentage text
                    html.Span(
                        percent_str,
                        style={'fontSize': '0.8em', 'color': '#666', 'marginLeft': 'auto', 'marginRight': '16px', 'whiteSpace': 'nowrap'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px',
                           'padding': '4px 8px', 'border': '1px solid #ccc', 'whiteSpace': 'nowrap',
                           'width': '100%', 'height': '40px', 'minHeight': '40px', 'boxSizing': 'border-box', 'overflow': 'hidden'})

                children.append(row)
            return children
        except Exception:
            import traceback
            traceback.print_exc()
            return []

    @app.callback(
        [Output('label-name-store', 'data'),
         Output('manual-labels-store', 'data', allow_duplicate=True)],
        Input({'type': 'label-name-input', 'index': ALL}, 'value'),
        State({'type': 'label-name-input', 'index': ALL}, 'id'),
        State('label-name-store', 'data'),
        prevent_initial_call=True
    )
    def sync_label_names(new_names, ids, current_store):
        if current_store is None:
            current_store = {}

        triggered_id = dash.callback_context.triggered_id
        if not isinstance(triggered_id, dict) or triggered_id.get('type') != 'label-name-input':
            return dash.no_update, dash.no_update

        old_label_name = str(triggered_id.get('index'))
        if old_label_name == 'Unlabeled':
            # Protect origin category from rename/delete.
            return dash.no_update, dash.no_update

        # Resolve the edited value for the triggered label input.
        new_label_name = None
        for name_val, id_dict in zip(new_names, ids):
            if str(id_dict.get('index')) == old_label_name:
                new_label_name = (name_val or '').strip()
                break

        if new_label_name is None:
            return dash.no_update, dash.no_update

        updated_store = current_store.copy()

        # Keep display-name store behavior for non-rename custom names.
        if new_label_name and new_label_name != old_label_name:
            updated_store[old_label_name] = new_label_name
        else:
            updated_store.pop(old_label_name, None)

        # Empty or unchanged text does not perform cache relabel operations.
        if not new_label_name or new_label_name == old_label_name:
            return updated_store, dash.no_update

        changed = False
        cache_updates = []  # Collect updates for batch operation
        cache_deletions = []  # Collect deletions for batch operation
        
        for cache_key in list(MANUAL_LABELS_CACHE):
            cache_label = MANUAL_LABELS_CACHE[cache_key]
            if cache_label != old_label_name:
                continue

            changed = True
            if new_label_name == 'Unlabeled':
                # Delete label assignment by removing per-second keys.
                cache_deletions.append(cache_key)
            else:
                # Rename/merge (case-sensitive by design).
                cache_updates.append((cache_key, new_label_name))

        if not changed:
            return updated_store, dash.no_update

        # Apply cache changes with thread-safe operations
        with MANUAL_LABELS_LOCK:
            # Apply updates
            for cache_key, new_label in cache_updates:
                MANUAL_LABELS_CACHE[cache_key] = new_label
            
            # Apply deletions
            for key in cache_deletions:
                if key in MANUAL_LABELS_CACHE:
                    del MANUAL_LABELS_CACHE[key]

        # Timestamp token is enough to trigger dependent callbacks.
        return updated_store, str(time.time())

    @app.callback(
        Output('label-color-store', 'data'),
        [Input({'type': 'label-color-picker', 'index': ALL}, 'value'),
         Input('manual-labels-store', 'data')],
        [State({'type': 'label-color-picker', 'index': ALL}, 'id'),
         State('label-color-store', 'data')],
        prevent_initial_call=True
    )
    def sync_label_colors(colors, manual_labels_trigger, ids, current_color_store):
        """
        Single Source of Truth for label colors.
        Auto-assigns colors from CLUSTER_COLORS palette and locks 'Unlabeled' to gray.
        """
        color_map = current_color_store.copy() if current_color_store else {}
        
        # Get all current labels from cache
        with MANUAL_LABELS_LOCK:
            all_labels = sorted(set(MANUAL_LABELS_CACHE[key] for key in list(MANUAL_LABELS_CACHE)))
        if 'Unlabeled' not in all_labels:
            all_labels.append('Unlabeled')
        
        # Sort labels alphabetically, but put "Unlabeled" at the end
        sorted_labels = [lbl for lbl in all_labels if lbl != 'Unlabeled']
        sorted_labels = sorted(sorted_labels)
        if 'Unlabeled' in all_labels:
            sorted_labels.append('Unlabeled')
        
        # Auto-assign colors for new labels using CLUSTER_COLORS
        for i, label in enumerate(sorted_labels):
            label_str = str(label)
            if label == 'Unlabeled':
                # IRONCLAD RULE: Unlabeled is always gray
                color_map[label_str] = '#D9D9D9'
            elif label_str not in color_map:
                # Assign from CLUSTER_COLORS palette using deterministic index
                color_index = i % len(CLUSTER_COLORS)
                color_map[label_str] = CLUSTER_COLORS[color_index]
        
        # Handle user color picker changes
        if colors and ids:
            for color, id_dict in zip(colors, ids):
                label_str = str(id_dict['index'])
                if label_str != 'Unlabeled':  # Protect "Unlabeled" from user changes
                    color_map[label_str] = color
                else:
                    # Force Unlabeled back to gray if user tries to change it
                    color_map[label_str] = '#D9D9D9'
        
        return color_map

