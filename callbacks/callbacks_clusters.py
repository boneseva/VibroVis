"""
Cluster-related callbacks: rendering, colors, names, and checkboxes.
"""
import time
from collections import Counter
import dash
from dash import Input, Output, State, html, dcc, ALL
import pandas as pd

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
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id'),
         State({'type': 'label-checkbox', 'index': ALL}, 'value'),
         State({'type': 'label-checkbox', 'index': ALL}, 'id')]
    )
    def render_cluster_controls(num_clusters, model_ready, preset_selected_ids, cluster_stats,
                                color_mode, manual_labels_trigger, label_names_data, label_colors_data,
                                current_colors, current_names, last_preset_time, 
                                current_checkbox_values, current_checkbox_ids,
                                current_label_checkbox_values=None, current_label_checkbox_ids=None):
        try:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty: dff = initial_df
            if dff.empty: return []

            is_manual = (color_mode == 'manual')
            triggered_ids = []
            try:
                if dash.callback_context and dash.callback_context.triggered:
                    triggered_ids = [t['prop_id'] for t in dash.callback_context.triggered]
            except Exception:
                pass
            
            is_mode_change = any('color-mode-radio' in t_id for t_id in triggered_ids)
            is_reset_trigger = any('num-cluster-dropdown' in t_id for t_id in triggered_ids) or \
                               any('model-data-ready-signal' in t_id for t_id in triggered_ids) or \
                               is_mode_change

            if is_manual:
                # -----------------------------------------------------------------
                # MANUAL LABELS MODE: Display human-readable manual labels in legend
                # -----------------------------------------------------------------
                with MANUAL_LABELS_LOCK:
                    cache_keys = list(MANUAL_LABELS_CACHE.iterkeys())
                    unique_labels = []
                    for key in cache_keys:
                        try:
                            unique_labels.append(MANUAL_LABELS_CACHE[key])
                        except KeyError:
                            continue
                    unique_labels = sorted(set(unique_labels))

                if 'manual_label' in dff.columns:
                    for l in dff['manual_label'].dropna().unique():
                        if l not in unique_labels:
                            unique_labels.append(l)
                    unique_labels = sorted(set(unique_labels))

                if 'Unlabeled' not in unique_labels:
                    unique_labels.append('Unlabeled')
                if 'Unlabeled' in unique_labels:
                    unique_labels.remove('Unlabeled')
                    unique_labels.append('Unlabeled')

                previously_selected = set()
                previously_rendered = set()
                if (not is_reset_trigger) and current_label_checkbox_values and current_label_checkbox_ids:
                    for val, id_dict in zip(current_label_checkbox_values, current_label_checkbox_ids):
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

                    if lbl == 'Unlabeled':
                        default_color = '#D9D9D9'
                    elif label_colors_data and lbl_str in label_colors_data:
                        default_color = label_colors_data[lbl_str]
                    else:
                        non_unlabeled = [l for l in unique_labels if l != 'Unlabeled']
                        idx = non_unlabeled.index(lbl) if lbl in non_unlabeled else 0
                        default_color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]

                    color_val = default_color
                    name_val = label_names_data.get(lbl_str, lbl_str) if label_names_data else lbl_str

                    if lbl_str in previously_rendered:
                        is_checked = (lbl_str in previously_selected)
                    else:
                        is_checked = True

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
                        dcc.Checklist(
                            id={'type': 'label-checkbox', 'index': lbl},
                            options=[{'label': '', 'value': 'on'}],
                            value=['on'] if is_checked else [],
                            style={'margin': '0', 'padding': '0', 'display': 'flex', 'transform': 'scale(1)', 'marginRight': '8px'}
                        ),
                        dcc.Input(
                            id={'type': 'label-color-picker', 'index': lbl},
                            type='color',
                            value=color_val,
                            disabled=(lbl == 'Unlabeled'),
                            style={'width': '40px', 'height': '24px', 'padding': '0', 'border': 'none',
                                   'cursor': 'pointer' if lbl != 'Unlabeled' else 'not-allowed',
                                   'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0,
                                   'opacity': 0.6 if lbl == 'Unlabeled' else 1.0}
                        ),
                        dcc.Input(
                            id={'type': 'label-name-input', 'index': lbl},
                            type='text',
                            value=name_val,
                            debounce=True,
                            placeholder=lbl_str,
                            style={'flex': '1', 'minWidth': '80px', 'border': 'none', 'backgroundColor': 'transparent',
                                   'fontSize': '0.85em', 'color': '#333', 'marginLeft': '8px', 'textAlign': 'left'}
                        ),
                        html.Span(
                            percent_str,
                            style={'fontSize': '0.8em', 'color': '#666', 'marginLeft': 'auto', 'marginRight': '16px', 'whiteSpace': 'nowrap'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px',
                               'padding': '4px 8px', 'border': '1px solid #ccc', 'whiteSpace': 'nowrap',
                               'width': '100%', 'height': '40px', 'minHeight': '40px', 'boxSizing': 'border-box', 'overflow': 'hidden'})
                    children.append(row)
                return children

            # -----------------------------------------------------------------
            # CLUSTER MODE: Display cluster controls, mapping to label if applicable
            # -----------------------------------------------------------------
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

            # Check if any cluster unambiguously corresponds 100% to a single manual label
            cluster_to_single_label = {}
            with MANUAL_LABELS_LOCK:
                has_cached_labels = bool(MANUAL_LABELS_CACHE)
            if has_cached_labels and 'cluster_id' in dff.columns:
                if 'manual_label' not in dff.columns:
                    dff_labeled = apply_manual_labels_efficiently(dff.copy())
                else:
                    dff_labeled = dff
                for c in clusters:
                    c_rows = dff_labeled[dff_labeled['cluster_id'] == c]
                    if not c_rows.empty and 'manual_label' in c_rows.columns:
                        non_unlabeled = c_rows[c_rows['manual_label'] != 'Unlabeled']['manual_label'].unique()
                        unlabeled_cnt = (c_rows['manual_label'] == 'Unlabeled').sum()
                        if len(non_unlabeled) == 1 and unlabeled_cnt == 0:
                            cluster_to_single_label[c] = str(non_unlabeled[0])

            shared_label_counts = Counter(cluster_to_single_label.values())

            children = []
            is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)
            effective_preset_ids = preset_selected_ids if is_preset_active else None

            previously_selected = set()
            previously_rendered = set()
            should_preserve = (not is_preset_active) and (not is_reset_trigger) and current_checkbox_values and current_checkbox_ids
            
            if should_preserve:
                for val, id_dict in zip(current_checkbox_values, current_checkbox_ids):
                    if val and 'on' in val:
                        previously_selected.add(str(id_dict['index']))
                for id_dict in current_checkbox_ids:
                    previously_rendered.add(str(id_dict['index']))

            total_clips = 0
            if cluster_stats and 'total' in cluster_stats:
                total_clips = cluster_stats['total']

            for i, c in enumerate(clusters):
                try:
                    c_idx = int(c) if not hasattr(c, 'item') else c.item()
                except Exception:
                    c_idx = c
                c_label_str = str(c_idx)

                try:
                    default_color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
                except Exception:
                    default_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

                color_val = current_colors.get(c_label_str, default_color) if current_colors else default_color

                # Display name:
                # 1. Custom user name in cluster-name-store takes priority
                # 2. Cluster unambiguously corresponding to a single manual label
                #    (with cluster ID prefix if multiple clusters share the same label)
                # 3. Numeric cluster ID
                if current_names and c_label_str in current_names:
                    name_val = current_names[c_label_str]
                elif c in cluster_to_single_label or c_idx in cluster_to_single_label:
                    mapped_lbl = cluster_to_single_label.get(c, cluster_to_single_label.get(c_idx))
                    if shared_label_counts.get(mapped_lbl, 0) > 1:
                        name_val = f"{c_label_str} ({mapped_lbl})"
                    else:
                        name_val = mapped_lbl
                else:
                    name_val = c_label_str

                is_checked = True
                c_key_for_check = str(c_idx)
                 
                if is_preset_active:
                     if effective_preset_ids is not None:
                         is_checked = c_idx in effective_preset_ids
                elif should_preserve:
                    if c_key_for_check in previously_selected:
                        is_checked = True
                    elif c_key_for_check in previously_rendered:
                        is_checked = False
                    else:
                        is_checked = True

                percent_str = ""
                if cluster_stats and c_label_str in cluster_stats and total_clips > 0:
                    count = cluster_stats[c_label_str]
                    pct = (count / total_clips) * 100
                    percent_str = f" ({pct:.1f}%)"

                row = html.Div([
                    dcc.Checklist(
                        id={'type': 'cluster-checkbox', 'index': c_idx},
                        options=[{'label': '', 'value': 'on'}],
                        value=['on'] if is_checked else [],
                        style={'margin': '0', 'padding': '0', 'display': 'flex', 'transform': 'scale(1)', 'marginRight': '8px'}
                    ),
                    dcc.Input(
                        id={'type': 'cluster-color-picker', 'index': c_idx},
                        type='color',
                        value=color_val,
                        style={'width': '40px', 'height': '24px', 'padding': '0', 'border': 'none', 'cursor': 'pointer',
                               'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0}
                    ),
                    dcc.Input(
                        id={'type': 'cluster-name-input', 'index': c_idx},
                        type='text',
                        value=name_val,
                        debounce=True,
                        placeholder=c_label_str,
                        style={'flex': '1', 'minWidth': '80px', 'border': 'none', 'backgroundColor': 'transparent',
                               'fontSize': '0.85em', 'color': '#333', 'marginLeft': '8px', 'textAlign': 'left'}
                    ),
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
        Output({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
        Input('all-or-none-cluster', 'value'),
        State({'type': 'cluster-checkbox', 'index': ALL}, 'options'),
        prevent_initial_call=True
    )
    def handle_cluster_select_all(cluster_select_all, cluster_opts):
        is_selected = bool(cluster_select_all and len(cluster_select_all) > 0)
        return [['on'] if is_selected else [] for _ in (cluster_opts or [])]

    @app.callback(
        Output({'type': 'label-checkbox', 'index': ALL}, 'value'),
        [Input('all-or-none-cluster', 'value'),
         Input('all-or-none-label', 'value')],
        State({'type': 'label-checkbox', 'index': ALL}, 'options'),
        prevent_initial_call=True
    )
    def handle_label_select_all(cluster_select_all, label_select_all, label_opts):
        triggered_id = 'all-or-none-cluster'
        try:
            if dash.callback_context and dash.callback_context.triggered:
                triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        except Exception:
            pass
        if triggered_id == 'all-or-none-label':
            is_selected = bool(label_select_all and len(label_select_all) > 0)
        else:
            is_selected = bool(cluster_select_all and len(cluster_select_all) > 0)
        return [['on'] if is_selected else [] for _ in (label_opts or [])]

    @app.callback(
        [Output('cluster-legend-header-label', 'children'),
         Output('cluster-legend-tooltip-text', 'children')],
        Input('color-mode-radio', 'value')
    )
    def update_legend_header(color_mode):
        if color_mode == 'manual':
            return "Manual Labels", "Select specific manual labels to display. Click the color block to change the label color."
        return "Clusters", "Select specific clusters to display from the chosen model. Click the color block to change the cluster color."

    @app.callback(
        Output('label-filter-section', 'style'),
        Input('color-mode-radio', 'value')
    )
    def toggle_label_filter_section(color_mode):
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
    def render_label_controls(*args):
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
                try:
                    with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                        _log.write(f"{time.time()}\tclusters_rename_update\t{cache_key}\t{new_label}\n")
                except Exception:
                    pass
            
            # Apply deletions
            for key in cache_deletions:
                if key in MANUAL_LABELS_CACHE:
                    try:
                        del MANUAL_LABELS_CACHE[key]
                        with open('debug_label_writes.log', 'a', encoding='utf-8') as _log:
                            _log.write(f"{time.time()}\tclusters_rename_delete\t{key}\tDELETED\n")
                    except Exception:
                        pass

        # Timestamp token is enough to trigger dependent callbacks.
        return updated_store, str(time.time())

    @app.callback(
        Output('label-color-store', 'data'),
        [Input({'type': 'label-color-picker', 'index': ALL}, 'value'),
         Input('manual-labels-store', 'data'),
         Input('model-data-ready-signal', 'data')],
        [State({'type': 'label-color-picker', 'index': ALL}, 'id'),
         State('label-color-store', 'data')],
        prevent_initial_call=True
    )
    def sync_label_colors(colors, manual_labels_trigger, model_ready, ids, current_color_store):
        """
        Single Source of Truth for label colors.
        Auto-assigns colors sequentially from CLUSTER_COLORS (same palette and order as
        clusters: first label created → CLUSTER_COLORS[0], second → CLUSTER_COLORS[1], …).
        'Unlabeled' is always locked to gray (#D9D9D9).
        Fires on startup (model-data-ready-signal) so pre-existing labels get colors.
        """
        color_map = current_color_store.copy() if current_color_store else {}

        # Always lock Unlabeled to gray
        color_map['Unlabeled'] = '#D9D9D9'

        # Get all current labels from cache
        with MANUAL_LABELS_LOCK:
            cache_keys = list(MANUAL_LABELS_CACHE.iterkeys())
            all_labels = []
            for key in cache_keys:
                try:
                    label = MANUAL_LABELS_CACHE[key]
                    all_labels.append(label)
                except KeyError:
                    continue
            all_labels = sorted(set(l for l in all_labels if l and l != 'Unlabeled'))

        # Track which palette colors are already claimed so new labels get
        # the next UNUSED color (no duplicates, same sequential walk as clusters).
        already_assigned = {
            v for k, v in color_map.items()
            if k != 'Unlabeled' and v != '#D9D9D9'
        }

        for label in all_labels:
            label_str = str(label)
            if label_str in color_map:
                continue  # already has a color — keep it
            # Pick the next CLUSTER_COLORS entry not yet used by another label
            assigned = False
            for candidate in CLUSTER_COLORS:
                if candidate != '#D9D9D9' and candidate not in already_assigned:
                    color_map[label_str] = candidate
                    already_assigned.add(candidate)
                    assigned = True
                    break
            if not assigned:
                # All palette colors used → wrap around
                color_map[label_str] = CLUSTER_COLORS[len(already_assigned) % len(CLUSTER_COLORS)]

        # Handle user color-picker changes (overrides auto-assignment)
        if colors and ids:
            for color, id_dict in zip(colors, ids):
                label_str = str(id_dict['index'])
                if label_str != 'Unlabeled':
                    color_map[label_str] = color
                else:
                    color_map[label_str] = '#D9D9D9'

        return color_map

