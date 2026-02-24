"""
Cluster-related callbacks: rendering, colors, names, and checkboxes.
"""
import time
import dash
from dash import Input, Output, State, html, dcc, ALL
import pandas as pd
import hashlib # Added for stable coloring

from .callbacks_constants import MODEL_DATA_CACHE, initial_df, CLUSTER_COLORS, MANUAL_LABELS_CACHE


def register_cluster_callbacks(app):
    


    @app.callback(
        Output('cluster-list-container', 'children'),
        [Input('num-cluster-dropdown', 'value'),
         Input('model-data-ready-signal', 'data'),
         Input('preset-cluster-selection', 'data'),
         Input('cluster-stats-store', 'data'),
         Input('color-mode-radio', 'value'),
         Input('manual-labels-store', 'data')],
        [State('cluster-color-store', 'data'),
         State('cluster-name-store', 'data'),
         State('last-preset-load-time', 'data'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id')]
    )
    def render_cluster_controls(num_clusters, model_ready, preset_selected_ids, cluster_stats,
                                color_mode, manual_labels_trigger,
                                current_colors, current_names, last_preset_time, 
                                current_checkbox_values, current_checkbox_ids):
        try:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty: dff = initial_df
            if dff.empty: return []

            is_manual = (color_mode == 'manual')
            clusters = []

            if is_manual:
                 # Gather all unique labels
                 # We can get them from the cache or from the current dataframe if they adhere to the current filter context
                 # For a global list of all used labels, we use the cache keys
                 # But it's better to show only labels relevant to the current filtering context if possible, 
                 # OR show all labels that exist in the cache. 
                 # Let's show labels that are effectively in the MANUAL_LABELS_CACHE for now.
                 unique_labels = sorted(set(MANUAL_LABELS_CACHE.values()))
                 if 'Unlabeled' not in unique_labels:
                     unique_labels.append('Unlabeled')
                 # Move Unlabeled to end
                 if 'Unlabeled' in unique_labels:
                     unique_labels.remove('Unlabeled')
                     unique_labels.append('Unlabeled')
                 
                 clusters = unique_labels
            else:
                if not num_clusters: return []
                try:
                    k = int(num_clusters)
                    if 'cluster_num' in dff.columns:
                        clusters = sorted(dff[dff['cluster_num'] == k]['cluster_id'].dropna().unique())
                    else:
                        clusters = sorted(dff['cluster_id'].dropna().unique())
                except:
                    return []

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
            
            # For coloring manual labels consistently
            # We can reuse the CLUSTER_COLORS by hashing the string or just index
            
            for i, c in enumerate(clusters):
                # c can be int (cluster id) or string (label)
                c_label_str = str(c)
                if isinstance(c, (int, float, str)):
                     c_idx = c
                else:
                     # Handle numpy types
                     try:
                         c_idx = c.item()
                     except:
                         c_idx = c
                
                # Further ensure basic types for Dash ID
                if hasattr(c_idx, 'dtype'):
                    c_idx = c_idx.item()
                
                # Determine Color
                default_color = '#888888'
                if is_manual:
                     if c == 'Unlabeled':
                         default_color = '#dddddd'
                     else:
                         # Stable hash independent of list order
                         hash_val = int(hashlib.md5(c_label_str.encode('utf-8')).hexdigest(), 16)
                         default_color = CLUSTER_COLORS[hash_val % len(CLUSTER_COLORS)]
                else:
                     try:
                         c_int = int(c)
                         default_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
                     except:
                         default_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

                color_val = current_colors[c_label_str] if current_colors and c_label_str in current_colors else default_color
                
                # Name override (only really useful for standard clusters, but allowed for manual too)
                name_val = current_names[c_label_str] if current_names and c_label_str in current_names else c_label_str

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
                        # New item (e.g. newly created label in Manual Mode)
                        if is_manual and c_label_str != 'Unlabeled':
                             is_checked = True
                        else:
                             # For standard clusters or Unlabeled, maybe default to True? 
                             # Or keep consistent with old logic (default True anyway).
                             is_checked = True
                
                # Formatting percent
                percent_str = ""
                # Don't show percentage for "Unlabeled" if manual mode
                should_show_pct = True
                if is_manual and c_label_str == 'Unlabeled':
                    should_show_pct = False

                if should_show_pct:
                   if cluster_stats and c_label_str in cluster_stats and total_clips > 0:
                       count = cluster_stats[c_label_str]
                       pct = (count / total_clips) * 100
                       percent_str = f" ({pct:.1f}%)"
                   elif cluster_stats:
                        # Avoid showing 0.0% for missing keys (stale stats or fully filtered)
                        percent_str = ""
                
                # Determine if we should disable Unlabeled color picker? Maybe allow it.

                row = html.Div([
                    dcc.Checklist(
                        id={'type': 'cluster-checkbox', 'index': c_idx},
                        options=[{'label': '', 'value': 'on'}],
                        value=['on'] if is_checked else [],
                        style={'margin': '0', 'padding': '0', 'display': 'flex'}
                    ),
                    dcc.Input(
                        id={'type': 'cluster-color-picker', 'index': c_idx},
                        type='color',
                        value=color_val,
                        style={'width': '20px', 'height': '20px', 'padding': '0', 'border': 'none', 'cursor': 'pointer',
                               'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0}
                    ),
                    dcc.Input(
                        id={'type': 'cluster-name-input', 'index': c_idx},
                        type='text',
                        value=name_val,
                        debounce=True,
                        placeholder=c_label_str,
                        style={'width': '30px', 'border': 'none', 'backgroundColor': 'transparent', 'fontSize': '0.85em',
                               'color': '#333', 'marginLeft': '3px', 'objectFit': 'contain', 'textAlign': 'right'}
                    ),
                    html.Span(
                        percent_str,
                        style={'fontSize': '0.8em', 'color': '#666', 'marginLeft': '2px', 'whiteSpace': 'pre'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px',
                          'padding': '2px 8px', 'border': '1px solid #ccc', 'whiteSpace': 'nowrap',
                          'width': 'calc(50% - 4px)', 'boxSizing': 'border-box', 'overflow': 'hidden'})

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
    def handle_cluster_select_all(select_all_value, options_list):
        is_selected = (len(select_all_value) > 0)

        new_values = []
        for _ in options_list:
            if is_selected:
                new_values.append(['on'])
            else:
                new_values.append([])
        return new_values

