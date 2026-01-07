"""
Cluster-related callbacks: rendering, colors, names, and checkboxes.
"""
import time
import dash
from dash import Input, Output, State, html, dcc, ALL
import pandas as pd

from .callbacks_constants import MODEL_DATA_CACHE, initial_df, CLUSTER_COLORS


def register_cluster_callbacks(app):
    


    @app.callback(
        Output('cluster-list-container', 'children'),
        [Input('num-cluster-dropdown', 'value'),
         Input('model-data-ready-signal', 'data'),
         Input('preset-cluster-selection', 'data'),
         Input('cluster-stats-store', 'data')],
        [State('cluster-color-store', 'data'),
         State('cluster-name-store', 'data'),
         State('last-preset-load-time', 'data'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id')]
    )
    def render_cluster_controls(num_clusters, model_ready, preset_selected_ids, cluster_stats,
                                current_colors, current_names, last_preset_time, 
                                current_checkbox_values, current_checkbox_ids):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: dff = initial_df
        if not num_clusters or dff.empty: return []

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
        
        # Reset selection if K changes or Model changes
        is_reset_trigger = any('num-cluster-dropdown' in t_id for t_id in triggered_ids) or \
                           any('model-data-ready-signal' in t_id for t_id in triggered_ids)

        # Map previous selection state
        previously_selected = set()
        should_preserve = (not is_preset_active) and (not is_reset_trigger) and current_checkbox_values and current_checkbox_ids
        
        if should_preserve:
            for val, id_dict in zip(current_checkbox_values, current_checkbox_ids):
                if val and 'on' in val:
                    previously_selected.add(int(id_dict['index']))
                    
        # If this is a fresh load (no previous selection state and no preset), default to all selected
        first_load = (not current_checkbox_values)

        total_clips = 0
        if cluster_stats and 'total' in cluster_stats:
            total_clips = cluster_stats['total']
        
        for c in clusters:
            c_int = int(c)
            c_str = str(c_int)

            default_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
            color_val = current_colors[c_str] if current_colors and c_str in current_colors else default_color
            name_val = current_names[c_str] if current_names and c_str in current_names else c_str

            is_checked = True
            if is_preset_active:
                 if effective_preset_ids is not None:
                    is_checked = c_int in effective_preset_ids
            elif should_preserve:
                is_checked = c_int in previously_selected
            
            # Formatting percent
            percent_str = ""
            if cluster_stats and c_str in cluster_stats and total_clips > 0:
                count = cluster_stats[c_str]
                pct = (count / total_clips) * 100
                percent_str = f" ({pct:.1f}%)"
            elif cluster_stats:
                percent_str = " (0.0%)"

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
                           'marginLeft': '5px', 'backgroundColor': 'transparent', 'flexShrink': 0}
                ),
                dcc.Input(
                    id={'type': 'cluster-name-input', 'index': c_int},
                    type='text',
                    value=name_val,
                    debounce=True,
                    placeholder=c_str,
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

