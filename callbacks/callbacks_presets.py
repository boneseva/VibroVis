"""
Preset save/load callbacks.
"""
import time
import os
import json
import dash
from dash import Input, Output, State, ALL
import pandas as pd

from .callbacks_constants import MODEL_DATA_CACHE


def register_preset_callbacks(app):
    
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
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id')],
        prevent_initial_call=True
    )
    def save_preset(n_clicks, name,
                    loc, micro, rec, chans, model, k, max_pts,
                    merge_on, merge_th, clip_th,
                    freq_scale, win_size, win_over, bins, min_f, max_f, cmap, db,
                    colors, names, dates, hours,
                    cluster_vals, cluster_ids):

        if not name:
            return "Please enter a name.", dash.no_update, dash.no_update

        selected_clusters = []
        for val, id_dict in zip(cluster_vals, cluster_ids):
            if val and len(val) > 0 and 'on' in val[0]:
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
            'selected_clusters': selected_clusters
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
         Output('last-preset-load-time', 'data')],
        Input('preset-load-btn', 'n_clicks'),
        State('preset-load-dropdown', 'value'),
        prevent_initial_call=True
    )
    def load_preset(n_clicks, preset_name):
        if not preset_name:
            return [dash.no_update] * 24

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
                time.time()
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

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            return dash.no_update

        return list(available_dates_set)

