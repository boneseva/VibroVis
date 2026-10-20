"""
Filter-related callbacks: dropdowns, sliders, and filter controls.
"""
import time
import dash
from dash import Input, Output, State, ALL
import pandas as pd
import numpy as np

from . import callbacks_constants
from .callbacks_constants import MODEL_DATA_CACHE
import citimon_map


def register_filter_callbacks(app):
    
    @app.callback(
        [Output('merge-threshold-container', 'style'),
         Output('clip-count-threshold-container', 'style'),
         Output('clip-count-threshold', 'value')],
        Input('merge-switch', 'on'))
    def toggle_merge_sliders(merge_on):
        style = {'display': 'block'} if merge_on else {'display': 'none'}
        return style, style, 1 if merge_on else 1

    @app.callback(
        [Output('clip-count-threshold', 'max'), Output('clip-count-threshold', 'marks')],
        Input('clip-count-max-store', 'data'))
    def update_clip_count_slider(max_clip_count):
        max_val = max_clip_count or 1
        marks = {i: str(i) for i in range(1, max_val + 1, max(1, max_val // 10))}
        if max_val > 1: marks[1] = '1'; marks[max_val] = str(max_val)
        return max_val, marks

    @app.callback(
        [Output('merge-threshold', 'max'),
         Output('merge-threshold', 'marks'),
         Output('merge-threshold', 'value', allow_duplicate=True)],
        Input('merge-max-store', 'data'),
        State('merge-threshold', 'value'),
        prevent_initial_call=True)
    def update_merge_slider(max_merge, current_val):
        max_val = max_merge or 1
        marks = {i: str(i) for i in range(0, max_val + 1, max(1, max_val // 10))}
        if max_val > 0: marks[0] = '0'; marks[max_val] = str(max_val)
        
        # Preserve current value if within range
        if current_val is not None and current_val <= max_val:
            return max_val, marks, dash.no_update
            
        # Default to 2.5% of max value, at least 1
        default_val = max(1, int(max_val * 0.025))
        
        return max_val, marks, default_val

    @app.callback(
        [Output('microlocation-dropdown', 'options', allow_duplicate=True),
         Output('microlocation-dropdown', 'value', allow_duplicate=True)],
        Input('derived-model-metadata', 'data'),
        [State('microlocation-dropdown', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_microlocation_options(metadata, current_micros, last_preset_time):
        if isinstance(metadata, dict) and 'microlocations' in metadata:
            micros = [m for m in metadata['microlocations'] if str(m).strip() and str(m).strip().lower() != 'default']
        else:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty: return [], []
            micros = [m for m in sorted(dff['microlocation'].dropna().unique()) if str(m).strip() and str(m).strip().lower() != 'default']

        options = [{'label': m, 'value': m} for m in micros]

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            return options, dash.no_update

        return options, micros

    @app.callback(
        [Output('recorder-type-dropdown', 'options', allow_duplicate=True),
         Output('recorder-type-dropdown', 'value', allow_duplicate=True)],
        [Input('derived-model-metadata', 'data'),
         Input('microlocation-dropdown', 'value')],
        [State('recorder-type-dropdown', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_recorder_options(metadata, selected_microlocation, current_recorders, last_preset_time):
        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty: return [], []

        # If no specific microlocation filter is applied, fast-path using metadata
        if (selected_microlocation is None or (isinstance(metadata, dict) and set(selected_microlocation) == set(metadata.get('microlocations', [])))) and isinstance(metadata, dict) and 'recorders' in metadata:
            recorders = metadata['recorders']
        else:
            mask = pd.Series(True, index=dff.index)
            if selected_microlocation is not None:
                if not selected_microlocation: # Empty list
                    return [], []
                mask &= dff['microlocation'].isin(selected_microlocation)
            if not mask.any(): return [], []
            recorders = sorted(dff.loc[mask, 'recorder_type'].dropna().unique())

        options = [{'label': m, 'value': m} for m in recorders]

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            return options, dash.no_update

        return options, recorders

    @app.callback(
        [Output('channel-checklist', 'options', allow_duplicate=True),
         Output('channel-checklist', 'value', allow_duplicate=True)],
        Input('derived-model-metadata', 'data'),
        State("all-or-none-channel", "value"),
        State('channel-checklist', 'value'),
        prevent_initial_call=True
    )
    def update_channel_options_and_values(metadata, select_all, current_channels):
        if isinstance(metadata, dict) and 'channels' in metadata:
            channels = metadata['channels']
        else:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty:
                dff = callbacks_constants.initial_df

            if dff.empty:
                return [], []

            channels = sorted(dff['channel'].dropna().unique())

        options = [{'label': str(c), 'value': c} for c in channels]

        if current_channels is not None:
            valid_channels = [c for c in current_channels if c in channels]
            if len(valid_channels) == len(current_channels):
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
        [Output('hour-slider', 'min', allow_duplicate=True),
         Output('hour-slider', 'max', allow_duplicate=True),
         Output('hour-slider', 'value', allow_duplicate=True),
         Output('hour-slider', 'marks', allow_duplicate=True)],
        Input('derived-model-metadata', 'data'),
        [State('hour-slider', 'value'),
         State('last-preset-load-time', 'data')],
        prevent_initial_call=True
    )
    def update_hour_slider_configuration(metadata, current_value, last_preset_time):
        if isinstance(metadata, dict) and 'hour_min' in metadata and 'hour_max' in metadata:
            min_h = metadata['hour_min']
            max_h = metadata['hour_max']
        else:
            dff = MODEL_DATA_CACHE.get('df')
            if dff is None or dff.empty:
                return 0, 24, [0, 24], {h: f"{h:02d}:00" for h in range(0, 25, 6)}

            min_val = dff['start_hour_float'].min()
            max_val = dff['start_hour_float'].max()
            if pd.isna(min_val) or pd.isna(max_val):
                return 0, 24, [0, 24], {h: f"{h:02d}:00" for h in range(0, 25, 6)}

            min_h = int(min_val)
            max_h = int(max_val) + 1

        duration = max_h - min_h
        tick_step = 6
        if duration <= 6:
            tick_step = 2
        elif duration <= 12:
            tick_step = 3
        elif duration <= 18:
            tick_step = 4

        marks = {h: f"{int(h):02d}:00" for h in range(min_h, max_h + 1, tick_step)}

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)

        if is_preset_active:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        return min_h, max_h, [min_h, max_h], marks

    @app.callback(
        [Output('project-metadata-store', 'data'),
         Output('dataset-metadata-store', 'data', allow_duplicate=True)],
        Input('project-dropdown', 'value'),
        prevent_initial_call=True
    )
    def load_project_metadata_store(selected_project):
        import read_data
        if not selected_project:
            return None, None
        payload = read_data.get_project_metadata_summary(selected_project)
        if not payload or not payload.get('locations'):
            return None, None
        return payload, payload

    @app.callback(
        [Output('location-dropdown', 'options', allow_duplicate=True),
         Output('location-dropdown', 'value', allow_duplicate=True)],
        Input('project-metadata-store', 'data'),
        State('location-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_location_options_from_metadata(meta_data, current_loc):
        if meta_data and isinstance(meta_data, dict) and 'locations' in meta_data:
            locations = meta_data['locations']
        else:
            initial_df = callbacks_constants.initial_df
            locations = sorted(initial_df['location'].dropna().unique()) if initial_df is not None and not initial_df.empty else []

        options = [{'label': str(loc), 'value': loc} for loc in locations]
        if current_loc in locations:
            return options, dash.no_update
        return options, locations[0] if locations else None

    @app.callback(
        [Output('model-dropdown', 'options', allow_duplicate=True),
         Output('model-dropdown', 'value', allow_duplicate=True)],
        [Input('location-dropdown', 'value'),
         Input('project-metadata-store', 'data')],
        State('model-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_model_options(selected_location, meta_data, current_model):
        if not selected_location:
            return [], None

        if meta_data and isinstance(meta_data, dict) and 'records' in meta_data and meta_data['records']:
            dff_records = pd.DataFrame(meta_data['records'])
            dff_loc = dff_records[dff_records['location'] == selected_location]
            models = sorted(dff_loc['model_name'].dropna().unique())
        else:
            initial_df = callbacks_constants.initial_df
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
         Input('model-dropdown', 'value'),
         Input('project-metadata-store', 'data')],
        State('num-cluster-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_num_cluster_options(selected_location, selected_model, meta_data, current_k):
        if not selected_location or not selected_model:
            return [], None

        if meta_data and isinstance(meta_data, dict) and 'records' in meta_data and meta_data['records']:
            dff_records = pd.DataFrame(meta_data['records'])
            dff_model = dff_records[
                (dff_records['location'] == selected_location) &
                (dff_records['model_name'] == selected_model)
            ]
            num_clusters = sorted(dff_model['cluster_num'].dropna().unique())
        else:
            initial_df = callbacks_constants.initial_df
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
        Output('num-bins', 'value', allow_duplicate=True),
        Input('fft-window-size', 'value'),
        prevent_initial_call=True
    )
    def update_num_bins_from_window_size(window_size):
        if not window_size:
            return dash.no_update
        try:
            val = int(window_size)
            return max(1, val // 8)
        except:
            return dash.no_update

    @app.callback(
        [Output('location-map-container', 'children'),
         Output('location-map-container', 'style')],
        [Input('location-dropdown', 'value'),
         Input('project-dropdown', 'value')],
        prevent_initial_call=False
    )
    def update_location_map_preview(selected_location, selected_project):
        if not citimon_map.is_citimon_project(selected_project):
            return None, {'display': 'none'}

        comp = citimon_map.render_citimon_map_component(selected_location)
        container_style = {
            'display': 'flex',
            'marginLeft': '8px',
            'flexShrink': '0',
            'alignSelf': 'flex-end',
            'marginBottom': '2px',
        }
        return comp, container_style
