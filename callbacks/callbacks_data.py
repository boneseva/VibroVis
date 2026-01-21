"""
Data loading, caching, and audio serving callbacks.
"""
import time
import os
import sys
import flask
import soundfile as sf
import io
import pandas as pd
import numpy as np
import warnings
from contextlib import redirect_stderr
import json
import base64
import datetime
from dash import Input, Output, State, callback_context, ALL
from dash import dcc

import read_data
import utils
from .callbacks_constants import MODEL_DATA_CACHE, MERGED_DATA_CACHE, initial_df, MANUAL_LABELS_CACHE

# Suppress mpg123 decoder warnings (these are non-critical)
warnings.filterwarnings('ignore', category=UserWarning)


def merge_clips_vectorized(dff, merge_threshold):
    if dff.empty:
        return dff
    if 'clip_count' not in dff.columns:
        dff = dff.copy()
        dff['clip_count'] = 1
    else:
        dff = dff.copy()
    
    if dff['file_name'].dtype != 'category':
        dff['file_name'] = dff['file_name'].astype('category')
    if dff['channel'].dtype != 'category':
        dff['channel'] = dff['channel'].astype('category')
    
    dff = dff.sort_values(['file_name', 'channel', 'clip_time'])
    
    # Ensure numeric consistency for math
    dff['clip_time'] = pd.to_numeric(dff['clip_time'], errors='coerce').fillna(0.0)
    # Handle variable clip durations (user confirmed they vary and overlap)
    # Default to 5.0 only if widely missing, but respect existing data
    if 'clip_duration' not in dff.columns:
        dff['clip_duration'] = 5.0
    else:
        dff['clip_duration'] = pd.to_numeric(dff['clip_duration'], errors='coerce').fillna(5.0)

    same_context = (
            (dff['mp3_file'] == dff['mp3_file'].shift(1)) &
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
        'start_hour_float': ('start_hour_float', 'first'),
        'location': ('location', 'first'),
        'day_dt': ('day_dt', 'first'),
        'microlocation': ('microlocation', 'first'),
    }
    grouped = dff.groupby(merge_groups).agg(**agg_dict).reset_index(drop=True)
    grouped['clip_duration'] = grouped['clip_end'] - grouped['clip_time']
    grouped = grouped.drop(columns=['clip_end'])
    return grouped


def register_data_callbacks(app):
    
    @app.server.route("/audio_segment_normalized/<path:filename>/<int:channel>/<float:start>/<float:end>")
    def serve_audio_segment_normalized(filename, channel, start, end):
        try:
            # Suppress stderr from mpg123 decoder (redirects to devnull)
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
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
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sf.write(buf, segment, samplerate, format='mp3')
            buf.seek(0)
            return flask.send_file(buf, mimetype="audio/mp3")
        except Exception as e:
            print(f"Error serving audio segment {filename}: {e}")
            return flask.abort(500)

    @app.callback(
        Output('model-data-ready-signal', 'data'),
        Input('model-dropdown', 'value'),
        Input('location-dropdown', 'value')
    )
    def load_data_into_server_cache(selected_model, selected_location):
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
                                'start_dt', 'time_of_day', 'recorder_type', 'wav_file']
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
                    
                    if 'day_dt' in model_df.columns:
                        model_df['day_dt_str'] = pd.to_datetime(model_df['day_dt']).dt.strftime('%Y-%m-%d')
                    
                    if 'cluster_id' in model_df.columns:
                        model_df['cluster_id_str'] = model_df['cluster_id'].astype(str)
                    
                    if 'clip_count' not in model_df.columns:
                        model_df['clip_count'] = 1
                    
                    if 'file_name' in model_df.columns and model_df['file_name'].dtype != 'category':
                        model_df['file_name'] = model_df['file_name'].astype('category')
                    if 'channel' in model_df.columns and model_df['channel'].dtype != 'category':
                        model_df['channel'] = model_df['channel'].astype('category')
                    
                    MODEL_DATA_CACHE['df'] = model_df
                    print(f"Cached {len(model_df)} rows.")
                except Exception as e:
                    print(f"Error loading data: {e}")
                    MODEL_DATA_CACHE['df'] = None
        return time.time()

    @app.callback(
        Output("download-csv", "data"),
        Input("export-csv-btn", "n_clicks"),
        [State('model-dropdown', 'value'),
         State('location-dropdown', 'value'),
         State('channel-checklist', 'value'),
         State('num-cluster-dropdown', 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         State({'type': 'cluster-checkbox', 'index': ALL}, 'id'),
         State('date-dropdown', 'data'),
         State('hour-slider', 'value'),
         State('merge-switch', 'on'),
         State('merge-threshold', 'value'),
         State('clip-count-threshold', 'value'),
         State('microlocation-dropdown', 'value'),
         State('recorder-type-dropdown', 'value')],
        prevent_initial_call=True
    )
    def export_filtered_data_to_csv(n_clicks, selected_model, selected_location,
                                     selected_channels, selected_num_clusters,
                                     cluster_checkbox_values, cluster_checkbox_ids, selected_dates,
                                     hour_range, merge_on, merge_threshold,
                                     clip_count_threshold, selected_microlocations,
                                     selected_recorders):
        """Export filtered data (not sampled) to CSV."""
        if n_clicks == 0:
            return dash.no_update

        # Get the raw data
        dff_raw = MODEL_DATA_CACHE.get('df')
        if dff_raw is None or dff_raw.empty:
            return dash.no_update

        # Replicate the filtering logic from update_figure
        # Determine active k
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

        # For export, always use raw data (not pre-merged) so we can apply our own merge logic
        # that ignores spatial position and always merges based on cluster, file, and channel
        dff_base = dff_raw

        # Apply filters
        mask = pd.Series(True, index=dff_base.index)

        if active_k is not None:
            try:
                mask &= (dff_base['cluster_num'] == active_k)
            except:
                pass

        if selected_microlocations:
            mask &= dff_base['microlocation'].isin(selected_microlocations)

        if selected_recorders:
            mask &= dff_base['recorder_type'].isin(selected_recorders)

        if selected_channels:
            mask &= dff_base['channel'].isin(selected_channels)

        if selected_dates:
            if 'day_dt_str' in dff_base.columns:
                mask &= dff_base['day_dt_str'].isin(selected_dates)
            else:
                valid_dates_in_data = set(pd.to_datetime(dff_base['day_dt']).dt.strftime('%Y-%m-%d'))
                relevant_dates = [d for d in selected_dates if d in valid_dates_in_data]
                if relevant_dates:
                    selected_datetimes = pd.to_datetime(relevant_dates).normalize()
                    mask &= dff_base['day_dt'].isin(selected_datetimes)

        if hour_range:
            mask &= (dff_base['start_hour_float'] >= hour_range[0]) & (dff_base['start_hour_float'] <= hour_range[1])

        if clip_count_threshold and clip_count_threshold > 1:
            mask &= (dff_base['clip_count'] >= int(clip_count_threshold))

        # Get filtered data (this is dff_macro - full filtered, not sampled)
        dff_filtered = dff_base[mask].copy()

        # Apply cluster filter if checkboxes are used
        selected_clusters = []
        if cluster_checkbox_values and cluster_checkbox_ids:
            # Process cluster checkboxes similar to callbacks_presets.py
            for val, id_dict in zip(cluster_checkbox_values, cluster_checkbox_ids):
                if val and len(val) > 0 and 'on' in val:
                    try:
                        selected_clusters.append(int(id_dict['index']))
                    except:
                        pass

        if selected_clusters:
            dff_filtered = dff_filtered[dff_filtered['cluster_id'].isin(selected_clusters)]

        if dff_filtered.empty:
            return dash.no_update

        if not dff_filtered.empty and 'file_name' in dff_filtered.columns and 'channel' in dff_filtered.columns:
            dff_filtered = dff_filtered.sort_values(['file_name', 'channel', 'cluster_id', 'clip_time']).reset_index(drop=True)
            
            merge_groups = []
            current_group = 0
            prev_file = None
            prev_channel = None
            prev_cluster = None
            prev_end = None
            
            for idx, row in dff_filtered.iterrows():
                file_name = row['file_name']
                channel = row['channel']
                cluster_id = row['cluster_id']
                clip_time = row['clip_time']
                clip_duration = row.get('clip_duration', 0)
                clip_end = clip_time + clip_duration
                
                can_merge = (
                    file_name == prev_file and
                    channel == prev_channel and
                    cluster_id == prev_cluster and
                    prev_end is not None and
                    clip_time <= prev_end
                )
                
                if not can_merge:
                    current_group += 1
                
                merge_groups.append(current_group)
                prev_file = file_name
                prev_channel = channel
                prev_cluster = cluster_id
                prev_end = clip_end
            
            dff_filtered['merge_group'] = merge_groups
            dff_filtered['clip_end'] = dff_filtered['clip_time'] + dff_filtered['clip_duration']
            
            agg_dict = {
                'clip_time': ('clip_time', 'first'),
                'clip_end': ('clip_end', 'last'),
                'file_name': ('file_name', 'first'),
                'channel': ('channel', 'first'),
                'cluster_id': ('cluster_id', 'first'),
                'start_dt': ('start_dt', 'first'),
                'day_dt': ('day_dt', 'first'),
                'clip_count': ('clip_count', 'sum') if 'clip_count' in dff_filtered.columns else ('clip_time', 'count')
            }
            
            for col in ['wav_file', 'mp3_file', 'recorder_type', 'microlocation', 'model_name', 'cluster_num']:
                if col in dff_filtered.columns:
                    agg_dict[col] = (col, 'first')
            
            dff_merged = dff_filtered.groupby('merge_group').agg(**agg_dict).reset_index(drop=True)
            dff_merged['clip_duration'] = dff_merged['clip_end'] - dff_merged['clip_time']
            dff_merged = dff_merged.drop(columns=['clip_end'])
            dff_filtered = dff_merged

        export_df = pd.DataFrame()

        if 'wav_file' in dff_filtered.columns:
            export_df['wav_file'] = dff_filtered['wav_file'].astype(str)
        elif 'mp3_file' in dff_filtered.columns:
            mp3_str = dff_filtered['mp3_file'].astype(str)
            export_df['wav_file'] = mp3_str.str.replace(r'_ch\d+\.mp3$', '.wav', regex=True)
        elif 'file_name' in dff_filtered.columns:
            export_df['wav_file'] = dff_filtered['file_name'].astype(str)
        else:
            export_df['wav_file'] = ''
            
        if 'day_dt' in dff_filtered.columns and 'start_dt' in dff_filtered.columns:
            day_dt = pd.to_datetime(dff_filtered['day_dt'])
            start_dt = pd.to_datetime(dff_filtered['start_dt'])
            start_time_str = start_dt.dt.strftime('%H:%M:%S')
            day_str = day_dt.dt.strftime('%Y-%m-%d')
            
            wav_start_datetime = pd.to_datetime(day_str + ' ' + start_time_str, errors='coerce')
            export_df['date_time'] = wav_start_datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
            
            def format_seconds_as_time(seconds):
                """Convert seconds to HH:MM:SS format."""
                if pd.isna(seconds) or seconds < 0:
                    return ''
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f'{hours:02d}:{minutes:02d}:{secs:02d}'
            
            if 'clip_time' in dff_filtered.columns:
                export_df['timestamp_start'] = dff_filtered['clip_time'].apply(format_seconds_as_time)
            else:
                export_df['timestamp_start'] = ''
            
            if 'clip_duration' in dff_filtered.columns and 'clip_time' in dff_filtered.columns:
                clip_end_time = dff_filtered['clip_time'] + dff_filtered['clip_duration']
                export_df['timestamp_end'] = clip_end_time.apply(format_seconds_as_time)
            else:
                export_df['timestamp_end'] = ''
        else:
            export_df['date_time'] = ''
            export_df['timestamp_start'] = ''
            export_df['timestamp_end'] = ''

        # Apply manual labels
        if 'clip_duration' not in dff_filtered.columns:
             dff_filtered['clip_duration'] = 5.0
        
        # We need to import the function dynamically or use utils if available
        # It is available as utils.apply_manual_labels_efficiently
        dff_filtered = utils.apply_manual_labels_efficiently(dff_filtered)
        
        export_df['manual_label'] = dff_filtered['manual_label'] if 'manual_label' in dff_filtered.columns else 'Unlabeled'


        export_df['cluster_id'] = dff_filtered['cluster_id'].astype(str) if 'cluster_id' in dff_filtered.columns else ''
        export_df = export_df.sort_values(['wav_file', 'date_time'])

        export_df = export_df.sort_values(['wav_file', 'date_time'])

        return dcc.send_data_frame(export_df.to_csv, "filtered_data_export.csv", index=False)

    @app.callback(
        Output("download-labels-json", "data"),
        Input("btn-save-labels", "n_clicks"),
        prevent_initial_call=True
    )
    def save_manual_labels_to_file(n_clicks):
        """Save manual labels to a JSON file."""
        if not n_clicks:
             return dash.no_update
        
        if not MANUAL_LABELS_CACHE:
            return dash.no_update
            
        # Convert tuple keys to string keys for JSON serialization
        # Key format: (loc, micro, f_base, chan, sec)
        # We will use valid separators, e.g. "||"
        json_data = {}
        for k, v in MANUAL_LABELS_CACHE.items():
            # k is tuple
            # Convert elements to string
            key_str = "||".join(str(x) for x in k)
            json_data[key_str] = v
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_labels_{timestamp}.json"
        
        return dict(content=json.dumps(json_data, indent=2), filename=filename)

    @app.callback(
        [Output('manual-labels-store', 'data', allow_duplicate=True),
         Output('label-saved-msg', 'children', allow_duplicate=True)],
        Input('upload-labels-data', 'contents'),
        prevent_initial_call=True
    )
    def load_manual_labels_from_file(contents):
        """Load manual labels from a JSON file."""
        if not contents:
            return dash.no_update, dash.no_update
            
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded.decode('utf-8'))
            
            count = 0
            for k_str, label in data.items():
                # Parse key
                parts = k_str.split('||')
                if len(parts) == 5:
                    loc = parts[0]
                    micro = parts[1]
                    f_base = parts[2]
                    chan = int(parts[3])
                    sec = int(parts[4])
                    
                    # Store in cache
                    MANUAL_LABELS_CACHE[(loc, micro, f_base, chan, sec)] = label
                    count += 1
            
            msg = f"Loaded {count} labels."
            # Trigger update by sending timestamp
            return {'updated_at': time.time()}, msg
            
        except Exception as e:
            print(f"Error loading manual labels: {e}")
            return dash.no_update, f"Error: {str(e)}"

    @app.callback(
        Output('confirm-reset-labels', 'displayed'),
        Input('btn-reset-labels', 'n_clicks'),
        prevent_initial_call=True
    )
    def display_reset_confirm(n_clicks):
        if n_clicks:
            return True
        return False

    @app.callback(
        [Output('manual-labels-store', 'data', allow_duplicate=True),
         Output('label-saved-msg', 'children', allow_duplicate=True)],
        Input('confirm-reset-labels', 'submit_n_clicks'),
        prevent_initial_call=True
    )
    def reset_manual_labels(submit_n_clicks):
        if not submit_n_clicks:
             return dash.no_update, dash.no_update
        
        MANUAL_LABELS_CACHE.clear()
        
        # Trigger update
        return {'updated_at': time.time(), 'cleared': True}, "Labels Reset!"

