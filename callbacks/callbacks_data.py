"""
Data loading, caching, and audio serving callbacks.
"""
import time
import os
import sys
import hashlib
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
import dash
from dash import Input, Output, State, callback_context, ALL
from dash import dcc

import read_data
import utils
from callbacks.callbacks_constants import MODEL_DATA_CACHE, MERGED_DATA_CACHE, initial_df, MANUAL_LABELS_CACHE, MANUAL_LABELS_LOCK

# Suppress mpg123 decoder warnings (these are non-critical)
warnings.filterwarnings('ignore', category=UserWarning)

LOCAL_LABELS_SCHEMA_VERSION = 1
LOCAL_LABELS_SOFT_LIMIT_BYTES = 4_000_000


def _key_tuple_to_string(key_tuple):
    return "||".join(str(x) for x in key_tuple)


def _string_to_key_tuple(key_str):
    parts = str(key_str).split('||')
    if len(parts) != 5:
        return None
    try:
        return parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
    except (TypeError, ValueError):
        return None


def _entry_to_key_tuple(entry):
    if not isinstance(entry, (list, tuple)) or len(entry) < 6:
        return None
    try:
        return str(entry[0]), str(entry[1]), str(entry[2]), int(entry[3]), int(entry[4])
    except (TypeError, ValueError):
        return None


def _build_local_labels_payload():
    entries = []
    for key_tuple in MANUAL_LABELS_CACHE:
        label = MANUAL_LABELS_CACHE[key_tuple]
        loc, micro, f_base, chan, sec = key_tuple
        entries.append([str(loc), str(micro), str(f_base), int(chan), int(sec), str(label)])
    entries.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    payload: dict[str, object] = {
        'v': LOCAL_LABELS_SCHEMA_VERSION,
        'e': entries,
        'n': len(entries)
    }
    payload_raw = json.dumps(payload, separators=(',', ':'), ensure_ascii=True)
    payload['h'] = hashlib.md5(payload_raw.encode('utf-8')).hexdigest()
    payload['b'] = len(payload_raw.encode('utf-8'))
    return payload


def _deserialize_labels_payload(payload):
    result = {}
    if not isinstance(payload, dict):
        return result

    # Compact local-storage schema
    if isinstance(payload.get('e'), list):
        for entry in payload['e']:
            key_tuple = _entry_to_key_tuple(entry)
            if key_tuple is None:
                continue
            result[key_tuple] = str(entry[5])
        return result

    # Portable export schema
    labels_obj = payload.get('labels')
    if isinstance(labels_obj, dict):
        for key_str, label in labels_obj.items():
            key_tuple = _string_to_key_tuple(key_str)
            if key_tuple is None:
                continue
            result[key_tuple] = str(label)
        return result

    # Legacy flat schema: {"loc||micro||file||chan||sec": "label"}
    reserved = {'v', 'e', 'n', 'h', 'b', 'format', 'version', 'created_at', 'labels'}
    for key_str, label in payload.items():
        if key_str in reserved:
            continue
        key_tuple = _string_to_key_tuple(key_str)
        if key_tuple is None:
            continue
        result[key_tuple] = str(label)

    return result


def _build_import_session(incoming_labels, source_name):
    non_conflicts = []
    conflicts = []

    for key_tuple, incoming_label in incoming_labels.items():
        key_entry = [key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3], key_tuple[4], str(incoming_label)]
        current_label = MANUAL_LABELS_CACHE.get(key_tuple)

        if current_label is None or str(current_label) == str(incoming_label):
            non_conflicts.append(key_entry)
        else:
            conflicts.append([
                key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3], key_tuple[4],
                str(current_label), str(incoming_label)
            ])

    return {
        'source': source_name,
        'incoming_total': len(incoming_labels),
        'non_conflicts': non_conflicts,
        'conflicts': conflicts,
        'resolutions': [None] * len(conflicts),
        'cursor': 0,
    }


def _next_unresolved_index(resolutions, start_idx=0):
    for idx in range(max(0, int(start_idx)), len(resolutions)):
        if resolutions[idx] is None:
            return idx
    return None


def _format_conflict_text(session):
    conflicts = session.get('conflicts', [])
    resolutions = session.get('resolutions', [])
    cursor = _next_unresolved_index(resolutions, session.get('cursor', 0))
    if cursor is None or cursor >= len(conflicts):
        return ""

    loc, micro, f_base, chan, sec, current_label, incoming_label = conflicts[cursor]
    total = len(conflicts)
    return (
        f"Conflict {cursor + 1}/{total} | {f_base} ch{chan} @ {sec}s | "
        f"Location: {loc}/{micro} | Current: '{current_label}' | Incoming: '{incoming_label}'"
    )


def _apply_import_session(session):
    applied_count = 0
    kept_conflicts = 0
    used_conflicts = 0

    for entry in session.get('non_conflicts', []):
        key_tuple = _entry_to_key_tuple(entry)
        if key_tuple is None:
            continue
        MANUAL_LABELS_CACHE[key_tuple] = str(entry[5])
        applied_count += 1

    conflicts = session.get('conflicts', [])
    resolutions = session.get('resolutions', [])
    for idx, conflict in enumerate(conflicts):
        resolution = resolutions[idx] if idx < len(resolutions) else 'keep'
        if resolution != 'use':
            kept_conflicts += 1
            continue

        key_tuple = (str(conflict[0]), str(conflict[1]), str(conflict[2]), int(conflict[3]), int(conflict[4]))
        MANUAL_LABELS_CACHE[key_tuple] = str(conflict[6])
        applied_count += 1
        used_conflicts += 1

    return applied_count, used_conflicts, kept_conflicts


def _build_portable_export_payload():
    labels = {}
    # Sort the cache keys for consistent ordering with thread safety
    with MANUAL_LABELS_LOCK:
        sorted_keys = sorted(MANUAL_LABELS_CACHE.iterkeys())
        for key_tuple in sorted_keys:
            try:
                label = MANUAL_LABELS_CACHE[key_tuple]
                if str(label) == 'Unlabeled':
                    continue  # skip unlabeled entries from export
                labels[_key_tuple_to_string(key_tuple)] = str(label)
            except KeyError:
                # Key was deleted during iteration, skip it
                continue

    return {
        'format': 'vibrovis-manual-labels',
        'version': 1,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'labels': labels,
        'count': len(labels)
    }


def merge_clips_vectorized(dff, merge_threshold):
    """
    Merge consecutive clips that share the same context (file, channel, cluster) 
    and are within spatial and temporal thresholds.
    
    CRITICAL FIXES:
    - Fixed sorting to include cluster_id (was missing!)
    - Added floating-point tolerance for temporal continuity
    - Fixed context field consistency (file_name vs mp3_file)
    - Added NaN handling for spatial distance calculations
    """
    if dff.empty:
        return dff
    if 'clip_count' not in dff.columns:
        dff = dff.copy()
        dff['clip_count'] = 1
    else:
        dff = dff.copy()
    
    # Convert to category for performance
    if dff['file_name'].dtype != 'category':
        dff['file_name'] = dff['file_name'].astype('category')
    if dff['channel'].dtype != 'category':
        dff['channel'] = dff['channel'].astype('category')
    
    # CRITICAL FIX: Add cluster_id to sort - this was missing and breaking adjacency logic!
    dff = dff.sort_values(['file_name', 'channel', 'cluster_id', 'clip_time']).reset_index(drop=True)
    
    # Ensure numeric consistency for math
    dff['clip_time'] = pd.to_numeric(dff['clip_time'], errors='coerce').fillna(0.0)
    
    # Handle variable clip durations (user confirmed they vary and overlap)
    if 'clip_duration' not in dff.columns:
        dff['clip_duration'] = 5.0
    else:
        dff['clip_duration'] = pd.to_numeric(dff['clip_duration'], errors='coerce').fillna(5.0)

    # FIXED: Use consistent field (file_name) in both sorting and context checks
    same_context = (
        (dff['file_name'] == dff['file_name'].shift(1)) &
        (dff['channel'] == dff['channel'].shift(1)) &
        (dff['cluster_id'] == dff['cluster_id'].shift(1))
    )
    
    # Spatial proximity calculation with NaN handling
    prev_points = dff[['x', 'y']].shift(1).to_numpy()
    curr_points = dff[['x', 'y']].to_numpy()
    
    # Handle NaN values in first row after shift
    valid_spatial = ~(np.isnan(prev_points).any(axis=1) | np.isnan(curr_points).any(axis=1))
    distances = np.full(len(dff), np.inf)  # Default to infinity (no merge)
    distances[valid_spatial] = np.linalg.norm(
        curr_points[valid_spatial] - prev_points[valid_spatial], axis=1
    )
    spatial_proximity = distances < merge_threshold
    
    # CRITICAL FIX: Add floating-point tolerance for temporal continuity
    prev_clip_end = dff['clip_time'].shift(1) + dff['clip_duration'].shift(1)
    # Use small epsilon to handle floating-point precision issues
    temporal_continuity = dff['clip_time'] <= (prev_clip_end + 1e-6)
    
    # Combine all merge conditions
    merge_mask = same_context & temporal_continuity & spatial_proximity
    
    # Create merge groups - clips with merge_mask=True belong to previous group
    merge_groups = (~merge_mask).cumsum()
    
    # Add debug validation
    original_count = len(dff)
    potential_merges = merge_mask.sum()
    
    # Create clip_end for aggregation
    dff['clip_end'] = dff['clip_time'] + dff['clip_duration']

    agg_dict = {
        'x': ('x', 'mean'), 'y': ('y', 'mean'),
        'clip_time': ('clip_time', 'first'), 'clip_end': ('clip_end', 'max'),
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
    
    # Debug output to verify merging is working
    final_count = len(grouped)
    if potential_merges > 0:
        print(f"MERGE DEBUG: {original_count} clips -> {final_count} after merging "
              f"({potential_merges} potential merges, threshold={merge_threshold})")
    
    return grouped
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

        if selected_microlocations is not None:
            if not selected_microlocations:
                mask &= False
            else:
                mask &= dff_base['microlocation'].isin(selected_microlocations)

        if selected_recorders is not None:
            if not selected_recorders:
                mask &= False
            else:
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
            
            # Ensure clip_duration is present
            if 'clip_duration' not in dff_filtered.columns:
                dff_filtered['clip_duration'] = 5.0
            else:
                 dff_filtered['clip_duration'] = pd.to_numeric(dff_filtered['clip_duration'], errors='coerce').fillna(5.0)

            # OPTIMIZATION: Vectorized merge group calculation (replacing iterrows loop)
            # Create shifted columns for comparison with previous row
            prev_file = dff_filtered['file_name'].shift(1)
            prev_channel = dff_filtered['channel'].shift(1)
            prev_cluster = dff_filtered['cluster_id'].shift(1)
            prev_clip_end = (dff_filtered['clip_time'] + dff_filtered['clip_duration']).shift(1)
            
            # Vectorized comparison: can this row merge with the previous one?
            same_context = (
                (dff_filtered['file_name'] == prev_file) &
                (dff_filtered['channel'] == prev_channel) &
                (dff_filtered['cluster_id'] == prev_cluster)
            )
            temporal_overlap = dff_filtered['clip_time'] <= prev_clip_end
            can_merge = same_context & temporal_overlap & prev_clip_end.notna()
            
            # Create merge groups: increment group number where can_merge is False
            # This preserves the exact same grouping logic as the original loop
            merge_groups = (~can_merge).cumsum()
            dff_filtered['merge_group'] = merge_groups
            dff_filtered['clip_end'] = dff_filtered['clip_time'] + dff_filtered['clip_duration']
            
            agg_dict = {
                'clip_time': ('clip_time', 'first'),
                'clip_end': ('clip_end', 'max'),
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
        Output('label-load-dropdown', 'options'),
        Input('label-last-action', 'data'),
        prevent_initial_call=False
    )
    def update_label_dropdown(_):
        if not os.path.exists('labels'):
            os.makedirs('labels')
        files = [f.replace('.json', '') for f in os.listdir('labels') if f.endswith('.json')]
        return [{'label': p, 'value': p} for p in sorted(files)]

    @app.callback(
        [Output('label-save-message', 'children'),
         Output('label-last-action', 'data'),
         Output('label-save-name', 'value')],
        Input('label-save-btn-server', 'n_clicks'),
        State('label-save-name', 'value'),
        prevent_initial_call=True
    )
    def save_manual_labels_server_side(n_clicks, name):
        """Save manual labels to a JSON file on the server."""
        if not n_clicks:
             return dash.no_update, dash.no_update, dash.no_update
        
        if not name:
             return "Please enter a name.", dash.no_update, dash.no_update
        
        if not MANUAL_LABELS_CACHE:
             return "No labels to save.", dash.no_update, dash.no_update
            
        # Convert tuple keys to string keys for JSON serialization
        # Key format: (loc, micro, f_base, chan, sec)
        json_data = {}
        for k in MANUAL_LABELS_CACHE:
            v = MANUAL_LABELS_CACHE[k]
            # k is tuple
            key_str = "||".join(str(x) for x in k)
            json_data[key_str] = v

        if not os.path.exists('labels'):
            os.makedirs('labels')
            
        try:
            with open(f"labels/{name}.json", 'w') as f:
                json.dump(json_data, f, indent=4)
            return f"Saved '{name}' successfully!", time.time(), ""
        except Exception as e:
            return f"Error saving: {str(e)}", dash.no_update, dash.no_update

    @app.callback(
        Output('label-saved-msg', 'children', allow_duplicate=True),
        Input('label-load-btn', 'n_clicks'),
        State('label-load-dropdown', 'value'),
        prevent_initial_call=True
    )
    def load_manual_labels_server_side(n_clicks, name):
        """Load manual labels from a JSON file on the server."""
        if not n_clicks or not name:
            return dash.no_update
            
        path = f"labels/{name}.json"
        if not os.path.exists(path):
            return dash.no_update, "File not found."
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            count = 0
            # Optional: clear cache first? User requested "Reset" button kept, so maybe not auto-clear.
            # But loading usually implies "set state to this". 
            # Current behavior of load_from_file was Append/Overwrite existing keys but keep others?
            # Let's check previous implementation: 
            # "MANUAL_LABELS_CACHE[(loc...)] = label" -> UpSert.
            
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
            
            msg = f"Loaded '{name}' ({count} labels)."
            return msg
        except Exception as e:
            print(f"Error loading manual labels: {e}")
            return dash.no_update, f"Error: {str(e)}"

    @app.callback(
        Output('label-backup-msg', 'children'),
        Input('local-labels-store', 'data'),
        prevent_initial_call=False
    )
    def update_local_backup_hint(local_payload):
        if MANUAL_LABELS_CACHE:
            return ""
        parsed = _deserialize_labels_payload(local_payload)
        if not parsed:
            return ""
        return f"Browser backup found ({len(parsed)} labels). Use 'Restore from Browser Backup' to import it."

    @app.callback(
        Output('local-labels-store', 'data'),
        Input('local-labels-store', 'data'),
        State('local-labels-store', 'data'),
        prevent_initial_call=True
    )
    def backup_manual_labels_to_browser(trigger_data, existing_local_payload):
        # We don't need to trigger off manual-labels-store anymore with simple global dict
        return dash.no_update

    @app.callback(
        [Output('download-labels-json', 'data'),
         Output('label-saved-msg', 'children', allow_duplicate=True)],
        Input('download-labels-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_manual_labels_json(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update
        if not MANUAL_LABELS_CACHE:
            return dash.no_update, "No labels to download."

        payload = _build_portable_export_payload()
        file_name = f"manual_labels_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return dcc.send_string(json.dumps(payload, indent=2), file_name), f"Downloaded {payload['count']} labels."

    @app.callback(
        [Output('manual-labels-store', 'data', allow_duplicate=True),
         Output('label-saved-msg', 'children', allow_duplicate=True),
         Output('label-import-session-store', 'data'),
         Output('label-conflict-panel', 'style'),
         Output('label-conflict-text', 'children')],
        [Input('restore-local-labels-btn', 'n_clicks'),
         Input('upload-labels-json', 'contents')],
        [State('local-labels-store', 'data'),
         State('upload-labels-json', 'filename')],
        prevent_initial_call=True
    )
    def start_restore_or_upload(restore_clicks, upload_contents, local_payload, upload_filename):
        triggered = callback_context.triggered
        if not triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        trigger_id = triggered[0]['prop_id'].split('.')[0]
        incoming = {}
        source_name = ""

        if trigger_id == 'restore-local-labels-btn':
            if not restore_clicks:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            incoming = _deserialize_labels_payload(local_payload)
            source_name = 'browser backup'

        elif trigger_id == 'upload-labels-json':
            if not upload_contents:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            try:
                _, content_string = upload_contents.split(',', 1)
                decoded = base64.b64decode(content_string)
                upload_json = json.loads(decoded.decode('utf-8'))
            except Exception as ex:
                return f"Upload failed: {str(ex)}", None, {'display': 'none'}, ""

            incoming = _deserialize_labels_payload(upload_json)
            source_name = f"uploaded file '{upload_filename or 'labels.json'}'"

        if not incoming:
            return f"No valid labels found in {source_name or 'source'}.", None, {'display': 'none'}, ""

        session = _build_import_session(incoming, source_name)
        conflict_count = len(session.get('conflicts', []))

        if conflict_count == 0:
            applied_count, _, _ = _apply_import_session(session)
            msg = f"Imported {applied_count} labels from {source_name}."
            return (
                msg,
                None,
                {'display': 'none'},
                ""
            )

        session['cursor'] = 0
        conflict_text = _format_conflict_text(session)
        msg = f"Found {conflict_count} conflicts from {source_name}. Resolve them below."
        return (
            msg,
            session,
            {'display': 'block', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px', 'marginBottom': '8px'},
            conflict_text
        )

    @app.callback(
        [Output('label-saved-msg', 'children', allow_duplicate=True),
         Output('label-import-session-store', 'data', allow_duplicate=True),
         Output('label-conflict-panel', 'style', allow_duplicate=True),
         Output('label-conflict-text', 'children', allow_duplicate=True)],
        [Input('label-conflict-keep-btn', 'n_clicks'),
         Input('label-conflict-use-btn', 'n_clicks'),
         Input('label-conflict-keep-all-btn', 'n_clicks'),
         Input('label-conflict-use-all-btn', 'n_clicks'),
         Input('label-conflict-cancel-btn', 'n_clicks')],
        State('label-import-session-store', 'data'),
        prevent_initial_call=True
    )
    def resolve_label_conflicts(keep_clicks, use_clicks, keep_all_clicks, use_all_clicks, cancel_clicks, session):
        if not session or not isinstance(session, dict):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered = callback_context.triggered
        if not triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        trigger_id = triggered[0]['prop_id'].split('.')[0]

        conflicts = session.get('conflicts', [])
        resolutions = session.get('resolutions', [])
        if len(resolutions) != len(conflicts):
            resolutions = [None] * len(conflicts)
            session['resolutions'] = resolutions

        if trigger_id == 'label-conflict-cancel-btn':
            return "Restore/import canceled.", None, {'display': 'none'}, ""

        cursor = _next_unresolved_index(resolutions, session.get('cursor', 0))
        if cursor is None:
            applied_count, used_conflicts, kept_conflicts = _apply_import_session(session)
            msg = f"Import complete: {applied_count} labels applied ({used_conflicts} replaced, {kept_conflicts} kept)."
            return msg, None, {'display': 'none'}, ""

        if trigger_id == 'label-conflict-keep-btn':
            resolutions[cursor] = 'keep'
        elif trigger_id == 'label-conflict-use-btn':
            resolutions[cursor] = 'use'
        elif trigger_id == 'label-conflict-keep-all-btn':
            for idx in range(cursor, len(resolutions)):
                if resolutions[idx] is None:
                    resolutions[idx] = 'keep'
        elif trigger_id == 'label-conflict-use-all-btn':
            for idx in range(cursor, len(resolutions)):
                if resolutions[idx] is None:
                    resolutions[idx] = 'use'
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        next_idx = _next_unresolved_index(resolutions, cursor + 1)
        if next_idx is None:
            applied_count, used_conflicts, kept_conflicts = _apply_import_session(session)
            msg = f"Import complete: {applied_count} labels applied ({used_conflicts} replaced, {kept_conflicts} kept)."
            return msg, None, {'display': 'none'}, ""

        session['cursor'] = next_idx
        session['resolutions'] = resolutions
        return dash.no_update, f"Resolved {next_idx}/{len(conflicts)} conflicts.", session, {'display': 'block', 'border': '1px solid #ddd', 'borderRadius': '4px', 'padding': '8px', 'marginBottom': '8px'}, _format_conflict_text(session)

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
        Output('label-saved-msg', 'children', allow_duplicate=True),
        Input('confirm-reset-labels', 'submit_n_clicks'),
        prevent_initial_call=True
    )
    def reset_manual_labels(submit_n_clicks):
        if not submit_n_clicks:
             return dash.no_update
        
        MANUAL_LABELS_CACHE.clear()

        return "Labels Reset!"
