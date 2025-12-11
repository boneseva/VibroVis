"""
Table view callbacks: data table rendering and interactions.
"""
import time
import dash
from dash import Input, Output, State, dash_table, callback_context
import pandas as pd
import re

from .callbacks_constants import server_cache, CLUSTER_COLORS
import utils
import numpy as np


def merge_by_cluster(dff):
    """Merge rows by cluster_id only, ignoring position. Groups consecutive rows with same cluster_id."""
    if dff.empty:
        return dff
    
    dff = dff.copy()
    
    # Ensure clip_count exists
    if 'clip_count' not in dff.columns:
        dff['clip_count'] = 1
    
    # Sort by cluster_id and clip_time for consistent merging
    dff = dff.sort_values(['cluster_id', 'clip_time'])
    
    # Group consecutive rows with same cluster_id
    dff['clip_end'] = dff['clip_time'] + dff['clip_duration']
    
    # Create merge groups: same cluster_id and overlapping/adjacent clips
    same_cluster = (dff['cluster_id'] == dff['cluster_id'].shift(1))
    prev_clip_end = dff['clip_time'].shift(1) + dff['clip_duration'].shift(1)
    temporal_continuity = dff['clip_time'] <= prev_clip_end
    merge_mask = same_cluster & temporal_continuity
    merge_groups = (~merge_mask).cumsum()
    
    # Aggregate grouped rows
    agg_dict = {
        'clip_time': ('clip_time', 'first'),
        'clip_end': ('clip_end', 'last'),
        'clip_count': ('clip_count', 'sum'),
        'cluster_id': ('cluster_id', 'first'),
        'file_name': ('file_name', 'first'),
        'channel': ('channel', 'first'),
        'start_dt': ('start_dt', 'first'),
        'day_dt': ('day_dt', 'first'),
        'mp3_file': ('mp3_file', 'first'),
    }
    
    # Add row_idx if it exists, otherwise use index
    if 'row_idx' in dff.columns:
        agg_dict['row_idx'] = ('row_idx', 'first')
    else:
        # Create row_idx from index if it doesn't exist
        dff['row_idx'] = dff.index
    
    # Only include columns that exist
    available_agg = {k: v for k, v in agg_dict.items() if k in dff.columns}
    
    grouped = dff.groupby(merge_groups).agg(**available_agg).reset_index(drop=True)
    grouped['clip_duration'] = grouped['clip_end'] - grouped['clip_time']
    grouped = grouped.drop(columns=['clip_end'])
    
    return grouped


def register_table_callbacks(app):
    
    @app.callback(
        [Output('view-mode-store', 'data'),
         Output('view-toggle-btn', 'children')],
        Input('view-toggle-btn', 'n_clicks'),
        State('view-mode-store', 'data')
    )
    def toggle_view_mode(n_clicks, current_mode):
        """Toggle between scatter plot and table view."""
        if n_clicks is None or n_clicks == 0:
            return 'scatter', '📋 Table'
        new_mode = 'table' if current_mode == 'scatter' else 'scatter'
        button_text = '📊 Scatter' if new_mode == 'table' else '📋 Table'
        return new_mode, button_text
    
    @app.callback(
        [Output('scatter', 'style'),
         Output('table-view-container', 'style')],
        Input('view-mode-store', 'data')
    )
    def switch_view_display(view_mode):
        """Show/hide scatter plot or table based on view mode."""
        if view_mode == 'table':
            return {'display': 'none'}, {'display': 'flex', 'flexDirection': 'column'}
        else:
            return {'display': 'block'}, {'display': 'none'}
    
    @app.callback(
        Output('table-column-visibility-store', 'data'),
        Input('table-column-selector', 'value'),
        prevent_initial_call=True
    )
    def update_column_visibility(selected_columns):
        """Update the column visibility store when user selects columns."""
        if selected_columns is None or len(selected_columns) == 0:
            # Default to all columns if nothing selected
            return ['date_time', 'timestamp_start', 'timestamp_end', 'cluster_id', 'wav_file', 'clip_count']
        return selected_columns
    
    @app.callback(
        [Output('table-full-data-store', 'data'),
         Output('table-style-store', 'data')],
        [Input('filtered-data', 'data'),
         Input('view-mode-store', 'data'),
         Input('cluster-color-store', 'data')],
        prevent_initial_call=False
    )
    def generate_table_data(filtered_data_cache_key, view_mode, cluster_colors):
        """Generate full table data with all columns. Only runs when filtered data changes."""
        # Only generate data when in table view mode
        if view_mode != 'table':
            return [], []
        
        if not filtered_data_cache_key:
            return [], []
        
        dff = server_cache.get(filtered_data_cache_key)
        if dff is None or dff.empty:
            return [], []
        
        # Prepare table data - format similar to export
        # Remove duplicates based on row_idx to avoid showing the same row multiple times
        if 'row_idx' in dff.columns:
            dff = dff.drop_duplicates(subset=['row_idx'], keep='first')
        
        table_data = []
        for idx, row in dff.iterrows():
            # After merging, use row_idx from the row (which points to first row of merged group)
            original_row_idx = row.get('row_idx', idx) if 'row_idx' in row and pd.notna(row.get('row_idx')) else idx
            # Calculate date_time
            date_time = ''
            timestamp_start = ''
            timestamp_end = ''
            
            if 'day_dt' in row and 'start_dt' in row and pd.notna(row['day_dt']) and pd.notna(row['start_dt']):
                day_dt = pd.to_datetime(row['day_dt'])
                start_dt = pd.to_datetime(row['start_dt'])
                start_time_str = start_dt.strftime('%H:%M:%S')
                day_str = day_dt.strftime('%Y-%m-%d')
                wav_start_datetime = pd.to_datetime(day_str + ' ' + start_time_str, errors='coerce')
                if pd.notna(wav_start_datetime):
                    date_time = wav_start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format timestamps
            def format_seconds_as_time(seconds):
                if pd.isna(seconds) or seconds < 0:
                    return ''
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f'{hours:02d}:{minutes:02d}:{secs:02d}'
            
            if 'clip_time' in row and pd.notna(row['clip_time']):
                timestamp_start = format_seconds_as_time(row['clip_time'])
            
            if 'clip_duration' in row and 'clip_time' in row:
                if pd.notna(row['clip_time']) and pd.notna(row['clip_duration']):
                    clip_end_time = row['clip_time'] + row['clip_duration']
                    timestamp_end = format_seconds_as_time(clip_end_time)
            
            # Get cluster_id
            cluster_id = int(row['cluster_id']) if 'cluster_id' in row and pd.notna(row['cluster_id']) else 0
            
            # Get wav_file
            wav_file = ''
            if 'wav_file' in row and pd.notna(row['wav_file']):
                wav_file = str(row['wav_file'])
            elif 'mp3_file' in row and pd.notna(row['mp3_file']):
                mp3_str = str(row['mp3_file'])
                wav_file = re.sub(r'_ch\d+\.mp3$', '.wav', mp3_str)
            elif 'file_name' in row and pd.notna(row['file_name']):
                wav_file = str(row['file_name'])
            
            # Get clip_count
            clip_count = int(row['clip_count']) if 'clip_count' in row and pd.notna(row['clip_count']) else 1
            
            # Create sortable date_time field (combines date_time and timestamp_start for proper sorting)
            sortable_date_time = ''
            if date_time:
                try:
                    dt = pd.to_datetime(date_time)
                    if timestamp_start:
                        time_parts = timestamp_start.split(':')
                        if len(time_parts) == 3:
                            extra_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                            dt = dt + pd.Timedelta(seconds=extra_seconds)
                    sortable_date_time = dt.timestamp()
                except:
                    sortable_date_time = date_time + ' ' + (timestamp_start if timestamp_start else '00:00:00')
            
            # Get playback data directly from merged row
            mp3_file = row.get('mp3_file', '') if 'mp3_file' in row and pd.notna(row.get('mp3_file')) else ''
            clip_time = float(row.get('clip_time', 0)) if 'clip_time' in row and pd.notna(row.get('clip_time')) else 0.0
            clip_duration = float(row.get('clip_duration', 0)) if 'clip_duration' in row and pd.notna(row.get('clip_duration')) else 0.0
            channel = int(row.get('channel', 0)) if 'channel' in row and pd.notna(row.get('channel')) else 0
            file_name = row.get('file_name', '') if 'file_name' in row and pd.notna(row.get('file_name')) else ''
            
            # Extract additional columns
            location = str(row.get('location', '')) if 'location' in row and pd.notna(row.get('location')) else ''
            microlocation = str(row.get('microlocation', '')) if 'microlocation' in row and pd.notna(row.get('microlocation')) else ''
            model_name = str(row.get('model_name', '')) if 'model_name' in row and pd.notna(row.get('model_name')) else ''
            recorder_type = str(row.get('recorder_type', '')) if 'recorder_type' in row and pd.notna(row.get('recorder_type')) else ''
            cluster_num = int(row.get('cluster_num', 0)) if 'cluster_num' in row and pd.notna(row.get('cluster_num')) else 0
            x_coord = float(row.get('x', 0)) if 'x' in row and pd.notna(row.get('x')) else 0.0
            y_coord = float(row.get('y', 0)) if 'y' in row and pd.notna(row.get('y')) else 0.0
            temperature = float(row.get('temperature', 0)) if 'temperature' in row and pd.notna(row.get('temperature')) else None
            hour = float(row.get('start_hour_float', 0)) if 'start_hour_float' in row and pd.notna(row.get('start_hour_float')) else 0.0
            
            # Create row with all columns
            clean_row = {
                'play_button': '▶',
                'date_time': date_time if date_time else '',
                'timestamp_start': timestamp_start if timestamp_start else '',
                'timestamp_end': timestamp_end if timestamp_end else '',
                'duration': clip_duration,
                'cluster_id': cluster_id if cluster_id is not None else 0,
                'cluster_num': cluster_num,
                'wav_file': wav_file if wav_file else '',
                'file_name': file_name if file_name else '',
                'clip_count': clip_count if clip_count is not None else 1,
                'location': location,
                'microlocation': microlocation,
                'model_name': model_name,
                'channel': channel,
                'recorder_type': recorder_type,
                'x': x_coord,
                'y': y_coord,
                'temperature': temperature if temperature is not None else '',
                'hour': hour,
                '_sortable_date_time': sortable_date_time,
                '_mp3_file': mp3_file,
                '_clip_time': clip_time,
                '_clip_duration': clip_duration,
                '_channel': channel,
                '_file_name': file_name,
                '_row_idx': original_row_idx
            }
            table_data.append(clean_row)
        
        # Sort by date_time by default
        if table_data:
            table_data = sorted(table_data, key=lambda x: x.get('_sortable_date_time', 0))
        
        # Create style conditions for cluster colors
        style_data_conditional = []
        if cluster_colors:
            for cluster_id_str, color in cluster_colors.items():
                try:
                    cluster_id_int = int(cluster_id_str)
                    style_data_conditional.append({
                        'if': {'filter_query': f'{{cluster_id}} = {cluster_id_int}'},
                        'backgroundColor': color,
                        'color': 'white' if _is_dark_color(color) else 'black'
                    })
                except:
                    pass
        
        # Add selected row styling
        style_data_conditional.append({
            'if': {'state': 'selected'},
            'backgroundColor': 'rgba(0, 116, 217, 0.3) !important',
            'border': '2px solid rgba(0, 116, 217, 0.5)',
            'fontWeight': 'bold'
        })
        
        return table_data, style_data_conditional
    
    @app.callback(
        [Output('data-table', 'columns'),
         Output('data-table', 'data'),
         Output('data-table', 'style_data_conditional'),
         Output('table-data-store', 'data')],
        [Input('table-full-data-store', 'data'),
         Input('table-style-store', 'data'),
         Input('table-column-visibility-store', 'data'),
         Input('view-mode-store', 'data')],
        prevent_initial_call=False
    )
    def update_table_columns(full_table_data, style_data_conditional, visible_columns, view_mode):
        """Update visible columns only. Fast - just filters the columns array without regenerating data."""
        # Define all available columns
        all_columns = {
            'play_button': {'name': '', 'id': 'play_button', 'presentation': 'markdown'},
            'date_time': {'name': 'Date/Time', 'id': 'date_time', 'type': 'datetime'},
            'timestamp_start': {'name': 'Start', 'id': 'timestamp_start'},
            'timestamp_end': {'name': 'End', 'id': 'timestamp_end'},
            'duration': {'name': 'Duration (s)', 'id': 'duration', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            'cluster_id': {'name': 'Cluster', 'id': 'cluster_id'},
            'cluster_num': {'name': 'Cluster Num (k)', 'id': 'cluster_num'},
            'wav_file': {'name': 'WAV File', 'id': 'wav_file'},
            'file_name': {'name': 'File Name', 'id': 'file_name'},
            'clip_count': {'name': 'Count', 'id': 'clip_count'},
            'location': {'name': 'Location', 'id': 'location'},
            'microlocation': {'name': 'Microlocation', 'id': 'microlocation'},
            'model_name': {'name': 'Model Name', 'id': 'model_name'},
            'channel': {'name': 'Channel', 'id': 'channel'},
            'recorder_type': {'name': 'Recorder Type', 'id': 'recorder_type'},
            'x': {'name': 'X Coordinate', 'id': 'x', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            'y': {'name': 'Y Coordinate', 'id': 'y', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            'temperature': {'name': 'Temperature (°C)', 'id': 'temperature', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            'hour': {'name': 'Hour', 'id': 'hour', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        }
        
        # Handle empty states
        if view_mode != 'table' or not full_table_data:
            if visible_columns is None:
                visible_columns = ['date_time', 'timestamp_start', 'timestamp_end', 'cluster_id', 'wav_file', 'clip_count']
            cols = [all_columns['play_button']]
            for col_id in visible_columns:
                if col_id in all_columns and col_id != 'play_button':
                    cols.append(all_columns[col_id])
            return cols, [], (style_data_conditional if style_data_conditional else []), {}
        
        # Filter columns based on user selection - FAST: just filter the columns array
        if visible_columns is None:
            visible_columns = ['date_time', 'timestamp_start', 'timestamp_end', 'cluster_id', 'wav_file', 'clip_count']
        
        # Always include play_button, then add selected columns
        columns = [all_columns['play_button']]
        for col_id in visible_columns:
            if col_id in all_columns and col_id != 'play_button':
                columns.append(all_columns[col_id])
        
        # Use the full table data as-is - no regeneration needed!
        # The data already contains all columns, Dash DataTable will only display the ones in the columns array
        table_data = full_table_data
        
        # Create row index mapping for compatibility
        row_index_mapping = {}
        for i, row in enumerate(table_data):
            original_idx = row.get('_row_idx', None)
            row_index_mapping[str(i)] = original_idx
        
        return columns, table_data, (style_data_conditional if style_data_conditional else []), row_index_mapping
    
    @app.callback(
        Output('data-table', 'data', allow_duplicate=True),
        Input('data-table', 'sort_by'),
        State('data-table', 'data'),
        prevent_initial_call=True
    )
    def handle_table_sort(sort_by, current_data):
        """Handle custom sorting for the table, with secondary sort by timestamp_start when sorting by date_time."""
        if not sort_by or not current_data:
            return dash.no_update
        
        # Convert to DataFrame for easier sorting
        df = pd.DataFrame(current_data)
        if df.empty:
            return dash.no_update
        
        # Determine sort columns and directions
        sort_columns = []
        for sort_entry in sort_by:
            col_id = sort_entry['column_id']
            ascending = sort_entry['direction'] == 'asc'
            
            # If sorting by date_time, use the hidden sortable field which already includes timestamp_start
            if col_id == 'date_time':
                # The _sortable_date_time field already combines date_time + timestamp_start
                sort_columns.append(('_sortable_date_time', ascending))
            else:
                sort_columns.append((col_id, ascending))
        
        # Perform sorting
        if sort_columns:
            df = df.sort_values([col for col, _ in sort_columns], ascending=[asc for _, asc in sort_columns], na_position='last')
        
        # Note: _sortable_date_time is kept in data but not displayed (not in columns definition)
        return df.to_dict('records')
    
    @app.callback(
        [Output('spectrogram-raw-data-store', 'data', allow_duplicate=True),
         Output('fft-warning', 'children', allow_duplicate=True)],
        [Input('data-table', 'active_cell'),
         Input('data-table', 'data'),
         Input('frequency-scale', 'value'),
         Input('fft-window-size', 'value'),
         Input('window-overlap', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value')],
        prevent_initial_call=True
    )
    def handle_table_row_click(active_cell, table_data, frequency_scale, fft_window_size, window_overlap,
                               window_type, min_freq, max_freq, num_bins):
        """Handle play button click to show audio and spectrogram."""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, ""
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # Respond to clicks on any cell in the table row
        row_idx = None
        if 'active_cell' in trigger_id and active_cell:
            # Trigger on any cell click in the table (not just play_button)
            row_idx = active_cell.get('row')
        
        if row_idx is None or not table_data or row_idx >= len(table_data):
            return dash.no_update, ""
        
        # Get playback data directly from table row (stored as hidden fields)
        row = table_data[row_idx]
        mp3_file = row.get('_mp3_file', '')
        clip_time = row.get('_clip_time', 0.0)
        clip_duration = row.get('_clip_duration', 0.0)
        channel = row.get('_channel', 0)
        file_name = row.get('_file_name', '')
        cluster_id = row.get('cluster_id', 0)
        
        if not mp3_file:
            return dash.no_update, "Error: No audio file found for this row."
        
        # Use the same logic as scatter plot click
        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=mp3_file,
            clip_time=clip_time,
            clip_duration=clip_duration,
            channel=channel,
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
        
        audio_path = f"/audio_segment_normalized/{mp3_file}/{channel}/{clip_time}/{clip_time + clip_duration}"
        info = f"{file_name} at {clip_time:.2f}s (cluster {cluster_id})"
        
        return {'x': t.tolist(), 'y': f.tolist(), 'z': Sxx_db.tolist(), 'audio_path': audio_path, 'info': info,
                '_rev': time.time_ns()}, ""
    
    @app.callback(
        Output("download-table-csv", "data"),
        Input("export-table-btn", "n_clicks"),
        [State('data-table', 'data'),
         State('data-table', 'filter_query'),
         State('table-data-store', 'data')],
        prevent_initial_call=True
    )
    def export_table_data(n_clicks, table_data, filter_query, table_data_store):
        """Export filtered table data to CSV."""
        if n_clicks == 0 or not table_data:
            return dash.no_update
        
        # Apply any active filters from the table
        # Note: filter_query is a string like "{cluster_id} = 4"
        # For now, export all visible table data (filtering is handled by DataTable)
        export_df = pd.DataFrame(table_data)
        
        # Remove internal/hidden columns (those starting with '_')
        hidden_cols = [col for col in export_df.columns if col.startswith('_')]
        if hidden_cols:
            export_df = export_df.drop(columns=hidden_cols)
        
        # Sort by date_time and wav_file
        if 'date_time' in export_df.columns:
            export_df = export_df.sort_values(['wav_file', 'date_time'])
        
        from dash import dcc
        return dcc.send_data_frame(export_df.to_csv, "table_export.csv", index=False)


def _is_dark_color(color):
    """Check if a color is dark (for text contrast)."""
    if not color or len(color) < 7:
        return False
    try:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return brightness < 128
    except:
        return False

