"""
Plot-related callbacks: scatter plot, spectrogram, and histogram.
"""
import time
import dash
from dash import Input, Output, State, callback_context, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import uuid

import utils
from .callbacks_constants import MODEL_DATA_CACHE, MERGED_DATA_CACHE, server_cache, initial_df, CLUSTER_COLORS

# Performance profiling
ENABLE_PROFILING = True
from .callbacks_data import merge_clips_vectorized


def register_plot_callbacks(app):
    
    @app.callback(
        [Output('scatter', 'figure'),
         Output('filtered-data', 'data'),
         Output('clip-count-max-store', 'data'),
         Output('merge-max-store', 'data'),
         Output('sampled-indices-store', 'data'),
         Output('anim-ranges-store', 'data'),
         Output('sampling-info-display', 'children'),
         Output('cluster-stats-store', 'data')],
        [Input('model-data-ready-signal', 'data'),
         Input('channel-checklist', 'value'),
         Input('num-cluster-dropdown', 'value'),
         Input({'type': 'cluster-checkbox', 'index': ALL}, 'value'),
         Input('cluster-color-store', 'data'),
         Input('date-dropdown', 'data'),
         Input('hour-slider', 'value'),
         Input('max-points', 'value'),
         Input('resample-btn', 'n_clicks'),
         Input('merge-switch', 'on'),
         Input('merge-threshold', 'value'),
         Input('clip-count-threshold', 'value'),
         Input('microlocation-dropdown', 'value'),
         Input('recorder-type-dropdown', 'value')],
        [State('scatter', 'figure'),
         State('sampled-indices-store', 'data'),
         State('location-dropdown', 'value'),
         State('last-preset-load-time', 'data')]
    )
    def update_figure(model_ready_signal, selected_channels, selected_num_clusters,
                      cluster_checkbox_values, cluster_colors_data,
                      selected_dates, hour_range, max_points, n_clicks, merge_on, merge_threshold,
                      clip_count_threshold,
                      selected_microlocations, selected_recorders, current_figure_state, stored_indices,
                      selected_location, last_preset_time):

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

        is_fresh_load = any('model-data-ready-signal' in t_id for t_id in all_triggered_ids)
        is_resample_click = any('resample-btn' in t_id for t_id in all_triggered_ids)
        is_k_change = any('num-cluster-dropdown' in t_id for t_id in all_triggered_ids)
        is_cluster_checkbox_click = any('cluster-checkbox' in t_id for t_id in all_triggered_ids)

        dff_raw = MODEL_DATA_CACHE.get('df')
        t_last = checkpoint('cache_get') or t_last

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
                return fig, None, 1, 100, [], None, "", {}

        if dff_raw is None:
            fig = go.Figure()
            fig.update_layout(
                annotations=[{"text": "Select a model and location to begin.", "xref": "paper", "yref": "paper",
                              "showarrow": False, "font": {"size": 16}}],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(visible=False);
            fig.update_yaxes(visible=False)
            return fig, None, 1, 100, [], None, "", {}

        is_preset_active = last_preset_time and (time.time() - last_preset_time < 6.0)
        should_force_defaults = is_fresh_load and not is_preset_active

        if should_force_defaults:
            selected_channels = []
            selected_num_clusters = None
            selected_microlocations = []
            selected_recorders = []
            selected_dates = []
            hour_range = None
            clip_count_threshold = 1
            cluster_checkbox_values = []
            stored_indices = None

        if is_fresh_load:
            stored_indices = None

        selected_clusters = []
        has_checkbox_inputs = False

        if not should_force_defaults and ctx.inputs_list and len(ctx.inputs_list) > 3:
            cluster_inputs = ctx.inputs_list[3]
            if isinstance(cluster_inputs, list) and len(cluster_inputs) > 0:
                has_checkbox_inputs = True
                for input_item in cluster_inputs:
                    val = input_item.get('value')
                    id_dict = input_item.get('id')
                    if val and 'on' in val:
                        try:
                            selected_clusters.append(int(id_dict['index']))
                        except:
                            pass

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

        dff_base = None
        max_distance = 100
        if merge_on:
            current_cache_key = (active_k, merge_threshold)
            stored_cache_key = MERGED_DATA_CACHE.get('key')
            if current_cache_key != stored_cache_key or MERGED_DATA_CACHE.get('df') is None:
                if active_k is None:
                    dff_base = pd.DataFrame()
                else:
                    dff_to_merge = dff_raw[dff_raw['cluster_num'] == active_k].copy()
                    merged_dff = merge_clips_vectorized(dff_to_merge, merge_threshold)
                    MERGED_DATA_CACHE['df'] = merged_dff
                    MERGED_DATA_CACHE['key'] = current_cache_key
                    dff_base = merged_dff
            else:
                dff_base = MERGED_DATA_CACHE.get('df')

            if not dff_base.empty:
                min_v, max_v = dff_base['x'].min(), dff_base['x'].max()
                min_y, max_y = dff_base['y'].min(), dff_base['y'].max()
                max_distance = int(np.sqrt((max_v - min_v) ** 2 + (max_y - min_y) ** 2))
        else:
            dff_base = dff_raw

        t_last = checkpoint('merge_logic') or t_last
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

        t_last = checkpoint('mask_creation') or t_last
        dff_macro = dff_base[mask]
        total_clips_at_location = len(dff_base)

        total_clips_available = dff_macro['clip_count'].sum() if not dff_macro.empty else 0
        t_last = checkpoint('filtering') or t_last

        dff_sampled = dff_macro
        new_indices_to_store = dash.no_update

        # Calculate cluster stats for percentages
        cluster_stats = {}
        if not dff_macro.empty:
            total_filtered_clips = dff_macro['clip_count'].sum()
            stats_series = dff_macro.groupby('cluster_id')['clip_count'].sum()
            cluster_stats = stats_series.to_dict()
            # Convert keys to string to ensure JSON compatibility and matching
            cluster_stats = {str(k): int(v) for k, v in cluster_stats.items()}
            cluster_stats['total'] = int(total_filtered_clips)
        else:
            cluster_stats = {'total': 0}

        should_resample = is_fresh_load or is_resample_click or is_k_change or not stored_indices

        if not should_resample:
            other_filters = ['merge-switch', 'merge-threshold', 'date-dropdown', 'hour-slider',
                             'microlocation-dropdown']
            if any(f in t_id for t_id in all_triggered_ids for f in other_filters):
                should_resample = True

        if max_points and dff_macro['clip_count'].sum() > max_points:
            if should_resample:
                rng = np.random.default_rng(n_clicks if is_resample_click else 42)
                shuffled_indices = rng.permutation(dff_macro.index)
                if len(shuffled_indices) > 0:
                    shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                    cumulative_counts = np.cumsum(shuffled_counts)
                    cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                    final_indices = shuffled_indices[:cutoff_idx]
                    if final_indices.size == 0 and len(dff_macro) > 0: final_indices = shuffled_indices[:1]
                    dff_sampled = dff_macro.loc[final_indices]
                    new_indices_to_store = dff_sampled['row_idx'].tolist()
                else:
                    dff_sampled = dff_macro
                    new_indices_to_store = []
            else:
                dff_sampled = dff_macro[dff_macro['row_idx'].isin(stored_indices)]

                if dff_sampled.empty and not dff_macro.empty:
                    rng = np.random.default_rng(42)
                    shuffled_indices = rng.permutation(dff_macro.index)
                    if len(shuffled_indices) > 0:
                        shuffled_counts = dff_macro.loc[shuffled_indices, 'clip_count'].values
                        cumulative_counts = np.cumsum(shuffled_counts)
                        cutoff_idx = np.searchsorted(cumulative_counts, max_points, side='right')
                        final_indices = shuffled_indices[:cutoff_idx]
                        if final_indices.size == 0 and len(dff_macro) > 0: final_indices = shuffled_indices[:1]
                        dff_sampled = dff_macro.loc[final_indices]
                        new_indices_to_store = dff_sampled['row_idx'].tolist()

        else:
            new_indices_to_store = []
            dff_sampled = dff_macro

        if has_checkbox_inputs and not should_force_defaults:
            should_apply_checkbox_filter = is_cluster_checkbox_click or is_preset_active

            if should_apply_checkbox_filter:
                if not dff_sampled.empty and selected_clusters:
                    dff_sampled = dff_sampled[dff_sampled['cluster_id'].isin(selected_clusters)]

        clips_visible = dff_sampled['clip_count'].sum() if not dff_sampled.empty else 0
        clips_sampled = dff_sampled['clip_count'].sum() if not dff_sampled.empty else 0

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
            return fig, None, 1, max_distance, new_indices_to_store, None, "", cluster_stats

        t_last = checkpoint('sampling') or t_last
        dff = dff_sampled.reset_index(drop=True).copy()
        dff['plot_id'] = dff.index
        dff['marker_size'] = 10 + 1 * (dff['clip_count'] - 1)
        
        if 'cluster_id_str' not in dff.columns:
            dff['cluster_id_str'] = dff['cluster_id'].astype(str)
        
        dff['cache_key'] = str(uuid.uuid4())

        if 'start_hour_float' not in dff.columns:
            dff['start_hour_float'] = 0
        if 'day_int' not in dff.columns:
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
        unique_clusters = dff['cluster_id_str'].unique()
        for cid in unique_clusters:
            if cluster_colors_data and cid in cluster_colors_data:
                final_color_map[cid] = cluster_colors_data[cid]
            else:
                try:
                    c_int = int(float(cid))
                    final_color_map[cid] = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
                except:
                    final_color_map[cid] = '#888888'

        t_last = checkpoint('color_map') or t_last
        
        fig = px.scatter(
            dff, x="x", y="y", color="cluster_id_str", size="marker_size",
            color_discrete_map=final_color_map,
            hover_data=["clip_count", "file_name", "clip_time", "channel"],
            custom_data=["cluster_id_str", "row_idx", "plot_id", "start_hour_float", "day_int", "clip_count", "file_name", "clip_time", "channel", "date_time_str"]
        )
        fig.update_traces(
            marker={'sizeref': 1, 'sizemode': 'diameter'},
            hovertemplate='<b>Cluster:</b> %{customdata[0]}<br>' +
                         '<b>File:</b> %{customdata[6]}<br>' +
                         '<b>Clip Count:</b> %{customdata[5]}<br>' +
                         '<b>Time:</b> %{customdata[7]:.2f}s<br>' +
                         '<b>Date & Time:</b> %{customdata[9]}<br>' +
                         '<b>Channel:</b> %{customdata[8]}<extra></extra>'
        )
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
            print(timing_str)

        return fig, dff['cache_key'].iloc[
            0], max_clip_count, max_distance, new_indices_to_store, ranges_data, sampling_text, cluster_stats

    @app.callback(
        Output('spectrogram-raw-data-store', 'data'),
        Output('fft-warning', 'children'),
        [Input("scatter", "clickData"),
         Input('filtered-data', 'data'),
         Input('frequency-scale', 'value'),
         Input('fft-window-size', 'value'),
         Input('window-overlap', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value')],
        prevent_initial_call=True)
    def compute_spectrogram_data(clickData, filtered_data_cache_key,
                                 frequency_scale, fft_window_size, window_overlap,
                                 window_type, min_freq, max_freq, num_bins):

        if not clickData or not filtered_data_cache_key:
            return dash.no_update, ""

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
            return dash.no_update, "Error: Filtered data not found in cache."

        point = clickData["points"][0]
        plot_id = point["customdata"][2]

        try:
            row = dff.loc[plot_id]
        except KeyError:
            return dash.no_update, "Error: Clicked point not found. Please re-filter."

        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=row['mp3_file'],
            clip_time=float(row['clip_time']),
            clip_duration=float(row['clip_duration']),
            channel=int(row['channel']),
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

        start_time = float(row['clip_time'])
        audio_path = f"/audio_segment_normalized/{row['mp3_file']}/{int(row['channel'])}/{start_time}/{start_time + float(row['clip_duration'])}"
        info = f"{row['file_name']} at {start_time:.2f}s (cluster {row['cluster_id']})"

        return {'x': t.tolist(), 'y': f.tolist(), 'z': Sxx_db.tolist(), 'audio_path': audio_path, 'info': info,
                '_rev': time.time_ns()}, ""

    @app.callback(
        [Output("info", "children", allow_duplicate=True),
         Output("audio-player", "src", allow_duplicate=True),
         Output("spectrogram-plot", "figure"),
         Output('spectrogram-plot-container', 'key')],
        [Input('spectrogram-raw-data-store', 'data'),
         Input('colormap', 'value'),
         Input('db-floor', 'value')],
        prevent_initial_call=True)
    def update_spectrogram_plot_from_cache(data, colormap, db_floor):
        if not data:
            data = {'x': [], 'y': [], 'z': [], 'info': 'No data', 'audio_path': '', '_rev': time.time_ns()}

        z_data = np.array(data['z'])
        if z_data.size > 0:
            z_data[z_data < db_floor] = db_floor

        fig = go.Figure(data=go.Heatmap(
            x=data['x'], y=data['y'], z=z_data.tolist(),
            colorscale=colormap,
            zmin=db_floor,
            zmax=np.max(z_data) if z_data.size > 0 else 0,
            colorbar=dict(title='dB')
        ))

        fig.update_layout(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Frequency (Hz)", type='log'),
            margin=dict(l=40, r=10, t=20, b=80), uirevision=data['_rev']
        )
        return data['info'], data['audio_path'], fig, str(data['_rev'])

    @app.callback(
        Output('cluster-histogram', 'figure'),
        [Input("scatter", "clickData"),
         Input('histogram-type-dropdown', 'value'),
         Input('histogram-time-scale-store', 'data'),
         Input('cluster-color-store', 'data'),
         State('filtered-data', 'data')])
    def show_histogram_for_clicked_cluster(clickData, hist_type, time_scale, cluster_colors, filtered_data_cache_key):
        if not clickData or not filtered_data_cache_key:
            fig = go.Figure()
            title = "Time of Day" if time_scale == 'daily' else "Week of Year"
            rng = [0, 1440] if time_scale == 'daily' else [1, 53]
            fig.update_layout(
                xaxis=dict(title=title, range=rng),
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

        cluster_id = str(clickData["points"][0]["customdata"][0])
        if 'cluster_id_str' in dff.columns:
            cluster_df = dff[dff['cluster_id_str'] == cluster_id].copy()
        else:
            cluster_df = dff[dff['cluster_id'].astype(str) == cluster_id].copy()

        bar_color = '#CCCCCC'
        if cluster_colors and cluster_id in cluster_colors:
            bar_color = cluster_colors[cluster_id]
        else:
            try:
                c_int = int(float(cluster_id))
                bar_color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]
            except:
                pass

        if cluster_df.empty: return go.Figure().update_layout(title_text=f"No data for cluster {cluster_id}")

        if time_scale == 'daily':
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

    @app.callback(
        Output('histogram-time-scale-store', 'data'),
        Input('histogram-click-wrapper', 'n_clicks'),
        State('histogram-time-scale-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_histogram_time_scale(n_clicks, current_scale):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return 'yearly' if current_scale == 'daily' else 'daily'

