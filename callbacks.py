import time
import dash
from dash import Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import os
import flask
import soundfile as sf
import io
import pandas as pd
import numpy as np
import uuid

import read_data
import utils

initial_df = pd.DataFrame()

# A server-side cache for the main DataFrame. This is the key to instantaneous filtering.
MODEL_DATA_CACHE = {'df': None}
# This cache is for the filtered data used by secondary callbacks (histogram, spectrogram).
server_cache = {}


def set_initial_data(df):
    global initial_df
    initial_df = df


def merge_clips_vectorized(dff, merge_threshold):
    if dff.empty:
        return dff

    if 'clip_count' not in dff.columns:
        dff['clip_count'] = 1

    dff['file_name'] = dff['file_name'].astype('category')
    dff['channel'] = dff['channel'].astype('category')

    dff = dff.sort_values(['file_name', 'channel', 'clip_time']).copy()

    same_context = (
            (dff['file_name'] == dff['file_name'].shift(1)) &
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
        'x': ('x', 'mean'),
        'y': ('y', 'mean'),
        'clip_time': ('clip_time', 'first'),
        'clip_end': ('clip_end', 'last'),
        'clip_count': ('clip_count', 'sum'),
        'cluster_id': ('cluster_id', 'first'),
        'file_name': ('file_name', 'first'),
        'channel': ('channel', 'first'),
        'start_dt': ('start_dt', 'first'),
        'time_of_day': ('time_of_day', 'first'),
        'model_name': ('model_name', 'first'),
        'mp3_file': ('mp3_file', 'first'),
    }

    grouped = dff.groupby(merge_groups).agg(**agg_dict).reset_index(drop=True)

    grouped['clip_duration'] = grouped['clip_end'] - grouped['clip_time']
    grouped = grouped.drop(columns=['clip_end'])

    grouped['row_idx'] = -1

    return grouped


def register_callbacks(dash_app):
    global app
    app = dash_app

    cluster_ids = range(20)
    cluster_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                      '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                      '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']
    color_map = {str(cid): cluster_colors[i % len(cluster_colors)] for i, cid in enumerate(cluster_ids)}

    @app.server.route("/audio_segment_normalized/<path:filename>/<int:channel>/<float:start>/<float:end>")
    def serve_audio_segment_normalized(filename, channel, start, end):
        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=filename,
            clip_time=start,
            clip_duration=(end - start),
            channel=channel
        )
        if segment is None:
            return flask.abort(404)

        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment * (0.99 / peak)

        buf = io.BytesIO()
        sf.write(buf, segment, samplerate, format='mp3')
        buf.seek(0)
        return flask.send_file(buf, mimetype="audio/mp3")

    @app.callback(
        Output('model-data-ready-signal', 'data'),
        Input('model-dropdown', 'value')
    )
    def load_data_into_server_cache(selected_model):
        if not selected_model:
            MODEL_DATA_CACHE['df'] = None
            return time.time()

        data_path = read_data.SAVE_PATH
        if not os.path.exists(data_path):
            print(f"FATAL: Source data not found at {data_path}.")
            MODEL_DATA_CACHE['df'] = None
            return time.time()

        print(f"Loading data into server cache for model: {selected_model}...")

        cols_to_load = [
            'x', 'y', 'location', 'microlocation', 'model_name', 'channel',
            'cluster_num', 'cluster_id', 'day_dt', 'start_hour_float',
            'file_name', 'clip_time', 'clip_duration', 'mp3_file', 'row_idx',
            'start_dt', 'time_of_day'
        ]

        try:
            model_df = pd.read_parquet(
                data_path,
                filters=[('model_name', '==', selected_model)],
                columns=cols_to_load
            )
            MODEL_DATA_CACHE['df'] = model_df
            print(f"Cached {len(model_df)} rows.")
        except Exception as e:
            print(f"Error loading data for model {selected_model}: {e}")
            MODEL_DATA_CACHE['df'] = None

        return time.time()

    @app.callback(
        [Output('scatter', 'figure'),
         Output('filtered-data', 'data'),
         Output('clip-count-max-store', 'data'),
         Output('merge-max-store', 'data')],
        [Input('model-data-ready-signal', 'data'),
         Input('channel-checklist', 'value'),
         Input('num-cluster-dropdown', 'value'),
         Input('cluster-checklist', 'value'),
         Input('date-dropdown', 'value'),
         Input('hour-slider', 'value'),
         Input('max-points', 'value'),
         Input('resample-btn', 'n_clicks'),
         Input('merge-switch', 'on'),
         Input('merge-threshold', 'value'),
         Input('clip-count-threshold', 'value'),
         Input('location-dropdown', 'value'),
         Input('microlocation-dropdown', 'value')]
    )
    def update_figure(model_ready_signal, selected_channels, selected_num_clusters, selected_clusters, selected_dates,
                      hour_range, max_points, n_clicks, merge_on, merge_threshold, clip_count_threshold,
                      selected_location, selected_microlocations):

        dff = MODEL_DATA_CACHE.get('df')
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'
        
        if dff is None or dff.empty:
            fig = go.Figure()
            fig.update_layout(
                annotations=[{
                    "text": "Select a model to begin.",
                    "xref": "paper", "yref": "paper",
                    "showarrow": False, "font": {"size": 16}
                }],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig, None, 1, 100

        dff_filtered = dff.copy()

        if selected_location:
            dff_filtered = dff_filtered[dff_filtered['location'] == selected_location]
        if selected_microlocations:
            dff_filtered = dff_filtered[dff_filtered['microlocation'].isin(selected_microlocations)]
        if selected_num_clusters:
            dff_filtered = dff_filtered[dff_filtered['cluster_num'] == int(selected_num_clusters)]
        if selected_channels:
            dff_filtered = dff_filtered[dff_filtered['channel'].isin(selected_channels)]
        if selected_clusters:
            dff_filtered = dff_filtered[dff_filtered['cluster_id'].isin(selected_clusters)]

        if selected_dates:
            selected_datetimes = pd.to_datetime(selected_dates).normalize()
            dff_filtered = dff_filtered[dff_filtered['day_dt'].isin(selected_datetimes)]

        if hour_range:
            dff_filtered = dff_filtered[
                (dff_filtered['start_hour_float'] >= hour_range[0]) &
                (dff_filtered['start_hour_float'] <= hour_range[1])
                ]

        dff = dff_filtered

        if dff.empty:
            fig = go.Figure()
            fig.update_layout(
                annotations=[{
                    "text": "No data found for the selected filters.",
                    "xref": "paper", "yref": "paper",
                    "showarrow": False, "font": {"size": 16}
                }],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            return fig, None, 1, 100

        if merge_on and not dff.empty:
            dff = merge_clips_vectorized(dff.copy(), merge_threshold)

        if "clip_count" not in dff.columns:
            dff["clip_count"] = 1
        if clip_count_threshold and clip_count_threshold > 1:
            dff = dff[dff["clip_count"] >= int(clip_count_threshold)]

        if max_points and dff['clip_count'].sum() > max_points:
            random_state = n_clicks if triggered_id == 'resample-btn' else 42
            
            dff_shuffled = dff.sample(frac=1, random_state=random_state)
            cumulative_clips = dff_shuffled['clip_count'].cumsum()
            
            # Find the index of the first row that makes the sum exceed max_points
            cutoff_index = np.searchsorted(cumulative_clips.values, max_points, side='right')
            
            dff = dff_shuffled.iloc[:cutoff_index]

            if dff.empty and not dff_shuffled.empty:
                dff = dff_shuffled.iloc[:1]

        if dff.empty:
            return go.Figure(), None, 1, 100

        min_value_x = dff['x'].min()
        max_value_x = dff['x'].max()
        min_value_y = dff['y'].min()
        max_value_y = dff['y'].max()
        max_distance = int(np.sqrt((max_value_x - min_value_x) ** 2 + (max_value_y - min_value_y) ** 2))

        dff = dff.reset_index(drop=True)
        dff['plot_id'] = dff.index

        dff['marker_size'] = 15 + 15 * np.sqrt(dff['clip_count'] - 1)

        dff['cluster_id'] = dff['cluster_id'].astype(str)

        cache_key = str(uuid.uuid4())
        dff['cache_key'] = cache_key

        fig = px.scatter(
            dff, x="x", y="y", color="cluster_id", size="marker_size", color_discrete_map=color_map,
            hover_data=["clip_count", "file_name", "clip_time", "channel"],
            custom_data=["cluster_id", "row_idx", "plot_id"]
        )
        fig.update_traces(marker={'sizeref': 1, 'sizemode': 'diameter'})

        fig.update_layout(
            showlegend=False,
            margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        max_clip_count = int(dff['clip_count'].max()) if not dff.empty else 1

        server_cache[cache_key] = dff

        return fig, cache_key, max_clip_count, max_distance

    @app.callback(
        Output('spectrogram-cache', 'data'),
        Output('fft-warning', 'children'),
        [Input("scatter", "clickData"),
         Input('filtered-data', 'data'),
         Input('frequency-scale', 'value'),
         Input('fft-window-size', 'value'),
         Input('window-overlap', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value'),
         Input('colormap', 'value'),
         Input('db-floor', 'value')],
        prevent_initial_call=True)
    def compute_spectrogram_data(clickData, filtered_data_cache_key,
                                 frequency_scale, fft_window_size, window_overlap,
                                 window_type, min_freq, max_freq, num_bins, colormap, db_floor):
        if not clickData or not filtered_data_cache_key:
            return dash.no_update, ""

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
            return dash.no_update, "Error: Filtered data not found in cache."

        point = clickData["points"][0]
        plot_id = point["customdata"][2]
        row = dff.loc[plot_id]

        segment, samplerate = utils.load_audio_segment(
            mp3_file_relative_path=row['mp3_file'],
            clip_time=float(row['clip_time']),
            clip_duration=float(row['clip_duration']),
            channel=int(row['channel'])
        )
        if segment is None:
            return dash.no_update, "Error: Could not load audio segment."

        f, t, Sxx_db = utils.compute_spectrogram(
            segment=segment, samplerate=samplerate, scale=frequency_scale,
            fft_window_size=fft_window_size, window_overlap=window_overlap,
            window_type=window_type, min_freq=min_freq, max_freq=max_freq,
            num_bins=num_bins, db_floor=db_floor
        )
        if Sxx_db.size == 0:
            return dash.no_update, "Warning: Spectrogram computation failed."

        start_time = float(row['clip_time'])
        audio_path = f"/audio_segment_normalized/{row['mp3_file']}/{int(row['channel'])}/{start_time}/{start_time + float(row['clip_duration'])}"
        info = f"{row['file_name']} at {start_time:.2f}s (cluster {row['cluster_id']})"

        return {'x': t.tolist(), 'y': f.tolist(), 'z': Sxx_db.tolist(), 'audio_path': audio_path, 'info': info,
                '_rev': time.time_ns(), 'colormap': colormap}, ""

    @app.callback(
        [Output("info", "children", allow_duplicate=True),
         Output("audio-player", "src", allow_duplicate=True),
         Output("spectrogram-plot", "figure"),
         Output('spectrogram-plot-container', 'key')],
        Input('spectrogram-cache', 'data'),
        prevent_initial_call=True)
    def update_spectrogram_from_cache(data):
        if not data:
            data = {'x': [], 'y': [], 'z': [], 'info': 'No data', 'audio_path': '', '_rev': time.time_ns(),
                    'colormap': 'Viridis'}

        fig = go.Figure(data=go.Heatmap(
            x=data['x'], y=data['y'], z=data['z'], colorscale=data.get('colormap', 'Viridis'),
            colorbar=dict(title='dB')
        ))
        fig.update_layout(
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Frequency (Hz)", type='log'),
            margin=dict(l=40, r=10, t=20, b=80), uirevision=data['_rev']
        )
        return data['info'], data['audio_path'], fig, str(data['_rev'])

    @app.callback(
        Output("cluster-checklist", "value"),
        Input("all-or-none-cluster", "value"),
        State("cluster-checklist", "options"))
    def select_all_none_cluster(all_selected, options):
        return [opt["value"] for opt in options] if all_selected else []

    @app.callback(
        Output("channel-checklist", "value"),
        Input("all-or-none-channel", "value"),
        State("channel-checklist", "options"))
    def select_all_none_channel(all_selected, options):
        return [opt["value"] for opt in options] if all_selected else []

    @app.callback(
        [Output('merge-threshold-container', 'style'),
         Output('clip-count-threshold-container', 'style'),
         Output('clip-count-threshold', 'value')],
        Input('merge-switch', 'on'))
    def toggle_merge_sliders(merge_on):
        style = {'display': 'block'} if merge_on else {'display': 'none'}
        return style, style, 1 if merge_on else 1

    @app.callback(
        Output('cluster-histogram', 'figure'),
        [Input('scatter', 'clickData'),
         Input('histogram-type-dropdown', 'value'),
         State('filtered-data', 'data')])
    def show_histogram_for_clicked_cluster(clickData, hist_type, filtered_data_cache_key):
        if not clickData or not filtered_data_cache_key:
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(title="Time of Day", range=[360, 1200]),
                yaxis_title="Count",
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[{
                    "text": "Click a point to see its histogram",
                    "xref": "paper", "yref": "paper",
                    "showarrow": False, "font": {"size": 14}
                }]
            )
            return fig

        dff = server_cache.get(filtered_data_cache_key)
        if dff is None:
            return go.Figure().update_layout(title_text="Error: Data not found in cache.")

        cluster_id = clickData["points"][0]["customdata"][0]
        cluster_df = dff[dff['cluster_id'] == cluster_id].copy()

        if cluster_df.empty:
            return go.Figure().update_layout(title_text=f"No data for cluster {cluster_id}")

        bin_width = 30

        cluster_df['time_of_day_minutes'] = cluster_df['time_of_day'].dt.hour * 60 + cluster_df['time_of_day'].dt.minute
        cluster_df['bin'] = (cluster_df['time_of_day_minutes'] // bin_width) * bin_width

        cluster_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377',
                          '#F2C894', '#BADCBD']
        color_map = {str(cid): cluster_colors[i % len(cluster_colors)] for i, cid in enumerate(range(20))}

        if hist_type == 'presence':
            presence = cluster_df.drop_duplicates(subset=['bin', 'location', 'channel']).groupby(
                'bin').size().reset_index(name='present')
            all_bins_df = pd.DataFrame({'bin': np.arange(0, 1440, bin_width)})
            presence = all_bins_df.merge(presence, on='bin', how='left').fillna(0)
            fig = px.bar(presence, x='bin', y='present',
                         color_discrete_sequence=[color_map.get(str(cluster_id), '#CCCCCC')])
            fig.update_layout(yaxis_title="Presence Count")
        else:
            fig = px.histogram(cluster_df, x='time_of_day_minutes',
                               color_discrete_sequence=[color_map.get(str(cluster_id), '#CCCCCC')])
            fig.update_traces(xbins=dict(start=0, end=1440, size=bin_width))
            fig.update_layout(yaxis_title="Clip Count")

        fig.update_layout(
            xaxis=dict(title="Time of Day", range=[360, 1200], tickvals=np.arange(360, 1201, 60),
                       ticktext=[f"{h // 60:02d}:00" for h in np.arange(360, 1201, 60)]),
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @app.callback(
        [Output('clip-count-threshold', 'max'),
         Output('clip-count-threshold', 'marks')],
        Input('clip-count-max-store', 'data'))
    def update_clip_count_slider(max_clip_count):
        max_val = max_clip_count or 1
        marks = {i: str(i) for i in range(1, max_val + 1, max(1, max_val // 10))}
        if max_val > 1:
            marks[1] = '1'
            marks[max_val] = str(max_val)
        return max_val, marks

    @app.callback(
        [Output('merge-threshold', 'max'),
         Output('merge-threshold', 'marks')],
        Input('merge-max-store', 'data'))
    def update_merge_slider(max_merge):
        max_val = max_merge or 1
        marks = {i: str(i) for i in range(1, max_val + 1, max(1, max_val // 10))}
        if max_val > 1:
            marks[1] = '1'
            marks[max_val] = str(max_val)
        return max_val, marks

    @app.callback(
        Output('microlocation-dropdown', 'options'),
        Input('location-dropdown', 'value'))
    def update_microlocation_options(selected_location):
        if not selected_location: return []
        dff = initial_df
        micros = sorted(dff[dff['location'] == selected_location]['microlocation'].dropna().unique())
        return [{'label': m, 'value': m} for m in micros]

    @app.callback(
        Output('cluster-checklist', 'options'),
        Input('num-cluster-dropdown', 'value'))
    def update_cluster_options(num_clusters):
        if not num_clusters: return []
        return [{'label': str(c), 'value': c} for c in range(int(num_clusters))]

    @app.callback(
        Output('date-dropdown', 'options'),
        [Input('location-dropdown', 'value'),
         Input('microlocation-dropdown', 'value')])
    def update_date_options(location, microlocations):
        if not location: return []
        dff = initial_df
        mask = (dff['location'] == location)
        if microlocations:
            mask &= dff['microlocation'].isin(microlocations)
        dates = sorted(dff.loc[mask, 'day_dt'].dropna().unique())
        return [{'label': pd.to_datetime(d).strftime('%Y-%m-%d'), 'value': pd.to_datetime(d).strftime('%Y-%m-%d')} for d
                in dates]

    @app.callback(
        [Output('audio-player', 'autoPlay'),
         Output('autoplay-toggle-btn', 'children'),
         Output('autoplay-toggle-btn', 'className')],
        Input('autoplay-toggle-btn', 'n_clicks'),
        State('audio-player', 'autoPlay'))
    def toggle_autoplay(n_clicks, current_autoplay):
        if n_clicks is None or n_clicks == 0:
            return False, "Autoplay: OFF", 'app-button autoplay-off'

        new_autoplay = not current_autoplay
        label = "Autoplay: ON" if new_autoplay else "Autoplay: OFF"

        class_name = 'app-button autoplay-on' if new_autoplay else 'app-button autoplay-off'

        return new_autoplay, label, class_name

    @app.callback(
        Output('filter-container', 'className'),
        Output('toggle-filters-btn', 'children'),
        Input('toggle-filters-btn', 'n_clicks'),
        State('filter-container', 'className'),
        prevent_initial_call=True
    )
    def toggle_filter_panel(n_clicks, current_class):
        if current_class == 'filters-expanded':
            return 'filters-collapsed', '<'
        else:
            return 'filters-expanded', '>'