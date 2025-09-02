import time

import dash
from dash import Input, Output, State
import plotly.express as px
import os
import flask
import soundfile as sf
import io
import plotly.io as pio

import read_data
from read_data import DATA_DIR

import numpy as np

from scipy.signal import spectrogram
from io import StringIO

def merge_clips_vectorized(dff, merge_threshold):
    if dff.empty:
        return dff

    if 'clip_count' not in dff.columns:
        dff['clip_count'] = 1

    dff = dff.sort_values(['file_name', 'channel', 'clip_time']).copy()

    dff['prev_file'] = dff['file_name'].shift(1)
    dff['prev_channel'] = dff['channel'].shift(1)
    dff['prev_cluster'] = dff['cluster_id'].shift(1)
    dff['prev_clip_end'] = dff['clip_time'].shift(1) + dff['clip_duration'].shift(1)

    prev_points = dff[['x', 'y']].shift(1).to_numpy()
    curr_points = dff[['x', 'y']].to_numpy()
    distances = np.linalg.norm(curr_points - prev_points, axis=1)

    same_context = (
        (dff['file_name'] == dff['prev_file']) &
        (dff['channel'] == dff['prev_channel']) &
        (dff['cluster_id'] == dff['prev_cluster'])
    )
    dff['prev_clip_time'] = dff['clip_time'].shift(1)
    temporal_continuity = dff['clip_time'] <= dff['prev_clip_time'].add(dff['clip_duration'].shift(1),
                                                                        fill_value=0) + 1.0
    spatial_proximity = distances < merge_threshold

    merge_mask = same_context & temporal_continuity & spatial_proximity
    merge_groups = (~merge_mask).cumsum()

    dff['row_idx_merging'] = np.arange(len(dff))
    dff['clip_end'] = dff['clip_time'] + dff['clip_duration']
    agg_dict = {
        'x': ('x', 'mean'),
        'y': ('y', 'mean'),
        'clip_time': ('clip_time', 'first'),
        'clip_end': ('clip_end', 'last'),
        'clip_count': ('clip_count', 'sum'),
        'x_sum': ('x', 'sum'),
        'y_sum': ('y', 'sum'),
        'cluster_id': ('cluster_id', 'first'),
        'file_name': ('file_name', 'first'),
        'channel': ('channel', 'first'),
        'start_dt': ('start_dt', 'first'),
        'row_idx_merging_min': ('row_idx_merging', 'min'),
        'row_idx_merging_max': ('row_idx_merging', 'max'),
        'model_name': ('model_name', 'first'),
        'mp3_file': ('mp3_file', 'first'),
        'row_idx': ('row_idx', list)
    }

    grouped = dff.groupby(merge_groups).agg(**agg_dict).reset_index(drop=True)
    grouped['clip_duration'] = grouped['clip_end'] - grouped['clip_time']
    grouped = grouped.drop(columns=['clip_end'])

    return grouped


def precompute_cluster_histograms(dff):
    dff = dff.copy()
    hist_cache = {}
    for cluster in dff['cluster_id'].unique():
        cluster_int = int(cluster)
        cluster_df = dff[dff['cluster_id'] == cluster]
        fig = px.histogram(
            cluster_df,
            x='time_of_day',
            title=f'Cluster {cluster_int}',
            nbins=12,
        )
        fig.update_layout(bargap=0.1, xaxis=dict(dtick=1), showlegend=False)
        hist_cache[str(cluster_int)] = pio.to_json(fig)
    return hist_cache

app = None

def register_callbacks(dash_app):
    global app
    app = dash_app

    cluster_ids = range(20)
    cluster_colors = ['#4E79A7',  # muted blue
    '#F28E2B',  # orange
    '#E15759',  # soft red
    '#76B7B2',  # teal
    '#EDC948',  # yellow
    '#B07AA1',  # muted purple
    '#FF9DA7',  # light coral
    '#A6A377',  # olive green
    '#F2C894',  # peach
    '#BADCBD'   # muted light green
                      ]
    color_map = {str(cid): cluster_colors[i % len(cluster_colors)] for i, cid in enumerate(cluster_ids)}

    @app.server.route("/audio_segment_normalized/<path:filename>/<int:channel>/<float:start>/<float:end>")
    def serve_audio_segment_normalized(filename, channel, start, end):
        abs_path = os.path.join(DATA_DIR, filename)
        abs_path = abs_path.replace("mp3/mp3", "mp3")
        try:
            data, samplerate = sf.read(abs_path)
        except Exception as e:
            print(f"Error loading audio file: {abs_path}\n{e}")
            return flask.abort(404)
        channel_data = data
        pad = 1.0
        segment_start = max(0, start - pad)
        segment_end = min(len(channel_data) / samplerate, end + pad)
        start_sample = int(segment_start * samplerate)
        end_sample = int(segment_end * samplerate)
        segment = channel_data[start_sample:end_sample]

        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment * (0.99 / peak)

        buf = io.BytesIO()
        try:
            sf.write(buf, segment, samplerate, format='mp3')
        except Exception as e:
            print(f"Error writing audio segment: {e}")
            return flask.abort(500)
        buf.seek(0)
        return flask.send_file(buf, mimetype="audio/mp3")

    @app.callback(
         Output('spectrogram-cache', 'data'),
        Output('fft-warning', 'children'),
        [Input("scatter", "clickData"),
         Input('filtered-data', 'data'),
         Input('fft-window-size', 'value'),
         Input('fft-step-size', 'value'),
         Input('nfft', 'value'),
         Input('window-type', 'value'),
         Input('min-freq', 'value'),
         Input('max-freq', 'value'),
         Input('num-bins', 'value'),
         Input('colormap', 'value'),
         Input('db-floor', 'value'),
         ],
        prevent_initial_call=True)
    def compute_spectrogram_data(clickData, filtered_json, fft_window_size, fft_step_size,
                                 nfft, window_type, min_freq, max_freq, num_bins, colormap, db_floor):
        if not clickData or not filtered_json:
            return dash.no_update

        dff = pd.read_json(StringIO(filtered_json), orient='split')
        point = clickData["points"][0]
        row_idx = point["customdata"][1]

        if isinstance(row_idx, list):
            mask = dff['row_idx'].apply(lambda x: any(r in x for r in row_idx) if isinstance(x, list) else x in row_idx)
        else:
            mask = dff['row_idx'].apply(lambda x: row_idx in x if isinstance(x, list) else x == row_idx)

        row = dff[mask]
        if row.empty:
            return dash.no_update
        row = row.iloc[0]

        start = float(row['clip_time'])
        end = start + float(row['clip_duration'])
        channel = int(row['channel'])
        abs_path = os.path.join(DATA_DIR, row['mp3_file'])
        abs_path = abs_path.replace("mp3/mp3", "mp3")
        print(abs_path)
        data, samplerate = sf.read(abs_path)
        if data.ndim == 1:
            channel_data = data
        else:
            channel_data = data[:, channel - 1]
        pad = 1.0
        segment_start = max(0, start - pad)
        segment_end = min(len(channel_data) / samplerate, end + pad)
        start_sample = int(segment_start * samplerate)
        end_sample = int(segment_end * samplerate)
        segment = channel_data[start_sample:end_sample]
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment * (0.99 / peak)

        if fft_window_size is None or fft_step_size is None or fft_window_size <= fft_step_size:
            fft_window_size = 320
            fft_step_size = 160
        if nfft is None or nfft < fft_window_size:
            nfft = fft_window_size
        if min_freq is None: min_freq = 50
        if max_freq is None: max_freq = 5000
        if num_bins is None: num_bins = 512
        if db_floor is None: db_floor = -100
        if window_type is None: window_type = 'hann'
        if colormap is None: colormap = 'Viridis'

        noverlap = fft_window_size - fft_step_size

        f, t, Sxx = spectrogram(
            segment,
            fs=samplerate,
            nperseg=fft_window_size,
            noverlap=noverlap,
            nfft=nfft,
            window=window_type
        )

        log_f = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_bins)
        Sxx_log = np.zeros((len(log_f), Sxx.shape[1]))
        for i in range(Sxx.shape[1]):
            Sxx_log[:, i] = np.interp(log_f, f, Sxx[:, i])
        z = 10 * np.log10(np.maximum(Sxx_log, 1e-10))
        z = np.clip(z, db_floor, None)
        z = np.nan_to_num(z, nan=db_floor, posinf=0, neginf=db_floor)

        audio_path = f"/audio_segment_normalized/{row['mp3_file']}/{channel}/{start}/{end}"
        info = f"{row['file_name']} at {row['clip_time']}s (cluster {row['cluster_id']})"

        if np.ptp(z) <= 1:
            return dash.no_update
        if np.isnan(z).all():
            return dash.no_update

        return {
            # 'x': (t + segment_start).tolist(),
            'x': t.tolist(),
            'y': log_f.tolist(),
            'z': z.tolist(),
            'audio_path': audio_path,
            'info': info,
            '_rev': time.time_ns(),
            'colormap': colormap,
        }, ""


    @app.callback(
        [Output("info", "children", allow_duplicate=True),
         Output("audio-player", "src", allow_duplicate=True),
         Output("spectrogram-plot", "figure"),
         Output('spectrogram-plot-container', 'key')],
        Input('spectrogram-cache', 'data'),
        prevent_initial_call=True)
    def update_spectrogram_from_cache(data):
        import plotly.graph_objects as go
        import numpy as np
        import time

        if not data:
            data['x'] = [0,0,0,0,0,0]
            data['y'] = [0,0,0,0,0,0]
            data['z'] = [[0, 0, 0, 0, 0, 0]]

        x = np.array(data['x']).tolist()
        y = np.array(data['y']).tolist()
        z = np.array(data['z']).tolist()

        fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale= data.get('colormap', 'Viridis'),
            zmin=np.nanmin(z) if z else -100,
            zmax=np.nanmax(z) if z else 0,
            colorbar=dict(title='dB'),
            uid=str(time.time_ns())
        ))

        fig.update_layout(
            xaxis=dict(
                title="Time (s)",
                autorange=True,
                fixedrange=False,
                rangeslider_visible=False,
                range=None
            ),
            yaxis=dict(
                title="Frequency (Hz)",
                type='log',
                autorange=True,
                fixedrange=False,
                exponentformat='power',
                range=None
            ),
            margin=dict(l=40, r=10, t=20, b=40),
            height=200,
            uirevision=f"rev_{data['_rev']}_{time.time_ns()}"
        )

        fig['layout']['meta'] = f"ts_{time.time_ns()}"
        return data['info'], data['audio_path'], fig, str(data['_rev'])

    @app.callback(
        Output("cluster-checklist", "value"),
        [Input("all-or-none-cluster", "value")],
        [State("cluster-checklist", "options")],
    )
    def select_all_none(all_selected, options):
        all_or_none = []
        all_or_none = [option["value"] for option in options if all_selected]
        return all_or_none

    @app.callback(
        Output("channel-checklist", "value"),
        [Input("all-or-none-channel", "value")],
        [State("channel-checklist", "options")],
    )
    def select_all_none(all_selected, options):
        all_or_none = []
        all_or_none = [option["value"] for option in options if all_selected]
        return all_or_none

    @app.callback(
        Output('histogram-cache', 'data'),
        Input('model-dropdown', 'value'),
        Input('channel-checklist', 'value'),
        Input('num-cluster-dropdown', 'value'),
        Input('histogram-type-dropdown', 'value'),
        Input('location-dropdown', 'value'),
        # Input('histogram-bin-width', 'value')     # Optional: bin width slider if you want
    )
    def update_histogram_cache(
            selected_models,
            selected_channels,
            selected_num_clusters,
            histogram_type,  # 'count' or 'presence'
            selected_locations
            # bin_width      # Set to 30 if not user-controlled
    ):
        bin_width = 30  # Default bin width in minutes; replace with bin_width if using slider
        dff = read_data.df.copy()
        dff = dff[
            dff['model_name'].isin([selected_models] if isinstance(selected_models, str) else selected_models)]
        dff = dff[dff['location'].isin([selected_locations] if isinstance(selected_locations, str) else selected_locations)]
        if selected_channels:
            dff = dff[dff['channel'].isin(selected_channels)]
        if selected_num_clusters:
            dff = dff[
                dff['cluster_num'].isin(
                    [selected_num_clusters] if isinstance(selected_num_clusters, int) else selected_num_clusters)]

        dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
        dff['time_of_day_minutes'] = (
                dff['time_of_day'].dt.hour * 60 +
                dff['time_of_day'].dt.minute +
                dff['time_of_day'].dt.second / 60
        )
        # For presence histogram, create a discrete bin column
        dff['bin'] = (dff['time_of_day_minutes'] // bin_width) * bin_width

        hist_cache = {}
        for cluster in dff['cluster_id'].unique():
            cluster_int = int(cluster)
            cluster_df = dff[dff['cluster_id'] == cluster]
            if histogram_type == 'presence':
                # Presence mode: 1 if any detection for (bin, location, channel), summed per bin
                presence = (
                    cluster_df
                    .drop_duplicates(subset=['bin', 'location', 'channel'])
                    .groupby('bin')
                    .size()
                    .reset_index(name='present')
                )
                # Expand to include all bins (set missing to 0)
                import numpy as np
                all_bins = np.arange(0, 1440, bin_width)
                all_bins_df = pd.DataFrame({'bin': all_bins})
                presence = all_bins_df.merge(presence, on='bin', how='left').fillna(0)

                fig = px.bar(
                    presence,
                    x='bin',
                    y='present',
                    title=None,
                    color_discrete_sequence=[color_map[str(cluster)]],
                )
                fig.update_traces(width=bin_width)
                fig.update_layout(
                    xaxis=dict(
                        title="Time of Day",
                        dtick=bin_width,
                        range=[360, 1200],
                        tickvals=[i for i in range(360, 1200, 60)],
                        ticktext=[f"{i // 60:02d}:{i % 60:02d}" for i in range(360, 1200, 60)]
                    ),
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=10),
                    xaxis_title='Time of Day',
                    plot_bgcolor='#f9f9f9',
                    paper_bgcolor='#f9f9f9'
                )
            else:
                # Count mode: classic histogram by minute-of-day
                fig = px.histogram(
                    cluster_df,
                    x='time_of_day_minutes',
                    title=None,
                    color_discrete_sequence=[color_map[str(cluster)]],
                )
                fig.update_traces(xbins=dict(start=0, end=1440, size=bin_width))
                fig.update_layout(
                    xaxis=dict(
                        title="Time of Day",
                        dtick=bin_width,
                        range=[360, 1200],
                        tickvals=[i for i in range(360, 1200, 60)],
                        ticktext=[f"{i // 60:02d}:{i % 60:02d}" for i in range(360, 1200, 60)]
                    ),
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=10),
                    xaxis_title='Time of Day',
                    yaxis_title=None,
                    yaxis=dict(showticklabels=True, showgrid=False, zeroline=False),
                    plot_bgcolor='#f9f9f9',
                    paper_bgcolor='#f9f9f9'
                )
            hist_cache[str(cluster_int)] = pio.to_json(fig)
        return hist_cache

    def to_plain(val):
        if isinstance(val, (np.ndarray, pd.Series)):
            return val.tolist()
        if hasattr(val, 'to_pydatetime'):  # For pandas.Timestamp
            return val.to_pydatetime()
        if hasattr(val, 'item'):
            return val.item()
        return val

    def clean_dict(d):
        return {k: to_plain(v) for k, v in d.items()}

    @app.callback(
        [Output('scatter', 'figure'),
         Output('filtered-data', 'data'),
         Output('clip-count-max-store', 'data')],
        [Input('model-dropdown', 'value'),
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
         Input('microlocation-dropdown', 'value'),
         Input('selected-point-index', 'data'),
         Input('label-filter-checklist', 'value'),
         Input('color-by-radio', 'value')],
        [State('labels-store', 'data')],
    )
    def update_figure(selected_models, selected_channels, selected_num_clusters, selected_clusters, selected_dates,
                      hour_range, max_points,
                      n_clicks, merge_on, merge_threshold, clip_count_threshold, selected_location,
                      selected_microlocations, selected_point_index, label_filter, color_by, labels_store):
        print("Called update_figure")
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        dff = read_data.df.copy()

        if not selected_channels or not selected_clusters:
            empty_fig = px.scatter(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
            return empty_fig, pd.DataFrame().to_json(date_format='iso', orient='split'), None

        if selected_models:
            dff = dff[
                dff['model_name'].isin([selected_models] if isinstance(selected_models, str) else selected_models)]
        if selected_num_clusters:
            dff = dff[
                dff['cluster_num'].isin(
                    [selected_num_clusters] if isinstance(selected_num_clusters, int) else selected_num_clusters)]
        if selected_channels:
            dff = dff[dff['channel'].isin(selected_channels)]
        if selected_clusters:
            dff = dff[dff['cluster_id'].isin(selected_clusters)]
        if selected_dates:
            selected_dates = pd.to_datetime(selected_dates, format="%Y-%m-%d")
            dff = dff[dff['day_dt'].isin(selected_dates)]
        if hour_range:
            dff = dff[(dff['start_hour_float'] >= hour_range[0]) & (dff['start_hour_float'] <= hour_range[1])]

        if selected_location:
            dff = dff[dff['location'] == selected_location]
        if selected_microlocations:
            dff = dff[dff['microlocation'].isin(selected_microlocations)]

        print(f"Data points before merge: {len(dff)}")
        if merge_on and merge_threshold is not None and not dff.empty:
            dff = merge_clips_vectorized(dff, merge_threshold)
            print(f"Data points after merge: {len(dff)}")

        if "clip_count" not in dff.columns:
            dff["clip_count"] = 1

        if clip_count_threshold:
            try:
                clip_count_threshold = int(clip_count_threshold)
            except Exception:
                clip_count_threshold = 1
            dff = dff[dff["clip_count"] >= clip_count_threshold]

        dff["clip_count"] = dff["clip_count"].astype(int)

        labels_filter = label_filter or ['unlabeled', 'background']
        labels_data = labels_store or {}

        dff['label'] = dff['row_idx'].astype(str).map(lambda rid: labels_data.get(rid, 'unlabeled'))
        dff = dff[dff['label'].isin(labels_filter)]

        # Define triggers for resampling
        filter_trigger_ids = {
            'model-dropdown',
            'channel-checklist',
            'num-cluster-dropdown',
            'cluster-checklist',
            'date-dropdown',
            'hour-slider',
            'max-points',
            'clip-count-threshold',
            'location-dropdown',
            'microlocation-dropdown',
            'label-filter-checklist',
        }
        should_resample = trigger_id in filter_trigger_ids or trigger_id == 'resample-btn'

        if max_points and len(dff) > max_points and should_resample:
            is_labeled = dff['label'] != 'unlabeled'
            labeled_points = dff[is_labeled]
            unlabeled_points = dff[~is_labeled]

            n_labeled = len(labeled_points)
            labeled_points_needed = min(n_labeled, max_points)
            labeled_points = labeled_points.iloc[:labeled_points_needed]

            remaining = max_points - len(labeled_points)
            if remaining > 0 and len(unlabeled_points) > 0:
                unlabeled_sample = unlabeled_points.sample(
                    n=remaining,
                    weights="clip_count",
                    random_state=42 + (n_clicks or 0)
                )
                dff = pd.concat([labeled_points, unlabeled_sample], ignore_index=True)
            else:
                dff = labeled_points
            dff = dff.reset_index(drop=True)

        base_size = 10
        scale = 5

        def marker_size_func(clip_count):
            return base_size + scale * clip_count

        dff['marker_size'] = dff['clip_count'].apply(marker_size_func)
        dff['cluster_id'] = dff['cluster_id'].astype(str)

        if color_by == 'label':
            dff['color_label'] = dff['label'].astype(str)
            unique_labels = dff['label'].unique()
            palette = px.colors.qualitative.Safe  # or any palette
            color_map_labels = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
            color_map_labels['background'] = 'gray'  # ensure background is gray
            custom_color_map = color_map_labels

        else:
            dff['color_label'] = dff.apply(
                lambda row: 'background' if row['label'] == 'background' else str(row['cluster_id']), axis=1)
            custom_color_map = color_map.copy()
            custom_color_map['background'] = 'gray'

        fig = px.scatter(
            dff,
            x="x",
            y="y",
            color="color_label",
            size="marker_size",
            color_discrete_map=custom_color_map,
            hover_data=["clip_count", "file_name", "clip_time", "model_name", "channel", "label"],
            custom_data=["cluster_id", "row_idx"],
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            showlegend=False
        )

        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        fig.update_layout(uirevision=str(hash(dff.to_json())))
        fig.update_coloraxes(showscale=False)

        max_clip_count = int(dff['clip_count'].max()) if not dff.empty else 1

        return fig, dff.to_json(date_format='iso', orient='split'), max_clip_count

    @app.callback(
        Output('label-filter-checklist', 'options'),
        Input('labels-store', 'data')
    )
    def update_label_filter_options(labels_store):
        if not labels_store:
            return []

        unique_labels = sorted(set(label.lower() for label in labels_store.values()))

        options = [{'label': label.capitalize(), 'value': label} for label in unique_labels if label != 'unlabeled']
        options.insert(0, {'label': 'Unlabeled', 'value': 'unlabeled'})
        print("Label filter options:", options)

        return options

    @app.callback(
        Output('selected-point-index', 'data'),
        Input('scatter', 'clickData'),
        prevent_initial_call=True
    )
    def store_selected_row_idx(clickData):
        if not clickData:
            return dash.no_update

        stable_row_idx = clickData["points"][0]["customdata"][1]
        print("clicked stable row_idx:", stable_row_idx)

        if isinstance(stable_row_idx, list):
            return [int(idx) for idx in stable_row_idx]

        return int(stable_row_idx)

    @app.callback(
        Output('existing-label-dropdown', 'options'),
        Input('labels-store', 'data')
    )
    def update_existing_labels(labels_store):
        unique_labels = sorted(set(label.lower() for label in labels_store.values()))

        if 'background' not in unique_labels:
            unique_labels.append('background')

        options = [{'label': label.capitalize(), 'value': label} for label in unique_labels if label != 'unlabeled']
        print("Existing label options:", options)
        return options

    @app.callback(
        Output('labels-store', 'data'),
        Input('apply-manual-label-btn', 'n_clicks'),
        State('selected-point-index', 'data'),
        State('existing-label-dropdown', 'value'),
        State('new-label-input', 'value'),
        State('labels-store', 'data'),
        prevent_initial_call=True
    )
    def apply_manual_label(n_clicks, selected_point_index, selected_existing_label, typed_new_label, current_labels):
        if selected_point_index is None:
            return dash.no_update

        if not isinstance(selected_point_index, list):
            selected_point_index = [selected_point_index]

        print("Applying label to row_idx:", selected_point_index)

        label_to_apply = typed_new_label.strip() if typed_new_label and typed_new_label.strip() else selected_existing_label
        if not label_to_apply:
            return dash.no_update

        print(f"Label to apply: '{label_to_apply}'")

        if current_labels is None or isinstance(current_labels, list):
            current_labels = {}

        full_df = read_data.df.copy()
        matched_original_mask = full_df['row_idx'].apply(
            lambda x: any(r == x for r in selected_point_index) if not isinstance(x, list) else any(
                r in x for r in selected_point_index)
        )
        original_clips = full_df.loc[matched_original_mask]

        if original_clips.empty:
            return dash.no_update

        all_to_label = pd.DataFrame()
        for _, clip in original_clips.iterrows():
            file_name = clip['file_name']
            channel = clip['channel']
            clip_time = clip['clip_time']
            match_mask = (
                    (full_df['file_name'] == file_name) &
                    (full_df['channel'] == channel) &
                    (full_df['clip_time'] == clip_time)
            )
            all_to_label = pd.concat([all_to_label, full_df.loc[match_mask]], ignore_index=True)

        print(f"Matched points:")
        print(all_to_label[['row_idx', 'file_name', 'channel', 'clip_time', 'model_name']])

        label_to_apply = label_to_apply.lower()

        for row_idx in all_to_label['row_idx']:
            current_labels[str(row_idx)] = label_to_apply

        print(f"Applied manual label '{label_to_apply}' to {len(all_to_label)} clips")

        return current_labels

    @app.callback(
        Output('merge-threshold-container', 'style'),
        Output('clip-count-threshold-container', 'style'),
        Output('clip-count-threshold', 'value'),
        Input('merge-switch', 'on')
    )
    def toggle_merge_sliders(merge_on):
        if merge_on:
            return {'marginTop': '0.5em'}, {'marginTop': '0.5em'}, 2
        else:
            return {'display': 'none'}, {'display': 'none'}, 1

    @app.callback(
        Output('cluster-histogram', 'figure'),
        Input('scatter', 'clickData'),
        Input('histogram-cache', 'data')
    )
    def show_histogram_for_clicked_cluster(clickData, hist_cache):
        import plotly.io as pio
        import pandas as pd
        import plotly.express as px

        if not clickData or not hist_cache:
            empty_df = pd.DataFrame({"time_of_day": [], "count": []})
            empty_fig = px.histogram(empty_df, x="time_of_day")
            return empty_fig

        cluster_id = int(clickData["points"][0]["customdata"][0])
        cluster_id_str = str(cluster_id)
        if cluster_id_str in hist_cache:
            return pio.from_json(hist_cache[cluster_id_str])
        else:
            empty_df = pd.DataFrame({"time_of_day": [], "count": []})
            empty_fig = px.histogram(empty_df, x="time_of_day")
            return empty_fig

    @app.callback(
        Output('clip-count-threshold', 'max'),
        Output('clip-count-threshold', 'marks'),
        [
            Input('clip-count-max-store', 'data')
        ],
    )
    def update_clip_count_slider(max_clip_count):
        if max_clip_count is None or not isinstance(max_clip_count, int):
            max_clip_count = 1
        if max_clip_count <= 10:
            marks = {i: str(i) for i in range(1, max_clip_count + 1)}
        else:
            step = max(1, max_clip_count // 5)
            marks = {i: str(i) for i in range(1, max_clip_count + 1, step)}
            if max_clip_count not in marks:
                marks[max_clip_count] = str(max_clip_count)

        return max_clip_count, marks

    @app.callback(
        Output('microlocation-dropdown', 'options'),
        Input('location-dropdown', 'value')
    )
    def update_microlocation_options(selected_location):
        dff = read_data.df.copy()
        possible_micros = sorted(dff[dff['location'] == selected_location]['microlocation'].dropna().unique())
        options = [{'label': str(m), 'value': m} for m in possible_micros]

        return options

    @app.callback(
        Output('cluster-checklist', 'options'),
        Input('num-cluster-dropdown', 'value'))
    def update_cluster_options(selected_num_clusters):
        options = [{'label': str(c), 'value': c} for c in range(selected_num_clusters)]
        value = [str(c) for c in range(selected_num_clusters)]
        return options

    @app.callback(
        Output('date-dropdown', 'options'),
        Input('location-dropdown', 'value'),
        Input('microlocation-dropdown', 'value')
    )
    def update_date_options(selected_location, selected_microlocation):
        dff = read_data.df.copy()

        if selected_location is None:
            return []

        if not selected_microlocation:
            location_mask = dff['location'] == selected_location
        else:
            # Handle single or multi-select microlocation
            if isinstance(selected_microlocation, str):
                micro_mask = dff['microlocation'] == selected_microlocation
            else:
                micro_mask = dff['microlocation'].isin(selected_microlocation)
            location_mask = (dff['location'] == selected_location) & micro_mask

        filtered = dff[location_mask]
        possible_dates = sorted(filtered['day_dt'].dropna().unique())

        options = [
            {'label': str(pd.to_datetime(date).date()), 'value': str(pd.to_datetime(date).date())}
            for date in possible_dates
        ]
        return options

    @app.callback(
        [Output('audio-player', 'autoPlay'),
         Output('autoplay-toggle-btn', 'children'),
         Output('autoplay-toggle-btn', 'style')],
        Input('autoplay-toggle-btn', 'n_clicks'),
        State('audio-player', 'autoPlay')
    )
    def toggle_autoplay(n_clicks, current_autoplay):
        if n_clicks is None:
            n_clicks = 0
        new_autoplay = not current_autoplay if current_autoplay is not None else True
        label = "Autoplay: ON" if new_autoplay else "Autoplay: OFF"
        style = {'background-color': '#BADCBD'} if new_autoplay else {'background-color': 'lightgray'}
        return new_autoplay, label, style

    import pathlib
    import pandas as pd

    LABEL_SAVE_PATH = pathlib.Path("data/cache/saved_labels.parquet")

    @app.callback(
        Output('save-status', 'children'),
        Input('save-button', 'n_clicks'),
        State('labels-store', 'data'),
        prevent_initial_call=True
    )
    def save_labels(n_clicks, labels_data):
        if not labels_data:
            return "No labels to save."

        try:
            df = pd.DataFrame(list(labels_data.items()), columns=['row_idx', 'label'])
            df['row_idx'] = df['row_idx'].astype(int)

            # Save to fixed file path
            df.to_parquet(LABEL_SAVE_PATH, index=False)

            return f"Labels saved successfully to {LABEL_SAVE_PATH}"
        except Exception as e:
            return f"Error saving labels: {e}"