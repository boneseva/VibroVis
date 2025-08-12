import time
from random import random

import dash
import numpy as np
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import ast
import os
import flask
from io import StringIO
import dash_daq as daq
from scipy.spatial.distance import euclidean
import soundfile as sf
import io
import plotly.io as pio

DATA_DIR = os.path.abspath(r"data")

def precompute_cluster_histograms(dff):
    dff = dff.copy()
    hist_cache = {}
    for cluster in dff['cluster_id'].unique():
        cluster_int = int(cluster)
        cluster_df = dff[dff['cluster_id'] == cluster]
        fig = px.histogram(
            cluster_df,
            x='time_of_day',
            title=f'Cluster {cluster_int} counts by time of day',
            nbins=12
        )
        fig.update_layout(bargap=0.1, xaxis=dict(dtick=1), showlegend=False)
        hist_cache[str(cluster_int)] = pio.to_json(fig)
    return hist_cache

def load_positions_tsv(tsv_dir):
    df = pd.DataFrame()
    tsv_root = os.path.normpath(tsv_dir)
    for root, dirs, files in os.walk(tsv_dir):
        for dir in dirs:
            if dir == 'positions':
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith('.tsv'):
                        tsv_path = os.path.join(root, dir, file)
                        dff = pd.read_csv(tsv_path, sep='\t')
                        dff['day_str'] = dff['day'].astype(str).str.zfill(6)
                        dff['day_dt'] = pd.to_datetime(dff['day_str'], format='%y%m%d')
                        dff[['x', 'y']] = dff['embedding'].apply(lambda s: pd.Series(ast.literal_eval(s)))
                        dff['abs_file_name'] = dff['file_name'].apply(lambda x: os.path.relpath(os.path.normpath(os.path.join(root, x)), tsv_root))
                        dff['start_dt'] = pd.to_datetime(dff['start_time'], format='%H%M%S')
                        dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
                        c = file.split('.')[-3].split('_')[-1]
                        dff['cluster_num'] = c if c.isnumeric() else 10
                        df = pd.concat([df, dff], ignore_index=True)
    return df

df = load_positions_tsv(DATA_DIR)

app = dash.Dash(__name__)
app.title = "VibroScape Visualization Tool"

app.layout = html.Div([
dcc.Store(id='click-counter', data=0),
    html.Div([
        # LEFT: Scatter plot, info, audio
        html.Div([
            dcc.Graph(id='scatter', style={"width": "100%", "height": "80vh"}),
            html.Div(id="info", style={
                'marginTop': '2em', "padding": "1em", "font-size": "large",
                'background': '#f9f9f9', 'borderRadius': '10px', 'boxShadow': '0 2px 8px #eee'
            }),
            html.Div([
                html.Div(
                    dcc.Graph(
                        id='spectrogram-plot',
                        style={'width': '100%', 'height': '200px', 'marginBottom': '1em'},
                        config={'displayModeBar': False}
                    ),
                    id='spectrogram-plot-container'
                ),
                dcc.Store(id='spectrogram-cache'),

                html.Audio(id='audio-player', controls=True, style={'width': '100%'})
            ], id='audio-player-container')
        ], style={'flex': '2', 'minWidth': '0', 'padding': '1em', 'margin': '0px'}),

        html.Div([
            html.Div([
                html.H4("Filters", style={'marginBottom': '1em'}),
                html.Div([
                    html.Label("Model"),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': str(m), 'value': m} for m in sorted(df['model_name'].unique())],
                        multi=False,
                        placeholder="Select a model"
                    ),
                    html.Label("Channels", style={'marginTop': '1em'}),
                    dcc.Dropdown(
                        id='channel-dropdown',
                        options=[{'label': str(c), 'value': c} for c in sorted(df['channel'].unique())],
                        multi=True,
                        placeholder="Select channel(s)"
                    ),
                    html.Label("Number of clusters", style={'marginTop': '1em'}),
                    dcc.Dropdown(
                        id='cluster-dropdown',
                        options=[{'label': str(c), 'value': c} for c in sorted(df['cluster_num'].unique())],
                        multi=True,
                        placeholder="Select number of clusters"
                    ),
                    html.Label("Date range", style={'marginTop': '1em'}),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=df['day_dt'].min(),
                        max_date_allowed=df['day_dt'].max(),
                        start_date=df['day_dt'].min(),
                        end_date=df['day_dt'].max(),
                        display_format='DD-MM',
                        style={'marginTop': '0.5em'}
                    ),
                    html.Label("Hour range", style={'marginTop': '1em'}),
                    html.Div(
                        dcc.RangeSlider(
                            id='hour-slider',
                            min=0, max=23,
                            value=[0, 23],
                            marks={h: f"{h}:00" for h in range(0, 24, 6)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        style={'marginTop': '0.5em'}
                    ),
                ], style={'marginBottom': '2em'}),
                html.H4("Sampling", style={'marginBottom': '1em'}),
                html.Div([
                    html.Label("Max points"),
                    dcc.Input(
                        id='max-points',
                        type='number',
                        min=1,
                        placeholder='Max points',
                        style={'width': '60%', 'marginRight': '1em'}
                    ),
                    html.Button('Resample', id='resample-btn', n_clicks=0, style={'marginTop': '1em'}),
                ], style={'marginBottom': '2em'}),
                html.H4("Merging", style={'marginBottom': '1em'}),
                html.Div([
                    daq.BooleanSwitch(
                        id='merge-switch',
                        on=False,
                        label='Merge consecutive clips',
                        labelPosition='left',
                        style={'marginBottom': '1em'}
                    ),
                    html.Label("Merge threshold", style={'marginTop': '1em'}),
                    html.Div(
                        dcc.Slider(
                            id='merge-threshold',
                            min=0, max=100, step=1, value=10,
                            marks={i: str(i) for i in range(0, 101, 20)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        id='merge-threshold-container',
                        style={'marginTop': '0.5em'}
                    ),
                    html.Label("Clip count threshold", style={'marginTop': '1em'}),
                    html.Div(
                        dcc.Slider(
                            id='clip-count-threshold',
                            min=1, max=20, step=1, value=1,
                            marks={i: str(i) for i in range(1, 21, 5)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        id='clip-count-threshold-container',
                        style={'marginTop': '0.5em'}
                    ),
                ], style={'marginBottom': '2em'}),
                html.Div([
                    html.H4("Histogram", style={'marginBottom': '1em'}),
                    html.Label("Bin width (minutes):"),
                    dcc.Slider(
                        id='histogram-bin-width',
                        min=15,
                        max=60,
                        step=None,
                        value=30,
                        marks={15: "15 min", 30: "30 min", 45: "45 min", 60: "60 min"},
                        tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'marginBottom': '2em'}),
            ], style={
                'padding': '1em', 'background': '#f9f9f9',
                'borderRadius': '10px', 'boxShadow': '0 2px 8px #eee',
                'marginBottom': '1em'
            }),

            html.Div([
                dcc.Graph(id='cluster-histogram', style={"width": "100%", "height": "32vh"}),
            ], style={
                'padding': '1em', 'background': '#fff',
                'borderRadius': '10px', 'boxShadow': '0 2px 8px #eee',
                'marginBottom': '1em'
            }),

            dcc.Store(id='histogram-cache'),
        ], style={
            'flex': '1.5',
            'maxWidth': '600px',
            'padding': '2vw 2vw 2vw 0',
            'background': '#f4f6fa',
            'borderLeft': '1px solid #eaeaea',
            'height': '100vh',
            'overflowY': 'auto',
            'boxSizing': 'border-box'
        }),
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100vw',
        'height': '100vh',
        'alignItems': 'flex-start'
    }),
    dcc.Store(id='filtered-data'),
])

@app.server.route("/audio_segment_normalized/<path:filename>/<int:channel>/<float:start>/<float:end>")
def serve_audio_segment_normalized(filename, channel, start, end):
    abs_path = os.path.join(DATA_DIR, filename)
    try:
        data, samplerate = sf.read(abs_path)
    except Exception as e:
        print(f"Error loading audio file: {abs_path}\n{e}")
        return flask.abort(404)
    channel_data = data[:, channel-1]
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
        sf.write(buf, segment, samplerate, format='WAV')
    except Exception as e:
        print(f"Error writing audio segment: {e}")
        return flask.abort(500)
    buf.seek(0)
    return flask.send_file(buf, mimetype="audio/wav")

@app.callback(
    Output('spectrogram-cache', 'data'),
    [Input("scatter", "clickData"),
     Input('filtered-data', 'data')],
    prevent_initial_call=True
)
def compute_spectrogram_data(clickData, filtered_json):
    import numpy as np
    import soundfile as sf
    from scipy.signal import spectrogram
    import os
    from io import StringIO

    if not clickData or not filtered_json:
        return dash.no_update

    dff = pd.read_json(StringIO(filtered_json), orient='split')
    point = clickData["points"][0]
    idx = point["pointIndex"]
    if idx >= len(dff):
        return dash.no_update

    row = dff.iloc[idx]
    start = float(row['clip_time'])
    end = start + float(row['clip_duration'])
    channel = int(row['channel'])
    abs_path = os.path.join(DATA_DIR, row['abs_file_name'])
    data, samplerate = sf.read(abs_path)
    if data.ndim == 1:
        channel_data = data
    else:
        channel_data = data[:, channel-1]
    pad = 1.0
    segment_start = max(0, start - pad)
    segment_end = min(len(channel_data) / samplerate, end + pad)
    start_sample = int(segment_start * samplerate)
    end_sample = int(segment_end * samplerate)
    segment = channel_data[start_sample:end_sample]
    peak = np.max(np.abs(segment))
    if peak > 0:
        segment = segment * (0.99 / peak)

    num_bins = 512
    f, t, Sxx = spectrogram(segment, fs=samplerate)
    min_freq = 50
    max_freq = 5000
    log_f = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_bins)
    Sxx_log = np.zeros((len(log_f), Sxx.shape[1]))
    for i in range(Sxx.shape[1]):
        Sxx_log[:, i] = np.interp(log_f, f, Sxx[:, i])
    z = 10 * np.log10(np.maximum(Sxx_log, 1e-10))
    z = np.nan_to_num(z, nan=-100, posinf=100, neginf=-100)

    audio_path = f"/audio_segment_normalized/{row['abs_file_name']}/{channel}/{start}/{end}"
    info = f"{row['file_name']} at {row['clip_time']}s (cluster {row['cluster_id']})"

    print("Called compute_spectrogram_data")
    assert not np.isnan(z).all(), "All spectrogram values are NaN"
    assert np.ptp(z) > 1, "Insufficient dynamic range in spectrogram"


    return {
       # 'x': (t + segment_start).tolist(),
        'x': t.tolist(),
        'y': log_f.tolist(),
        'z': z.tolist(),
        'audio_path': audio_path,
        'info': info,
        '_rev': time.time_ns()
    }


@app.callback(
    [Output("info", "children", allow_duplicate=True),
     Output("audio-player", "src", allow_duplicate=True),
     Output("spectrogram-plot", "figure"),
     Output('spectrogram-plot-container', 'key')],
    Input('spectrogram-cache', 'data'),
    prevent_initial_call=True
)
def update_spectrogram_from_cache(data):
    import plotly.graph_objects as go
    import numpy as np
    import time

    if not data:
        return dash.no_update

    # 1. Convert ALL data to native Python types
    y = np.array(data['y']).tolist()
    z = np.array(data['z']).tolist()
    time_step = 0.032
    x_relative = (np.arange(len(z[0])) * time_step).tolist() if z else []

    # 2. Create COMPLETELY new figure with atomic updates
    fig = go.Figure(data=go.Heatmap(
        x=x_relative,
        y=y,
        z=z,
        colorscale='Viridis',
        zmin=np.nanmin(z) if z else -100,
        zmax=np.nanmax(z) if z else 0,
        colorbar=dict(title='dB'),
        uid=str(time.time_ns())  # Nanosecond timestamp
    ))

    # 3. Force clean layout state with double reset
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
        uirevision=f"rev_{data['_rev']}_{time.time_ns()}"  # Combined unique ID
    )

    # 4. Nuclear DOM refresh
    fig['layout']['meta'] = f"ts_{time.time_ns()}"  # Force new DOM element

    print(f"X: {x_relative[:3]}... | UIRev: {fig.layout.uirevision} | Meta: {fig.layout.meta}")

    return data['info'], data['audio_path'], fig, str(data['_rev'])


@app.callback(
    Output('histogram-cache', 'data'),
    Input('model-dropdown', 'value'),
    Input('channel-dropdown', 'value'),
    Input('cluster-dropdown', 'value'),
    Input('histogram-bin-width', 'value')
)
def update_histogram_cache(selected_models, selected_channels, selected_clusters, bin_width):
    dff = df.copy()
    if selected_models:
        dff = dff[dff['model_name'].isin([selected_models] if isinstance(selected_models, str) else selected_models)]
    if selected_channels:
        dff = dff[dff['channel'].isin(selected_channels)]
    if selected_clusters:
        dff = dff[dff['cluster_num'].isin(selected_clusters)]
    dff['time_of_day'] = dff['start_dt'] + pd.to_timedelta(dff['clip_time'], unit='s')
    dff['time_of_day_minutes'] = (
        dff['time_of_day'].dt.hour * 60 +
        dff['time_of_day'].dt.minute +
        dff['time_of_day'].dt.second / 60
    )
    hist_cache = {}
    for cluster in dff['cluster_id'].unique():
        cluster_int = int(cluster)
        cluster_df = dff[dff['cluster_id'] == cluster]
        fig = px.histogram(
            cluster_df,
            x='time_of_day_minutes',
            title=f'Cluster {cluster_int} counts by time of day'
        )
        fig.update_traces(xbins=dict(start=0, end=1440, size=bin_width))
        fig.update_layout(
            bargap=0.1,
            xaxis=dict(
                title="Time of Day",
                dtick=bin_width,
                range=[360, 1200],
                tickvals=[i for i in range(360, 1200, bin_width)],
                ticktext=[f"{i//60:02d}:{i%60:02d}" for i in range(360, 1200, bin_width)]
            ),
            showlegend=False
        )
        hist_cache[str(cluster_int)] = pio.to_json(fig)
    return hist_cache

@app.callback(
    [Output('scatter', 'figure'),
     Output('filtered-data', 'data')],
    [Input('model-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
    Input('cluster-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('hour-slider', 'value'),
     Input('max-points', 'value'),
     Input('resample-btn', 'n_clicks'),
     Input('merge-switch', 'on'),
     Input('merge-threshold', 'value'),
     State('scatter', 'relayoutData')],
    [State('clip-count-threshold', 'value'),
     # State('click-counter', 'data')
    ]
)
def update_figure(selected_models, selected_channels, selected_clusters, start_date, end_date, hour_range, max_points, n_clicks, merge_on, merge_threshold, relayoutData, clip_count_threshold):
    dff = df.copy()
    if selected_models:
        dff = dff[dff['model_name'].isin([selected_models] if isinstance(selected_models, str) else selected_models)]
    if selected_clusters:
        dff = dff[dff['cluster_num'].isin([selected_clusters] if isinstance(selected_clusters, str) else selected_clusters)]
    if selected_channels:
        dff = dff[dff['channel'].isin(selected_channels)]
    if start_date and end_date:
        dff = dff[(dff['day_dt'] >= pd.to_datetime(start_date)) & (dff['day_dt'] <= pd.to_datetime(end_date))]
    if hour_range:
        hours = dff['start_time'].astype(str).str[:2].astype(int)
        dff = dff[(hours >= hour_range[0]) & (hours <= hour_range[1])]
    if merge_on and not dff.empty:
        dff = dff.sort_values(['file_name', 'channel', 'clip_time']).reset_index(drop=True)
        merged = []
        current = dff.iloc[0].copy()
        current['x_sum'] = current['x']
        current['y_sum'] = current['y']
        current['clip_count'] = 1
        for i in range(1, len(dff)):
            prev = dff.iloc[i - 1]
            this = dff.iloc[i]
            can_merge = (
                this['file_name'] == prev['file_name'] and
                this['channel'] == prev['channel'] and
                this['cluster_id'] == prev['cluster_id'] and
                euclidean([this['x'], this['y']], [prev['x'], prev['y']]) < merge_threshold and
                this['clip_time'] == prev['clip_time'] + prev['clip_duration']
            )
            if can_merge:
                current['clip_duration'] += this['clip_duration']
                current['clip_count'] += 1
                current['x_sum'] += this['x']
                current['y_sum'] += this['y']
            else:
                current['x'] = current['x_sum'] / current['clip_count']
                current['y'] = current['y_sum'] / current['clip_count']
                merged.append(current)
                current = this.copy()
                current['x_sum'] = current['x']
                current['y_sum'] = current['y']
                current['clip_count'] = 1
        current['x'] = current['x_sum'] / current['clip_count']
        current['y'] = current['y_sum'] / current['clip_count']
        merged.append(current)
        dff = pd.DataFrame(merged)
        if clip_count_threshold is not None:
            try:
                clip_count_threshold = int(clip_count_threshold)
            except Exception:
                clip_count_threshold = 1
            dff = dff[dff["clip_count"] >= clip_count_threshold]
    if "clip_count" not in dff.columns:
        dff["clip_count"] = 1
    dff["clip_count"] = dff["clip_count"].astype(float)
    if max_points and len(dff) > max_points:
        dff = dff.sample(
            n=max_points,
            weights="clip_count",
            random_state=42 + (n_clicks or 0)
        ).reset_index(drop=True)
    base_size = 20
    scale = 5
    dff['marker_size'] = dff['clip_count'].apply(lambda c: base_size if c == 1 else base_size + scale * (c - 1))
    fig = px.scatter(
        dff, x="x", y="y", color="cluster_id",
        hover_data=["clip_count", "file_name", "clip_time", "model_name", "channel"],
        custom_data=["cluster_id"]
    )
    fig.update_traces(marker=dict(size=dff['marker_size']))
    fig.update_layout(uirevision=str(hash(dff.to_json())))
    fig.update_coloraxes(showscale=False)
    return fig, dff.to_json(date_format='iso', orient='split')

@app.callback(
    Output('merge-threshold-container', 'style'),
    Output('clip-count-threshold-container', 'style'),
    Input('merge-switch', 'on')
)
def toggle_merge_sliders(merge_on):
    if merge_on:
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}

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
        empty_fig.update_layout(title="Click a point to see cluster histogram")
        return empty_fig

    cluster_id = int(clickData["points"][0]["customdata"][0])
    cluster_id_str = str(cluster_id)
    if cluster_id_str in hist_cache:
        return pio.from_json(hist_cache[cluster_id_str])
    else:
        empty_df = pd.DataFrame({"time_of_day": [], "count": []})
        empty_fig = px.histogram(empty_df, x="time_of_day")
        empty_fig.update_layout(title="No histogram for selected cluster")
        return empty_fig

@app.callback(
    [Output('clip-count-threshold', 'max'),
     Output('clip-count-threshold', 'marks'),
     Output('clip-count-threshold', 'value')],
    [Input('merge-switch', 'on'),
     Input('filtered-data', 'data')],
    [State('clip-count-threshold', 'value')]
)
def update_clip_count_slider(merge_on, filtered_json, current_value):
    import pandas as pd
    from io import StringIO
    if not merge_on or not filtered_json:
        return 10, {i: str(i) for i in range(1, 11)}, 1
    dff = pd.read_json(StringIO(filtered_json), orient='split')
    if 'clip_count' in dff.columns and not dff.empty:
        max_clip_count = int(dff['clip_count'].max())
        step = max(1, max_clip_count // 4)
        marks = {i: str(i) for i in range(1, max_clip_count, step)}
        marks[max_clip_count] = str(max_clip_count)
        value = current_value if current_value and current_value <= max_clip_count else 1
        return max_clip_count, marks, value
    else:
        return 10, {i: str(i) for i in range(1, 11)}, 1


if __name__ == "__main__":
    app.run(debug=True)
