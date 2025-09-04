import dash_daq as daq
from dash import html, dcc
import pandas as pd


def create_layout(df):
    min_hour = df['start_hour_float'].min()
    max_hour = df['start_hour_float'].max()

    return html.Div([
        html.Div([
            html.Div([
                html.Button(">", id="toggle-filters-btn", n_clicks=0, className="panel-toggle-button"),

                dcc.Graph(id='scatter'),
                html.Div(id='fft-warning', style={'color': 'red', 'margin': '0.1em'}),

                html.Div([
                    html.Div(id="info"),
                    html.Button(
                        "Autoplay: OFF",
                        id='autoplay-toggle-btn',
                        n_clicks=0,
                        className='app-button autoplay-off'
                    ),
                ], className='info-autoplay-row'),

                html.Div([
                    html.Audio(id='audio-player', controls=True, autoPlay=False),
                    html.Div(
                        dcc.Graph(id='spectrogram-plot', config={'displayModeBar': False}),
                        id='spectrogram-plot-container'
                    ),
                ], id='audio-spectrogram-container')
            ], id='scatter-audio-container'),

            # --- RIGHT PANEL (Filters) ---
            html.Div([
                # Scrolling container for all the filter controls
                html.Div([
                    html.H4("Filters", id='filter-title'),
                    html.Label("Location", style={'marginTop': '1em'}),
                    dcc.Dropdown(id='location-dropdown', options=[{'label': str(loc), 'value': loc} for loc in
                                                                  sorted(df['location'].dropna().unique())],
                                 value=sorted(df['location'].dropna().unique())[0] if df[
                                                                                          'location'].nunique() > 0 else None),
                    html.Label("Microlocation", style={'marginTop': '1em'}),
                    dcc.Dropdown(id='microlocation-dropdown', options=[], multi=True, value=[]),
                    html.Label("Model", style={'marginTop': '1em'}),
                    dcc.Dropdown(id='model-dropdown',
                                 options=[{'label': str(m), 'value': m} for m in sorted(df['model_name'].unique())],
                                 value=sorted(df['model_name'].unique())[0]),
                    html.Label("Channels", style={'marginTop': '1em'}),
                    dcc.Checklist(id="all-or-none-channel", options=[{"label": "Select All", "value": "All"}],
                                  value=["All"], labelStyle={"display": "inline-block"}),
                    dcc.Checklist(id='channel-checklist',
                                  options=[{'label': str(c), 'value': c} for c in sorted(df['channel'].unique())],
                                  value=df['channel'].unique().tolist(), labelStyle={"display": "inline-block"}),
                    html.Label("Number of clusters", style={'marginTop': '1em'}),
                    dcc.Dropdown(id='num-cluster-dropdown',
                                 options=[{'label': str(c), 'value': c} for c in sorted(df['cluster_num'].unique())],
                                 value=sorted(df['cluster_num'].unique())[0]),
                    html.Label("Clusters", style={'marginTop': '1em'}),
                    dcc.Checklist(id="all-or-none-cluster", options=[{"label": "Select All", "value": "All"}],
                                  value=["All"], labelStyle={"display": "inline-block"}),
                    dcc.Checklist(id='cluster-checklist', options=[{'label': str(int(c)), 'value': int(c)} for c in
                                                                   sorted(df['cluster_id'].unique())],
                                  value=df['cluster_id'].unique().tolist(), labelStyle={"display": "inline-block"}),
                    html.Label("Dates", style={'marginTop': '1em'}),
                    dcc.Dropdown(id='date-dropdown', options=[{'label': d, 'value': d} for d in
                                                              sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique())],
                                 multi=True, value=sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique())),
                    html.Label("Hour range", style={'marginTop': '1em'}),
                    dcc.RangeSlider(id='hour-slider', min=int(min_hour), max=int(max_hour) + 1,
                                    value=[min_hour, max_hour], step=0.25,
                                    marks={h: f"{int(h):02d}:00" for h in range(int(min_hour), int(max_hour) + 2, 2)}),
                    html.H4("Sampling", id='sampling-title'),
                    dcc.Input(id='max-points', type='number', min=1, value=10000,
                              style={'width': '80px', 'marginRight': '1em'}),
                    html.Button('Resample', id='resample-btn', n_clicks=0, className='app-button'),
                    html.H4("Merging", id='merging-title'),
                    daq.BooleanSwitch(id='merge-switch', on=False, label='Merge consecutive clips',
                                      labelPosition='left'),
                    html.Div(id='merge-threshold-container', children=[html.Label("Merge threshold"),
                                                                       dcc.Slider(id='merge-threshold', min=0, max=100,
                                                                                  step=1, value=10,
                                                                                  marks={i: str(i) for i in
                                                                                         range(0, 101, 20)})]),
                    html.Div(id='clip-count-threshold-container', children=[html.Label("Clip count threshold"),
                                                                            dcc.Slider(id='clip-count-threshold', min=1,
                                                                                       max=1, step=1, value=1,
                                                                                       marks={i: str(i) for i in
                                                                                              range(1, 11)})]),
                    html.Details([
                        html.Summary("Advanced Options", className='section-title'),
                        html.Div(id='advanced-options-content', children=[
                            html.Div([html.Label("Frequency Scale"), dcc.Dropdown(id='frequency-scale', options=[
                                {'label': 'Log', 'value': 'log'}, {'label': 'Mel', 'value': 'mel'},
                                {'label': 'Linear', 'value': 'linear'}], value='mel')], className='spectrogram-row'),
                            html.Div([html.Label("Window Size"), dcc.Dropdown(id='fft-window-size',
                                                                              options=[{'label': str(s), 'value': s} for
                                                                                       s in
                                                                                       [256, 512, 1024, 2048, 4096]],
                                                                              value=4096)],
                                     className='spectrogram-row'),
                            html.Div([html.Label("Window Overlap"), html.Div(
                                dcc.Slider(id='window-overlap', min=0, max=0.9, step=0.05, value=0.9,
                                           marks={i / 10: f'{int(i * 10)}%' for i in range(0, 10)}),
                                className='slider-container')], className='spectrogram-row'),
                            html.Div([html.Label("Frequency Bins"), dcc.Input(id='num-bins', type='number', value=512)],
                                     className='spectrogram-row'),
                            html.Div([html.Label("Window Function"), dcc.Dropdown(id='window-type', options=[
                                {'label': 'Hann', 'value': 'hann'}, {'label': 'Hamming', 'value': 'hamming'}],
                                                                                  value='hann')],
                                     className='spectrogram-row'),
                            html.Div([html.Label("Min Freq (Hz)"), dcc.Input(id='min-freq', type='number', value=50)],
                                     className='spectrogram-row'),
                            html.Div([html.Label("Max Freq (Hz)"), dcc.Input(id='max-freq', type='number', value=5000)],
                                     className='spectrogram-row'),
                            html.Div([html.Label("Colormap"), dcc.Dropdown(id='colormap',
                                                                           options=[{'label': c, 'value': c} for c in
                                                                                    ['Viridis', 'Plasma', 'Inferno',
                                                                                     'Magma']], value='Viridis')],
                                     className='spectrogram-row'),
                            html.Div([html.Label("DB Floor"), dcc.Input(id='db-floor', type='number', value=-100)],
                                     className='spectrogram-row'),
                        ])
                    ]),
                ], id='filter-panel-container'),

                html.Div([
                    html.Div(className='histogram-header', children=[
                        html.H4("Histogram", id='histogram-title'),
                        # Histogram type dropdown moved here
                        dcc.Dropdown(
                            id='histogram-type-dropdown',
                            options=[
                                {'label': 'Clip Count', 'value': 'count'},
                                {'label': 'Presence', 'value': 'presence'}
                            ],
                            value='count',
                            style={'width': '220px'}
                        )
                    ]),
                    dcc.Graph(id='cluster-histogram', config={'displayModeBar': False}, style={'height': '89%'}),
                ], id='histogram-container'),

            ], id='filter-container', className='filters-expanded'),
        ], id='main-content'),

        dcc.Store(id='filtered-data'),
        dcc.Store(id='spectrogram-cache'),
        dcc.Store(id='clip-count-max-store'),
        dcc.Store(id='histogram-cache'),
    ], id='main-container')