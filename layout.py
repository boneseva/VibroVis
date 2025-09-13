import dash_daq as daq
from dash import html, dcc
import pandas as pd


def create_layout(df):
    min_hour = df['start_hour_float'].min() if not df['start_hour_float'].empty else 0
    max_hour = df['start_hour_float'].max() if not df['start_hour_float'].empty else 24

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
                html.Div([
                    html.A(
                        "ℹ️",
                        href="https://docs.google.com/document/d/e/2PACX-1vSxvnYGbOE4oblvbkKrpfleLwe92h3irOA3eVr757FLZAfHqwbSBH6hcNKTqfj64_gvWBcZzeWjs8DC/pub", 
                        target="_blank",  # new tab
                        className="guide-title",
                        title="Help & Documentation",
                    )
                ], className="guide-div"),
                
                html.Div([
                    # html.H4("Recordings", id='filter-title'),
                    html.Details([
                    html.Summary("Recordings", className='section-title'),
                    html.Div(id='recordings-filter-content', children=[
                        html.Div([
                            html.Label("Location"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Filter data by the primary recording location.", className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Dropdown(id='location-dropdown', options=[{'label': str(loc), 'value': loc} for loc in
                                                                      sorted(df['location'].dropna().unique())],
                                     value=sorted(df['location'].dropna().unique())[0] if df[
                                                                                              'location'].nunique() > 0 else None),

                        html.Div([
                            html.Label("Microlocation"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Filter by specific sub-locations or microphone positions.",
                                          className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Dropdown(id='microlocation-dropdown', options=[], multi=True, value=[]),


                        html.Div([
                            html.Label("Channels"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Select which audio channels to display.", className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Checklist(id="all-or-none-channel", options=[{"label": "Select All", "value": "All"}],
                                      value=["All"], labelStyle={"display": "inline-block"}),
                        dcc.Checklist(id='channel-checklist',
                                      options=[{'label': str(c), 'value': c} for c in sorted(df['channel'].unique())],
                                      value=df['channel'].unique().tolist(), labelStyle={"display": "inline-block"}),

                        html.Div([
                            html.Label("Dates"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Filter the data by specific recording dates.", className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Dropdown(id='date-dropdown', options=[{'label': d, 'value': d} for d in
                                                                  sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique())],
                                     multi=True, value=sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique())),

                        html.Div([
                            html.Label("Hour range"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Filter by the time of day.", className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.RangeSlider(id='hour-slider', min=int(min_hour), max=int(max_hour) + 1,
                                        value=[min_hour, max_hour], step=0.25,
                                        marks={h: f"{int(h):02d}:00" for h in range(int(min_hour), int(max_hour) + 2, 2)}),
                    ])
                    ], open=True),

                    html.Details([
                    html.Summary("Model", className='section-title'),
                    # html.H4("Model", id='model-filter-title'),
                    html.Div(id='model-filter-content', children=[
                        html.Div([
                            html.Label("Model"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Select the machine learning model used for detection.", className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Dropdown(id='model-dropdown',
                                     options=[{'label': str(m), 'value': m} for m in sorted(df['model_name'].unique())],
                                     value=sorted(df['model_name'].unique())[0]),

                        html.Div([
                            html.Label("Number of clusters"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Choose the clustering model (e.g., k=10 or k=20 clusters).",
                                          className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Dropdown(id='num-cluster-dropdown',
                                     options=[{'label': str(c), 'value': c} for c in sorted(df['cluster_num'].unique())],
                                     value=sorted(df['cluster_num'].unique())[0]),

                        html.Div([
                            html.Label("Clusters"),
                            html.Div(className="tooltip-container", children=[
                                html.Span(" ⓘ", className="info-icon"),
                                html.Span("Select specific clusters to display from the chosen model.",
                                          className="tooltip-text")
                            ])
                        ], className="label-with-info", style={'marginTop': '1em'}),
                        dcc.Checklist(id="all-or-none-cluster", options=[{"label": "Select All", "value": "All"}],
                                      value=["All"], labelStyle={"display": "inline-block"}),
                        dcc.Checklist(id='cluster-checklist', options=[{'label': str(int(c)), 'value': int(c)} for c in
                                                                       sorted(df['cluster_id'].unique())],
                                      value=df['cluster_id'].unique().tolist(), labelStyle={"display": "inline-block"}),
                    ])
                    ], open=True),



                    # html.H4("Sampling and Merging", id='sampling-title'),
                    html.Details([
                    html.Summary("Sampling and Merging", className='section-title'),
                        html.Div(id='sampling-merging-content', children=[
                            html.Div([
                                html.Label("Max points"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Limits the number of points displayed on the scatter plot to improve performance.",
                                        className="tooltip-text")
                                ])
                            ], className="label-with-info"),
                            dcc.Input(id='max-points', type='number', min=1, value=10000,
                                      style={'width': '80px', 'marginRight': '1em'}),
                            html.Button('Resample', id='resample-btn', n_clicks=0, className='app-button'),

                            html.Div([
                                    html.Label("Merge consecutive clips"),
                                    html.Div(className="tooltip-container", children=[
                                        html.Span(" ⓘ", className="info-icon"),
                                        html.Span(
                                            "Merges consecutive clips that are in the same cluster and close enough in the space into one point.",
                                            className="tooltip-text")
                                    ])
                                ], className="label-with-info", style={'marginTop': '1em'}),
                            daq.BooleanSwitch(id='merge-switch', on=False, label='', labelPosition='top'),

                            html.Div(id='merge-threshold-container', children=[
                                html.Div([
                                    html.Label("Merge threshold"),
                                    html.Div(className="tooltip-container", children=[
                                        html.Span(" ⓘ", className="info-icon"),
                                        html.Span(
                                            "Controls how close clips must be to be merged. Higher values = more merging.",
                                            className="tooltip-text")
                                    ])
                                ], className="label-with-info"),
                                dcc.Slider(id='merge-threshold', min=0, max=100,
                                           step=1, value=1,
                                           marks={i: str(i) for i in range(0, 101, 20)})
                            ]),

                            html.Div(id='clip-count-threshold-container', children=[
                                html.Div([
                                    html.Label("Clip count threshold"),
                                    html.Div(className="tooltip-container", children=[
                                        html.Span(" ⓘ", className="info-icon"),
                                        html.Span(
                                            "Only show merged points that contain at least this many individual clips.",
                                            className="tooltip-text")
                                    ])
                                ], className="label-with-info"),
                                dcc.Slider(id='clip-count-threshold', min=1,
                                           max=1, step=1, value=1,
                                           marks={i: str(i) for i in range(1, 11)})
                            ])])
                    ], open=True),

                    html.Details([
                        html.Summary("Spectrogram Settings", className='section-title'),
                        html.Div(id='advanced-options-content', children=[

                            html.Div([
                                html.Label("Frequency Scale"),
                                dcc.Dropdown(id='frequency-scale', options=[
                                    {'label': 'Log', 'value': 'log'}, {'label': 'Mel', 'value': 'mel'},
                                    {'label': 'Linear', 'value': 'linear'}], value='mel'),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Visual scale for the frequency axis of the spectrogram.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Window Size"),
                                dcc.Dropdown(id='fft-window-size',
                                             options=[{'label': str(s), 'value': s} for s in
                                                      [256, 512, 1024, 2048, 4096]],
                                             value=4096),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Size of the FFT window. Larger windows give better frequency resolution but worse time resolution.",
                                        className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Window Overlap"),
                                dcc.Slider(id='window-overlap', min=0, max=0.9, step=0.05, value=0.9,
                                           marks={i / 10: f'{int(i * 10)}%' for i in range(0, 10, 2)}),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Percentage of overlap between FFT windows. Higher overlap results in a smoother spectrogram.",
                                        className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Frequency Bins"),
                                dcc.Input(id='num-bins', type='number', value=512),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Number of frequency bins to display (for Mel/Log scales).",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Window Function"),
                                dcc.Dropdown(id='window-type', options=[
                                    {'label': 'Hann', 'value': 'hann'}, {'label': 'Hamming', 'value': 'hamming'}],
                                             value='hann'),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Windowing function to apply before FFT to reduce spectral leakage.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Min Freq (Hz)"),
                                dcc.Input(id='min-freq', type='number', value=50),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Minimum frequency to display on the spectrogram.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Max Freq (Hz)"),
                                dcc.Input(id='max-freq', type='number', value=5000),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Maximum frequency to display on the spectrogram.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Colormap"),
                                dcc.Dropdown(id='colormap',
                                             options=[{'label': c, 'value': c} for c in
                                                      ['Viridis', 'Plasma', 'Inferno', 'Magma']],
                                             value='Viridis'),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Color scheme for the spectrogram's intensity.", className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("dB Floor"),
                                dcc.Input(id='db-floor', type='number', value=-100),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Minimum decibel level to display; values below this are clipped.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),
                        ])
                    ]),
                ], id='filter-panel-container'),

                html.Div([
                    html.Div(className='histogram-header', children=[
                        html.H4("Histogram", id='histogram-title'),
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
        dcc.Store(id='merge-max-store'),
        dcc.Store(id='histogram-cache'),
        dcc.Store(id='model-data-ready-signal'),
    ], id='main-container')