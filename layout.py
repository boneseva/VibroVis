from dash import html, dcc
import dash_daq as daq


def create_layout(df):
    min_hour = df['start_hour_float'].min()
    max_hour = df['start_hour_float'].max()
    return html.Div([
        html.Div([
            # In your layout:
            html.Div(id='fft-warning', style={'color': 'red', 'marginTop': '1em'}),

            html.Div([
                dcc.Graph(id='scatter'),
                html.Div(id="info"),
                html.Div([
                    html.Audio(id='audio-player', controls=True, autoPlay=True),
                    html.Div(
                        dcc.Graph(
                            id='spectrogram-plot',
                            config={'displayModeBar': False}
                        ),
                        id='spectrogram-plot-container'
                    ),
                    dcc.Store(id='spectrogram-cache'),
                ], id='audio-spectrogram-container')
            ], id='scatter-audio-container'),

            html.Div([
                html.Div([
                    html.H4("Filters", id='filter-title'),
                    html.Div([
                        html.Label("Location", style={'marginTop': '1em'}),
                        dcc.Dropdown(
                            id='location-dropdown',
                            options=[{'label': str(loc), 'value': loc} for loc in
                                     sorted(df['location'].dropna().unique())],
                            multi=False,
                            placeholder="Select a location",
                            value=sorted(df['location'].dropna().unique())[0]
                        ),
                        html.Label("Microlocation", style={'marginTop': '1em'}),
                        dcc.Dropdown(
                            id='microlocation-dropdown',
                            options=[],
                            multi=True,
                            placeholder="Select microlocation(s)",
                            value=[],
                            style={'marginBottom': '1em'},
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[{'label': str(m), 'value': m} for m in sorted(df['model_name'].unique())],
                            multi=False,
                            placeholder="Select a model",
                            value=sorted(df['model_name'].unique())[0]
                        ),
                        html.Label("Channels", style={'marginTop': '1em'}),
                        dcc.Checklist(
                            id="all-or-none-channel",
                            options=[{"label": "Select All", "value": "All"}],
                            value=["All"],
                            labelStyle={"display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id='channel-checklist',
                            options=[{'label': str(c), 'value': c} for c in sorted(df['channel'].unique())],
                            value=df['channel'].unique().tolist(),
                            labelStyle={"display": "inline-block"},
                        ),
                        html.Label("Number of clusters", style={'marginTop': '1em'}),
                        dcc.Dropdown(
                            id='num-cluster-dropdown',
                            options=[{'label': str(c), 'value': c} for c in sorted(df['cluster_num'].unique())],
                            multi=False,
                            placeholder="Select number of clusters",
                            value=sorted(df['cluster_num'].unique())[0]
                        ),
                        html.Label("Clusters", style={'marginTop': '1em'}),
                        dcc.Checklist(
                            id="all-or-none-cluster",
                            options=[{"label": "Select All", "value": "All"}],
                            value=["All"],
                            labelStyle={"display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id='cluster-checklist',
                            options=[{'label': str(int(c)), 'value': int(c)} for c in sorted(df['cluster_id'].unique())],
                            value=df['cluster_id'].unique().tolist(),
                            labelStyle={"display": "inline-block"}
                        ),
                        html.Label("Dates", style={'marginTop': '1em', 'marginRight': '0.5em'}),
                        dcc.Dropdown(
                            id='date-dropdown',
                            options=[
                                {'label': d, 'value': d}
                                for d in sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique())
                            ],
                            multi=True,
                            value=sorted(df['day_dt'].dt.strftime('%Y-%m-%d').unique()),
                            placeholder="Select date(s)...",
                            style={'marginTop': '0.5em'}
                        ),
                        html.Label("Hour range", style={'marginTop': '1em', 'marginRight': '0.5em'}),
                        html.Div(
                            dcc.RangeSlider(
                                id='hour-slider',
                                min=int(min_hour),
                                max=int(max_hour)+1,
                                value=[min_hour, max_hour],
                                step=0.25,
                                marks={h: f"{int(h):02d}:00" for h in range(int(min_hour), int(max_hour) + 1, 1)},
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False,
                                    "transform": "hourMinute"
                                },
                            ),
                            style={'marginTop': '0.5em'}
                        ),
                    ]),
                    html.H4("Sampling", id='sampling-title'),
                    html.Div([
                        html.Label("Max points", style={'marginRight': '0.5em'}),
                        dcc.Input(
                            id='max-points',
                            type='number',
                            min=1,
                            placeholder='Max points',
                            value=10000,
                            style={'width': '40%', 'marginRight': '1em'}
                        ),
                        html.Button('Resample', id='resample-btn', n_clicks=0),
                    ]),
                    html.H4("Merging", id='merging-title'),
                    html.Div([
                        dcc.Loading(
                            id="merge-loading",
                            type="circle",
                            fullscreen=False,
                            children=[
                                dcc.Store(id='merged-cache'),
                                daq.BooleanSwitch(
                                    id='merge-switch',
                                    on=False,
                                    label='Merge consecutive clips',
                                    labelPosition='left',
                                    style={},
                                    persistence_type='memory'
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Merge threshold", style={}),
                                        dcc.Slider(
                                            id='merge-threshold',
                                            min=0, max=100, step=1, value=10,
                                            marks={i: str(i) for i in range(0, 101, 20)},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        )
                                    ],
                                    id='merge-threshold-container',
                                    style={}
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Clip count threshold", style={}),
                                        dcc.Slider(
                                            id='clip-count-threshold',
                                            min=1, max=1, step=1, value=1,
                                            marks={i: str(i) for i in range(1, 11)},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        dcc.Store(id='clip-count-max-store')
                                    ],
                                    id='clip-count-threshold-container',
                                    style={}
                                ),
                            ],
                            style={'position': 'relative'}
                        )
                    ], id='merge-section-container', style={'position': 'relative'}),

                    html.Div([
                        html.Details([
                            html.Summary("Advanced Options", className='section-title'),
                            html.Div([
                                html.Label("Spectrogram", style={'marginTop': '1em'}),
                                html.Div([
                                    html.Label("FFT Window Size", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='fft-window-size',
                                        type='number',
                                        min=64,
                                        max=4096,
                                        step=1,
                                        value=320,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),
                                html.Div([
                                    html.Label("FFT Step Size", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='fft-step-size',
                                        type='number',
                                        min=16,
                                        max=4096,
                                        step=1,
                                        value=160,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),
                                html.Div([
                                    html.Label("FFT Padding (nfft)", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='nfft',
                                        type='number',
                                        min=64,
                                        max=8192,
                                        step=1,
                                        value=512,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Window Function", className='spectrogram-labels'),
                                    dcc.Dropdown(
                                        id='window-type',
                                        options=[
                                            {'label': 'Hann', 'value': 'hann'},
                                            {'label': 'Hamming', 'value': 'hamming'},
                                            {'label': 'Blackman', 'value': 'blackman'},
                                            {'label': 'Rectangular', 'value': 'boxcar'}
                                        ],
                                        value='hann',
                                        style={'width': '120px', 'display': 'inline-block', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Min Frequency (Hz)", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='min-freq',
                                        type='number',
                                        min=0,
                                        max=20000,
                                        step=1,
                                        value=50,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Max Frequency (Hz)", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='max-freq',
                                        type='number',
                                        min=100,
                                        max=24000,
                                        step=1,
                                        value=5000,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Number of Frequency Bins", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='num-bins',
                                        type='number',
                                        min=64,
                                        max=2048,
                                        step=1,
                                        value=512,
                                        style={'width': '120px', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Spectrogram Colormap", className='spectrogram-labels'),
                                    dcc.Dropdown(
                                        id='colormap',
                                        options=[
                                            {'label': 'Viridis', 'value': 'Viridis'},
                                            {'label': 'Plasma', 'value': 'Plasma'},
                                            {'label': 'Inferno', 'value': 'Inferno'},
                                            {'label': 'Magma', 'value': 'Magma'},
                                            {'label': 'Cividis', 'value': 'Cividis'}
                                        ],
                                        value='Viridis',
                                        style={'width': '120px', 'display': 'inline-block', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),

                                html.Div([
                                    html.Label("Dynamic Range Floor (dB)", className='spectrogram-labels'),
                                    dcc.Input(
                                        id='db-floor',
                                        type='number',
                                        min=-160,
                                        max=0,
                                        step=1,
                                        value=-100,
                                        style={'width': '120px', 'display': 'inline-block', 'marginBottom': 10, 'justifyContent': 'flex-end'}
                                    ),
                                ], style={'display': 'flex'}),
                            ])
                        ], id='advanced-options-container', open=False)
                    ])
                ], id='filter-panel-container'),

                html.Div([
                    # html.Label("Bin width (minutes):"),
                    # dcc.Slider(
                    #     id='histogram-bin-width',
                    #     min=15,
                    #     max=60,
                    #     step=None,
                    #     value=30,
                    #     marks={15: "15 min", 30: "30 min", 45: "45 min", 60: "60 min"},
                    #     tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div([
                        html.H4("Histogram", id='histogram-title', style={'margin': 0}),
                        dcc.Dropdown(
                            id='histogram-type-dropdown',
                            options=[
                                {'label': 'Count', 'value': 'count'},
                                {'label': 'Presence', 'value': 'presence'}
                            ],
                            value='count',
                            clearable=False,
                            style={'width': '180px'}
                        )
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'center',
                        'justifyContent': 'space-between',
                        'marginBottom': '0.3em'
                    }),
                    dcc.Graph(id='cluster-histogram',
                              config={
                                  'displayModeBar': False,
                                  'staticPlot': True
                              }),
                    dcc.Store(id='histogram-cache'),
                ], id='histogram-container'),

            ], id='filter-container')
        ], id='main-content'),
        dcc.Store(id='filtered-data'),
        dcc.Store(id='click-counter', data=0),
    ], id='main-container')
