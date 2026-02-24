# Updated layout.py
import os

import dash_daq as daq
from dash import html, dcc
import pandas as pd

CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def generate_date_selector(df, selected_dates=None):
    """
    Generates Date Selector.
    If selected_dates is provided (list of strings 'YYYY-MM-DD'),
    only checks those dates. Otherwise checks ALL.
    """
    if df is None or df.empty or 'day_dt' not in df.columns:
        return html.Div("No date data found.")

    dates = pd.to_datetime(df['day_dt']).dt.date.unique()
    dates = sorted(dates)

    selected_set = set(selected_dates) if selected_dates is not None else None

    if len(dates) < 10:
        all_values = [d.strftime('%Y-%m-%d') for d in dates]

        if selected_set is not None:
            current_value = [d for d in all_values if d in selected_set]
        else:
            current_value = all_values

        options = [{'label': d.strftime('%Y-%m-%d'), 'value': d.strftime('%Y-%m-%d')} for d in dates]

        return html.Div([
            dcc.Dropdown(
                id={'type': 'simple-date-dropdown', 'index': 0},
                options=options,
                value=current_value,
                multi=True,
                clearable=True,
                placeholder="Select dates..."
            )
        ], style={'padding': '5px 0'})

    year_dict = {}
    for date in dates:
        year = date.year
        month = date.month
        if year not in year_dict: year_dict[year] = {}
        if month not in year_dict[year]: year_dict[year][month] = []
        year_dict[year][month].append(date)

    year_elements = []
    for year in sorted(year_dict.keys()):
        month_elements = []
        for month in sorted(year_dict[year].keys()):
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B')

            day_options = []
            day_values = []

            current_month_dates = year_dict[year][month]
            for d in current_month_dates:
                d_str = d.strftime('%Y-%m-%d')
                day_options.append({'label': f" {d.day:02d}", 'value': d_str})

                if selected_set is None or d_str in selected_set:
                    day_values.append(d_str)

            month_select_all_val = []
            if len(day_values) == len(current_month_dates) and len(day_values) > 0:
                month_select_all_val = ['all']

            month_elements.append(
                html.Details([
                    html.Summary(
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'month-select-all', 'year': year, 'month': month},
                                options=[{'label': '', 'value': 'all'}],
                                value=month_select_all_val,
                                labelStyle={'display': 'inline-block', 'marginRight': '5px'},
                                style={'display': 'inline-block', 'marginRight': '5px'},
                                persistence=False
                            ),
                            html.Span(month_name, style={'cursor': 'pointer'})
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                        style={'cursor': 'pointer', 'paddingLeft': '30px', 'listStyle': 'none'}
                    ),
                    html.Div([
                        dcc.Checklist(
                            id={'type': 'date-checklist', 'year': year, 'month': month},
                            options=day_options,
                            value=day_values,
                            labelStyle={'display': 'block'},
                            style={'paddingLeft': '40px'},
                            persistence=False
                        )
                    ])
                ], open=False, style={'marginBottom': '5px'})
            )

        year_elements.append(
            html.Details([
                html.Summary(
                    html.Div([
                        dcc.Checklist(
                            id={'type': 'year-select-all', 'year': year},
                            options=[{'label': '', 'value': 'all'}],
                            value=['all'],
                            labelStyle={'display': 'inline-block', 'marginRight': '5px'},
                            style={'display': 'inline-block', 'marginRight': '5px'},
                            persistence=False
                        ),
                        html.Span(str(year), style={'cursor': 'pointer', 'fontWeight': 'bold'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    style={'cursor': 'pointer', 'listStyle': 'none'}
                ),
                html.Div(month_elements, style={'paddingLeft': '15px'})
            ], open=True, style={'marginBottom': '10px'})
        )

    return html.Div([
        html.Div(
            year_elements,
            style={'maxHeight': '300px', 'overflow': 'auto', 'border': '1px solid #ddd', 'padding': '10px',
                   'borderRadius': '4px'}
        ),
        dcc.Store(id='date-tree-structure', data=year_dict)
    ])

def get_available_presets():
    """Helper to list JSON files in the presets folder"""
    if not os.path.exists('presets'):
        os.makedirs('presets')
    files = [f.replace('.json', '') for f in os.listdir('presets') if f.endswith('.json')]
    return sorted(files)

def create_layout(df):
    """
    Creates the Dash layout.
    *** UPDATED *** to set smart defaults for dropdowns to prevent "No data" on load.
    """
    preset_options = [{'label': p, 'value': p} for p in get_available_presets()]

    all_locations = sorted(df['location'].dropna().unique())
    default_location = all_locations[0] if all_locations else None

    models_for_location = []
    default_model = None
    if default_location:
        models_for_location = sorted(df[df['location'] == default_location]['model_name'].dropna().unique())
        default_model = models_for_location[0] if models_for_location else None

    k_values = []
    default_k = None
    if default_location and default_model:
        k_values = sorted(df[
                              (df['location'] == default_location) &
                              (df['model_name'] == default_model)
                              ]['cluster_num'].dropna().unique())
        default_k = k_values[0] if k_values else None

    cluster_options = []
    default_clusters = []
    if default_k:
        clusters = sorted(df[df['cluster_num'] == default_k]['cluster_id'].dropna().unique())
        default_clusters = [int(c) for c in clusters]

        for c in clusters:
            c_int = int(c)
            color = CLUSTER_COLORS[c_int % len(CLUSTER_COLORS)]

            label_component = html.Span([
                html.Span("■", style={'color': color, 'fontSize': '1.5em', 'marginRight': '5px', 'lineHeight': '1'}),
                html.Span(str(c_int))
            ], style={'display': 'flex', 'alignItems': 'center'})

            cluster_options.append({'label': label_component, 'value': c_int})

    min_hour = df['start_hour_float'].min() if not df['start_hour_float'].empty else 0
    max_hour = df['start_hour_float'].max() if not df['start_hour_float'].empty else 24

    date_selector = generate_date_selector(df)

    return html.Div([

        html.Div([
            html.Div([

                html.Div([
                    html.Button(">", id="toggle-filters-btn", n_clicks=0, className="panel-toggle-button"),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                
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
                        dcc.Graph(id='spectrogram-plot', config={'displayModeBar': False}, style={'height': '100%'}),
                        id='spectrogram-plot-container'
                    ),
                ], id='audio-spectrogram-container')
            ], id='scatter-audio-container'),

            html.Div([
                
                html.Div([
                    html.Img(src="/assets/logo.svg", alt="VibroVis Logo",
                             style={'height': '80px', 'marginRight': '10px'}),
                    html.A(
                        "ℹ️",
                        href="https://docs.google.com/document/d/e/2PACX-1vSxvnYGbOE4oblvbkKrpfleLwe92h3irOA3eVr757FLZAfHqwbSBH6hcNKTqfj64_gvWBcZzeWjs8DC/pub",
                        target="_blank",
                        title="Help & Documentation",
                        style={
                            'marginLeft': 'auto',
                            'fontSize': '1.3em',
                            'textDecoration': 'none',
                            'lineHeight': '36px',
                            'cursor': 'pointer'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '10px 0',
                    'marginBottom': '10px',
                    'borderBottom': '2px solid #e0e0e0'
                }),
                
                html.Div([
                    html.Div([
                        # 1. Preset Dropdown (Takes up most space)
                        dcc.Dropdown(
                            id='preset-load-dropdown',
                            options=preset_options,
                            placeholder="Load a preset...",
                            style={'flex': '1', 'minWidth': '100px', 'fontSize': '0.9em'},
                            clearable=False
                        ),

                        # 2. Load Button
                        html.Button(
                            'Load',
                            id='preset-load-btn',
                            className='app-button',
                            style={'marginLeft': '5px', 'padding': '2px 10px', 'height': '36px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),

                    html.Details([
                        html.Summary("Save current settings as preset...", style={
                            'fontSize': '0.85em', 'color': '#888', 'cursor': 'pointer', 'marginBottom': '5px',
                            'userSelect': 'none'
                        }),
                        html.Div([
                            dcc.Input(
                                id='preset-save-name',
                                type='text',
                                placeholder="Preset name...",
                                className='input-field',
                                style={'flex': '1', 'marginRight': '5px'}
                            ),
                            html.Button('Save', id='preset-save-btn', className='app-button'),
                        ], style={'display': 'flex', 'marginBottom': '5px'}),
                        html.Div(id='preset-message', style={'fontSize': '0.8em', 'color': '#666'}),
                    ], style={'marginBottom': '10px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),

            #    html.Div([
                    html.Details([
                        html.Summary("Recordings", className='section-title'),
                        html.Div(id='recordings-filter-content', children=[
                            html.Div([
                                html.Label("Location"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Filter data by the primary recording location.",
                                              className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.Dropdown(
                                id='location-dropdown',
                                options=[{'label': str(loc), 'value': loc} for loc in all_locations],
                                value=default_location
                            ),

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
                                html.Label("Recorder type"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Filter by the type of recorder used.", className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.Dropdown(id='recorder-type-dropdown', options=[], multi=True, value=[]),

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
                                          options=[{'label': str(c), 'value': c} for c in
                                                   sorted(df['channel'].unique())],
                                          value=df['channel'].unique().tolist(),
                                          labelStyle={"display": "inline-block"}),

                            html.Div([
                                html.Label("Dates"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Filter by specific dates.", className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),

                            html.Div(id='date-tree-container', children=date_selector),

                            dcc.Store(id='date-dropdown', data=[d.strftime('%Y-%m-%d') for d in
                                                                pd.to_datetime(df['day_dt']).dt.date.unique()]),

                            html.Div([
                                html.Label("Hour range"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Filter by the time of day.", className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.RangeSlider(id='hour-slider', min=int(min_hour), max=int(max_hour) + 1,
                                            value=[min_hour, max_hour], step=0.5,
                                            marks={h: f"{int(h):02d}:00" for h in
                                                   range(int(min_hour), int(max_hour) + 2, 2)}
                                            ),
                        ])
                    ], open=True),

                    html.Details([
                        html.Summary("Model", className='section-title'),
                        html.Div(id='model-filter-content', children=[
                            html.Div([
                                html.Label("Model"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Select the machine learning model used for detection.",
                                              className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[{'label': str(m), 'value': m} for m in models_for_location],
                                value=default_model
                            ),

                            html.Div([
                                html.Label("Number of clusters"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Choose the clustering model (e.g., k=10 or k=20 clusters).",
                                              className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.Dropdown(
                                id='num-cluster-dropdown',
                                options=[{'label': str(int(c)), 'value': int(c)} for c in k_values],
                                value=default_k
                            ),

                            html.Div([
                                html.Label("Color by"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Choose to color points by their cluster assignment or by manual labels.",
                                              className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),

                            dcc.RadioItems(
                                id='color-mode-radio',
                                options=[
                                    {'label': 'Clusters', 'value': 'cluster'},
                                    {'label': 'Manual Labels', 'value': 'manual'}
                                ],
                                value='cluster',
                                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                            ),

                            html.Div([
                                html.Label("Clusters"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Select specific clusters to display from the chosen model. Click the color block to change the cluster color.",
                                              className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            dcc.Checklist(id="all-or-none-cluster", options=[{"label": "Select All", "value": "All"}],
                                          value=["All"], labelStyle={"display": "inline-block"}),
                            html.Div(
                                id='cluster-list-container',
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'flexWrap': 'wrap',
                                    'gap': '8px',
                                    'marginTop': '10px',
                                    'maxHeight': '300px',
                                    'overflowY': 'auto',
                                    'alignContent': 'flex-start'
                                }
                            ),
                        ])
                    ], open=True),

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
                            html.Div([
                                dcc.Input(id='max-points', type='number', min=1, value=10000,
                                          style={'width': '80px', 'marginRight': '1em'},
                            className='input-field'),
                                html.Button('Resample', id='resample-btn', n_clicks=0, className='app-button'),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),

                            html.Div(
                                id='sampling-info-display',
                                style={'fontSize': '0.85em', 'color': '#666', 'fontStyle': 'italic',
                                       'marginBottom': '10px'}
                            ),

                            html.Div([
                                html.Label("Merge consecutive clips"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Merges consecutive clips that are in the same cluster and close enough in the space into one point.",
                                        className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            daq.BooleanSwitch(
                                id='merge-switch',
                                on=False,
                                label='',
                                labelPosition='top',
                                persistence=False
                            ),

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
                                           step=1, value=15,
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
                            ]),

                            html.Div([
                                html.Label("Export filtered data"),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Export all currently filtered data (not sampled) to CSV. Data will be merged/concatenated.",
                                        className="tooltip-text")
                                ])
                            ], className="label-with-info", style={'marginTop': '1em'}),
                            html.Button('Export to CSV', id='export-csv-btn', n_clicks=0, className='app-button', style={'width': '80%'}),
                            dcc.Download(id="download-csv")
                            ])
                    ], open=True),

                    html.Details([
                        html.Summary("Labeling", className='section-title'),
                        html.Div(children=[
                            # Row 1: Input and Apply
                            html.Div([
                                 dcc.Input(id='manual-label-input', type='text', placeholder='Enter label...', list='manual-label-datalist', className='input-field', style={'flex': '1', 'marginRight': '5px'}),
                                 html.Datalist(id='manual-label-datalist'),
                                 html.Button('Apply', id='save-label-btn', n_clicks=0, className='app-button'),
                            ], style={'display': 'flex', 'marginBottom': '10px'}),

                            # Row 2: Load (Dropdown + Button)
                            html.Div([
                                dcc.Dropdown(
                                    id='label-load-dropdown',
                                    options=[],
                                    placeholder="Load labels...",
                                    style={'flex': '1', 'minWidth': '100px', 'fontSize': '0.9em'},
                                    clearable=False
                                ),
                                html.Button(
                                    'Load',
                                    id='label-load-btn',
                                    className='app-button',
                                    style={'marginLeft': '5px', 'padding': '2px 10px', 'height': '36px'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),

                            # Row 3: Save (Input + Button)
                            html.Details([
                                html.Summary("Save current labels...", style={
                                    'fontSize': '0.85em', 'color': '#888', 'cursor': 'pointer', 'marginBottom': '5px',
                                    'userSelect': 'none'
                                }),
                                html.Div([
                                    dcc.Input(
                                        id='label-save-name',
                                        type='text',
                                        placeholder='Label set name...',
                                        className='input-field',
                                        style={'flex': '1', 'marginRight': '5px'}
                                    ),
                                    html.Button('Save', id='label-save-btn-server', className='app-button'),
                                ], style={'display': 'flex', 'marginBottom': '5px'}),
                                html.Div(id='label-save-message', style={'fontSize': '0.8em', 'color': '#666'}),
                            ], style={'marginBottom': '10px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                            
                            # Row 4: Reset
                            html.Div([
                                 html.Button('Reset Labels', id='btn-reset-labels', className='app-button', style={'width': '80%', 'backgroundColor': '#d9534f', 'color': 'white'}),
                                 dcc.ConfirmDialog(
                                      id='confirm-reset-labels',
                                      message='Are you sure you want to RESET all manual labels?\n\nThis will clear all labels from memory.\nMake sure you have saved your labels if you want to keep them.',
                                 ),
                            ], style={'marginBottom': '5px'}),

                            # Status Message
                            html.Div(id='label-saved-msg', style={'color':'#28a745', 'fontSize':'0.9em', 'fontWeight': 'bold', 'textAlign': 'center', 'minHeight': '1.2em'})

                        ], style={'padding': '10px'})
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
                                             value=1024),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span(
                                        "Size of the FFT window. Larger windows give better frequency resolution but worse time resolution.",
                                        className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Window Overlap"),
                                dcc.Slider(id='window-overlap', min=0, max=0.9, step=0.05, value=0.5,
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
                                dcc.Input(id='num-bins', type='number', value=128,
                            className='input-field'),
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
                                dcc.Input(id='min-freq', type='number', value=50,
                            className='input-field'),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Minimum frequency to display on the spectrogram.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),

                            html.Div([
                                html.Label("Max Freq (Hz)"),
                                dcc.Input(id='max-freq', type='number', value=5000,
                            className='input-field'),
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
                                dcc.Input(id='db-floor', type='number', value=-100,
                            className='input-field'),
                                html.Div(className="tooltip-container", children=[
                                    html.Span(" ⓘ", className="info-icon"),
                                    html.Span("Minimum decibel level to display; values below this are clipped.",
                                              className="tooltip-text")
                                ])
                            ], className='spectrogram-row'),
                        ])
                    ]),
                
                    html.Details([
                        html.Summary("Time Animation", className='section-title'),
                        html.Div(children=[
                            
                            html.Div([
                                html.Label("Enable Animation"),
                                daq.BooleanSwitch(
                                    id='anim-enabled-switch',
                                    on=False,
                                    label='',
                                    labelPosition='top',
                                    persistence=False
                                )
                            ], className="label-with-info", style={'marginBottom': '15px'}),

                            html.Div([
                                html.Button("▶ Play", id="anim-play-btn", n_clicks=0, className="app-button", style={'width': '80%'}),
                            ], style={'marginBottom': '15px'}),

                            html.Div([
                                html.Label("Mode"),
                                dcc.Dropdown(
                                    id='anim-mode',
                                    options=[
                                        {'label': 'Daily (24h)', 'value': 'daily'}, 
                                        {'label': 'Yearly', 'value': 'yearly'}
                                    ],
                                    value='daily',
                                    clearable=False
                                )
                            ], style={'marginBottom': '15px'}),

                            html.Div([
                                html.Label("Speed (Interval ms)"),
                                html.Div([
                                    dcc.Input(
                                        id='anim-speed-slider',
                                        type='number',
                                        min=1,
                                        max=1000,
                                        step=1,
                                        value=100,
                                        style={'marginRight': '10px'},
                            className='input-field'
                                    ),
                                    html.Div(
                                        "(1 = Fastest, 1000 = Slowest)",
                                        style={'fontSize': '0.85em', 'fontStyle': 'italic'}
                                    )
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style={'marginBottom': '15px'}),

                            html.Div([
                                html.Label("Window Size"),
                                html.Div([
                                    dcc.Input(id='anim-window-size', type='number', value=2, min=0.1, step=0.1, style={'width': '70px'},
                            className='input-field'),
                                    html.Span(id='anim-unit-label', children=" (hours)", style={'marginLeft': '10px', 'fontSize': '0.9em'})
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style={'marginBottom': '15px'}),

                            html.Div(id='anim-time-display', style={'textAlign': 'center', 'fontWeight': 'bold', 'color': '#007bff', 'fontSize': '1.2em', 'marginBottom':'5px'}),
                            
                            dcc.Slider(id='anim-progress-slider', min=0, max=24, step=0.1, value=0, marks={}),
                            
                            dcc.Interval(id='anim-interval', interval=100, n_intervals=0, disabled=True),

                        ], style={'padding': '10px'})
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
                    html.Div(
                        dcc.Graph(id='cluster-histogram', config={'displayModeBar': False}, style={'height': '100%'}),
                        id='histogram-click-wrapper',
                        style={'height': '89%'}
                    ),
                ], id='histogram-container'),

            ], id='filter-container', className='filters-expanded'),
        ], id='main-content'),

        dcc.Store(id='histogram-time-scale-store', data='daily'),
        dcc.Store(id='filtered-data'),
        dcc.Store(id='spectrogram-raw-data-store'),
        dcc.Store(id='clip-count-max-store'),
        dcc.Store(id='merge-max-store'),
        dcc.Store(id='histogram-cache'),
        dcc.Store(id='model-data-ready-signal'),
        dcc.Store(id='sampled-indices-store'),
        dcc.Store(id='anim-ranges-store'),
        dcc.Store(id='is-anim-open-store', data=False),
        dcc.Store(id='cluster-color-store', data={}),
        dcc.Store(id='cluster-name-store', data={}),
        dcc.Store(id='preset-last-action'),
        dcc.Store(id='preset-cluster-selection'),
        dcc.Store(id='cluster-stats-store'),
        dcc.Store(id='last-preset-load-time', data=0),
        dcc.Store(id='manual-labels-store', data={}),
        dcc.Store(id='label-last-action'),
        dcc.Store(id='params-store'),
    ], id='main-container')