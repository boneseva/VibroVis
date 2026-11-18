"""
Date selector generation and related callbacks.
"""
import dash
from dash import Input, Output, State, html, dcc, ALL, MATCH
import pandas as pd

from .callbacks_constants import MODEL_DATA_CACHE


def generate_date_selector(df, selected_dates=None):
    """
    Generates Date Selector.
    If selected_dates is provided (list of strings 'YYYY-MM-DD'),
    only checks those dates. Otherwise checks ALL.
    """
    if df is None or df.empty:
        return html.Div("No date data found.")

    if 'day_dt_str' in df.columns:
        date_strings = sorted(df['day_dt_str'].unique())
        dates = [pd.to_datetime(d).date() for d in date_strings]
    elif 'day_dt' in df.columns:
        dates = pd.to_datetime(df['day_dt']).dt.date.unique()
        dates = sorted(dates)
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    else:
        return html.Div("No date data found.")

    selected_set = set(selected_dates) if selected_dates is not None else None

    if len(dates) < 10:
        all_values = date_strings
        options = [{'label': d, 'value': d} for d in date_strings]

        if selected_set is not None:
            current_value = [d for d in all_values if d in selected_set]
        else:
            current_value = all_values

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
                d_str = date_strings[dates.index(d)]
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


def register_date_selector_callbacks(app):
    
    @app.callback(
        [Output({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'value'),
         Output({'type': 'month-select-all', 'year': MATCH, 'month': MATCH}, 'value')],
        [Input({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'value'),
         Input({'type': 'month-select-all', 'year': MATCH, 'month': MATCH}, 'value')],
        [State({'type': 'date-checklist', 'year': MATCH, 'month': MATCH}, 'options')],
        prevent_initial_call=True
    )
    def sync_month_and_dates(selected_dates_in, month_select_in, options):
        triggered_id = dash.callback_context.triggered_id
        if isinstance(triggered_id, dict):
            triggered_type = triggered_id.get('type')
            if triggered_type == 'month-select-all':
                if options is None: return dash.no_update, dash.no_update
                all_dates = [opt['value'] for opt in options]
                return (all_dates, dash.no_update) if month_select_in == ['all'] else ([], dash.no_update)
            elif triggered_type == 'date-checklist':
                if options is None: return dash.no_update, dash.no_update
                selected_dates_set = set() if selected_dates_in is None else set(selected_dates_in)
                all_options_set = set(opt['value'] for opt in options)
                return (dash.no_update, ['all']) if selected_dates_set == all_options_set else (dash.no_update, [])
        return dash.no_update, dash.no_update

    @app.callback(
        Output({'type': 'year-select-all', 'year': MATCH}, 'value'),
        Input({'type': 'month-select-all', 'year': MATCH, 'month': ALL}, 'value'),
        prevent_initial_call=True
    )
    def update_year_checkbox(all_month_checkboxes):
        if not all_month_checkboxes: return []
        all_months_selected = all(['all' in month_val if month_val else False for month_val in all_month_checkboxes])
        return ['all'] if all_months_selected and len(all_month_checkboxes) > 0 else []

    @app.callback(
        Output({'type': 'month-container-wrapper', 'year': MATCH, 'month': MATCH}, 'style'),
        Input({'type': 'year-toggle', 'year': MATCH, 'month': MATCH}, 'n_clicks'),
        State({'type': 'month-container-wrapper', 'year': MATCH, 'month': MATCH}, 'style'),
        prevent_initial_call=True
    )
    def toggle_year_collapse(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return {'display': 'block'} if current_style and current_style.get('display') == 'none' else {'display': 'none'}

    @app.callback(
        Output({'type': 'day-grid-container', 'year': MATCH, 'month': MATCH}, 'style'),
        Input({'type': 'month-toggle', 'year': MATCH, 'month': MATCH}, 'n_clicks'),
        State({'type': 'day-grid-container', 'year': MATCH, 'month': MATCH}, 'style'),
        prevent_initial_call=True
    )
    def toggle_month_collapse(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0: return dash.no_update
        return {'display': 'block', 'paddingLeft': '20px'} if current_style and current_style.get(
            'display') == 'none' else {'display': 'none', 'paddingLeft': '20px'}

    @app.callback(
        Output('date-dropdown', 'data'),
        [Input({'type': 'date-checklist', 'year': ALL, 'month': ALL}, 'value'),
         Input({'type': 'simple-date-dropdown', 'index': ALL}, 'value')])
    def collect_selected_dates(tree_values, dropdown_values):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        trigger_id = ctx.triggered[0]['prop_id']

        if 'simple-date-dropdown' in trigger_id:
            if dropdown_values and isinstance(dropdown_values[0], list):
                return dropdown_values[0]
            return []

        selected_dates = []
        for dates in tree_values:
            if dates: selected_dates.extend(dates)
        return list(set(selected_dates))

    @app.callback(
        Output('date-tree-container', 'children'),
        [Input('model-data-ready-signal', 'data')],
        [State('date-dropdown', 'data')]
    )
    def update_date_tree_dynamically(model_ready_signal, stored_dates):
        if not model_ready_signal:
            return html.Div("Select a location...")

        dff = MODEL_DATA_CACHE.get('df')
        if dff is None or dff.empty:
            return html.Div("No data found.")

        dates_to_pass = None

        if stored_dates:
            if 'day_dt_str' in dff.columns:
                available_dates = set(dff['day_dt_str'].unique())
            else:
                available_dates = set(pd.to_datetime(dff['day_dt']).dt.strftime('%Y-%m-%d'))
            stored_set = set(stored_dates)

            if not available_dates.isdisjoint(stored_set):
                dates_to_pass = stored_dates
            else:
                dates_to_pass = None

        return generate_date_selector(dff, selected_dates=dates_to_pass)

