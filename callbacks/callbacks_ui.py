"""
UI-related callbacks: toggles, animations, and UI interactions.
"""
import dash
from dash import Input, Output, State
from datetime import datetime, timedelta
import numpy as np

from .callbacks_constants import MODEL_DATA_CACHE


def register_ui_callbacks(app):
    
    @app.callback(
        [Output('audio-player', 'autoPlay'),
         Output('autoplay-toggle-btn', 'children'),
         Output('autoplay-toggle-btn', 'className')],
        Input('autoplay-toggle-btn', 'n_clicks'),
        State('audio-player', 'autoPlay'))
    def toggle_autoplay(n_clicks, current_autoplay):
        if n_clicks is None or n_clicks == 0: return False, "Autoplay: OFF", 'app-button autoplay-off'
        new_autoplay = not current_autoplay
        return new_autoplay, "Autoplay: ON" if new_autoplay else "Autoplay: OFF", 'app-button autoplay-on' if new_autoplay else 'app-button autoplay-off'

    @app.callback(
        Output('filter-container', 'className'),
        Output('toggle-filters-btn', 'children'),
        Input('toggle-filters-btn', 'n_clicks'),
        State('filter-container', 'className'),
        prevent_initial_call=True
    )
    def toggle_filter_panel(n_clicks, current_class):
        return ('filters-collapsed', '<') if current_class == 'filters-expanded' else ('filters-expanded', '>')

    @app.callback(
        [Output('anim-progress-slider', 'min'),
         Output('anim-progress-slider', 'max'),
         Output('anim-progress-slider', 'marks'),
         Output('anim-progress-slider', 'value'),
         Output('anim-unit-label', 'children')],
        [Input('anim-mode', 'value'),
         Input('anim-ranges-store', 'data')],
        State('anim-progress-slider', 'value')
    )
    def update_animation_slider_config(mode, ranges, current_val):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
        if not ranges: return 0, 24, {0: '0h', 24: '24h'}, 0, " (hours)"

        if mode == 'yearly':
            min_val, max_val = map(int, ranges['yearly'])
            base_date = datetime(2024, 1, 1)
            total_days = max_val - min_val
            step = max(1, total_days // 10)
            marks = {}
            for i in range(min_val, max_val + 1, step):
                marks[i] = {'label': (base_date + timedelta(days=i - 1)).strftime('%b %d'),
                            'style': {'whiteSpace': 'nowrap'}}
            marks[min_val] = {'label': (base_date + timedelta(days=min_val - 1)).strftime('%b %d'),
                              'style': {'whiteSpace': 'nowrap'}}
            marks[max_val] = {'label': (base_date + timedelta(days=max_val - 1)).strftime('%b %d'),
                              'style': {'whiteSpace': 'nowrap'}}
            new_val = min_val if 'anim-mode' in trigger else max(min_val, min(current_val, max_val))
            return min_val, max_val, marks, new_val, " (days)"
        else:
            raw_min, raw_max = ranges['daily']
            min_limit = int(np.floor(raw_min));
            max_limit = int(np.ceil(raw_max))
            if min_limit == max_limit:
                if max_limit < 24:
                    max_limit += 1
                else:
                    min_limit = max(0, min_limit - 1)
            span = max_limit - min_limit
            step = max(1, span // 6)
            marks = {i: f"{i:02d}:00" for i in range(min_limit, max_limit + 1, step)}
            marks[min_limit] = f"{min_limit:02d}:00";
            marks[max_limit] = f"{max_limit:02d}:00"
            new_val = raw_min if 'anim-mode' in trigger else max(raw_min, min(current_val, raw_max))
            return min_limit, max_limit, marks, new_val, " (hours)"

    @app.callback(
        [Output('anim-interval', 'disabled'), Output('anim-play-btn', 'children')],
        Input('anim-play-btn', 'n_clicks'), State('anim-interval', 'disabled')
    )
    def toggle_animation(n_clicks, is_disabled):
        if n_clicks == 0: return True, "▶ Play"
        return (False, "⏸ Pause") if is_disabled else (True, "▶ Play")

    @app.callback(Output('anim-interval', 'interval'), Input('anim-speed-slider', 'value'))
    def update_anim_speed(value):
        return value

    app.clientside_callback(
        """
        function(n_intervals, current_val, mode, ranges) {
            if (!ranges) return current_val;
            let step = (mode === 'yearly') ? 1.0 : 0.1; 
            let next_val = current_val + step;
            let max_limit = (mode === 'yearly') ? ranges['yearly'][1] : ranges['daily'][1];
            let min_limit = (mode === 'yearly') ? ranges['yearly'][0] : ranges['daily'][0];
            if (mode === 'daily') { if (next_val > max_limit) { next_val = min_limit; } }
            else { if (next_val > max_limit) { next_val = min_limit; } }
            return next_val;
        }
        """,
        Output('anim-progress-slider', 'value', allow_duplicate=True),
        Input('anim-interval', 'n_intervals'),
        State('anim-progress-slider', 'value'),
        State('anim-mode', 'value'),
        State('anim-ranges-store', 'data'),
        prevent_initial_call=True
    )

    app.clientside_callback(
        """
        function(time_value, window_size, mode, is_enabled, figure) {
            if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
            const new_fig = JSON.parse(JSON.stringify(figure));
            if (!is_enabled) {
                for (let i = 0; i < new_fig.data.length; i++) {
                    const trace = new_fig.data[i];
                    if (trace.marker) trace.marker.opacity = null;
                    trace.selectedpoints = null;
                }
                return [new_fig, ""];
            }
            let time_str = "";
            if (mode === 'yearly') {
                const date = new Date(2024, 0, time_value); 
                const month = date.toLocaleString('default', { month: 'short' });
                const day = date.getDate();
                time_str = `${month} ${day}`;
            } else {
                let hours = Math.floor(time_value);
                let minutes = Math.floor((time_value - hours) * 60);
                time_str = hours.toString().padStart(2, '0') + ":" + minutes.toString().padStart(2, '0');
            }
            const half_window = window_size / 2.0;
            for (let i = 0; i < new_fig.data.length; i++) {
                const trace = new_fig.data[i];
                if (trace.customdata) {
                    const opacity_array = [];
                    const time_idx = (mode === 'yearly') ? 4 : 3;
                    for (let j = 0; j < trace.customdata.length; j++) {
                        const point_time = trace.customdata[j][time_idx];
                        let dist = Math.abs(point_time - time_value);
                        if (mode === 'daily') { if (dist > 12) dist = 24 - dist; }
                        if (dist <= half_window) {
                            let op = 1.0 - (dist / half_window);
                            opacity_array.push(Math.max(0.1, op)); 
                        } else { opacity_array.push(0.05); }
                    }
                    if (!trace.marker) trace.marker = {};
                    trace.marker.opacity = opacity_array;
                    trace.selectedpoints = null; 
                }
            }
            return [new_fig, time_str];
        }
        """,
        [Output('scatter', 'figure', allow_duplicate=True), Output('anim-time-display', 'children')],
        [Input('anim-progress-slider', 'value'), Input('anim-window-size', 'value'),
         Input('anim-mode', 'value'), Input('anim-enabled-switch', 'on')],
        State('scatter', 'figure'),
        prevent_initial_call=True
    )

