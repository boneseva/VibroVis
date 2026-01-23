from dash import Dash, html
from werkzeug.middleware.profiler import ProfilerMiddleware

from layout import create_layout
import read_data
from callbacks import set_initial_data, register_callbacks

app = Dash(__name__)
app.title = "VibroVis"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="shortcut icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="apple-touch-icon" href="/assets/favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="shortcut icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="apple-touch-icon" href="/assets/favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

initial_df = read_data.get_initial_data_for_layout()
set_initial_data(initial_df)

app.layout = create_layout(initial_df)

register_callbacks(app)

if __name__ == "__main__":
    # server = app.server
    #
    # PORT = 8050
    # ADDRESS = "0.0.0.0"
    #
    # app.run(port=PORT, host=ADDRESS, debug=True, use_reloader=False, dev_tools_ui=True)

    app.server.wsgi_app = ProfilerMiddleware(
        app.server.wsgi_app,
        sort_by=['cumtime']
    )

    app.run(port=8050, host="0.0.0.0", debug=False, threaded=False)
