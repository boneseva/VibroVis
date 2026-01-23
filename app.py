from dash import Dash, html
from werkzeug.middleware.profiler import ProfilerMiddleware

from layout import create_layout
import read_data
from callbacks import set_initial_data, register_callbacks

app = Dash(__name__)
app.title = "VibroVis"
server = app.server

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
    PORT = 8050
    ADDRESS = "0.0.0.0"

    display_port = PORT
    if ADDRESS == "0.0.0.0":
        display_host = "127.0.0.1"
    else:
        display_host = ADDRESS
        
    print(f"Dash is running on http://{display_host}:{display_port}/")
    print("WARNING: This is a PRODUCTION configuration. Debug mode is OFF.")
    
    app.run(port=PORT, host=ADDRESS, debug=False, use_reloader=False, dev_tools_ui=False)