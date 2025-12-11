from dash import Dash, html
from layout import create_layout
import read_data
from callbacks import set_initial_data, register_callbacks
import os

app = Dash(__name__)
app.title = "VibroVis"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" sizes="any">
        <link rel="shortcut icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="apple-touch-icon" href="/assets/favicon.svg">
        <style>
            link[rel="icon"] {
                image-rendering: -webkit-optimize-contrast;
                image-rendering: crisp-edges;
            }
        </style>
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
server = app.server

PORT = 8050
ADDRESS = "0.0.0.0"

if __name__ == "__main__":
    app.run(port=PORT, host=ADDRESS)