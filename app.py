import os
import re
from dash import Dash, html
from werkzeug.middleware.proxy_fix import ProxyFix

from layout import create_layout
import read_data
from callbacks import set_initial_data, register_callbacks

app = Dash(__name__)
app.title = "VibroVis"
server = app.server
server.secret_key = 'vibrovis-secure-key-change-this-in-env'
server.config['WTF_CSRF_ENABLED'] = False

# Apply ProxyFix to handle HTTPS headers correctly
server.wsgi_app = ProxyFix(server.wsgi_app, x_proto=1, x_host=1)


def _extract_logo_accent_color(svg_path: str, fallback: str = '#4b8af2') -> str:
    """Read the logo SVG and return the first fill color found in its <style> block."""
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_text = f.read()
        match = re.search(r'fill:\s*(#[0-9a-fA-F]{3,8})', svg_text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return fallback


_logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.svg')
_accent_color = _extract_logo_accent_color(_logo_path)
print(f'[VibroVis] Theme accent color extracted from logo: {_accent_color}')

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="shortcut icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="apple-touch-icon" href="/assets/favicon.svg">
        <style>
            :root {{ --color-accent: {_accent_color}; }}
        </style>
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
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
    
    app.run(port=PORT, host=ADDRESS, debug=False, use_reloader=False, dev_tools_ui=False)
   # app.run(port=PORT, host=ADDRESS, debug=True, use_reloader=True, dev_tools_ui=True)
