from dash import Dash
from layout import create_layout
import read_data
from read_data import df
import callbacks

app = Dash(__name__)
app.title = "VibroVis"
# app.css.append_css({
#     'external_url': [
#         'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
#     ]
# })
print(df.head())
app.layout = create_layout(df)
callbacks.register_callbacks(app)
server = app.server

PORT = 8050
ADDRESS = "0.0.0.0"
    
if __name__ == "__main__":
    
    app.run(port=PORT, host=ADDRESS)
