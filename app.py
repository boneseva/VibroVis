# In app.py
from dash import Dash, html
from layout import create_layout
import read_data
#from read_data import df
import callbacks

app = Dash(__name__)
app.title = "VibroVis"

initial_df = read_data.get_initial_data_for_layout()
callbacks.set_initial_data(initial_df)

app.layout = create_layout(initial_df)

callbacks.register_callbacks(app)
server = app.server

PORT = 8050
ADDRESS = "0.0.0.0"

if __name__ == "__main__":
    app.run(port=PORT, host=ADDRESS)
    # app.run_server(debug=True)