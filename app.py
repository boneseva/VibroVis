# In app.py
from dash import Dash, html
from layout import create_layout
import read_data
from read_data import df
import callbacks

app = Dash(__name__)
app.title = "VibroVis"

# Revert to the standard layout using the create_layout function
app.layout = create_layout(df)

callbacks.register_callbacks(app)

PORT = 8050
ADDRESS = "0.0.0.0"

if __name__ == "__main__":
    app.run(port=PORT, host=ADDRESS)