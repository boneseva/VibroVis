"""
Main callbacks module that imports and registers all callback modules.
"""
from .callbacks_data import register_data_callbacks
from .callbacks_clusters import register_cluster_callbacks
from .callbacks_plots import register_plot_callbacks
from .callbacks_filters import register_filter_callbacks
from .callbacks_ui import register_ui_callbacks
from .callbacks_presets import register_preset_callbacks
from .callbacks_date_selector import register_date_selector_callbacks


def register_callbacks(dash_app):
    """Register all callbacks from separate modules."""
    register_data_callbacks(dash_app)
    register_cluster_callbacks(dash_app)
    register_plot_callbacks(dash_app)
    register_ui_callbacks(dash_app)
    register_preset_callbacks(dash_app)
    register_filter_callbacks(dash_app)
    register_date_selector_callbacks(dash_app)

