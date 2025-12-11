"""
Loading state management callbacks - simple overlay indicators.
"""
from dash import Input, Output, State, callback_context, ALL


def register_loading_callbacks(app):
    """Register callbacks for loading overlay indicators."""
    
    @app.callback(
        Output('scatter-loading-overlay', 'style'),
        [Input('filtered-data', 'data'),
         Input('scatter', 'figure')],
        prevent_initial_call=True
    )
    def update_scatter_loading(filtered_data_key, scatter_fig):
        """Show loading overlay when data is being updated."""
        ctx = callback_context
        
        base_style = {
            'position': 'absolute',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'zIndex': 1000,
            'pointerEvents': 'none'
        }
        
        if not ctx.triggered:
            base_style['display'] = 'none'
            return base_style
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # If scatter figure was updated, hide spinner (update complete)
        if 'scatter.figure' in trigger_id:
            base_style['display'] = 'none'
            return base_style
        
        # If filtered-data changed, show spinner (data is being processed)
        if 'filtered-data' in trigger_id:
            base_style['display'] = 'block'
            return base_style
        
        base_style['display'] = 'none'
        return base_style
    
    @app.callback(
        Output('table-loading-overlay', 'style'),
        [Input('filtered-data', 'data'),
         Input('view-mode-store', 'data'),
         Input('data-table', 'data')],
        prevent_initial_call=True
    )
    def update_table_loading(filtered_data_key, view_mode, table_data):
        """Show loading overlay when table data is being updated."""
        ctx = callback_context
        
        base_style = {
            'position': 'absolute',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'zIndex': 1000,
            'pointerEvents': 'none'
        }
        
        # Only show spinner in table view mode
        if view_mode != 'table':
            base_style['display'] = 'none'
            return base_style
        
        if not ctx.triggered:
            # If we have table data, hide spinner
            if table_data and len(table_data) > 0:
                base_style['display'] = 'none'
            else:
                base_style['display'] = 'none'
            return base_style
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # If table data was updated, hide spinner (loading complete)
        if 'data-table.data' in trigger_id:
            base_style['display'] = 'none'
            return base_style
        
        # If filtered-data changed, show spinner (data is being processed)
        if 'filtered-data' in trigger_id:
            base_style['display'] = 'block'
            return base_style
        
        # If view mode changed to table, show spinner if no data yet
        if 'view-mode-store' in trigger_id:
            if not table_data or len(table_data) == 0:
                base_style['display'] = 'block'
            else:
                base_style['display'] = 'none'
            return base_style
        
        # Default: hide if we have data
        if table_data and len(table_data) > 0:
            base_style['display'] = 'none'
        else:
            base_style['display'] = 'none'
        return base_style

