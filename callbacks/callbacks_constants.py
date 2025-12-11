"""
Shared constants used across callback modules.
"""
import pandas as pd

# Global data stores
initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}
server_cache = {}

# Cluster color palette
CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def set_initial_data(df):
    """Set the initial dataframe."""
    global initial_df
    initial_df = df

