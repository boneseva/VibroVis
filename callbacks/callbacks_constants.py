"""
Shared constants used across callback modules.
"""
import pandas as pd
import diskcache as dc
import os
import threading

# Global data stores
initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}

# Server-side cache for filtered data (used by scatter plot and table)
server_cache = {}

# Thread-safe manual labels cache using diskcache for Gunicorn production
# This ensures all threads see the same state immediately
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'manual_labels')
MANUAL_LABELS_CACHE = dc.Cache(cache_dir)
MANUAL_LABELS_LOCK = threading.RLock()  # Reentrant lock for thread safety

# Cluster color palette
CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']

# In-memory map: label key → clip duration (seconds, float).
# Written at label-save time so apply_manual_labels_efficiently can do
# overlap-based matching between models that use different clip lengths.
# Plain dict — no lock needed (GIL protects simple writes; stale reads are benign).
CLIP_DURATION_CACHE: dict = {}


def set_initial_data(df):
    """Set the initial dataframe."""
    global initial_df
    initial_df = df
