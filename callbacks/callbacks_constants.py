"""
Shared constants used across callback modules.
"""
import pandas as pd
import math
import time
from collections import Counter, OrderedDict

# Global data stores
initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}

# OPTIMIZATION #1: Enhanced server_cache with LRU eviction to prevent memory leaks
class LRUCache:
    """LRU Cache with size limit and automatic cleanup of old entries."""
    
    def __init__(self, max_size=50, max_age_minutes=30):
        self.max_size = max_size
        self.max_age_seconds = max_age_minutes * 60
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key):
        """Get item and mark as recently used."""
        if key in self.cache:
            # Move to end (mark as recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def __setitem__(self, key, value):
        """Set item with automatic cleanup."""
        self._cleanup_old_entries()
        
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key, None)
            
        # Add new item
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def _cleanup_old_entries(self):
        """Remove entries older than max_age."""
        current_time = time.time()
        keys_to_remove = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.max_age_seconds
        ]
        
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __contains__(self, key):
        return key in self.cache

# Replace simple dict with LRU cache
server_cache = LRUCache(max_size=50, max_age_minutes=30)
MANUAL_LABELS_CACHE = {}

# Cluster color palette
CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def set_initial_data(df):
    """Set the initial dataframe."""
    global initial_df
    initial_df = df

# apply_manual_labels_efficiently MOVED TO utils.py
