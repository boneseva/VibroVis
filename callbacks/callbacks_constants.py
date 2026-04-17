"""
Shared constants used across callback modules.
"""
import pandas as pd
import math
import os
from collections import Counter
import diskcache

# Global data stores
initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}
server_cache = {}

# Persistent, process-shared manual labels cache
# This fixes the multi-worker deployment bug where labels would vanish
_CACHE_DIR = os.path.join(os.getcwd(), 'cache', 'manual_labels')
os.makedirs(_CACHE_DIR, exist_ok=True)

# Enhanced diskcache configuration for multi-worker deployments
MANUAL_LABELS_CACHE = diskcache.Cache(
    _CACHE_DIR, 
    size_limit=100*1024*1024,  # 100MB limit
    disk_min_file_size=0,      # Store all data in DB, no separate files
    timeout=30,                # 30 second timeout for locks
    retry=True,                # Retry on lock timeouts
    cull_limit=100,            # More aggressive cleanup
    eviction_policy='least-recently-used'  # Correct eviction policy name
)


def safe_cache_write(key, value, max_retries=3):
    """
    Safely write to the manual labels cache with retry logic.
    Prevents corruption in multi-worker environments.
    """
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            # Use cache transaction for atomic write
            with MANUAL_LABELS_CACHE.transact():
                MANUAL_LABELS_CACHE[key] = value
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter to prevent thundering herd
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Cache write retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)
            else:
                print(f"Cache write failed after {max_retries} attempts: {e}")
                # Fallback: try without transaction
                try:
                    MANUAL_LABELS_CACHE[key] = value
                    return True
                except Exception as e2:
                    print(f"Cache write fallback also failed: {e2}")
                    return False
    return False


def safe_cache_delete(key, max_retries=3):
    """
    Safely delete from the manual labels cache with retry logic.
    """
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            with MANUAL_LABELS_CACHE.transact():
                if key in MANUAL_LABELS_CACHE:
                    del MANUAL_LABELS_CACHE[key]
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Cache delete retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)
            else:
                print(f"Cache delete failed after {max_retries} attempts: {e}")
                try:
                    MANUAL_LABELS_CACHE.pop(key, None)
                    return True
                except Exception as e2:
                    print(f"Cache delete fallback also failed: {e2}")
                    return False
    return False


def safe_cache_batch_write(key_value_pairs, max_retries=3):
    """
    Safely write multiple key-value pairs to cache in a single transaction.
    Much more efficient for bulk operations.
    """
    import time
    import random
    
    if not key_value_pairs:
        return True
    
    for attempt in range(max_retries):
        try:
            with MANUAL_LABELS_CACHE.transact():
                for key, value in key_value_pairs:
                    MANUAL_LABELS_CACHE[key] = value
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Batch cache write retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)
            else:
                print(f"Batch cache write failed after {max_retries} attempts: {e}")
                # Fallback: write individually
                success_count = 0
                for key, value in key_value_pairs:
                    if safe_cache_write(key, value, max_retries=1):
                        success_count += 1
                print(f"Fallback individual writes: {success_count}/{len(key_value_pairs)} succeeded")
                return success_count == len(key_value_pairs)
    return False

# Cluster color palette
CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def set_initial_data(df):
    """Set the initial dataframe."""
    global initial_df
    initial_df = df

# apply_manual_labels_efficiently MOVED TO utils.py
