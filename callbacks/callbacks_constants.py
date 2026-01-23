"""
Shared constants used across callback modules.
"""
import pandas as pd
import math
from collections import Counter

# Global data stores
initial_df = pd.DataFrame()
MODEL_DATA_CACHE = {'df': None}
MERGED_DATA_CACHE = {'df': None, 'key': None}
server_cache = {}
MANUAL_LABELS_CACHE = {}

# Cluster color palette
CLUSTER_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#EDC948', '#B07AA1', '#FF9DA7', '#A6A377', '#F2C894',
                  '#BADCBD', '#59A14F', '#9C755F', '#BAB0AC', '#D37295', '#A0CBE8',
                  '#FFBE7D', '#9CD17D', '#D4B7A9', '#D9D9D9', '#FABFD2']


def set_initial_data(df):
    """Set the initial dataframe."""
    global initial_df
    initial_df = df

def apply_manual_labels_efficiently(dff):
    """
    Apply manual labels to dff using an inverted loop strategy.
    Instead of iterating all rows, iterate likely labeled groups.
    MOVED TO HELPER to be shared between callbacks.
    """
    if not MANUAL_LABELS_CACHE or dff.empty:
        dff['manual_label'] = 'Unlabeled'
        return dff
        
    # Initialize with Unlabeled
    dff['manual_label'] = 'Unlabeled'
    
    # Fast optimization: pre-calculate label map
    label_map = {}
    if MANUAL_LABELS_CACHE:
        # Key format: (location, microlocation, f_base, chan, sec)
        for key_tuple, label in MANUAL_LABELS_CACHE.items():
            if len(key_tuple) == 5:
                loc, micro, f_base, chan, sec = key_tuple
                
                # Group by (loc, micro, f_base, chan)
                group_key = (loc, micro, f_base, chan)
                if group_key not in label_map:
                    label_map[group_key] = {}
                label_map[group_key][sec] = label
            # Add backwards compatibility check if needed, or simply let old keys fail/drop
    
    if not label_map:
        dff['manual_label'] = 'Unlabeled'
        return dff
        
    # Initialize with Unlabeled
    dff['manual_label'] = 'Unlabeled'
    
    # Iterate over labeled files only
    for (loc, micro, f_base, chan), sec_map in label_map.items():
        # Fast filter: channel is usually integer or char
        try:
             # Mask: Channel AND File Basename AND Location AND Microlocation
             
             # Channel match
             if dff['channel'].dtype.name == 'category':
                  mask = (dff['channel'].astype(int) == int(chan))
             else:
                  mask = (dff['channel'] == int(chan))
             
             # Location match (if column exists, usually does)
             if 'location' in dff.columns:
                 # Handle NaN vs Unknown
                 # dff is a copy, safe to modify or just use fillna in comparison
                 mask &= (dff['location'].fillna('Unknown') == loc)
             
             # Microlocation match
             if 'microlocation' in dff.columns:
                 mask &= (dff['microlocation'].fillna('Unknown') == micro)
             
             # File Basename match
             mask &= dff['mp3_file'].astype(str).str.endswith(f_base)
             
             # Get indices
             indices = dff.index[mask]
             
             if len(indices) == 0:
                 continue
                 
             # Iterate only the relevant rows
             # This is much faster (e.g. 100 rows vs 700k)
             for idx in indices:
                 row_start = float(dff.at[idx, 'clip_time'])
                 row_dur = float(dff.at[idx, 'clip_duration']) if 'clip_duration' in dff.columns else 5.0
                 
                 # Majority vote using the sec_map directly
                 start_second = math.floor(row_start)
                 end_second = math.ceil(row_start + row_dur)
                 
                 found_labels = []
                 for sec in range(start_second, end_second):
                     l = sec_map.get(sec)
                     if l and l != 'Unlabeled':
                         found_labels.append(l)
                 
                 if found_labels:
                     # Filter valid (ignore 'Unlabeled' if it snuck in)
                     valid_labels = [l for l in found_labels if l != 'Unlabeled']
                     
                     # NEW LOGIC: Require >50% overlap
                     if len(valid_labels) >= (0.5 * row_dur):
                         # Majority vote
                         final_label = Counter(valid_labels).most_common(1)[0][0]
                         dff.at[idx, 'manual_label'] = final_label
                     else:
                          # Explicitly Unlabeled if threshold not met
                          dff.at[idx, 'manual_label'] = 'Unlabeled'
                     
        except Exception as e:
            # Fallback or ignore errors in optimization to allow other rows to proceed
            print(f"Error applying labels for {f_base}: {e}")
            continue

    return dff