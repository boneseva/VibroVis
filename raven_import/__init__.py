"""
Raven Pro selection-table import for Citimon.

Imported annotations live in their own SQLite database (see ``store.py``) and are
never written into, or promoted to, the manual-label store.

Modules:
- config:     configurable assumptions about Raven's export format and import options
- parser:     decode + parse Raven .txt/.tsv tables, collapse duplicate views
- clip_index: recording / clip lookup built from Citimon partition data
- matcher:    resolve each selection to a recording and compute clip overlaps
- store:      persistent annotation database (batches, selections, links)
- service:    preview / commit orchestration used by the UI and CLI
"""
