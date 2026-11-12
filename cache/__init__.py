"""
Cache package — partitioned cache reader and adapter abstractions.
"""
from cache.reader import (
    ProjectMetadata,
    BaseCacheReader,
    PartitionedCacheReader,
    LegacyCacheReader,
    CacheReaderAdapter,
    get_default_adapter,
    get_project_metadata,
    load_location_model_data,
    get_last_source_used,
)

__all__ = [
    "ProjectMetadata",
    "BaseCacheReader",
    "PartitionedCacheReader",
    "LegacyCacheReader",
    "CacheReaderAdapter",
    "get_default_adapter",
    "get_project_metadata",
    "load_location_model_data",
    "get_last_source_used",
]
