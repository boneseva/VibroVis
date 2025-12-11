"""
Callbacks package - exports main functions for app initialization.
"""
from .callbacks_constants import set_initial_data
from .callbacks import register_callbacks

__all__ = ['set_initial_data', 'register_callbacks']

