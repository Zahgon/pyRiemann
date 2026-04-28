"""Utility functions for dataset management."""

import os
from pathlib import Path


def get_data_path(dataset_name=None):
    """Get the base path for pyRiemann datasets.

    Resolves the root directory for dataset storage. Checks the
    ``PYRIEMANN_DATA_PATH`` environment variable first, then falls back
    to ``~/pyriemann_data``.

    Parameters
    ----------
    dataset_name : str | None, default=None
        Optional dataset subdirectory name. When provided, it is appended
        to the base path.

    Returns
    -------
    path : str
        Absolute path to the dataset directory.
    """
    pass
