"""Centralised configuration loader for the panda_control package.

Configuration values are read from ``config/default.yaml`` at import
time and cached for the lifetime of the process.  Every module in the
package can retrieve a setting with a single call::

    from panda_control.config import get as cfg

    width = cfg("camera", "width")       # -> 640
    eta   = cfg("robot", "apf", "eta")   # -> 0.005

The loader searches for the YAML file relative to *this* file's
directory so it works regardless of the working directory.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

_CONFIG: dict = {}


def _load() -> dict:
    """Read and cache the default YAML configuration file."""
    global _CONFIG
    if _CONFIG:
        return _CONFIG
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config", "default.yaml")
    with open(path, "r") as fh:
        _CONFIG = yaml.safe_load(fh)
    return _CONFIG


def get(*keys: str) -> Any:
    """Walk the configuration tree by *keys* and return the leaf value.

    Parameters
    ----------
    *keys : str
        Sequence of nested keys, e.g. ``get("robot", "apf", "eta")``.

    Returns
    -------
    Any
        The value stored at the requested path.

    Raises
    ------
    KeyError
        If any key in the chain does not exist.
    """
    cfg = _load()
    for k in keys:
        cfg = cfg[k]
    return cfg
