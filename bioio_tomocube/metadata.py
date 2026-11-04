from typing import Any, Dict

import numpy as np


def _plain(value: Any) -> Any:
    """h5py attribute value to a plain Python scalar, string or list."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _plain(value.flat[0]) if value.size == 1 else [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def group_attrs(obj: Any) -> Dict[str, Any]:
    return {key: _plain(value) for key, value in obj.attrs.items()}


def attr(obj: Any, key: str) -> Any:
    value = obj.attrs.get(key)
    return None if value is None else _plain(value)
