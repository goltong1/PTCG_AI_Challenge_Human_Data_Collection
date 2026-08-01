from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import IntEnum
from typing import Any


def to_plain_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, IntEnum):
        return int(obj)
    if isinstance(obj, list):
        return [to_plain_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_plain_dict(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return {field.name: to_plain_dict(getattr(obj, field.name)) for field in fields(obj)}
    return obj
