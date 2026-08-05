"""Typed text reports and restart state using a YAML scalar grammar."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .archive import atomic_text_writer, open_text_reader, parse_header


def plain_data(value):
    if is_dataclass(value) and not isinstance(value, type):
        return plain_data(asdict(value))
    if isinstance(value, np.ndarray):
        return [plain_data(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return plain_data(value.item())
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if not normalized_key:
                raise ValueError("Structured LunarOps text rejects empty mapping keys.")
            if normalized_key in result:
                raise ValueError(f"Structured LunarOps text has colliding mapping key {normalized_key!r}.")
            result[normalized_key] = plain_data(item)
        return result
    if isinstance(value, set):
        return sorted((plain_data(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Structured LunarOps text rejects non-finite floats.")
        return value
    raise TypeError(f"Structured LunarOps text cannot encode {type(value).__name__} objects.")


def write_structured_text(
    path: str | Path,
    artifact_type: str,
    payload: Mapping[str, object],
) -> Path:
    target = Path(path).expanduser()
    with atomic_text_writer(target, artifact_type) as stream:
        yaml.safe_dump(
            plain_data(payload),
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    return target


def read_structured_text(path: str | Path, artifact_type: str) -> dict[str, object]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, artifact_type)
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"LunarOps {artifact_type} payload must be a mapping: {source}")
    return payload


__all__ = ["plain_data", "read_structured_text", "write_structured_text"]
