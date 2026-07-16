"""Versioned LLROPS normal-point interchange files."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from llrops.base.epoch import Epoch

from .npt import NptDataset, NptRecord


SCHEMA_NAME = "llrops.normal_points"
SCHEMA_VERSION = 1
SUPPORTED_SUFFIXES = (
    ".llnpt",
    ".llnpt.gz",
    ".llrops.jsonl",
    ".llrops.jsonl.gz",
)


def is_llrops_npt_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _open_text(path: Path, mode: str):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def _record_mapping(record: NptRecord) -> dict[str, object]:
    return {
        "record_type": "normal_point",
        "station_name": record.station_name,
        "reflector_name": record.reflector_name,
        "transmit_epoch": record.transmit_epoch.to_dict(),
        "round_trip_time_s": record.round_trip_time_s,
        "uncertainty_two_way_s": record.uncertainty_two_way_s,
        "pressure_hpa": record.pressure_hpa,
        "temperature_k": record.temperature_k,
        "humidity_percent": record.humidity_percent,
        "wavelength_nm": record.wavelength_nm,
        "index": record.index,
        "station_code": record.station_code,
        "reflector_code": record.reflector_code,
    }


def _record_from_mapping(data: Mapping[str, object], *, line_number: int) -> NptRecord:
    if data.get("record_type") != "normal_point":
        raise ValueError(
            f"line {line_number}: expected record_type='normal_point'."
        )
    try:
        epoch_data = data["transmit_epoch"]
        if not isinstance(epoch_data, Mapping):
            raise TypeError("transmit_epoch must be an object")
        return NptRecord(
            station_name=data["station_name"],
            reflector_name=data["reflector_name"],
            transmit_epoch=Epoch.from_dict(epoch_data),
            round_trip_time_s=data["round_trip_time_s"],
            uncertainty_two_way_s=data["uncertainty_two_way_s"],
            pressure_hpa=data["pressure_hpa"],
            temperature_k=data["temperature_k"],
            humidity_percent=data["humidity_percent"],
            wavelength_nm=data["wavelength_nm"],
            index=data.get("index", 0),
            station_code=data.get("station_code"),
            reflector_code=data.get("reflector_code"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: invalid normal-point record: {exc}") from exc


def write_llrops_npt(dataset: NptDataset, path: str | Path) -> Path:
    """Atomically write a canonical LLROPS JSONL normal-point dataset."""
    if not isinstance(dataset, NptDataset):
        raise TypeError("dataset must be an NptDataset.")
    target = Path(path).expanduser()
    if not is_llrops_npt_file(target):
        raise ValueError(
            "LLROPS normal-point files must use .llnpt, .llnpt.gz, "
            ".llrops.jsonl, or .llrops.jsonl.gz."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.gz" if target.name.lower().endswith(".gz") else ".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    header = {
        "record_type": "header",
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "dataset_name": dataset.name,
        "n_records": len(dataset.records),
        "n_input_records": int(dataset.n_input_records),
        "n_invalid_records": int(dataset.n_invalid_records),
    }
    try:
        with _open_text(temporary, "wt") as stream:
            stream.write(json.dumps(header, ensure_ascii=True, allow_nan=False))
            stream.write("\n")
            for record in dataset.records:
                stream.write(
                    json.dumps(
                        _record_mapping(record),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                )
                stream.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_llrops_npt(path: str | Path) -> NptDataset:
    """Read and validate a canonical LLROPS JSONL normal-point dataset."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"LLROPS normal-point file not found: {source}")
    records: list[NptRecord] = []
    header = None
    with _open_text(source, "rt") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(data, Mapping):
                raise ValueError(f"line {line_number}: expected a JSON object.")
            if header is None:
                header = data
                if header.get("record_type") != "header":
                    raise ValueError("The first nonblank line must be an LLROPS header.")
                if header.get("schema") != SCHEMA_NAME:
                    raise ValueError(f"Unsupported LLROPS schema {header.get('schema')!r}.")
                if header.get("version") != SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported LLROPS normal-point schema version "
                        f"{header.get('version')!r}; expected {SCHEMA_VERSION}."
                    )
                continue
            records.append(_record_from_mapping(data, line_number=line_number))

    if header is None:
        raise ValueError(f"LLROPS normal-point file is empty: {source}")
    expected_records = int(header.get("n_records", len(records)))
    if expected_records != len(records):
        raise ValueError(
            f"LLROPS header declares {expected_records} records, found {len(records)}."
        )
    return NptDataset(
        records=records,
        name=header.get("dataset_name") or source.stem,
        n_input_records=int(header.get("n_input_records", len(records))),
        n_invalid_records=int(header.get("n_invalid_records", 0)),
    )


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_SUFFIXES",
    "is_llrops_npt_file",
    "read_llrops_npt",
    "write_llrops_npt",
]
