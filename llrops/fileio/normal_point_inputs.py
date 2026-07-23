"""Source-independent normal-point input discovery and dispatch."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .llrops_normal_point_file import SUPPORTED_SUFFIXES as LLROPS_NPT_SUFFIXES


SUPPORTED_MINI_SUFFIXES = (
    ".dat",
    ".mini",
    ".dat.txt",
    ".mini.txt",
    ".dat.gz",
    ".mini.gz",
)
SUPPORTED_CRD_SUFFIXES = (
    ".npt",
    ".crd",
    ".frd",
    ".npt.gz",
    ".crd.gz",
    ".frd.gz",
)
SUPPORTED_NORMAL_POINT_SUFFIXES = (
    *SUPPORTED_MINI_SUFFIXES,
    *SUPPORTED_CRD_SUFFIXES,
    *LLROPS_NPT_SUFFIXES,
)


def _has_suffix(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def is_mini_file(path: Path) -> bool:
    return _has_suffix(path, SUPPORTED_MINI_SUFFIXES)


def is_crd_file(path: Path) -> bool:
    return _has_suffix(path, SUPPORTED_CRD_SUFFIXES)


def is_normal_point_file(path: Path) -> bool:
    return _has_suffix(path, SUPPORTED_NORMAL_POINT_SUFFIXES)


def iter_input_files(path: Path) -> Iterable[Path]:
    path = path.expanduser()
    if path.is_file():
        if is_normal_point_file(path):
            yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and is_normal_point_file(child):
                yield child
        return
    raise FileNotFoundError(f"Input path does not exist: {path}")


def resolve_normal_point_inputs(inputs) -> list[Path]:
    """Expand input files/directories without converting or writing files."""
    values = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    seen: dict[Path, None] = {}
    for value in values:
        for path in iter_input_files(Path(str(value))):
            seen.setdefault(path.resolve())
    return sorted(seen)


def read_normal_points(path: str | Path, *, mini_io_log_path=None):
    """Read MINI, CRD, or LLROPS JSONL directly into an NptDataset."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Normal-point input file not found: {source}")

    from .crd import looks_like_crd_file, parse_crd_file
    from .llrops_normal_point_file import is_llrops_npt_file, read_llrops_npt
    from .mini import looks_like_mini_file, parse_mini_file

    if is_llrops_npt_file(source):
        return read_llrops_npt(source)
    if is_crd_file(source) or looks_like_crd_file(source):
        return parse_crd_file(source)
    if is_mini_file(source) or looks_like_mini_file(source):
        return parse_mini_file(source, mini_io_log_path=mini_io_log_path)
    raise ValueError(f"Unsupported normal-point input format: {source}")


__all__ = [
    "LLROPS_NPT_SUFFIXES",
    "SUPPORTED_CRD_SUFFIXES",
    "SUPPORTED_MINI_SUFFIXES",
    "SUPPORTED_NORMAL_POINT_SUFFIXES",
    "is_crd_file",
    "is_mini_file",
    "is_normal_point_file",
    "iter_input_files",
    "read_normal_points",
    "resolve_normal_point_inputs",
]
