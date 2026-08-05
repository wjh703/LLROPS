#!/usr/bin/env python3
"""Verify that built distributions contain the native IERS payload."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


SOURCE_NAMES = {
    path.name for path in Path(__file__).parents[1].joinpath("lunarops", "_external", "iers2010", "src").glob("*.F")
}


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(roots) != 1:
        raise SystemExit(f"{path}: expected one sdist root, found {sorted(roots)}")
    root = next(iter(roots))
    required = {
        f"{root}/lunarops/_external/iers2010/LICENSE",
        *(f"{root}/lunarops/_external/iers2010/src/{name}" for name in SOURCE_NAMES),
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    required = {
        "lunarops/_external/iers2010/LICENSE",
        *(f"lunarops/_external/iers2010/src/{name}" for name in SOURCE_NAMES),
    }
    missing = sorted(required - names)
    extensions = sorted(name for name in names if name.startswith("lunarops/_iers2010") and name.endswith(".so"))
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    if not extensions:
        raise SystemExit(f"{path}: missing compiled lunarops/_iers2010 extension")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.archives:
        if path.name.endswith(".tar.gz"):
            _check_sdist(path)
        elif path.name.endswith(".whl"):
            _check_wheel(path)
        else:
            raise SystemExit(f"unsupported archive: {path}")
        print(f"verified {path}")


if __name__ == "__main__":
    main()
