"""CSV/JSON serialization of observation rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def write_csv(results: Sequence[Mapping[str, object]], path) -> None:
    rows = [dict(result) for result in results]
    if not rows:
        raise ValueError("No observation results to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(results: Sequence[Mapping[str, object]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([dict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_grouped(
    results_by_source: Mapping[str, Sequence[Mapping[str, object]]],
    path,
) -> None:
    write_csv(
        [row for rows in results_by_source.values() for row in rows],
        path,
    )


def write_json_grouped(
    results_by_source: Mapping[str, Sequence[Mapping[str, object]]],
    path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(source): [dict(result) for result in rows]
        for source, rows in results_by_source.items()
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["write_csv", "write_csv_grouped", "write_json", "write_json_grouped"]
