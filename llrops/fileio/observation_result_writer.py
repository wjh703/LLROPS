"""CSV/JSON serialization of typed LLR observation results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from llrops.classes.observation import LlrObservationResult, ObservationOutputLevel


def _rows(
    results: Iterable[LlrObservationResult | Mapping[str, object]],
    level: ObservationOutputLevel | str,
) -> list[dict]:
    parsed_level = ObservationOutputLevel.parse(level)
    rows: list[dict] = []
    for result in results:
        if isinstance(result, Mapping):
            rows.append(dict(result))
        else:
            rows.append(result.to_row(parsed_level))
    return rows


def write_csv(
    results: Sequence[LlrObservationResult | Mapping[str, object]],
    path,
    *,
    level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
) -> None:
    rows = _rows(results, level)
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


def write_json(
    results: Sequence[LlrObservationResult | Mapping[str, object]],
    path,
    *,
    level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_rows(results, level), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv_grouped(
    results_by_source: Mapping[str, Sequence[LlrObservationResult | Mapping[str, object]]],
    path,
    *,
    level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
) -> None:
    merged = [
        result
        for source_results in results_by_source.values()
        for result in source_results
    ]
    write_csv(merged, path, level=level)


def write_json_grouped(
    results_by_source: Mapping[str, Sequence[LlrObservationResult | Mapping[str, object]]],
    path,
    *,
    level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
) -> None:
    parsed_level = ObservationOutputLevel.parse(level)
    payload = {
        str(source): _rows(results, parsed_level)
        for source, results in results_by_source.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["write_csv", "write_csv_grouped", "write_json", "write_json_grouped"]
