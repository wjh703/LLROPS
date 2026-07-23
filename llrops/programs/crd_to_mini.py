"""Convert CRD normal-point files to MINI files."""
from __future__ import annotations

from pathlib import Path
from typing import List

from llrops.config.context import RunContext
from llrops.programs.registry import program


@program("CrdToMini")
def crd_to_mini(config: dict, context: RunContext):
    from llrops.fileio.crd import convert_crd_to_mini
    from llrops.fileio.normal_point_inputs import is_crd_file, iter_input_files

    out_dir = context.resolve_path(config["outputDir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    converted: List[str] = []
    input_crd = config["inputCrd"]
    for item in input_crd if isinstance(input_crd, list) else [input_crd]:
        for path in iter_input_files(Path(str(item))):
            if not is_crd_file(path):
                continue
            mini_path = out_dir / (path.stem + ".mini")
            convert_crd_to_mini(path, mini_path)
            converted.append(str(mini_path))
    print(f"[CrdToMini] converted {len(converted)} file(s) -> {out_dir}")
    return converted


__all__ = ["crd_to_mini"]
