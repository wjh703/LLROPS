"""Config file loader: variables, substitution, loops, program sequence.

A run config (YAML or JSON) mirrors a GROOPS scenario file::

    variables:
      dataDir: /data/llr
      ephemeris: "{dataDir}/inpop21a.dat"

    globals:                      # shared class configs, built once per run
      ephemerides:   {type: calceph, file: "{ephemeris}"}
      earthRotation: {type: iersC04, file: "{dataDir}/eopc04.1962-now"}

    programs:
      - program: NormalPointsToLlrops
        inputNormalPoints: ["{dataDir}/crd"]
        outputFile: "{dataDir}/normal-points.llnpt.gz"
      - program: LlrResiduals
        loop: {variable: station, values: [APOLLO, GRASSE, WETTZELL]}
        inputNormalPoints: ["{dataDir}/normal-points.llnpt.gz"]
        outputCsv: "oc_{station}.csv"

``{name}`` placeholders are substituted recursively from ``variables`` (and
from loop variables inside a loop body).  CLI ``--set name=value`` overrides
entries in ``variables``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config_file(path) -> dict:
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return data


def substitute(value: Any, variables: Dict[str, Any]) -> Any:
    """Recursively substitute ``{name}`` placeholders in strings."""
    if isinstance(value, str):
        # Full-string placeholder keeps the native type of the variable.
        m = _PLACEHOLDER.fullmatch(value)
        if m and m.group(1) in variables:
            return variables[m.group(1)]

        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name not in variables:
                raise KeyError(f"Undefined config variable {{{name}}} in {value!r}")
            return str(variables[name])

        return _PLACEHOLDER.sub(_sub, value)
    if isinstance(value, list):
        return [substitute(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, variables) for k, v in value.items()}
    return value


def _parse_set_value(value: str) -> Any:
    """Parse one CLI ``--set`` value into a native config scalar/container.

    The command line has no type system, but config substitution keeps native
    variable types when a full string is ``{name}``.  Parsing here prevents
    common surprises such as ``--set enabled=false`` being treated as a truthy
    string, and allows small JSON/YAML lists or mappings for batch scripts.
    """
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None

    # Prefer JSON/YAML for quoted strings and containers.  This keeps
    # ``--set x="001"`` as the string ``001`` while ``--set x=1`` is int.
    if text.startswith(("'", '"', "[", "{")):
        try:
            return json.loads(text)
        except Exception:
            try:
                import yaml

                parsed = yaml.safe_load(text)
                if isinstance(parsed, (str, int, float, bool, list, dict)) or parsed is None:
                    return parsed
            except Exception:
                pass

    if re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:(?:[0-9]+\.[0-9]*)|(?:\.[0-9]+)|(?:[0-9]+))(?:[eE][+-]?[0-9]+)?", text):
        try:
            return float(text)
        except ValueError:
            pass

    return value


def parse_set_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects name=value, got {pair!r}")
        name, value = pair.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--set expects a non-empty variable name, got {pair!r}")
        overrides[name] = _parse_set_value(value)
    return overrides


def iter_program_calls(config: dict, overrides: Dict[str, Any] | None = None) -> Iterator[Tuple[str, dict, dict]]:
    """Yield ``(program_name, resolved_program_config, resolved_globals)``.

    Loop entries are expanded; ``enabled: false`` entries are skipped.
    """
    variables = dict(config.get("variables") or {})
    variables.update(overrides or {})

    globals_raw = config.get("globals") or {}

    for entry in config.get("programs") or []:
        if not isinstance(entry, dict) or "program" not in entry:
            raise ValueError(f"Each program entry needs a 'program' key: {entry!r}")
        if entry.get("enabled", True) in (False, "false", "no", 0):
            continue
        name = str(entry["program"])
        loop = entry.get("loop")
        body = {k: v for k, v in entry.items() if k not in ("program", "loop", "enabled")}
        if loop:
            loop_var = str(loop["variable"])
            for loop_value in loop["values"]:
                local_vars = dict(variables)
                local_vars[loop_var] = loop_value
                yield name, substitute(body, local_vars), substitute(globals_raw, local_vars)
        else:
            yield name, substitute(body, variables), substitute(globals_raw, variables)
