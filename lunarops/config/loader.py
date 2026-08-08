"""YAML config loader: variables, substitution, loops, program sequence.

A YAML run config mirrors a GROOPS scenario file::

    variables:
      dataDir: /data/llr
      ephemeris: "{dataDir}/inpop21a.dat"

    globals:                      # shared class configs, built once per run
      ephemerides:   {type: calceph, file: "{ephemeris}", lunarRelativisticScaleConvention: alreadyScaled}
      earthRotation: {type: iersC04, file: "{dataDir}/eopc04.1962-now"}

    programs:
      - program: NormalPointsConvert
        inputFilesNormalPoints: ["{dataDir}/crd"]
        outputFileNormalPoints: "{dataDir}/normalPoints.txt.gz"
      - program: LlrResiduals
        loop: {variable: station, values: [APOLLO, GRASSE, WETTZELL]}
        inputFilesNormalPoints: ["{dataDir}/normalPoints.txt.gz"]
        outputFileObservationResults: "oc_{station}.txt.gz"

``{name}`` placeholders are substituted recursively from ``variables`` (and
from loop variables inside a loop body).  CLI ``--set name=value`` overrides
entries in ``variables``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config_file(path) -> dict:
    path = Path(path).expanduser()
    if path.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(f"LunarOps configuration files must use .yml or .yaml: {path}")
    text = path.read_text(encoding="utf-8")
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    unknown = set(data) - {"variables", "globals", "programs"}
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}")
    for key in ("variables", "globals"):
        if data.get(key) is not None and not isinstance(data[key], Mapping):
            raise TypeError(f"Top-level {key!r} section must be a mapping.")
    if data.get("programs") is not None and not isinstance(data["programs"], list):
        raise TypeError("Top-level 'programs' section must be a list.")
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
    string, and allows small YAML lists or mappings for batch scripts.
    """
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None

    # Use the YAML scalar grammar for quoted strings and containers.  This keeps
    # ``--set x="001"`` as the string ``001`` while ``--set x=1`` is int.
    if text.startswith(("'", '"', "[", "{")):
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
    if not isinstance(config, Mapping):
        raise TypeError("Run configuration must be a mapping.")
    variables_raw = config.get("variables") or {}
    globals_raw = config.get("globals") or {}
    programs = config.get("programs") or []
    if not isinstance(variables_raw, Mapping):
        raise TypeError("Top-level 'variables' section must be a mapping.")
    if not isinstance(globals_raw, Mapping):
        raise TypeError("Top-level 'globals' section must be a mapping.")
    if not isinstance(programs, list):
        raise TypeError("Top-level 'programs' section must be a list.")

    variables = dict(variables_raw)
    variables.update(overrides or {})

    for entry in programs:
        if not isinstance(entry, dict) or "program" not in entry:
            raise ValueError(f"Each program entry needs a 'program' key: {entry!r}")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError("Program 'enabled' must be a YAML boolean.")
        if not enabled:
            continue
        if not isinstance(entry["program"], str) or not entry["program"].strip():
            raise ValueError("Program names must be non-empty strings.")
        name = entry["program"].strip()
        loop = entry.get("loop")
        body = {k: v for k, v in entry.items() if k not in ("program", "loop", "enabled")}
        if loop is not None:
            if not isinstance(loop, Mapping):
                raise TypeError("Program loop must be a mapping.")
            if set(loop) != {"variable", "values"}:
                raise ValueError("Program loop requires exactly 'variable' and 'values'.")
            loop_var = loop["variable"]
            loop_values = loop["values"]
            if not isinstance(loop_var, str) or _PLACEHOLDER.fullmatch("{" + loop_var + "}") is None:
                raise ValueError("Program loop variable must be a valid identifier.")
            if not isinstance(loop_values, list) or not loop_values:
                raise ValueError("Program loop values must be a non-empty list.")
            for loop_value in loop_values:
                local_vars = dict(variables)
                local_vars[loop_var] = loop_value
                yield (
                    name,
                    substitute(body, local_vars),
                    substitute(globals_raw, local_vars),
                )
        else:
            yield name, substitute(body, variables), substitute(globals_raw, variables)
