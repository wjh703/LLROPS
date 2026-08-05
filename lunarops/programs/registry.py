"""Declarative program and artifact registry.

The registry is deliberately strict.  A program owns a small, typed contract
and its callable is only entered after the contract has been checked.  This
keeps YAML scenario files inspectable in the same way as GROOPS program
chains, while leaving the scientific implementation in ordinary Python
functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, cast

from lunarops.config.context import RunContext

ProgramFunc = Callable[[dict, RunContext], object]


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    key: str
    artifact_type: str
    many: bool = False
    required: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    name: str
    summary: str
    inputs: tuple[ArtifactSlot, ...] = ()
    outputs: tuple[ArtifactSlot, ...] = ()
    required_keys: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        all_slots = (*self.inputs, *self.outputs)
        keys = [slot.key for slot in all_slots]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Program {self.name} declares duplicate artifact keys.")
        declared = [*keys, *self.required_keys, *self.optional_keys]
        duplicates = sorted({key for key in declared if declared.count(key) > 1})
        if duplicates:
            raise ValueError(f"Program {self.name} declares configuration keys more than once: {duplicates}")
        object.__setattr__(self, "required_keys", tuple(dict.fromkeys(self.required_keys)))
        object.__setattr__(self, "optional_keys", tuple(dict.fromkeys(self.optional_keys)))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if not self.name or not self.name.strip():
            raise ValueError("Program names must not be empty.")
        if not self.summary:
            raise ValueError(f"Program {self.name} needs a summary.")

    @property
    def slots(self) -> tuple[ArtifactSlot, ...]:
        return (*self.inputs, *self.outputs)

    @property
    def allowed_keys(self) -> frozenset[str]:
        return (
            frozenset(slot.key for slot in self.slots) | frozenset(self.required_keys) | frozenset(self.optional_keys)
        )

    def describe(self) -> dict[str, object]:
        def slot_data(slot: ArtifactSlot) -> dict[str, object]:
            return {
                "key": slot.key,
                "artifactType": slot.artifact_type,
                "many": slot.many,
                "required": slot.required,
                "description": slot.description,
            }

        return {
            "name": self.name,
            "summary": self.summary,
            "inputs": [slot_data(slot) for slot in self.inputs],
            "outputs": [slot_data(slot) for slot in self.outputs],
            "requiredKeys": list(self.required_keys),
            "optionalKeys": list(self.optional_keys),
        }


@dataclass(frozen=True, slots=True)
class RegisteredProgram:
    spec: ProgramSpec
    function: ProgramFunc


_PROGRAMS: Dict[str, RegisteredProgram] = {}


def program(
    spec_or_name: ProgramSpec | str,
    *,
    summary: str | None = None,
    inputs: Sequence[ArtifactSlot] = (),
    outputs: Sequence[ArtifactSlot] = (),
    required_keys: Sequence[str] = (),
    optional_keys: Sequence[str] = (),
):
    """Register a callable with a :class:`ProgramSpec`.

    ``@program(ProgramSpec(...))`` is the preferred spelling.  The keyword
    form is retained solely to keep declarations compact in program modules;
    it still creates a complete strict spec.
    """
    if isinstance(spec_or_name, ProgramSpec):
        spec = spec_or_name
    else:
        spec = ProgramSpec(
            name=str(spec_or_name),
            summary=summary or "",
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            required_keys=tuple(required_keys),
            optional_keys=tuple(optional_keys),
        )

    def _wrap(func: ProgramFunc) -> ProgramFunc:
        key = spec.name.casefold()
        if key in _PROGRAMS:
            raise RuntimeError(f"Program {spec.name!r} is already registered.")
        _PROGRAMS[key] = RegisteredProgram(spec, func)
        setattr(func, "program_name", spec.name)
        setattr(func, "program_spec", spec)
        return func

    return _wrap


def get_program(name: str) -> RegisteredProgram:
    try:
        return _PROGRAMS[str(name).casefold()]
    except KeyError:
        raise KeyError(f"Unknown program {name!r}. Available: {available_programs()}") from None


def program_specs() -> tuple[ProgramSpec, ...]:
    return tuple(
        sorted(
            (entry.spec for entry in _PROGRAMS.values()),
            key=lambda item: item.name.casefold(),
        )
    )


def validate_program_config(name: str, config: Mapping[str, object]) -> ProgramSpec:
    entry = get_program(name)
    spec = entry.spec
    if not isinstance(config, Mapping):
        raise TypeError(f"Program {spec.name} configuration must be a mapping.")
    unknown = set(config) - spec.allowed_keys
    if unknown:
        raise ValueError(f"{spec.name} has unknown configuration key(s): {sorted(str(key) for key in unknown)}")
    required = set(spec.required_keys)
    for slot in spec.slots:
        if slot.required:
            required.add(slot.key)
    missing = {key for key in required if key not in config or config[key] is None}
    if missing:
        raise ValueError(f"{spec.name} is missing required key(s): {sorted(missing)}")
    for slot in spec.slots:
        if slot.key not in config or config[slot.key] is None:
            continue
        value = config[slot.key]
        if slot.many:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{spec.name}.{slot.key} must be a list of paths.")
            if not value:
                raise ValueError(f"{spec.name}.{slot.key} must not be empty.")
        elif isinstance(value, (list, tuple, dict, set)):
            raise TypeError(f"{spec.name}.{slot.key} must be one path, not a collection.")
        values: Sequence[object] = cast(Sequence[object], value) if slot.many else [value]
        if any(not isinstance(item, (str, Path)) for item in values):
            raise TypeError(f"{spec.name}.{slot.key} paths must be strings.")
    return spec


_TEXT_ARTIFACT_HEADERS = {
    "NormalPointFile": "normalPoint",
    "ObservationResultFile": "observationResult",
    "ParameterVectorFile": "parameterVector",
    "AdjustmentStateFile": "adjustmentState",
    "AdjustmentReportFile": "adjustmentReport",
    "SolutionReportFile": "normalEquationSolutionReport",
    "NormalPointStatisticsFile": "normalPointStatistics",
    "ObservationResultStatisticsFile": "observationResultStatistics",
    "StationCatalogFile": "stationCatalog",
    "ReflectorCatalogFile": "reflectorCatalog",
    "ModelStateFile": "llrModelState",
    "ImportReportFile": "normalPointImportReport",
}
_GROUP_INFO_HEADERS = {
    "NormalEquationFile": "normalEquationInfo",
    "ObservationEquationFile": "observationEquationInfo",
    "CovarianceMatrixFile": "covarianceInfo",
}


def _slot_values(slot: ArtifactSlot, value: object) -> list[object]:
    if not slot.many:
        return [value]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Artifact slot {slot.key} must contain a sequence of paths.")
    return list(value)


def validate_program_artifacts(
    name: str,
    config: Mapping[str, object],
    context: RunContext,
    *,
    require_inputs: bool = True,
    available_artifacts: Mapping[Path, str] | None = None,
) -> None:
    """Validate paths and declared artifact types without running a program."""
    from lunarops.fileio.archive import is_binary_path, is_text_path, read_artifact_type

    spec = get_program(name).spec
    available = {Path(path).resolve(): artifact_type for path, artifact_type in (available_artifacts or {}).items()}
    input_keys = {slot.key for slot in spec.inputs}
    resolved_slots: dict[Path, str] = {}
    for slot in spec.slots:
        value = config.get(slot.key)
        if value is None:
            continue
        for raw_path in _slot_values(slot, value):
            path = context.resolve_path(raw_path)
            resolved = path.resolve()
            if resolved in resolved_slots:
                raise ValueError(f"{spec.name} reuses path {path} in both {resolved_slots[resolved]} and {slot.key}.")
            resolved_slots[resolved] = slot.key
            is_input = slot.key in input_keys
            generated_type = available.get(resolved)
            if is_input and generated_type is not None:
                if generated_type != slot.artifact_type:
                    raise ValueError(
                        f"{spec.name}.{slot.key} expects {slot.artifact_type}, but an earlier "
                        f"program produces {generated_type}: {path}"
                    )
                continue
            if is_input and require_inputs and not path.exists():
                raise FileNotFoundError(f"{spec.name}.{slot.key} does not exist: {path}")
            if slot.artifact_type == "ExternalNormalPointFile":
                continue
            if slot.artifact_type == "MatrixFile":
                if not (is_text_path(path) or is_binary_path(path)):
                    raise ValueError(f"{spec.name}.{slot.key} must use .txt[.gz] or .dat[.gz]: {path}")
                if is_input and require_inputs:
                    from lunarops.fileio.matrix import matrix_kind

                    matrix_kind(path)
                continue
            if slot.artifact_type in _TEXT_ARTIFACT_HEADERS:
                if not is_text_path(path):
                    raise ValueError(f"{spec.name}.{slot.key} must use .txt or .txt.gz: {path}")
                expected = _TEXT_ARTIFACT_HEADERS[slot.artifact_type]
                if is_input and require_inputs:
                    actual = read_artifact_type(path)
                    if expected is not None and actual != expected:
                        raise ValueError(f"{spec.name}.{slot.key} expects {expected!r}, found {actual!r}: {path}")
                continue
            if slot.artifact_type in _GROUP_INFO_HEADERS:
                if path.suffix:
                    raise ValueError(f"{spec.name}.{slot.key} file groups require an extensionless directory: {path}")
                if is_input and require_inputs:
                    if not path.is_dir():
                        raise ValueError(f"{spec.name}.{slot.key} must be a file-group directory: {path}")
                    actual = read_artifact_type(path / "info.txt")
                    expected = _GROUP_INFO_HEADERS[slot.artifact_type]
                    if actual != expected:
                        raise ValueError(f"{spec.name}.{slot.key} expects {expected!r}, found {actual!r}: {path}")
                continue
            raise RuntimeError(f"Program {spec.name} declares unknown artifact type {slot.artifact_type!r}.")


def run_program(name: str, config: dict, context: RunContext):
    entry = get_program(name)
    validate_program_config(entry.spec.name, config)
    validate_program_artifacts(entry.spec.name, config, context)
    return entry.function(config, context)


def available_programs() -> list[str]:
    return [spec.name for spec in program_specs()]


__all__ = [
    "ArtifactSlot",
    "ProgramSpec",
    "RegisteredProgram",
    "available_programs",
    "get_program",
    "program",
    "program_specs",
    "run_program",
    "validate_program_config",
    "validate_program_artifacts",
]
