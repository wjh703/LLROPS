"""Program registry (GROOPS ``programs`` analogue).

A program is one self-contained task: convert files, compute residuals,
build/solve normal equations, run an adjustment.  Programs are executed in
sequence from a run config by :mod:`llrops.cli` and share heavyweight objects
through the :class:`~llrops.config.context.RunContext`.
"""
from __future__ import annotations

from typing import Callable, Dict

from llrops.config.context import RunContext

ProgramFunc = Callable[[dict, RunContext], object]

_PROGRAMS: Dict[str, ProgramFunc] = {}


def program(name: str):
    def _wrap(func: ProgramFunc) -> ProgramFunc:
        _PROGRAMS[name.lower()] = func
        func.program_name = name
        return func

    return _wrap


def run_program(name: str, config: dict, context: RunContext):
    try:
        func = _PROGRAMS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown program {name!r}. Available: {sorted(p for p in _PROGRAMS)}"
        ) from None
    return func(config, context)


def available_programs():
    return sorted(func.program_name for func in _PROGRAMS.values())
