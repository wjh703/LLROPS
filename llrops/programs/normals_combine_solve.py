"""Combine normal-equation files by parameter name and solve them."""
from __future__ import annotations

import json

from llrops.config.context import RunContext
from llrops.programs.registry import program


@program("NormalsCombineSolve")
def normals_combine_solve(config: dict, context: RunContext):
    import numpy as np

    from llrops.fileio.normal_equations import NormalEquations

    stems = config.get("inputNormals") or []
    if not stems:
        raise ValueError("inputNormals is required")
    total = NormalEquations.load(context.resolve_path(stems[0]))
    for stem in stems[1:]:
        total = total.add(NormalEquations.load(context.resolve_path(stem)))

    x, Qxx, sigma0 = total.solve()
    solution = {
        "sigma0_post": sigma0,
        "obs_count": total.obs_count,
        "parameters": [
            {
                "name": str(name),
                "estimate": float(xi),
                "cofactor_sigma": float(np.sqrt(Qxx[i, i])),
                "formal_sigma": (
                    None
                    if sigma0 is None
                    else float(sigma0 * np.sqrt(Qxx[i, i]))
                ),
            }
            for i, (name, xi) in enumerate(zip(total.parameter_names, x))
        ],
    }
    if config.get("outputSolutionJson"):
        path = context.resolve_path(config["outputSolutionJson"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(solution, indent=2), encoding="utf-8")
    if config.get("outputNormals"):
        total.save(context.resolve_path(config["outputNormals"]))
    sigma0_text = "undefined" if sigma0 is None else f"{sigma0:.4f}"
    print(
        f"[NormalsCombineSolve] solved {len(total.parameter_names)} parameters, "
        f"sigma0={sigma0_text}"
    )
    return solution


__all__ = ["normals_combine_solve"]
