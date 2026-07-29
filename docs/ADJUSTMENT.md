# Adjustment reference

`LlrAdjustment` is the nonlinear estimator for reflector coordinates and other
registered parameter blocks. The supplied example is
`configs/llrops_reflector_bias_adjustment_detailed.yml`.

## Model

For each normal point, the linearized equation is

```text
l = A * delta_x + B * delta_bias + v
```

Input sigma remains the within-group relative precision. VCE estimates a
multiplicative group scale; it does not rewrite individual input sigmas. Bias
parameters belong in the function model and must be estimated before using
residual scatter to interpret a variance component.

The current stochastic model uses one Helmert VCE component per configured
station/equipment-era group. Each retained observation must match exactly one
component. Sparse groups fall back to a longer calibration window or a parent
group rather than producing an unconstrained fine-scale component.

## Iteration order

```text
linearize geometry
  -> initialize Bias and MAD group scales
  -> fixed-geometry IGGIII + Helmert VCE block
  -> apply parameter correction
  -> relinearize
```

IGGIII uses `k0 = 1.5` and `k1 = 6.0`. Zero-weight observations are excluded
from the current effective redundancy. The stochastic block is bounded by the
configured maximum iterations and variance-ratio limits. The final stochastic
model is frozen, final residuals/flags are computed, and the state is solved
once more for the reported result.

## Parametrizations

```yaml
parametrization:
  - type: reflectorPosition
    reflectors: [APOLLO11, APOLLO14, APOLLO15]
  - type: stationRangeBias
    per: station
```

`reflectorPosition` estimates three PA-frame coordinates per reflector.
`stationRangeBias` estimates additive one-way metres either per station or per
explicit `station+interval` block. Overlapping intervals activate multiple
columns and can make a system singular.

`globals.rangeBias` is different: it is a deterministic two-way forward-model
correction, for example `{type: inpop21a}` or `{type: none}`. Do not use it as a
substitute for estimated Bias parameters.

A custom forward table uses centimetres of two-way light distance and the same
left-closed, right-open interval convention:

```yaml
globals:
  rangeBias:
    type: table
    biases:
      - station: APOLLO
        start: 2020-01-01
        end: 2021-01-01
        biasCm: 1.25
```

## Stages and output

Stages can solve blocks separately and then jointly. The standard pattern is
reflector, Bias, and joint stages, with a smaller update factor available for a
known oscillating joint solution. Parameter convergence is evaluated per block.

`outputFileAdjustmentReport` contains termination status, iteration history,
stochastic diagnostics, parameter precision/correlation, and observation
summaries. `outputFileAdjustmentState` is the fingerprinted restart product.
`outputFileSolution`, `outputFileCovariance`, and
`outputFileNormalEquations` publish the numerical products independently.

At minimum, inspect parameter updates, Bias uncertainty, group redundancy and
scale, residual distributions, robust-factor counts, rank/condition diagnostics,
and sensitivity to interval boundaries. A robust weight is not by itself a
scientific data-rejection decision.

## References

The implementation follows the LLR/SLR use of robust systematic-error
estimation and variance components described by Sahin, Cross and Sellers
(1992), Yang et al. (1999), Teunissen and Amiri-Simkooei (2008), and Li et al.
(2024). These references motivate separating Bias from the stochastic model,
using multiplicative group scales, and avoiding fine VCE groups without enough
redundancy.
