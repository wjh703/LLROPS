# LLROPS v37 input-owned observation uncertainty

## Breaking changes

- Removed station-period uncertainty tables and their configurable model
  category.
- Removed program-level uncertainty selection. Every observation now uses the
  uncertainty stored in its input normal-point record.
- Removed table- and mode-specific uncertainty diagnostics from full O-C output.

## Input mapping

- MINI records contribute their two-way uncertainty field.
- CRD record 11 contributes bin RMS, converted from picoseconds to seconds.
- LLROPS normal-point files preserve the canonical `uncertainty_two_way_s`
  value without reclassification.
- The one-way range sigma used by estimation is always
  `0.5 * c * uncertainty_two_way_s` before adjustment quality control and
  robust/VCE weighting.
