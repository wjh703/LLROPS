"""llrops — GROOPS-inspired Lunar Laser Ranging processing system.

Layers (see docs/ARCHITECTURE.md):
  config/    class registry, config loader, run context
  base/      constants, unified epochs, validation, parameter names
  files/     MINI/CRD/catalog/normal-equation/table IO
  classes/   polymorphic model classes (ephemerides, frames, delays,
             displacements, parametrizations, observation equations)
  estimation/  generic adjustment and normal-equation accumulation
  programs/  one task per program, driven by a run config
"""
__version__ = "35.0.0.dev0"
