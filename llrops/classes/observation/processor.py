"""Dataset orchestration for LLR measurement evaluation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from llrops.fileio.normal_points import NptDataset

from .equations import ObservationEquation, ObservationOutputLevel
from .measurement import LlrMeasurement
from .resolver import CatalogSelection, ObservationResolver, ResolvedObservation

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except ImportError:  # pragma: no cover
    _tqdm = None


@dataclass(frozen=True, slots=True)
class ObservationProcessingOptions:
    station_name: str | None = None
    reflector_name: str | None = None
    min_elevation_deg: float = 0.0
    include_reflector_position_partial: bool = False
    show_progress: bool = False
    progress_description: str | None = None

    def __post_init__(self) -> None:
        min_elevation = float(self.min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
        object.__setattr__(self, "min_elevation_deg", min_elevation)

    @property
    def catalog_selection(self) -> CatalogSelection:
        return CatalogSelection(self.station_name, self.reflector_name)

    def with_progress(
        self,
        description: str | None,
        *,
        enabled: bool | None = None,
    ) -> "ObservationProcessingOptions":
        return replace(
            self,
            progress_description=description,
            show_progress=self.show_progress if enabled is None else bool(enabled),
        )


class LlrObservationProcessor:
    def __init__(self, resolver: ObservationResolver, measurement: LlrMeasurement) -> None:
        self.resolver = resolver
        self.model_state = resolver.model_state
        self.measurement = measurement

    @property
    def station_catalog(self):
        return self.resolver.station_catalog

    @property
    def reflector_catalog(self):
        return self.resolver.reflector_catalog

    @property
    def ephemeris_file(self) -> str:
        return str(self.measurement.ephemeris.source_file)

    def _with_progress(
        self,
        observations: Iterable[ResolvedObservation],
        *,
        total: int,
        options: ObservationProcessingOptions,
    ) -> Iterable[ResolvedObservation]:
        if not options.show_progress or total <= 0:
            return observations
        description = options.progress_description or "LLR observations"
        if _tqdm is not None:
            return _tqdm(
                observations,
                total=total,
                desc=description,
                unit="np",
                dynamic_ncols=True,
                smoothing=0.1,
            )

        def generator():
            for index, item in enumerate(observations, start=1):
                print(
                    f"\r{description}: {index}/{total}",
                    end="" if index < total else "\n",
                    flush=True,
                )
                yield item

        return generator()

    def _observations(
        self,
        dataset: NptDataset,
        options: ObservationProcessingOptions,
    ) -> Iterable[ResolvedObservation]:
        observations = self.resolver.validate(
            dataset.records, options.catalog_selection
        )
        return self._with_progress(
            observations,
            total=len(observations),
            options=options,
        )

    def equations(
        self,
        dataset: NptDataset,
        *,
        options: ObservationProcessingOptions | None = None,
    ) -> list[ObservationEquation]:
        options = options or ObservationProcessingOptions()
        return [
            self.measurement.evaluate(
                observation,
                min_elevation_deg=options.min_elevation_deg,
                include_reflector_position_partial=(
                    options.include_reflector_position_partial
                ),
            ).equation
            for observation in self._observations(dataset, options)
        ]

    def rows(
        self,
        dataset: NptDataset,
        *,
        options: ObservationProcessingOptions | None = None,
        level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
    ) -> list[dict[str, object]]:
        options = options or ObservationProcessingOptions()
        rows: list[dict[str, object]] = []
        for observation in self._observations(dataset, options):
            row = self.measurement.evaluate(
                observation,
                min_elevation_deg=options.min_elevation_deg,
                include_reflector_position_partial=(
                    options.include_reflector_position_partial
                ),
                output_level=level,
            ).row
            if row is None:
                raise RuntimeError("Measurement row was not generated.")
            rows.append(row)
        return rows


__all__ = ["LlrObservationProcessor", "ObservationProcessingOptions"]
