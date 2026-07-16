"""MPI master-worker backend (port of the v24 ``run_*_mpi.py`` scripts).

Launch any config under MPI — nothing else changes::

    mpirun -n 16 python -m llrops run config.yml --mpi
    srun python -m llrops run config.yml --mpi        # SLURM, multi-node

Architecture (identical to v24, generalized):

rank 0
    * Parses the config, loads MINI files once, runs the program sequence.
    * Programs that support MPI dispatch dynamic chunks of already parsed
      ``NptRecord`` objects to worker ranks and gather results.
rank 1..N
    * Enter a lightweight generic worker loop before the program registry is
      imported.  Rank 0 broadcasts each immutable observation spec once
      (resolved configs, catalogs, parsed EOP arrays); later tasks carry only
      ``specId``.  Before task dispatch, rank 0 explicitly initializes every
      worker and waits for a ready response.  Each worker opens its own
      process-local CALCEPH handle once per spec and caches the resulting
      processor; initialization time is excluded from observation throughput.

Per-iteration state (updated reflector positions from the Gauss-Newton loop)
travels with every task as ``catalogState``, so relinearization works exactly
as in the serial path.

Task kinds:

``observation_results``
    chunk of NptRecords -> typed observation results (used by ``LlrResiduals`` and by the
    equation sources of ``LlrAdjustment`` / ``LlrNormalEquations``).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

TAG_TASK = 101
TAG_RESULT = 102
TAG_STOP = 103
TAG_BROADCAST_SPEC = 104
TAG_INITIALIZE_SPEC = 105
TAG_READY = 106

_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
    "uncertaintyModel",
)


def mpi_comm_world():
    """Import mpi4py lazily; raise a helpful error outside an MPI runtime."""
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "--mpi requires mpi4py and an MPI runtime; launch with "
            "mpirun/mpiexec/srun (e.g. 'srun python -m llrops run cfg.yml --mpi')."
        ) from exc
    return MPI


# ---------------------------------------------------------------------------
# observation spec: everything a worker needs to build its own observation processor
# ---------------------------------------------------------------------------

def _prepare_shared_resources(merged: dict, context) -> dict:
    """Build rank-0-only immutable resources for one observation spec.

    Text tables are parsed once on rank 0 and serialized as compact arrays.
    Process-local native handles (notably CALCEPH) are intentionally excluded;
    they cannot be safely pickled or shared between MPI processes.
    """
    resources: dict = {}
    earth_rotation_config = merged.get("earthRotation")
    if earth_rotation_config is not None:
        from llrops.classes.builders import ensure_registered
        from llrops.classes.frames import C04EarthOrientation
        from llrops.config.registry import normalize_class_config

        cfg = normalize_class_config(earth_rotation_config)
        if str(cfg["type"]).strip().lower() == "iersc04":
            ensure_registered()
            earth_orientation = context.create_class(
                "earthRotation",
                earth_rotation_config,
                cache=True,
            )
            if not isinstance(earth_orientation, C04EarthOrientation):
                raise TypeError(
                    "MPI earthRotation resource preparation expected "
                    "C04EarthOrientation."
                )
            context.shared["earthOrientation"] = earth_orientation
            resources["earthRotation"] = earth_orientation.to_mpi_payload()
    return resources


def make_observation_spec(config: dict, context, datasets) -> dict:
    """Resolve one picklable observation specification on rank 0.

    The complete specification is broadcast once to every worker and then
    referenced by ``specId`` in individual tasks.  This avoids repeatedly
    pickling catalogs and EOP arrays for every small chunk.
    """
    merged: dict = {}
    for category in _MODEL_CATEGORIES:
        value = context.class_config(category, config)
        if value is not None:
            merged[category] = value

    station_catalog = context.shared.get("stationCatalog")
    if station_catalog is None:
        from llrops.fileio.catalogs import load_station_catalog

        station_catalog = load_station_catalog(
            config.get("stationCatalog", context.global_class_configs.get("stationCatalog"))
        )
        context.shared["stationCatalog"] = station_catalog
    reflector_catalog = context.shared.get("reflectorCatalog")
    if reflector_catalog is None:
        from llrops.fileio.catalogs import load_reflector_catalog

        reflector_catalog = load_reflector_catalog(
            config.get("reflectorCatalog", context.global_class_configs.get("reflectorCatalog"))
        )
        context.shared["reflectorCatalog"] = reflector_catalog

    spec_id = f"{id(context)}-{hash(repr(sorted(merged.items(), key=lambda kv: kv[0])))}"
    return {
        "specId": spec_id,
        "programConfig": merged,
        "workingDir": str(context.working_dir),
        "stationCatalog": station_catalog,
        "reflectorCatalog": reflector_catalog,
        "sharedResources": _prepare_shared_resources(merged, context),
    }


def build_worker_processor(spec: dict, shared_class_cache: Optional[dict] = None):
    from llrops.config.context import RunContext
    from llrops.classes.builders import build_observation_processor

    context = RunContext(
        variables={},
        global_class_configs={},
        working_dir=spec.get("workingDir", "."),
    )
    context.shared["mpiResources"] = dict(spec.get("sharedResources") or {})
    if shared_class_cache is not None:
        # Reuse immutable/heavy classes (CALCEPH ephemeris, Earth-orientation source, immutable
        # observation components) without caching the mutable LlrObservationProcessor itself.
        context._cache = shared_class_cache
    return build_observation_processor(
        context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )


def _processor_for_task(cache: dict, spec: dict):
    """Return one cached processor per observation spec on each worker rank.

    The v31 worker rebuilt the light-time processor for every tiny MPI chunk.
    Heavy CALCEPH/EOP objects were cached, but the processor/resolver/reducer
    graph was still reconstructed for every task and then closed immediately.
    Keep the processor warm for the lifetime of the worker, matching the MPI
    architecture documented at the top of this module.
    """
    processor_key = ("processor", spec["specId"])
    if processor_key not in cache:
        shared_class_cache = cache.setdefault(("sharedClassCache", spec["specId"]), {})
        cache[processor_key] = build_worker_processor(spec, shared_class_cache)
    return cache[processor_key]


def _initialized_processor_for_task(cache: dict, spec: dict):
    """Return a processor created by the explicit initialization phase."""
    processor_key = ("processor", spec["specId"])
    try:
        return cache[processor_key]
    except KeyError:
        raise RuntimeError(
            f"MPI observation processor {spec['specId']!r} was not initialized "
            "before observation task dispatch."
        ) from None


def _close_cached_objects(cache: dict) -> None:
    def _walk(value):
        if isinstance(value, dict):
            for item in value.values():
                _walk(item)
            return
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    for obj in cache.values():
        _walk(obj)


def snapshot_catalog_state(context) -> dict:
    """Pickle-light snapshot of the mutable per-iteration model state."""
    import numpy as np

    reflectors = context.shared.get("reflectorCatalog") or {}
    return {
        "reflectorPositions": {
            str(key): [float(x) for x in np.asarray(rec.moon_fixed_xyz_m, dtype=float).reshape(3)]
            for key, rec in reflectors.items()
        }
    }


def _apply_catalog_state(processor, catalog_state: Optional[dict]) -> None:
    if not catalog_state:
        return
    import numpy as np
    from dataclasses import replace

    positions = catalog_state.get("reflectorPositions") or {}
    if positions:
        new_catalog = {}
        for key, rec in processor.reflector_catalog.items():
            if key in positions:
                rec = replace(rec, moon_fixed_xyz_m=np.asarray(positions[key], dtype=float))
            new_catalog[key] = rec
        processor.reflector_catalog = new_catalog


# ---------------------------------------------------------------------------
# task handlers (executed on worker ranks; rank 0 falls back to them serially)
# ---------------------------------------------------------------------------

def _observation_spec_for_payload(payload: dict, cache: dict) -> dict:
    spec_id = str(payload["specId"])
    try:
        return cache[("observationSpec", spec_id)]
    except KeyError:
        raise RuntimeError(
            f"MPI observation spec {spec_id!r} was not broadcast before task dispatch."
        ) from None


def _handle_observation_results(payload: dict, cache: dict):
    """NptRecord chunk -> typed observation results or lightweight table rows."""
    from llrops.fileio.npt import NptDataset
    from llrops.classes.observation import ObservationOutputLevel, ObservationProcessingOptions

    spec = _observation_spec_for_payload(payload, cache)
    processor = _initialized_processor_for_task(cache, spec)
    _apply_catalog_state(processor, payload.get("catalogState"))
    options = ObservationProcessingOptions(**payload["options"]).with_progress(
        None,
        enabled=False,
    )
    records = list(payload["records"])
    local = NptDataset(
        records=records,
        name=f"mpi-task-{payload['taskId']}",
        n_input_records=len(records),
        n_invalid_records=0,
    )
    results = processor.process(
        local,
        source_name=payload["sourceName"],
        options=options,
    )
    response = {
        "sourceName": payload["sourceName"],
        "startIndex": payload["startIndex"],
        "nRecords": len(records),
    }
    if payload.get("returnRows"):
        level = ObservationOutputLevel.parse(payload.get("outputLevel", "standard"))
        response["rows"] = [result.to_row(level) for result in results]
    else:
        response["results"] = results
    return response



TASK_HANDLERS: Dict[str, Callable[[dict, dict], object]] = {
    "observation_results": _handle_observation_results,
}


# ---------------------------------------------------------------------------
# runtime: generic dynamic master-worker scheduler
# ---------------------------------------------------------------------------

class MpiRuntime:
    """One communicator, one worker loop, one dynamic scheduler.

    Programs never talk MPI directly — they call :func:`mpi_observation_results`, which
    routes through :meth:`map_tasks`.  Workers stay alive across programs and
    Gauss-Newton iterations, keeping their CALCEPH/EOP model caches warm.
    """

    def __init__(self, comm=None) -> None:
        self._MPI = mpi_comm_world()
        self.comm = comm if comm is not None else self._MPI.COMM_WORLD
        self._prepared_spec_ids: set[str] = set()
        self._initialized_spec_ids: set[str] = set()
        self._serial_cache: dict = {}

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def has_workers(self) -> bool:
        return self.size > 1

    def prepare_observation_spec(self, spec: dict) -> bool:
        """Broadcast one immutable observation spec exactly once.

        Workers remain in the generic receive loop.  A small point-to-point
        control message moves every rank into the same collective ``bcast``;
        the complete spec (catalogs plus parsed EOP arrays) is then transferred
        once and cached by ``specId``.  Individual tasks carry only the ID.
        """
        if not self.is_master:
            raise RuntimeError("Only rank 0 may broadcast observation specs.")
        spec_id = str(spec["specId"])
        if spec_id in self._prepared_spec_ids:
            return False

        if self.has_workers:
            command = {"kind": "observationSpec", "specId": spec_id}
            for worker in range(1, self.size):
                self.comm.send(command, dest=worker, tag=TAG_BROADCAST_SPEC)
            self.comm.bcast(spec, root=0)
        else:
            self._serial_cache[("observationSpec", spec_id)] = spec
        self._prepared_spec_ids.add(spec_id)
        return True

    def initialize_observation_workers(
        self,
        spec_id: str,
        *,
        quiet: bool = False,
    ) -> bool:
        """Construct and cache one observation processor on every worker.

        Initialization is a separate synchronization phase rather than a side
        effect of the first observation task.  Consequently CALCEPH opening,
        processor construction, and other one-time model setup are reported
        independently and are excluded from the timed observation throughput.
        """
        if not self.is_master:
            raise RuntimeError("Only rank 0 may initialize observation workers.")

        spec_id = str(spec_id)
        if spec_id not in self._prepared_spec_ids:
            raise RuntimeError(
                f"MPI observation spec {spec_id!r} must be broadcast before initialization."
            )
        if spec_id in self._initialized_spec_ids:
            return False

        if not self.has_workers:
            spec = self._serial_cache[("observationSpec", spec_id)]
            _processor_for_task(self._serial_cache, spec)
            self._initialized_spec_ids.add(spec_id)
            return True

        worker_ranks = list(range(1, self.size))
        if not quiet:
            print(
                f"[MPI] initializing {len(worker_ranks)} worker(s) "
                "(processor + process-local CALCEPH)...",
                flush=True,
            )

        command = {"kind": "initializeObservationSpec", "specId": spec_id}
        started = time.perf_counter()
        for worker in worker_ranks:
            self.comm.send(command, dest=worker, tag=TAG_INITIALIZE_SPEC)

        status = self._MPI.Status()
        pending = set(worker_ranks)
        failures: list[tuple[int, str]] = []
        worker_elapsed: dict[int, float] = {}
        while pending:
            is_error, response_spec_id, detail = self.comm.recv(
                source=self._MPI.ANY_SOURCE,
                tag=TAG_READY,
                status=status,
            )
            worker = status.Get_source()
            if worker not in pending:
                raise RuntimeError(
                    f"Unexpected duplicate MPI initialization response from rank {worker}."
                )
            pending.remove(worker)
            if str(response_spec_id) != spec_id:
                failures.append(
                    (
                        worker,
                        "MPI initialization response spec ID mismatch: "
                        f"expected {spec_id!r}, received {response_spec_id!r}.",
                    )
                )
            elif is_error:
                failures.append((worker, str(detail)))
            elif isinstance(detail, dict):
                worker_elapsed[worker] = float(detail.get("elapsedSeconds", 0.0))

        elapsed = time.perf_counter() - started
        if failures:
            worker, remote_traceback = failures[0]
            extra = "" if len(failures) == 1 else f" ({len(failures)} workers failed)"
            raise RuntimeError(
                f"MPI worker rank {worker} failed while initializing observation "
                f"spec {spec_id!r}{extra}:\n{remote_traceback}"
            )

        self._initialized_spec_ids.add(spec_id)
        if not quiet:
            slowest = max(worker_elapsed.values(), default=0.0)
            print(
                f"[MPI] {len(worker_ranks)}/{len(worker_ranks)} worker(s) ready; "
                f"initialization {elapsed:.2f} s, slowest worker {slowest:.2f} s.",
                flush=True,
            )
        return True

    # -- worker side ---------------------------------------------------------
    def worker_loop(self) -> None:
        status = self._MPI.Status()
        cache: dict = {}
        try:
            while True:
                message = self.comm.recv(source=0, tag=self._MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == TAG_STOP:
                    break
                if tag == TAG_BROADCAST_SPEC:
                    command = message or {}
                    if command.get("kind") != "observationSpec":
                        raise RuntimeError(
                            f"Rank {self.rank} received invalid broadcast command {command!r}."
                        )
                    spec = self.comm.bcast(None, root=0)
                    spec_id = str(command["specId"])
                    if str(spec.get("specId")) != spec_id:
                        raise RuntimeError(
                            "MPI observation-spec broadcast ID mismatch: "
                            f"command={spec_id!r}, payload={spec.get('specId')!r}."
                        )
                    cache[("observationSpec", spec_id)] = spec
                    continue
                if tag == TAG_INITIALIZE_SPEC:
                    command = message or {}
                    spec_id = str(command.get("specId", ""))
                    if command.get("kind") != "initializeObservationSpec" or not spec_id:
                        self.comm.send(
                            (
                                True,
                                spec_id,
                                f"Rank {self.rank} received invalid initialization "
                                f"command {command!r}.",
                            ),
                            dest=0,
                            tag=TAG_READY,
                        )
                        continue
                    started = time.perf_counter()
                    try:
                        spec = cache[("observationSpec", spec_id)]
                        _processor_for_task(cache, spec)
                        self.comm.send(
                            (
                                False,
                                spec_id,
                                {"elapsedSeconds": time.perf_counter() - started},
                            ),
                            dest=0,
                            tag=TAG_READY,
                        )
                    except Exception:
                        self.comm.send(
                            (True, spec_id, traceback.format_exc()),
                            dest=0,
                            tag=TAG_READY,
                        )
                    continue
                if tag != TAG_TASK:
                    raise RuntimeError(
                        f"Rank {self.rank} received unexpected MPI tag {tag}"
                    )
                kind, payload = message
                task_id = payload.get("taskId")
                try:
                    result = TASK_HANDLERS[kind](payload, cache)
                    self.comm.send((False, task_id, result), dest=0, tag=TAG_RESULT)
                except Exception:
                    self.comm.send(
                        (True, task_id, traceback.format_exc()), dest=0, tag=TAG_RESULT
                    )
        finally:
            _close_cached_objects(cache)

    # -- master side -----------------------------------------------------------
    def map_tasks(
        self,
        kind: str,
        payloads: Sequence[dict],
        *,
        progress_desc: Optional[str] = None,
        progress_total: Optional[int] = None,
        quiet: bool = False,
    ) -> List[object]:
        """Dynamically schedule payloads over worker ranks; return results in
        task order.  Worker exceptions re-raise here with the remote traceback.

        With no workers (size 1) tasks run serially in-process — the same
        fallback the v24 scripts implemented.
        """
        payloads = [dict(p, taskId=i) for i, p in enumerate(payloads)]
        n_tasks = len(payloads)
        if n_tasks == 0:
            return []

        if not self.has_workers:
            return [TASK_HANDLERS[kind](p, self._serial_cache) for p in payloads]

        status = self._MPI.Status()
        results: Dict[int, object] = {}
        worker_ranks = list(range(1, self.size))
        next_task = 0
        completed = 0
        completed_units = 0

        def send_next(worker_rank: int) -> None:
            nonlocal next_task
            if next_task >= n_tasks:
                return
            self.comm.send((kind, payloads[next_task]), dest=worker_rank, tag=TAG_TASK)
            next_task += 1

        desc = progress_desc or kind
        total = progress_total if progress_total is not None else n_tasks
        if not quiet:
            print(
                f"\r{desc}: 0/{total} "
                f"(starting, tasks 0/{n_tasks}, ranks {self.size})",
                end="",
                flush=True,
            )

        # Start formal work only after the explicit worker-initialization
        # barrier has completed. Prime workers before starting throughput
        # timing so initial task sends/serialization do not dilute the rate.
        for worker in worker_ranks[: min(len(worker_ranks), n_tasks)]:
            send_next(worker)

        rate_started = time.perf_counter()
        last_report = rate_started
        last_report_units = 0
        while completed < n_tasks:
            is_error, task_id, result = self.comm.recv(
                source=self._MPI.ANY_SOURCE, tag=TAG_RESULT, status=status
            )
            worker = status.Get_source()
            if is_error:
                raise RuntimeError(
                    f"MPI worker rank {worker} failed on {kind} task {task_id}:\n{result}"
                )
            results[int(task_id)] = result
            completed += 1
            if isinstance(result, dict):
                completed_units += int(result.get("nRecords", 1))

            now = time.perf_counter()
            if not quiet and (now - last_report >= 2.0 or completed == n_tasks):
                elapsed = max(1.0e-9, now - rate_started)
                done = completed_units if progress_total is not None else completed
                window_elapsed = max(1.0e-9, now - last_report)
                window_done = done - last_report_units
                avg_rate = done / elapsed
                recent_rate = window_done / window_elapsed
                print(
                    f"\r{desc}: {done}/{total} "
                    f"({avg_rate:.2f}/s avg, {recent_rate:.2f}/s recent, "
                    f"active {elapsed:.1f}s, tasks {completed}/{n_tasks}, "
                    f"ranks {self.size})",
                    end="" if completed < n_tasks else "\n",
                    flush=True,
                )
                last_report = now
                last_report_units = done

            send_next(worker)

        return [results[i] for i in range(n_tasks)]

    def shutdown(self) -> None:
        if not self.is_master:
            return
        if self.has_workers:
            for worker in range(1, self.size):
                self.comm.send(None, dest=worker, tag=TAG_STOP)
        _close_cached_objects(self._serial_cache)
        self._serial_cache.clear()
        self._prepared_spec_ids.clear()
        self._initialized_spec_ids.clear()


# ---------------------------------------------------------------------------
# program-facing helpers
# ---------------------------------------------------------------------------

def chunk_dataset_tasks(datasets, chunksize: int) -> List[dict]:
    """Chunk already parsed NptRecords of every source (v24 ``_make_tasks``)."""
    chunk = max(1, int(chunksize))
    tasks: List[dict] = []
    for source_name, dataset in datasets.items():
        records = list(dataset.records)
        for start in range(0, len(records), chunk):
            tasks.append(
                {
                    "sourceName": str(source_name),
                    "startIndex": start,
                    "records": records[start : start + chunk],
                }
            )
    return tasks


def _observation_task_payloads(
    spec_id: str,
    datasets,
    options,
    *,
    chunksize: int,
    catalog_state: Optional[dict],
    return_rows: bool = False,
    output_level: str = "standard",
) -> list[dict]:
    options_dict = asdict(options)
    return [
        dict(
            task,
            specId=str(spec_id),
            options=options_dict,
            catalogState=catalog_state,
            returnRows=bool(return_rows),
            outputLevel=str(output_level),
        )
        for task in chunk_dataset_tasks(datasets, chunksize)
    ]


def _prepare_and_initialize_spec(
    runtime: MpiRuntime,
    spec: dict,
    *,
    quiet: bool,
) -> None:
    prepared = runtime.prepare_observation_spec(spec)
    if prepared and not quiet:
        earth_rotation = (spec.get("sharedResources") or {}).get("earthRotation") or {}
        sample_count = len(earth_rotation.get("mjdUtc", ()))
        detail = f", EOP samples={sample_count}" if sample_count else ""
        print(
            f"[MPI] observation spec broadcast to {runtime.size - 1} worker(s){detail}.",
            flush=True,
        )
    runtime.initialize_observation_workers(spec["specId"], quiet=quiet)


def mpi_observation_results(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: Optional[dict] = None,
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> Dict[str, list]:
    """Compute typed observation results for every dataset over MPI."""
    _prepare_and_initialize_spec(runtime, spec, quiet=quiet)
    tasks = _observation_task_payloads(
        spec["specId"],
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
    )
    total = sum(len(dataset.records) for dataset in datasets.values())
    results = runtime.map_tasks(
        "observation_results",
        tasks,
        progress_desc=progress_desc,
        progress_total=total,
        quiet=quiet,
    )
    results_by_source: Dict[str, list] = {str(name): [] for name in datasets}
    for result in results:
        results_by_source[result["sourceName"]].extend(result["results"])
    for source_results in results_by_source.values():
        source_results.sort(key=lambda item: item.normal_point_index)
    return results_by_source


def mpi_observation_rows(
    runtime: MpiRuntime,
    spec: dict,
    datasets,
    options,
    *,
    output_level: str = "standard",
    chunksize: int = 8,
    catalog_state: Optional[dict] = None,
    progress_desc: str = "O-C normal points",
    quiet: bool = False,
) -> Dict[str, list[dict]]:
    """Compute lightweight O-C table rows over MPI.

    Residual-table production does not need to ship full typed observation
    objects, partial arrays, or Epoch instances back to rank 0.  Returning rows
    restores the old MPI transport shape while leaving adjustment code on the
    typed-result path.
    """
    _prepare_and_initialize_spec(runtime, spec, quiet=quiet)
    tasks = _observation_task_payloads(
        spec["specId"],
        datasets,
        options,
        chunksize=chunksize,
        catalog_state=catalog_state,
        return_rows=True,
        output_level=output_level,
    )
    total = sum(len(dataset.records) for dataset in datasets.values())
    results = runtime.map_tasks(
        "observation_results",
        tasks,
        progress_desc=progress_desc,
        progress_total=total,
        quiet=quiet,
    )
    rows_by_source: Dict[str, list[dict]] = {str(name): [] for name in datasets}
    for result in results:
        rows_by_source[result["sourceName"]].extend(result["rows"])
    for rows in rows_by_source.values():
        rows.sort(key=lambda row: int(row.get("normal_point_index", row.get("record_index", 0))))
    return rows_by_source
