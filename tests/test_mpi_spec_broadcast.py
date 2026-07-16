from llrops.parallel.mpi import (
    MpiRuntime,
    TAG_BROADCAST_SPEC,
    TAG_INITIALIZE_SPEC,
    TAG_READY,
    TAG_STOP,
)


class _FakeStatus:
    def __init__(self):
        self.source = None
        self.tag = None

    def Get_source(self):
        return self.source

    def Get_tag(self):
        return self.tag


class _FakeComm:
    def __init__(self, *, size, ready_responses=None):
        self._size = size
        self.sent = []
        self.broadcasts = []
        self.ready_responses = list(ready_responses or [])

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self._size

    def send(self, value, *, dest, tag):
        self.sent.append((dest, tag, value))

    def recv(self, *, source, tag, status):
        assert tag == TAG_READY
        worker, value = self.ready_responses.pop(0)
        status.source = worker
        status.tag = tag
        return value

    def bcast(self, value, *, root):
        self.broadcasts.append((root, value))
        return value


class _FakeMPI:
    ANY_SOURCE = -1
    ANY_TAG = -2
    Status = _FakeStatus


def _runtime(comm):
    runtime = object.__new__(MpiRuntime)
    runtime._MPI = _FakeMPI()
    runtime.comm = comm
    runtime._prepared_spec_ids = set()
    runtime._initialized_spec_ids = set()
    runtime._serial_cache = {}
    return runtime


def test_observation_spec_is_broadcast_once_and_tasks_can_use_id():
    comm = _FakeComm(size=4)
    runtime = _runtime(comm)
    spec = {"specId": "abc", "sharedResources": {}}

    assert runtime.prepare_observation_spec(spec) is True
    assert runtime.prepare_observation_spec(spec) is False
    assert [(dest, tag) for dest, tag, _ in comm.sent] == [
        (1, TAG_BROADCAST_SPEC),
        (2, TAG_BROADCAST_SPEC),
        (3, TAG_BROADCAST_SPEC),
    ]
    assert comm.broadcasts == [(0, spec)]


def test_all_workers_initialize_and_report_ready_once():
    spec_id = "abc"
    comm = _FakeComm(
        size=4,
        ready_responses=[
            (2, (False, spec_id, {"elapsedSeconds": 2.0})),
            (1, (False, spec_id, {"elapsedSeconds": 1.0})),
            (3, (False, spec_id, {"elapsedSeconds": 3.0})),
        ],
    )
    runtime = _runtime(comm)
    runtime._prepared_spec_ids.add(spec_id)

    assert runtime.initialize_observation_workers(spec_id, quiet=True) is True
    assert runtime.initialize_observation_workers(spec_id, quiet=True) is False
    assert [(dest, tag) for dest, tag, _ in comm.sent] == [
        (1, TAG_INITIALIZE_SPEC),
        (2, TAG_INITIALIZE_SPEC),
        (3, TAG_INITIALIZE_SPEC),
    ]
    assert spec_id in runtime._initialized_spec_ids
    assert comm.ready_responses == []


def test_worker_initialization_failure_is_raised_after_ready_collection():
    spec_id = "broken"
    comm = _FakeComm(
        size=3,
        ready_responses=[
            (1, (True, spec_id, "remote traceback")),
            (2, (False, spec_id, {"elapsedSeconds": 0.5})),
        ],
    )
    runtime = _runtime(comm)
    runtime._prepared_spec_ids.add(spec_id)

    try:
        runtime.initialize_observation_workers(spec_id, quiet=True)
    except RuntimeError as exc:
        assert "rank 1" in str(exc)
        assert "remote traceback" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("initialization failure must propagate to rank 0")
    assert comm.ready_responses == []
    assert spec_id not in runtime._initialized_spec_ids


def test_single_rank_spec_uses_serial_cache(monkeypatch):
    runtime = _runtime(_FakeComm(size=1))
    spec = {"specId": "serial", "sharedResources": {}}
    processor = object()

    monkeypatch.setattr(
        "llrops.parallel.mpi._processor_for_task",
        lambda cache, prepared_spec: cache.setdefault(
            ("processor", prepared_spec["specId"]), processor
        ),
    )

    assert runtime.prepare_observation_spec(spec) is True
    assert runtime.initialize_observation_workers("serial", quiet=True) is True
    assert runtime._serial_cache[("observationSpec", "serial")] is spec
    assert runtime._serial_cache[("processor", "serial")] is processor


class _FakeWorkerComm:
    def __init__(self, spec):
        self.spec = spec
        self.messages = [
            (
                TAG_BROADCAST_SPEC,
                {"kind": "observationSpec", "specId": spec["specId"]},
            ),
            (
                TAG_INITIALIZE_SPEC,
                {"kind": "initializeObservationSpec", "specId": spec["specId"]},
            ),
            (TAG_STOP, None),
        ]
        self.sent = []

    def Get_rank(self):
        return 1

    def Get_size(self):
        return 2

    def recv(self, *, source, tag, status):
        message_tag, value = self.messages.pop(0)
        status.source = 0
        status.tag = message_tag
        return value

    def bcast(self, value, *, root):
        assert value is None
        assert root == 0
        return self.spec

    def send(self, value, *, dest, tag):
        self.sent.append((dest, tag, value))


def test_worker_constructs_processor_before_ready(monkeypatch):
    spec = {"specId": "worker-spec", "sharedResources": {}}
    comm = _FakeWorkerComm(spec)
    runtime = _runtime(comm)
    processor = object()
    built = []

    def build(cache, prepared_spec):
        built.append(prepared_spec["specId"])
        cache[("processor", prepared_spec["specId"])] = processor
        return processor

    monkeypatch.setattr("llrops.parallel.mpi._processor_for_task", build)

    runtime.worker_loop()

    assert built == ["worker-spec"]
    assert len(comm.sent) == 1
    dest, tag, response = comm.sent[0]
    assert dest == 0
    assert tag == TAG_READY
    assert response[0] is False
    assert response[1] == "worker-spec"
    assert response[2]["elapsedSeconds"] >= 0.0


class _FakeProgressComm:
    def __init__(self):
        self._size = 3
        self.sent = []
        self.responses = [
            (1, (False, 0, {"nRecords": 10})),
            (2, (False, 1, {"nRecords": 10})),
            (1, (False, 2, {"nRecords": 10})),
        ]

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self._size

    def send(self, value, *, dest, tag):
        from llrops.parallel.mpi import TAG_TASK

        assert tag == TAG_TASK
        self.sent.append((dest, tag, value))

    def recv(self, *, source, tag, status):
        from llrops.parallel.mpi import TAG_RESULT

        assert source == _FakeMPI.ANY_SOURCE
        assert tag == TAG_RESULT
        worker, response = self.responses.pop(0)
        status.source = worker
        status.tag = tag
        return response


def test_progress_rate_clock_starts_after_initial_task_dispatch(monkeypatch, capsys):
    import llrops.parallel.mpi as mpi_module

    comm = _FakeProgressComm()
    runtime = _runtime(comm)
    sent_counts_at_clock_calls = []
    times = iter([100.0, 101.0, 103.0, 104.0])

    def fake_perf_counter():
        sent_counts_at_clock_calls.append(len(comm.sent))
        return next(times)

    monkeypatch.setattr(mpi_module.time, "perf_counter", fake_perf_counter)

    results = runtime.map_tasks(
        "observation_results",
        [{}, {}, {}],
        progress_desc="records",
        progress_total=30,
        quiet=False,
    )

    assert results == [{"nRecords": 10}, {"nRecords": 10}, {"nRecords": 10}]
    assert sent_counts_at_clock_calls[0] == 2
    out = capsys.readouterr().out
    assert "starting" in out
    assert "avg" in out
    assert "recent" in out
    assert "active 4.0s" in out
