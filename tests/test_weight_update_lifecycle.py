import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sync = _load_module("_atom_weight_sync_lifecycle_test", "atom/rollout/weight_sync.py")
_updater = _load_module(
    "_atom_weight_updater_lifecycle_test", "atom/rollout/weight_updater.py"
)
WeightUpdaterMixin = _updater.WeightUpdaterMixin
next_buffer_capacity = _sync.next_buffer_capacity


class _PlainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


class _Runner(WeightUpdaterMixin):
    def __init__(self):
        self.model = _PlainModel()
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.label = "lifecycle-test"
        self.clear_count = 0

    def clear_kv_cache(self):
        self.clear_count += 1

    def _invalidate_cudagraphs_after_weight_update(self):
        pass


class _CoreManager:
    def __init__(self):
        self.calls = []

    def broadcast_utility_command_sync(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return [{"cmd": cmd, "result": True}]


def test_capacity_reuses_and_grows_geometrically_with_mib_alignment():
    mib = 1 << 20
    assert next_buffer_capacity(4 * mib, 3 * mib) == 4 * mib
    assert next_buffer_capacity(4 * mib, 5 * mib) == 8 * mib
    assert next_buffer_capacity(0, mib + 1) == 2 * mib


def test_shm_sender_uses_explicit_lifecycle_and_grows_for_large_tensor():
    manager = _CoreManager()
    large = torch.arange((1 << 20) // 4 + 1, dtype=torch.float32)

    _sync.load_weights_via_shm(
        manager,
        [("weight", large)],
        bucket_size_mb=1,
    )

    commands = [cmd for cmd, _ in manager.calls]
    assert commands == [
        "begin_buffered_weight_update",
        "apply_weight_bucket_from_shm",
        "commit_buffered_weight_update",
    ]
    assert all("is_last" not in args for _, args in manager.calls)
    apply_args = manager.calls[1][1]
    assert apply_args["payload_bytes"] == large.nbytes
    assert apply_args["payload_bytes"] > 1 << 20


def test_empty_shm_update_still_commits_and_never_uses_clear_cache_command():
    manager = _CoreManager()
    _sync.load_weights_via_shm(manager, [], bucket_size_mb=1)
    assert [cmd for cmd, _ in manager.calls] == [
        "begin_buffered_weight_update",
        "commit_buffered_weight_update",
    ]


def test_ipc_generation_reuse_replacement_and_stale_rejection(monkeypatch):
    runner = _Runner()
    monkeypatch.setitem(sys.modules, "atom.rollout.weight_sync", _sync)
    monkeypatch.setattr(
        _sync,
        "rebuild_ipc_handle",
        lambda handle, device_id=None: handle,
    )
    source0 = torch.zeros(16, dtype=torch.uint8)
    values = torch.tensor([1.0, 2.0])
    source0[: values.nbytes].copy_(values.view(torch.uint8))
    meta = {
        "weight": {
            "shape": tuple(values.shape),
            "dtype": str(values.dtype),
            "offset": 0,
            "nbytes": values.nbytes,
        }
    }

    runner.begin_buffered_weight_update("update-1", "ipc")
    prepared = runner.prepare_ipc_weight_buffer("update-1", 0, 16, source0)
    assert prepared["reused"] is False
    assert runner.prepare_ipc_weight_buffer(
        "update-1", 0, 16, source0
    )["reused"]

    runner.apply_weight_bucket_from_ipc("update-1", 0, meta, values.nbytes)
    source0.zero_()
    torch.testing.assert_close(runner.model.weight, values)

    source1 = torch.zeros(32, dtype=torch.uint8)
    prepared = runner.prepare_ipc_weight_buffer("update-1", 1, 32, source1)
    assert prepared["reused"] is False
    with pytest.raises(RuntimeError, match="stale IPC buffer generation"):
        runner.prepare_ipc_weight_buffer("update-1", 0, 16, source0)

    manifest = runner.commit_buffered_weight_update(
        "update-1", verify_full_load=False
    )
    assert manifest["update_id"] == "update-1"
    assert manifest["transport"] == "ipc"
    assert runner._ipc_buffer is None
    assert runner.clear_count == 1


def test_buffered_failure_fences_until_a_new_commit():
    runner = _Runner()
    runner.begin_buffered_weight_update("bad", "shm")
    runner.apply_weight_bucket([("unknown.weight", torch.ones(1))])
    with pytest.raises(RuntimeError, match="incomplete ATOM SHM weight reload"):
        runner.commit_buffered_weight_update("bad", verify_full_load=True)
    with pytest.raises(RuntimeError, match="serving is fenced"):
        runner.assert_weight_update_ready()

    runner.begin_buffered_weight_update("good", "shm")
    runner.apply_weight_bucket([("weight", torch.tensor([3.0, 4.0]))])
    runner.commit_buffered_weight_update("good", verify_full_load=True)
    runner.assert_weight_update_ready()
    torch.testing.assert_close(runner.model.weight, torch.tensor([3.0, 4.0]))
