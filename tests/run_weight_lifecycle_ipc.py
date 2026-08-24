"""Single-GPU, cross-process CUDA IPC validation for weight-update lifecycle."""

import importlib.util
import multiprocessing as mp
import sys
import traceback
import types
from pathlib import Path

import torch


def _load_module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sync = _load_module("_atom_weight_sync_ipc_test", "atom/rollout/weight_sync.py")
_atom = sys.modules.setdefault("atom", types.ModuleType("atom"))
_rollout = sys.modules.setdefault("atom.rollout", types.ModuleType("atom.rollout"))
_atom.rollout = _rollout
_rollout.weight_sync = _sync
sys.modules["atom.rollout.weight_sync"] = _sync
_updater = _load_module(
    "_atom_weight_updater_ipc_test", "atom/rollout/weight_updater.py"
)
WeightUpdaterMixin = _updater.WeightUpdaterMixin


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, device="cuda"))


class _Runner(WeightUpdaterMixin):
    def __init__(self):
        self.model = _Model()
        self.device = torch.device("cuda:0")
        self.rank = 0
        self.world_size = 1
        self.label = "ipc-lifecycle-child"
        self.clear_count = 0

    def clear_kv_cache(self):
        self.clear_count += 1

    def _invalidate_cudagraphs_after_weight_update(self):
        pass


def _receiver(connection, handle0, capacity0, handle1, capacity1):
    try:
        torch.cuda.set_device(0)
        runner = _Runner()
        runner.begin_buffered_weight_update("ipc-real", "ipc")
        runner.prepare_ipc_weight_buffer(
            "ipc-real", 0, capacity0, handle0
        )
        metadata = {
            "weight": {
                "shape": (4,),
                "dtype": "torch.float32",
                "offset": 0,
                "nbytes": 16,
            }
        }
        result = runner.apply_weight_bucket_from_ipc(
            "ipc-real", 0, metadata, payload_bytes=16
        )
        connection.send(("applied", runner.model.weight.detach().cpu().tolist(), result))
        assert connection.recv() == "source-overwritten"
        retained = runner.model.weight.detach().cpu()

        replacement = runner.prepare_ipc_weight_buffer(
            "ipc-real", 1, capacity1, handle1
        )
        stale_rejected = False
        try:
            runner.prepare_ipc_weight_buffer(
                "ipc-real", 0, capacity0, handle0
            )
        except RuntimeError as exc:
            stale_rejected = "stale IPC buffer generation" in str(exc)
        manifest = runner.commit_buffered_weight_update(
            "ipc-real", verify_full_load=True
        )
        connection.send(
            (
                "done",
                retained.tolist(),
                replacement,
                stale_rejected,
                manifest,
                runner.clear_count,
                runner._ipc_buffer is None,
            )
        )
    except Exception:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()


def _recv(connection, timeout=120):
    if not connection.poll(timeout):
        raise TimeoutError("timed out waiting for IPC lifecycle child")
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(message[1])
    return message


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")
    torch.cuda.set_device(0)
    from torch.multiprocessing.reductions import reduce_tensor

    source0 = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    source1 = torch.zeros(2 << 20, dtype=torch.uint8, device="cuda")
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    source0[: expected.nbytes].copy_(expected.view(torch.uint8))
    torch.cuda.synchronize()

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_receiver,
        args=(
            child,
            reduce_tensor(source0),
            source0.numel(),
            reduce_tensor(source1),
            source1.numel(),
        ),
    )
    process.start()
    applied = _recv(parent)
    assert applied[0] == "applied"
    assert applied[1] == [1.0, 2.0, 3.0, 4.0]

    source0.zero_()
    torch.cuda.synchronize()
    parent.send("source-overwritten")
    done = _recv(parent)
    assert done[0] == "done"
    assert done[1] == [1.0, 2.0, 3.0, 4.0]
    assert done[2]["reused"] is False
    assert done[3] is True
    assert done[4]["buckets"] == 1
    assert done[5] == 1
    assert done[6] is True
    process.join(120)
    assert process.exitcode == 0
    print(
        "IPC_LIFECYCLE_RESULT status=PASS "
        "generation_replace=PASS stale_reject=PASS source_reuse=PASS"
    )


if __name__ == "__main__":
    main()
