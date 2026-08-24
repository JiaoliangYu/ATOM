# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import gc
import itertools
import logging
import time
import uuid
from multiprocessing import shared_memory
from typing import Callable

import torch

logger = logging.getLogger("atom")
_BUFFER_ALIGNMENT = 1 << 20


def next_buffer_capacity(current_capacity, required_bytes, alignment=_BUFFER_ALIGNMENT):
    '''Grow at least 2x, align up, and reuse sufficient capacity.'''
    current_capacity = max(0, int(current_capacity))
    required_bytes = max(0, int(required_bytes))
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("buffer alignment must be positive")
    if required_bytes <= current_capacity:
        return current_capacity
    target = max(required_bytes, current_capacity * 2)
    return ((target + alignment - 1) // alignment) * alignment


def rebuild_ipc_handle(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    args = list(args)
    if device_id is not None:
        args[6] = device_id
    return func(*args)


class IPCWeightBufferPool:
    '''Persistent sender-side CUDA IPC buffers.'''

    def __init__(self):
        self.capacity = 0
        self.generation = -1
        self.num_gpus = 0
        self.device = None
        self.buffer = None
        self.per_gpu_buffers = {}
        self.ipc_handle = None
        self.ipc_handles = {}
        self._retired = []

    def ensure_capacity(self, required_bytes, *, device, num_gpus):
        device = torch.device(device)
        num_gpus = max(1, int(num_gpus))
        topology_changed = self.device != device or self.num_gpus != num_gpus
        if (
            self.buffer is not None
            and not topology_changed
            and self.capacity >= required_bytes
        ):
            return False

        capacity = next_buffer_capacity(self.capacity, required_bytes)
        if capacity == 0:
            capacity = _BUFFER_ALIGNMENT
        if self.buffer is not None:
            self._retired.append(
                (self.buffer, self.per_gpu_buffers, self.ipc_handle, self.ipc_handles)
            )

        from torch.multiprocessing.reductions import reduce_tensor

        buffer = torch.empty(capacity, dtype=torch.uint8, device=device)
        per_gpu_buffers = {}
        ipc_handles = {}
        ipc_handle = None
        if num_gpus > 1:
            for index in range(num_gpus):
                gpu_buffer = torch.empty(
                    capacity, dtype=torch.uint8, device=f"cuda:{index}"
                )
                per_gpu_buffers[index] = gpu_buffer
                ipc_handles[index] = reduce_tensor(gpu_buffer)
        else:
            ipc_handle = reduce_tensor(buffer)

        self.capacity = capacity
        self.generation += 1
        self.num_gpus = num_gpus
        self.device = device
        self.buffer = buffer
        self.per_gpu_buffers = per_gpu_buffers
        self.ipc_handle = ipc_handle
        self.ipc_handles = ipc_handles
        return True

    def retire_previous(self):
        '''Drop old owners only after receivers acknowledged new handles.'''
        if not self._retired:
            return
        self._retired.clear()
        gc.collect()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _initial_capacity_bytes(bucket_size_mb):
    capacity = int(bucket_size_mb) << 20
    if capacity <= 0:
        raise ValueError("bucket_size_mb must be positive")
    return capacity


def _abort_buffered_update(core_mgr, update_id, error):
    try:
        core_mgr.broadcast_utility_command_sync(
            "abort_buffered_weight_update", update_id=update_id, error=str(error)
        )
    except Exception:
        logger.exception("failed to abort buffered weight update %s", update_id)


def load_weights_via_shm(core_mgr, weights, bucket_size_mb=2048):
    '''Load weights through SHM with begin/apply/commit lifecycle.'''
    update_id = uuid.uuid4().hex
    initial_capacity = _initial_capacity_bytes(bucket_size_mb)
    total_params = 0
    shm = buffer = None
    capacity = 0
    core_mgr.broadcast_utility_command_sync(
        "begin_buffered_weight_update", update_id=update_id, transport="shm"
    )

    def replace_buffer(required_bytes):
        nonlocal shm, buffer, capacity
        capacity = next_buffer_capacity(
            capacity, max(initial_capacity, required_bytes)
        )
        if shm is not None:
            del buffer
            shm.close()
            shm.unlink()
        shm = shared_memory.SharedMemory(
            name=f"atom_weights_{uuid.uuid4().hex}", create=True, size=capacity
        )
        buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

    def flush(bucket_meta, payload_bytes):
        nonlocal total_params
        if not bucket_meta:
            return
        core_mgr.broadcast_utility_command_sync(
            "apply_weight_bucket_from_shm",
            update_id=update_id,
            shm_name=shm.name,
            bucket_meta=bucket_meta,
            payload_bytes=payload_bytes,
        )
        total_params += len(bucket_meta)

    try:
        offset = 0
        bucket_meta = {}
        for name, tensor in weights:
            tensor = tensor.cpu() if tensor.is_cuda else tensor
            tensor = tensor.contiguous()
            tensor_nbytes = tensor.nbytes
            if shm is None:
                replace_buffer(tensor_nbytes)
            if bucket_meta and offset + tensor_nbytes > capacity:
                flush(bucket_meta, offset)
                bucket_meta, offset = {}, 0
            if tensor_nbytes > capacity:
                replace_buffer(tensor_nbytes)
            buffer[offset : offset + tensor_nbytes].copy_(
                tensor.view(-1).view(torch.uint8)
            )
            bucket_meta[name] = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "offset": offset,
                "nbytes": tensor_nbytes,
            }
            offset += tensor_nbytes

        flush(bucket_meta, offset)
        core_mgr.broadcast_utility_command_sync(
            "commit_buffered_weight_update",
            update_id=update_id,
            verify_full_load=False,
        )
    except Exception as exc:
        _abort_buffered_update(core_mgr, update_id, exc)
        raise
    finally:
        if shm is not None:
            del buffer
            shm.close()
            shm.unlink()
    logger.info("load_weights_via_shm: done - %d params", total_params)


def load_weights_via_ipc(
    core_mgr,
    weights,
    bucket_size_mb=2048,
    num_gpus=1,
    buffer_pool=None,
):
    '''Load weights through persistent CUDA IPC buffers with explicit acks.'''
    start_time = time.time()
    update_id = uuid.uuid4().hex
    initial_capacity = _initial_capacity_bytes(bucket_size_mb)
    pool = buffer_pool or IPCWeightBufferPool()
    total_params = 0
    prepared_generation = None
    core_mgr.broadcast_utility_command_sync(
        "begin_buffered_weight_update", update_id=update_id, transport="ipc"
    )

    def prepare(required_bytes, device):
        nonlocal prepared_generation
        changed = pool.ensure_capacity(
            max(initial_capacity, required_bytes),
            device=device,
            num_gpus=num_gpus,
        )
        if changed or prepared_generation != pool.generation:
            core_mgr.broadcast_utility_command_sync(
                "prepare_ipc_weight_buffer",
                update_id=update_id,
                generation=pool.generation,
                capacity=pool.capacity,
                ipc_handle=pool.ipc_handle,
                ipc_handles=pool.ipc_handles or None,
            )
            prepared_generation = pool.generation
            pool.retire_previous()

    def flush(bucket_meta, payload_bytes):
        nonlocal total_params
        if not bucket_meta:
            return
        torch.cuda.synchronize(pool.device)
        if pool.num_gpus > 1:
            source = pool.buffer[:payload_bytes]
            for target in pool.per_gpu_buffers.values():
                target[:payload_bytes].copy_(source, non_blocking=True)
            for index in pool.per_gpu_buffers:
                torch.cuda.synchronize(index)
        core_mgr.broadcast_utility_command_sync(
            "apply_weight_bucket_from_ipc",
            update_id=update_id,
            generation=pool.generation,
            bucket_meta=bucket_meta,
            payload_bytes=payload_bytes,
        )
        total_params += len(bucket_meta)

    try:
        weights_iter = iter(weights)
        try:
            first_item = next(weights_iter)
        except StopIteration:
            core_mgr.broadcast_utility_command_sync(
                "commit_buffered_weight_update",
                update_id=update_id,
                verify_full_load=False,
            )
            return
        device = (
            first_item[1].device
            if first_item[1].is_cuda
            else torch.device("cuda:0")
        )
        offset = 0
        bucket_meta = {}
        for name, tensor in itertools.chain([first_item], weights_iter):
            tensor = tensor.contiguous()
            if not tensor.is_cuda:
                tensor = tensor.to(device)
            tensor_nbytes = tensor.nbytes
            if pool.buffer is None:
                prepare(tensor_nbytes, device)
            if bucket_meta and offset + tensor_nbytes > pool.capacity:
                flush(bucket_meta, offset)
                bucket_meta, offset = {}, 0
            if tensor_nbytes > pool.capacity or prepared_generation is None:
                prepare(tensor_nbytes, device)
            pool.buffer[offset : offset + tensor_nbytes].copy_(
                tensor.view(-1).view(torch.uint8), non_blocking=True
            )
            bucket_meta[name] = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "offset": offset,
                "nbytes": tensor_nbytes,
            }
            offset += tensor_nbytes

        flush(bucket_meta, offset)
        core_mgr.broadcast_utility_command_sync(
            "commit_buffered_weight_update",
            update_id=update_id,
            verify_full_load=False,
        )
    except Exception as exc:
        _abort_buffered_update(core_mgr, update_id, exc)
        raise
    logger.info(
        "load_weights_via_ipc: done - %d params in %.2fs",
        total_params,
        time.time() - start_time,
    )
