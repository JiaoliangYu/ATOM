# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger("atom")


@dataclass
class WeightBucketResult:
    """Result of applying one bucket of HF-named weights."""

    updated: int = 0
    received_names: set[str] = field(default_factory=set)
    loaded_internal: set[str] = field(default_factory=set)
    skipped_names: set[str] = field(default_factory=set)
    ignored_scale_names: set[str] = field(default_factory=set)
    packed_shards: dict[str, set[object]] = field(default_factory=dict)
    packed_expected: dict[str, set[object]] = field(default_factory=dict)


@dataclass
class WeightUpdateTransaction:
    """Cross-bucket state for one atomic-at-the-serving-boundary reload."""

    version: int | None
    update_id: str | None = None
    transport: str = "rdma"
    buckets: int = 0
    payload_bytes: int = 0
    received_names: set[str] = field(default_factory=set)
    loaded_internal: set[str] = field(default_factory=set)
    skipped_names: set[str] = field(default_factory=set)
    ignored_scale_names: set[str] = field(default_factory=set)
    packed_shards: dict[str, set[object]] = field(default_factory=dict)
    packed_expected: dict[str, set[object]] = field(default_factory=dict)


class WeightUpdaterMixin:
    """Mixin providing weight update capabilities for ModelRunner.

    Host class must provide:
      - self.model (nn.Module)
      - self.device (torch.device)
      - self.rank (int) — TP rank
      - self.world_size (int) — TP size
      - self.label (str)
      - self.clear_kv_cache() — method
    """

    def _invalidate_cudagraphs_after_weight_update(self) -> None:
        """Drop stale CUDA graphs after online weight updates.

        Recapture is intentionally deferred to ``resume_memory``/wake-up, where
        MemoryManagerMixin verifies that both weights and KV cache are resident
        on GPU.  This avoids recapturing against an incomplete post-update
        memory state while preventing stale graph replay.
        """
        if getattr(self, "enforce_eager", False):
            return

        # No-eager policy: weights are updated in-place (param.data.copy_ and
        # in-place shuffle preserve the parameter storage/address), so captured
        # CUDA graphs read the refreshed values from the same addresses and stay
        # valid. Keep the graphs resident instead of dropping+recapturing them,
        # which under expandable_segments faults during post-wake graph capture.
        return

        torch.cuda.synchronize()
        graphs = getattr(self, "graphs", None)
        if graphs:
            self._graphs_backup_keys = list(graphs.keys())
            graphs.clear()
        if hasattr(self, "graph_logits"):
            self.graph_logits.clear()
        if hasattr(self, "graph_aux_hidden"):
            self.graph_aux_hidden.clear()
        if hasattr(self, "graph_pool"):
            self.graph_pool = None
        tbo_graphs = getattr(getattr(self, "model", None), "tbo_graphs", None)
        if tbo_graphs is not None:
            tbo_graphs.clear()
        torch.cuda.empty_cache()
        logger.info(f"{self.label}: CUDA graphs invalidated after weight update")

    def _get_param_to_module_mapping(self) -> dict[str, tuple]:
        """
        Get or build the parameter name to module mapping.

        This mapping is cached after the first call to avoid expensive
        rebuilding on every weight update.

        Returns:
            Dict mapping parameter full name to (module, param_name, param) tuple
        """
        if not hasattr(self, "_param_to_module") or self._param_to_module is None:
            self._param_to_module = {}
            for module_name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = (
                        f"{module_name}.{param_name}" if module_name else param_name
                    )
                    self._param_to_module[full_name] = (module, param_name, param)
            logger.debug(
                f"{self.label}: Built param_to_module mapping with "
                f"{len(self._param_to_module)} parameters"
            )
        return self._param_to_module

    def _get_packed_modules_mapping(self) -> dict:
        if not hasattr(self, "_cached_packed_mapping"):
            self._cached_packed_mapping = (
                getattr(self.model, "packed_modules_mapping", None) or {}
            )
        return self._cached_packed_mapping

    def _get_packed_shard_order(self) -> dict[str, list]:
        """Build {target_suffix: [shard_id_0, shard_id_1, ...]} preserving declaration order."""
        if not hasattr(self, "_cached_packed_shard_order"):
            order: dict[str, list] = {}
            for _, (tgt, shard_id) in self._get_packed_modules_mapping().items():
                order.setdefault(tgt, []).append(shard_id)
            self._cached_packed_shard_order = order
        return self._cached_packed_shard_order

    def _resolve_packed_name(
        self, name: str, param_to_module: dict
    ) -> tuple[str, object, str] | None:
        """Try to resolve an HF name to an ATOM packed parameter.

        Returns (atom_full_name, shard_id, target_suffix) or None.
        """
        for src_suffix, (
            tgt_suffix,
            shard_id,
        ) in self._get_packed_modules_mapping().items():
            if src_suffix in name:
                atom_name = name.replace(src_suffix, tgt_suffix)
                if atom_name in param_to_module:
                    return atom_name, shard_id, tgt_suffix
        return None

    def _apply_packed_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """Handle a single incoming weight that belongs to a packed (fused) module.

        For FP8 params, shards are accumulated in a float32 buffer using the
        module's weight_loader (which handles GQA-aware TP sharding for QKV).
        Once all shards arrive, the buffer is requantized to FP8 in one shot.

        Returns:
            'updated'     – fused param fully updated (all shards received)
            'accumulated' – shard stored, waiting for remaining shards
            'skipped'     – not a packed param or lookup failed
        """
        resolved = self._resolve_packed_name(name, param_to_module)
        if resolved is None:
            return "skipped"

        atom_name, shard_id, tgt_suffix = resolved
        module, param_name, param = param_to_module[atom_name]
        weight_loader = getattr(module, "weight_loader", None)
        if weight_loader is None:
            return "skipped"

        if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
            if not hasattr(self, "_packed_weight_accum"):
                self._packed_weight_accum = {}

            if atom_name not in self._packed_weight_accum:
                self._packed_weight_accum[atom_name] = {"shards": {}}

            self._packed_weight_accum[atom_name]["shards"][shard_id] = tensor.clone()

            expected = self._get_packed_shard_order().get(tgt_suffix, [])
            if set(self._packed_weight_accum[atom_name]["shards"].keys()) >= set(
                expected
            ):
                buf = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=torch.float32, device=self.device),
                    requires_grad=False,
                )
                wlp = getattr(param, "weight_loader_process", None)
                if wlp is None:
                    wlp = getattr(module, "weight_loader_process", None)
                if wlp is not None:
                    buf.weight_loader_process = wlp
                else:
                    def _weight_loader_process(param_data, loaded_weight):
                        if param_data.dtype != loaded_weight.dtype:
                            loaded_weight = loaded_weight.to(param_data.dtype)
                        if (
                            loaded_weight.shape != param_data.shape
                            and loaded_weight.numel() == param_data.numel()
                        ):
                            loaded_weight = loaded_weight.reshape(param_data.shape)
                        param_data.copy_(loaded_weight)

                    buf.weight_loader_process = _weight_loader_process

                for sid in expected:
                    shard_t = self._packed_weight_accum[atom_name]["shards"][sid]
                    shard_gpu = shard_t.to(device=self.device, dtype=torch.float32)
                    weight_loader(buf, shard_gpu, sid)

                requantized = self._requantize_fp8_weight(
                    module, param_name, param, buf.data
                )
                del self._packed_weight_accum[atom_name]
                if not requantized:
                    return "skipped"
                logger.debug(
                    f"{self.label}: FP8 packed weight updated: {atom_name} "
                    f"(composed from {len(expected)} shards)"
                )
                return "updated"
            return "accumulated"

        tensor_gpu = tensor.to(device=self.device)
        weight_loader(param, tensor_gpu, shard_id)
        return "updated"

    def _try_shard_weight(
        self,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> bool:

        param_shape = param.shape
        tensor_shape = tensor.shape

        if len(param_shape) != len(tensor_shape):
            return False

        # Find which dimension needs sharding
        shard_dim = None
        for dim in range(len(param_shape)):
            if tensor_shape[dim] == param_shape[dim] * tp_size:
                shard_dim = dim
                break
            elif tensor_shape[dim] != param_shape[dim]:
                # Dimension mismatch but not by tp_size factor
                return False

        if shard_dim is None:
            # No dimension needs sharding but shapes don't match
            return False

        # Shard the tensor along the identified dimension
        shard_size = param_shape[shard_dim]
        start_idx = tp_rank * shard_size

        tensor = tensor.to(device=self.device, dtype=param.dtype)
        sharded_tensor = tensor.narrow(shard_dim, start_idx, shard_size)
        param.data.copy_(sharded_tensor)

        return True

    @staticmethod
    def _is_fp8_param(module: torch.nn.Module, param: torch.nn.Parameter) -> bool:
        return (
            param.dtype.is_floating_point
            and param.element_size() < 2
            and getattr(module, "weight_scale", None) is not None
        )

    def _requantize_fp8_weight(
        self,
        module: torch.nn.Module,
        param_name: str,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
    ) -> bool:
        """Requantize a full-precision weight to FP8 with updated weight_scale.

        Called when FSDP sends float32/bfloat16 trained weights to an FP8 model.
        Computes new per-block (or per-tensor/per-token) scale factors and writes
        both the FP8 weight and scale into the module in place.
        """
        weight_scale = module.weight_scale
        fp8_dtype = param.dtype
        fp8_max = torch.finfo(fp8_dtype).max

        tensor_gpu = tensor.to(device=self.device, dtype=torch.float32)

        tp_size = self.world_size
        if tp_size > 1 and tensor_gpu.shape != param.shape:
            for dim in range(len(param.shape)):
                if tensor_gpu.shape[dim] == param.shape[dim] * tp_size:
                    shard_size = param.shape[dim]
                    tensor_gpu = tensor_gpu.narrow(
                        dim, self.rank * shard_size, shard_size
                    )
                    break

        if tensor_gpu.shape != param.shape:
            logger.warning(
                f"{self.label}: Shape mismatch in FP8 requantize for {param_name}: "
                f"param={param.shape}, tensor={tensor_gpu.shape}"
            )
            return False

        from aiter import QuantType as _QT

        quant_type = getattr(module, "quant_type", None)

        if quant_type is not None and quant_type.value == _QT.per_1x128.value:
            # Must match the load-time online_quantize_weight layout: a true
            # 128x128 block scale of shape (N//128, K//128). The previous code
            # produced a 1x128-along-K scale (N, K//128) and sliced it into the
            # (N//128, K//128) buffer, which is inconsistent with the blockscale
            # GEMM and collapses generation after the first weight update.
            from atom.quantization.quark.utils import (
                quantize_weight_to_fp8_128x128_blockscale,
            )

            q_weight, scale = quantize_weight_to_fp8_128x128_blockscale(
                tensor_gpu, fp8_dtype
            )
            param.data.copy_(q_weight)
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        elif quant_type is not None and quant_type.value == _QT.per_Tensor.value:
            amax = tensor_gpu.abs().max()
            scale = (amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.fill_(scale.item())

        elif quant_type is not None and quant_type.value == _QT.per_Token.value:
            row_amax = tensor_gpu.abs().amax(dim=-1, keepdim=True)
            scale = (row_amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        else:
            logger.warning(
                f"{self.label}: Unknown quant_type {quant_type} for FP8 requantize"
            )
            return False

        self._post_process_fp8_weight(module, param)
        logger.debug(
            f"{self.label}: FP8 requantized {param_name} on {type(module).__name__}, "
            f"quant_type={quant_type}, scale_shape={weight_scale.shape}"
        )
        return True

    def _post_process_fp8_weight(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
    ) -> None:
        """Post-process an FP8 weight after update: normalization and shuffle.

        Must be called after any FP8 weight write (both requantize and direct copy)
        to ensure the weight layout matches what ATOM's GEMM kernels expect.
        """
        weight_scale = getattr(module, "weight_scale", None)

        if (
            getattr(module, "need_normalize_e4m3fn_to_e4m3fnuz", False)
            and weight_scale is not None
        ):
            from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz

            param.data, weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                param.data, weight_scale.data
            )

        quant_type = getattr(module, "quant_type", None)
        if quant_type is None:
            return

        from aiter import QuantType as _QT
        from atom.utils import envs
        from atom.model_ops.utils import shuffle_weights

        needs_shuffle = False
        if quant_type.value == _QT.per_1x128.value:
            # Match LinearBase.process_weights_after_loading(): blockscale FP8
            # weights are only preshuffled when ATOM is configured to use the
            # preshuffle GEMM path. Forcing a shuffle here makes post-sync
            # weights use a different layout from initial online quantization.
            needs_shuffle = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        elif quant_type.value == _QT.per_1x32.value:
            needs_shuffle = True
        elif quant_type.value == _QT.per_Token.value:
            try:
                from atom.model_ops import dtypes

                needs_shuffle = param.dtype == dtypes.fp8
            except ImportError:
                needs_shuffle = param.element_size() < 2

        if needs_shuffle and param.dim() == 2:
            shuffle_weights(param)

    @staticmethod
    def _module_parameter_name(name: str, param_name: str, sibling: str) -> str:
        prefix = name[: -len(param_name)] if param_name and name.endswith(param_name) else ""
        return f"{prefix}{sibling}"

    def _record_fp8_side_effects(
        self,
        loaded_internal: set[str],
        name: str,
        module: torch.nn.Module,
        param_name: str,
        param: torch.nn.Parameter,
        *,
        scale_updated: bool,
    ) -> None:
        """Record parameters modified indirectly by FP8 post-processing."""
        loaded_internal.add(name)
        if not scale_updated or not self._is_fp8_param(module, param):
            return
        weight_scale = getattr(module, "weight_scale", None)
        if isinstance(weight_scale, torch.nn.Parameter):
            loaded_internal.add(
                self._module_parameter_name(name, param_name, "weight_scale")
            )

    def _apply_named_tensors(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> WeightBucketResult:
        """Apply one bucket without finalizing the cross-bucket lifecycle."""
        param_to_module = self._get_param_to_module_mapping()
        result = WeightBucketResult()

        for name, tensor in named_tensors:
            result.received_names.add(name)
            if name not in param_to_module:
                packed = self._resolve_packed_name(name, param_to_module)
                if packed is not None:
                    atom_name, shard_id, target_suffix = packed
                    result.packed_shards.setdefault(atom_name, set()).add(shard_id)
                    result.packed_expected[atom_name] = set(
                        self._get_packed_shard_order().get(target_suffix, [])
                    )
                packed_result = self._apply_packed_weight(name, tensor, param_to_module)
                if packed_result == "updated":
                    result.updated += 1
                    if packed is not None:
                        atom_name, _, _ = packed
                        module, param_name, param = param_to_module[atom_name]
                        self._record_fp8_side_effects(
                            result.loaded_internal,
                            atom_name,
                            module,
                            param_name,
                            param,
                            scale_updated=tensor.dtype != param.dtype,
                        )
                elif packed_result == "accumulated":
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    result.ignored_scale_names.add(name)
                else:
                    logger.debug(f"{self.label}: Unmatched parameter: {name}")
                    result.skipped_names.add(name)
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)
            loaded = False
            scale_updated = False

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                loaded = self._requantize_fp8_weight(
                    module, param_name, param, tensor
                )
                scale_updated = loaded
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                tensor = tensor.to(device=self.device)
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                loaded = True
            elif tensor.shape == param.shape:
                tensor = tensor.to(device=self.device, dtype=param.dtype)
                param.data.copy_(tensor)
                loaded = True
            elif weight_loader is not None and callable(weight_loader):
                try:
                    tensor = tensor.to(device=self.device)
                    weight_loader(param, tensor)
                    loaded = True
                except Exception as exc:
                    logger.warning(
                        f"{self.label}: weight_loader failed for {name}: {exc}"
                    )
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                loaded = tp_size > 1 and self._try_shard_weight(
                    param, tensor, tp_rank, tp_size
                )

            if loaded:
                result.updated += 1
                self._record_fp8_side_effects(
                    result.loaded_internal,
                    name,
                    module,
                    param_name,
                    param,
                    scale_updated=scale_updated,
                )
            else:
                if tensor.shape != param.shape:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                result.skipped_names.add(name)

        return result

    @staticmethod
    def _weight_update_label(transaction: WeightUpdateTransaction) -> str:
        if transaction.version is not None:
            return f"version={transaction.version}"
        return f"update_id={transaction.update_id}, transport={transaction.transport}"

    def _begin_weight_update_transaction(
        self,
        *,
        version: int | None,
        update_id: str | None,
        transport: str,
    ) -> WeightUpdateTransaction:
        active = getattr(self, "_weight_update_transaction", None)
        if active is not None:
            raise RuntimeError(
                "weight update "
                f"{self._weight_update_label(active)} is already in progress"
            )
        if hasattr(self, "_packed_weight_accum"):
            self._packed_weight_accum.clear()
        transaction = WeightUpdateTransaction(
            version=version,
            update_id=update_id,
            transport=transport,
        )
        self._weight_update_transaction = transaction
        return transaction

    def begin_weight_update(self, version: int) -> dict:
        """Begin a numeric, versioned RDMA weight update transaction."""
        version = int(version)
        last_version = getattr(self, "_last_started_weight_version", None)
        if last_version is not None and version <= last_version:
            raise RuntimeError(
                f"weight update version must increase: "
                f"last_started={last_version}, got={version}"
            )
        self._begin_weight_update_transaction(
            version=version, update_id=None, transport="rdma"
        )
        self._last_started_weight_version = version
        logger.info(f"{self.label}: began weight update version={version}")
        return {"version": version, "state": "receiving"}

    def begin_buffered_weight_update(self, update_id: str, transport: str) -> dict:
        """Begin an IPC/SHM update without consuming the RDMA version space."""
        update_id = str(update_id)
        transport = str(transport)
        if not update_id:
            raise ValueError("buffered weight update_id must not be empty")
        if transport not in ("ipc", "shm"):
            raise ValueError(f"unsupported buffered weight transport: {transport}")
        if update_id == getattr(self, "_last_started_buffered_update_id", None):
            raise RuntimeError(f"buffered weight update_id was already used: {update_id}")
        self._begin_weight_update_transaction(
            version=None, update_id=update_id, transport=transport
        )
        self._last_started_buffered_update_id = update_id
        logger.info(
            f"{self.label}: began buffered weight update "
            f"update_id={update_id}, transport={transport}"
        )
        return {
            "update_id": update_id,
            "transport": transport,
            "state": "receiving",
        }

    def apply_weight_bucket(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        payload_bytes: int = 0,
    ) -> dict:
        """Apply one bucket and retain packed/fused state for later buckets."""
        transaction = getattr(self, "_weight_update_transaction", None)
        if transaction is None:
            raise RuntimeError("no weight update transaction is in progress")
        try:
            bucket = self._apply_named_tensors(named_tensors)
        except Exception as exc:
            self._abort_active_weight_update(exc)
            raise
        transaction.buckets += 1
        transaction.payload_bytes += int(payload_bytes)
        transaction.received_names.update(bucket.received_names)
        transaction.loaded_internal.update(bucket.loaded_internal)
        transaction.skipped_names.update(bucket.skipped_names)
        transaction.ignored_scale_names.update(bucket.ignored_scale_names)
        for name, shards in bucket.packed_shards.items():
            transaction.packed_shards.setdefault(name, set()).update(shards)
        transaction.packed_expected.update(bucket.packed_expected)
        return {
            "version": transaction.version,
            "update_id": transaction.update_id,
            "transport": transaction.transport,
            "bucket": transaction.buckets,
            "updated": bucket.updated,
            "received": len(bucket.received_names),
            "loaded_internal": len(bucket.loaded_internal),
            "skipped": sorted(bucket.skipped_names),
            "ignored_scales": sorted(bucket.ignored_scale_names),
        }

    def _commit_active_weight_update(
        self,
        transaction: WeightUpdateTransaction,
        *,
        verify_full_load: bool,
    ) -> dict:
        try:
            incomplete_shards = {
                name: sorted(
                    transaction.packed_expected[name]
                    - transaction.packed_shards.get(name, set()),
                    key=str,
                )
                for name in transaction.packed_expected
                if transaction.packed_expected[name]
                - transaction.packed_shards.get(name, set())
            }
            if incomplete_shards:
                raise RuntimeError(
                    "incomplete packed shards after weight reload: "
                    f"{incomplete_shards}"
                )
            incomplete_packed = sorted(
                getattr(self, "_packed_weight_accum", {}).keys()
            )
            if incomplete_packed:
                raise RuntimeError(
                    "incomplete packed parameters after weight reload: "
                    f"{incomplete_packed[:20]}"
                )

            expected_names = {name for name, _ in self.model.named_parameters()}
            missing = sorted(expected_names - transaction.loaded_internal)
            if verify_full_load and (missing or transaction.skipped_names):
                reload_kind = (
                    "RDMA"
                    if transaction.transport == "rdma"
                    else transaction.transport.upper()
                )
                raise RuntimeError(
                    f"incomplete ATOM {reload_kind} weight reload: "
                    f"loaded {len(transaction.loaded_internal)}/{len(expected_names)} "
                    f"internal parameters from {len(transaction.received_names)} HF tensors; "
                    f"missing={missing[:20]}, "
                    f"skipped={sorted(transaction.skipped_names)[:20]}"
                )

            self.clear_kv_cache()
            self._invalidate_cudagraphs_after_weight_update()
        except Exception as exc:
            self._abort_active_weight_update(exc)
            raise

        manifest = {
            "version": transaction.version,
            "update_id": transaction.update_id,
            "transport": transaction.transport,
            "buckets": transaction.buckets,
            "bytes": transaction.payload_bytes,
            "received": len(transaction.received_names),
            "loaded_internal": len(transaction.loaded_internal),
            "loaded_internal_names": sorted(transaction.loaded_internal),
            "skipped": sorted(transaction.skipped_names),
            "ignored_scales": sorted(transaction.ignored_scale_names),
            "packed_shards": {
                name: sorted(shards, key=str)
                for name, shards in transaction.packed_shards.items()
            },
            "missing": missing,
        }
        self._weight_update_healthy = True
        self._weight_update_failure = None
        self._weight_update_transaction = None
        if hasattr(self, "_packed_weight_accum"):
            self._packed_weight_accum.clear()
        logger.info(
            f"{self.label}: committed weight update "
            f"{self._weight_update_label(transaction)}, "
            f"buckets={manifest['buckets']}, loaded={manifest['loaded_internal']}"
        )
        return manifest

    def commit_weight_update(self, version: int, verify_full_load: bool = True) -> dict:
        """Finalize a numeric RDMA reload."""
        version = int(version)
        transaction = getattr(self, "_weight_update_transaction", None)
        if transaction is None:
            raise RuntimeError("no weight update transaction is in progress")
        if transaction.version != version or transaction.transport != "rdma":
            error = RuntimeError(
                "weight update version mismatch: "
                f"active={self._weight_update_label(transaction)}, got={version}"
            )
            self._abort_active_weight_update(error)
            raise error
        manifest = self._commit_active_weight_update(
            transaction, verify_full_load=verify_full_load
        )
        self._last_committed_weight_version = version
        return manifest

    def commit_buffered_weight_update(
        self, update_id: str, verify_full_load: bool = False
    ) -> dict:
        """Finalize an IPC/SHM reload and release any receiver mapping."""
        transaction = self._require_buffered_weight_update(update_id)
        transport = transaction.transport
        manifest = self._commit_active_weight_update(
            transaction, verify_full_load=verify_full_load
        )
        if transport == "ipc":
            self._release_ipc_weight_buffer()
        return manifest

    def _abort_active_weight_update(self, error) -> dict:
        active = getattr(self, "_weight_update_transaction", None)
        version = active.version if active is not None else None
        update_id = active.update_id if active is not None else None
        transport = active.transport if active is not None else None
        self._weight_update_transaction = None
        self._weight_update_healthy = False
        self._weight_update_failure = str(error)
        if hasattr(self, "_packed_weight_accum"):
            self._packed_weight_accum.clear()
        if transport == "ipc":
            self._release_ipc_weight_buffer()
        label = (
            self._weight_update_label(active)
            if active is not None
            else "no active transaction"
        )
        logger.error(f"{self.label}: aborted weight update {label}: {error}")
        return {
            "version": version,
            "update_id": update_id,
            "transport": transport,
            "state": "aborted",
            "error": str(error),
        }

    def abort_weight_update(self, version: int, error) -> dict:
        """Discard transaction state and fence serving after a partial write."""
        active = getattr(self, "_weight_update_transaction", None)
        if active is not None and active.version not in (None, int(version)):
            error = RuntimeError(
                f"cannot abort version={version}; "
                f"active={self._weight_update_label(active)}"
            )
        return self._abort_active_weight_update(error)

    def abort_buffered_weight_update(self, update_id: str, error) -> dict:
        active = getattr(self, "_weight_update_transaction", None)
        if active is not None and active.update_id != str(update_id):
            error = RuntimeError(
                f"cannot abort update_id={update_id}; "
                f"active={self._weight_update_label(active)}"
            )
        return self._abort_active_weight_update(error)

    def _require_buffered_weight_update(
        self, update_id: str, transport: str | None = None
    ) -> WeightUpdateTransaction:
        transaction = getattr(self, "_weight_update_transaction", None)
        if transaction is None:
            raise RuntimeError("no weight update transaction is in progress")
        if transaction.version is not None or transaction.update_id != str(update_id):
            raise RuntimeError(
                f"buffered weight update mismatch: "
                f"active={self._weight_update_label(transaction)}, got={update_id}"
            )
        if transport is not None and transaction.transport != transport:
            raise RuntimeError(
                f"buffered weight transport mismatch: "
                f"active={transaction.transport}, got={transport}"
            )
        return transaction

    def assert_weight_update_ready(self) -> None:
        transaction = getattr(self, "_weight_update_transaction", None)
        if transaction is not None:
            raise RuntimeError(
                "ATOM serving is fenced while weight update "
                f"{self._weight_update_label(transaction)} is in progress"
            )
        if getattr(self, "_weight_update_healthy", True):
            return
        raise RuntimeError(
            "ATOM serving is fenced after a partial weight update; "
            "complete a newer full reload or restart the worker. "
            f"failure={getattr(self, '_weight_update_failure', 'unknown')}"
        )

    def get_weight_update_status(self) -> dict:
        transaction = getattr(self, "_weight_update_transaction", None)
        return {
            "healthy": getattr(self, "_weight_update_healthy", True),
            "active_version": transaction.version if transaction is not None else None,
            "active_update_id": (
                transaction.update_id if transaction is not None else None
            ),
            "active_transport": (
                transaction.transport if transaction is not None else None
            ),
            "last_committed_version": getattr(
                self, "_last_committed_weight_version", None
            ),
            "failure": getattr(self, "_weight_update_failure", None),
        }

    def update_weights(
        self, named_tensors: list[tuple[str, torch.Tensor]], clear_kv_cache: bool = True
    ) -> int:
        """
        Update model weights from named tensors.

        Called by RLHF frameworks after each training step to
        synchronize weights from training engine to inference engine.

        Supports both direct parameter names and HuggingFace-style names that
        map to ATOM's fused parameters (qkv_proj, gate_up_proj) via the model's
        packed_modules_mapping.

        Args:
            named_tensors: List of (parameter_name, tensor) tuples.
                           Tensors should be full (unsharded) weights.
            clear_kv_cache: Whether to clear KV cache after update

        Returns:
            Number of parameters successfully updated
        """
        result = self._apply_named_tensors(named_tensors)

        if clear_kv_cache:
            self.clear_kv_cache()

        if clear_kv_cache and hasattr(self, "_packed_weight_accum"):
            if self._packed_weight_accum:
                logger.warning(
                    f"{self.label}: Incomplete packed weight accumulators: "
                    f"{list(self._packed_weight_accum.keys())}"
                )
            self._packed_weight_accum.clear()

        if clear_kv_cache:
            self._invalidate_cudagraphs_after_weight_update()

        logger.info(
            f"{self.label}: Weight update complete - "
            f"updated={result.updated}, skipped={len(result.skipped_names)}, "
            f"ignored_scales={len(result.ignored_scale_names)}"
        )
        return result.updated

    def update_weights_from_shm(
        self,
        shm_name: str,
        bucket_meta: dict,
        is_last: bool = True,
    ) -> int:
        """
        Update model weights by reading tensor data from POSIX shared memory.

        Only lightweight metadata (shm_name, bucket_meta) is transmitted through
        the control path (EngineCore -> MessageQueue).  The heavy tensor payload
        resides in ``/dev/shm/<shm_name>`` and each ModelRunner maps it directly.

        Args:
            shm_name: Name of the POSIX shared-memory segment created by the
                       caller (LLMEngine).
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        from multiprocessing import shared_memory as _shm_mod
        from unittest.mock import patch

        # Open the existing shared-memory segment (do NOT unlink – caller owns it)
        with patch(
            "multiprocessing.resource_tracker.register",
            lambda *args, **kwargs: None,
        ):
            shm = _shm_mod.SharedMemory(name=shm_name)

        try:
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            param_to_module = self._get_param_to_module_mapping()

            updated = 0
            skipped = 0
            ignored_scales = 0

            for name, meta in bucket_meta.items():
                # Reconstruct a CPU tensor view from shared memory
                dtype_str = meta["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                offset = meta["offset"]
                nbytes = meta["nbytes"]
                tensor = (
                    buffer[offset : offset + nbytes]
                    .view(dtype=dtype)
                    .view(meta["shape"])
                )

                if name not in param_to_module:
                    result = self._apply_packed_weight(name, tensor, param_to_module)
                    if result == "updated":
                        updated += 1
                    elif result == "accumulated":
                        pass
                    elif "weight_scale" in name or "input_scale" in name:
                        ignored_scales += 1
                    else:
                        logger.debug(f"{self.label}: Unmatched parameter: {name}")
                        skipped += 1
                    continue

                module, param_name, param = param_to_module[name]
                weight_loader = getattr(module, "weight_loader", None)

                if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                    self._requantize_fp8_weight(module, param_name, param, tensor)
                    updated += 1
                elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                    tensor = tensor.to(device=self.device)
                    param.data.copy_(tensor)
                    self._post_process_fp8_weight(module, param)
                    updated += 1
                elif tensor.shape == param.shape:
                    tensor = tensor.to(device=self.device, dtype=param.dtype)
                    param.data.copy_(tensor)
                    updated += 1
                elif weight_loader is not None and callable(weight_loader):
                    try:
                        tensor = tensor.to(device=self.device)
                        weight_loader(param, tensor)
                        updated += 1
                    except Exception as e:
                        logger.warning(
                            f"{self.label}: weight_loader failed for {name}: {e}"
                        )
                        skipped += 1
                else:
                    tp_size = self.world_size
                    tp_rank = self.rank
                    if tp_size > 1 and self._try_shard_weight(
                        param, tensor, tp_rank, tp_size
                    ):
                        updated += 1
                    else:
                        logger.warning(
                            f"{self.label}: Shape mismatch for {name}: "
                            f"expected {param.shape}, got {tensor.shape}"
                        )
                        skipped += 1

            if is_last:
                self.clear_kv_cache()
                if hasattr(self, "_packed_weight_accum"):
                    if self._packed_weight_accum:
                        logger.warning(
                            f"{self.label}: Incomplete packed weight accumulators: "
                            f"{list(self._packed_weight_accum.keys())}"
                        )
                    self._packed_weight_accum.clear()
                self._invalidate_cudagraphs_after_weight_update()

            logger.info(
                f"{self.label}: SHM weight update bucket done - "
                f"updated={updated}, skipped={skipped}, "
                f"ignored_scales={ignored_scales}, is_last={is_last}"
            )
            return updated
        finally:
            shm.close()

    @staticmethod
    def _weight_bucket_views(buffer: torch.Tensor, bucket_meta: dict):
        named_tensors = []
        capacity = buffer.numel()
        for name, meta in bucket_meta.items():
            offset = int(meta["offset"])
            nbytes = int(meta["nbytes"])
            if offset < 0 or nbytes < 0 or offset + nbytes > capacity:
                raise RuntimeError(
                    f"weight bucket entry {name} is outside buffer capacity: "
                    f"offset={offset}, nbytes={nbytes}, capacity={capacity}"
                )
            dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
            tensor = (
                buffer[offset : offset + nbytes]
                .view(dtype=dtype)
                .view(meta["shape"])
            )
            named_tensors.append((name, tensor))
        return named_tensors

    def _synchronize_weight_update_device(self) -> None:
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _release_ipc_weight_buffer(self) -> None:
        if getattr(self, "_ipc_buffer", None) is None:
            self._ipc_buffer_update_id = None
            return
        self._synchronize_weight_update_device()
        self._ipc_buffer = None
        self._ipc_buffer_update_id = None
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def prepare_ipc_weight_buffer(
        self,
        update_id: str,
        generation: int,
        capacity: int,
        ipc_handle,
        ipc_handles: Optional[dict] = None,
    ) -> dict:
        """Map one explicit sender generation, replacing stale mappings safely."""
        self._require_buffered_weight_update(update_id, transport="ipc")
        generation = int(generation)
        capacity = int(capacity)
        current_generation = getattr(self, "_ipc_buffer_generation", -1)
        current_capacity = getattr(self, "_ipc_buffer_capacity", 0)
        current_buffer = getattr(self, "_ipc_buffer", None)
        if generation < current_generation:
            raise RuntimeError(
                f"stale IPC buffer generation: "
                f"current={current_generation}, got={generation}"
            )
        if generation == current_generation and capacity != current_capacity:
            raise RuntimeError(
                f"IPC capacity changed without a new generation: "
                f"current={current_capacity}, got={capacity}"
            )
        if generation == current_generation and current_buffer is not None:
            self._ipc_buffer_update_id = str(update_id)
            return {
                "update_id": str(update_id),
                "generation": generation,
                "capacity": capacity,
                "reused": True,
            }

        self._release_ipc_weight_buffer()
        from atom.rollout.weight_sync import rebuild_ipc_handle

        parallel_config = getattr(getattr(self, "config", None), "parallel_config", None)
        dp_rank_local = getattr(parallel_config, "data_parallel_rank_local", 0) or 0
        global_device_idx = dp_rank_local * self.world_size + self.rank
        local_device_idx = getattr(self.device, "index", None)
        if ipc_handles is not None and global_device_idx in ipc_handles:
            buffer = rebuild_ipc_handle(
                ipc_handles[global_device_idx], device_id=local_device_idx
            )
        else:
            buffer = rebuild_ipc_handle(ipc_handle, device_id=local_device_idx)
        if buffer.numel() < capacity:
            raise RuntimeError(
                f"mapped IPC buffer is too small: "
                f"mapped={buffer.numel()}, advertised={capacity}"
            )
        self._ipc_buffer = buffer
        self._ipc_buffer_generation = generation
        self._ipc_buffer_capacity = capacity
        self._ipc_buffer_update_id = str(update_id)
        return {
            "update_id": str(update_id),
            "generation": generation,
            "capacity": capacity,
            "reused": False,
        }

    def apply_weight_bucket_from_ipc(
        self,
        update_id: str,
        generation: int,
        bucket_meta: dict,
        payload_bytes: int = 0,
    ) -> dict:
        """Apply IPC views and synchronize before acknowledging sender reuse."""
        self._require_buffered_weight_update(update_id, transport="ipc")
        generation = int(generation)
        if getattr(self, "_ipc_buffer", None) is None:
            raise RuntimeError("IPC weight buffer has not been prepared")
        if self._ipc_buffer_update_id != str(update_id):
            raise RuntimeError(
                f"IPC buffer belongs to update_id={self._ipc_buffer_update_id}, "
                f"got={update_id}"
            )
        if generation != self._ipc_buffer_generation:
            raise RuntimeError(
                f"IPC generation mismatch: "
                f"prepared={self._ipc_buffer_generation}, got={generation}"
            )
        named_tensors = self._weight_bucket_views(self._ipc_buffer, bucket_meta)
        result = self.apply_weight_bucket(named_tensors, payload_bytes=payload_bytes)
        self._synchronize_weight_update_device()
        return result

    def apply_weight_bucket_from_shm(
        self,
        update_id: str,
        shm_name: str,
        bucket_meta: dict,
        payload_bytes: int = 0,
    ) -> dict:
        """Apply one SHM bucket inside the same explicit transaction."""
        self._require_buffered_weight_update(update_id, transport="shm")
        from multiprocessing import shared_memory as _shm_mod
        from unittest.mock import patch

        with patch(
            "multiprocessing.resource_tracker.register",
            lambda *args, **kwargs: None,
        ):
            shm = _shm_mod.SharedMemory(name=shm_name)
        try:
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            named_tensors = self._weight_bucket_views(buffer, bucket_meta)
            result = self.apply_weight_bucket(
                named_tensors, payload_bytes=payload_bytes
            )
            self._synchronize_weight_update_device()
            del named_tensors
            del buffer
            return result
        finally:
            shm.close()

    def update_weights_from_ipc(
        self,
        ipc_handle,
        bucket_meta: dict,
        is_last: bool = True,
        ipc_handles: Optional[dict] = None,
    ) -> int:
        """Update model weights by reading tensor data from a CUDA IPC shared buffer.

        The sender (typically the RLHF training process) has allocated a GPU
        buffer, copied weight data into it, and obtained a CUDA IPC handle via
        ``reduce_tensor()``.

        When ``ipc_handles`` (per-GPU) is provided, each ModelRunner opens
        ONLY its own GPU's handle — always same-GPU IPC, no cross-GPU
        ``hipIpcOpenMemHandle``.  This avoids the ROCm/MI300X crash where
        opening an IPC handle from a different physical GPU causes a
        "Memory access fault".

        When ``ipc_handles`` is ``None``, falls back to the original
        ``ipc_handle`` (single handle) behavior.

        Args:
            ipc_handle: CUDA IPC handle from ``reduce_tensor(buffer)`` in
                the sender process.  Used as fallback when ``ipc_handles``
                is not provided.
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).
            ipc_handles: Per-GPU IPC handles dict ``{device_index: handle}``.
                When provided, each ModelRunner opens the handle for its own
                GPU (same-GPU IPC, safe on ROCm).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        # Cache the IPC buffer mapping: only open once per weight-update cycle.
        if not hasattr(self, "_ipc_buffer") or self._ipc_buffer is None:
            from atom.rollout.weight_sync import rebuild_ipc_handle

            dp_rank_local = self.config.parallel_config.data_parallel_rank_local or 0
            global_device_idx = dp_rank_local * self.world_size + self.rank
            local_device_idx = self.device.index
            if ipc_handles is not None and global_device_idx in ipc_handles:
                self._ipc_buffer = rebuild_ipc_handle(
                    ipc_handles[global_device_idx], device_id=local_device_idx
                )
                logger.info(
                    f"{self.label}: opened per-GPU IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"global_device_idx={global_device_idx}, local_device_idx={local_device_idx}, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
            else:
                self._ipc_buffer = rebuild_ipc_handle(ipc_handle)
                logger.info(
                    f"{self.label}: opened IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
        buffer = self._ipc_buffer

        param_to_module = self._get_param_to_module_mapping()

        updated = 0
        skipped = 0
        ignored_scales = 0

        for name, meta in bucket_meta.items():
            dtype_str = meta["dtype"].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            offset = meta["offset"]
            nbytes = meta["nbytes"]

            # View into the IPC buffer (on sender's GPU), then copy to
            # this runner's device.  .to() always returns a new tensor
            # when the device differs; for same-device case we need an
            # explicit copy so the sender can safely overwrite the buffer.
            src = buffer[offset : offset + nbytes].view(dtype=dtype).view(meta["shape"])
            if src.device == self.device:
                tensor = src.clone()
            else:
                tensor = src.to(device=self.device)

            if name not in param_to_module:
                result = self._apply_packed_weight(name, tensor, param_to_module)
                if result == "updated":
                    updated += 1
                elif result == "accumulated":
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    ignored_scales += 1
                else:
                    logger.debug(f"{self.label}: Unmatched parameter: {name}")
                    skipped += 1
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(dtype=param.dtype)
                param.data.copy_(tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    weight_loader(param, tensor)
                    updated += 1
                except Exception as e:
                    logger.warning(
                        f"{self.label}: weight_loader failed for {name}: {e}"
                    )
                    skipped += 1
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                if tp_size > 1 and self._try_shard_weight(
                    param, tensor, tp_rank, tp_size
                ):
                    updated += 1
                else:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                    skipped += 1

        # Only release the IPC buffer mapping on the last bucket
        if is_last:
            self._ipc_buffer = None
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass  # ipc_collect may not be available on all platforms

            self.clear_kv_cache()
            if hasattr(self, "_packed_weight_accum"):
                if self._packed_weight_accum:
                    logger.warning(
                        f"{self.label}: Incomplete packed weight accumulators: "
                        f"{list(self._packed_weight_accum.keys())}"
                    )
                self._packed_weight_accum.clear()
            self._invalidate_cudagraphs_after_weight_update()

        logger.info(
            f"{self.label}: IPC weight update bucket done - "
            f"updated={updated}, skipped={skipped}, "
            f"ignored_scales={ignored_scales}, is_last={is_last}"
        )
        return updated
