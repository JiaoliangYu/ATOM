---
name: validate-model-pr
description: Validate an AITER or ATOM PR in a Docker environment against the supported four-GPU DeepSeek V4 Pro and DeepSeek R1 scenarios. Use when a user asks for five-scenario applicability, CI-parity accuracy, baseline and candidate E2E throughput, or a short validation trace.
metadata:
  version: 1.0.1
  scope: ATOM and AITER on four-GPU gfx1250 systems using /app/scripts launchers
  last_updated: 2026-09-21
---

# Validate a Model PR

Map the supplied PR to the supported scenarios before changing code or starting a measurement. Freeze the matching Atom CI accuracy criteria, measure valid baselines, integrate the pinned PR, run accuracy, and benchmark the candidate only after accuracy passes.

For a baseline-only request, perform the same discovery and baseline controls, then stop before PR integration.

## Supported scenarios

Do not silently substitute another model, concurrency, token length, or MoE mode.

| Model | Concurrency | ISL | OSL | MoE mode |
| --- | ---: | ---: | ---: | --- |
| DSV4_PRO | 2048 | 1024 | 1024 | EP4 |
| DSV4_PRO | 16 | 1024 | 1024 | TP4 |
| DSV4_PRO | 512 | 1024 | 1024 | TP4 |
| DS_r1 | 512 | 1024 | 1024 | EP4 |
| DS_r1 | 16 | 1024 | 1024 | TP4 |

`DSV4_PRO` means DeepSeek V4 Pro. `DS_r1` means DeepSeek R1. Resolve the concrete checkpoint and matching CI entry from the supplied container.

EP4 and TP4 describe MoE expert-parallel and tensor-parallel execution. Verify the effective server arguments, environment, and runtime topology; `-tp 4` or four visible GPUs alone does not prove the MoE mode.

Concurrency is maximum in-flight requests, not total request count. Record both values and keep the full workload identical between baseline and candidate.

## Non-negotiable controls

- For DSV4 Pro EP4 at concurrency 2048, every baseline and candidate performance run must use:

  ```bash
  bash /app/scripts/dsv4/bench_dsv4_conc2048.sh <model-path>
  ```

  Do not replace it with `scripts/run_benchmark.sh`, an inline benchmark command, or another client.

- For DSV4 Pro and DeepSeek R1 accuracy, explicitly start the matching server with `FAKE_EPLB=0`.
- For DeepSeek R1 TP4 at concurrency 16 performance and tracing, explicitly start the matching server with `FAKE_EPLB=0` and verify the effective runtime value is `false`.
- For performance and tracing in every other supported scenario, leave `FAKE_EPLB` unset and confirm that the matching server script defaults it to `1`. A default of `0` is a script defect that must be corrected and recorded; do not hide it with a command-line override.
- Never reuse an accuracy server for performance or tracing. Stop it and launch a fresh server for the next purpose.
- Use only total token throughput in tok/s for the E2E comparison. Do not substitute output throughput or requests per second.
- Store commands, identities, raw logs, structured outputs, and the final report in a persistent run-specific directory. Keep every scenario and baseline/candidate phase separate.
- Preserve existing local changes. Do not clear caches, delete artifacts, stop processes, or integrate code until the corresponding workflow step authorizes it.

## Route the task

1. Read [references/validation-workflow.md](references/validation-workflow.md) for applicability mapping, CI accuracy resolution, baseline/candidate execution, JIT cleanup, comparison, and reporting.
2. If the user requests a profiler trace, also read [references/trace.md](references/trace.md). A trace workload is diagnostic only and is not a valid 1k/1k throughput result.

## Required outcome

Report separately:

- scenario applicability and runnability;
- configuration and effective activation;
- server/rank readiness;
- accuracy metrics and CI gate result;
- valid second-run total token throughput;
- failures, blockers, and measurement uncertainty;
- exact artifact and log paths;
- final source revision and server state.

Do not treat health status, process exit, emitted rows, or a trace-derived capacity estimate as proof that accuracy or performance passed.
