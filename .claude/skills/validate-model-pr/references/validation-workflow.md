# Validation workflow

## 1. Establish identity and preserve state

- Record the container name, image tag and image ID, GPU architecture, runtime versions, and visible GPU allocation.
- Locate the active ATOM and AITER source trees and installed package paths. Record both Git commits, branches, remotes, and local changes.
- Verify that the server imports the source tree that will receive the PR.
- Check actual GPU utilization, VRAM, GPU PIDs, host processes, and container ownership. Do not interfere with another user's workload.
- Create one persistent run directory containing commands, versions, raw logs, parsed results, and the final summary.

## 2. Map the PR to scenarios

Complete this assessment before measuring a baseline, integrating the PR, or deleting compilation artifacts.

- Read the PR description, diff, tests, and examples. Pin its head commit and identify the target repository and dependency revisions.
- Trace changed operators and callers into the actual model paths. Do not infer applicability from a title, model name, or example benchmark shape alone.
- Inspect `/app/scripts` for server, eval, and benchmark commands, model variants, feature flags, dtypes, backends, MoE topology, and shape limits.
- Classify all five supported rows as `applicable`, `not_applicable`, or `uncertain`, with evidence and required activation flags.
- Record runnability separately. Missing weights, scripts, dependencies, or hardware are execution blockers, not evidence that a row is unrelated.
- Share the mapping and selected scope before starting measurements. By default, validate every confirmed applicable and runnable row unless the user narrows scope.
- If no PR was supplied for a baseline-only request, run the explicitly requested supported rows and record that applicability was not PR-filtered.

Do not ask the user to choose facts that source and runtime inspection can resolve. Ask only when a material uncertainty remains.

## 3. Freeze Atom CI accuracy criteria

- Start from a user-specified CI reference; otherwise use the original ATOM checkout recorded for the baseline.
- Inspect the workflow and configuration it invokes. Match the concrete model variant, task, dataset/version, sample count, prompt format, few-shot count, decoding settings, metric name, comparison operator, and threshold.
- Check that the available eval command matches those settings. Resolve mismatches before running. A threshold from a different evaluation setup is not a valid gate.
- For DSV4 Pro and DeepSeek R1, use an accuracy server launched with explicit `FAKE_EPLB=0`.
- Freeze the criteria before integrating the PR. A PR change to CI must not silently change its own acceptance criteria.
- Baseline accuracy is required only when CI requires comparison with a measured baseline or the user asks for it.
- If no matching CI definition can be established, report the missing mapping or reference. Do not invent a threshold or treat exit code zero as an accuracy pass.

The headline GSM8K metric used by Atom CI is normally `exact_match,flexible-extract`; also retain `exact_match,strict-match` when emitted.

## 4. Measure baseline performance

For every selected scenario:

1. Preserve the original source/runtime state and record model weights or revision, GPU allocation, server command, workload, seed, warmup settings, concurrency, ISL, OSL, and total requests.
2. Start the original performance server with `FAKE_EPLB` unset. Verify from the script and startup log that the effective value is `1` and that the intended MoE topology is active.
3. Verify every required model runner/rank is initialized. Port readiness alone is insufficient.
4. Run the required benchmark twice consecutively against the same server without restarting it.
5. Save both raw outputs. Treat run 1 as JIT/warmup only and retain only run 2's explicit total token throughput as the baseline result.
6. Resolve a failed, zero, null, or otherwise invalid baseline before changing code.

For DSV4 Pro EP4 concurrency 2048, use only the fixed client named in `SKILL.md`.

## 5. Integrate the PR

- Stop the baseline server.
- Apply the pinned PR changes to the recorded baseline with an appropriate Git method.
- Preserve existing local work and record the method, resulting revision/diff, dependency changes, conflict resolutions, and any additional fixes.
- At candidate startup, prove that the changed source and extensions are actually loaded. Report a path that cannot be exercised rather than claiming validation.

## 6. Clear affected AITER compilation artifacts

- Locate `aiter/jit` in the active package, accounting for repository/package layout.
- Determine affected extensions from the PR diff, JIT module definitions, dependencies, and artifact names.
- Remove the affected `build` directory and matching compiled `.so` files directly under `aiter/jit/`.
- Record exact paths. Do not remove every `.so` indiscriminately.
- If the extension mapping remains unclear, request the relevant module names before deleting anything.
- If no compiled extension is affected, record why.

## 7. Run candidate accuracy

1. Start a fresh server with the baseline scenario configuration and explicit `FAKE_EPLB=0`.
2. Verify the intended source and rebuilt extensions are loaded and all ranks are ready.
3. Preserve the existing warmup procedure.
4. Run the frozen eval command and capture its exit status, result JSON, and actual metrics.
5. Evaluate every required CI condition. A missing metric, evaluation error, or unmet threshold does not pass.
6. Stop the accuracy server whether the result passes or fails.

If accuracy does not pass, save the evidence and skip candidate performance.

## 8. Measure candidate performance

After accuracy passes:

1. Start a fresh performance server with `FAKE_EPLB` unset and verify the script default is `1`.
2. Use the same model, GPU allocation, server settings, workload, and measurement procedure as baseline.
3. Run the benchmark twice consecutively without restarting the server.
4. Save run 1 as JIT/warmup only and retain run 2's explicit total token throughput.
5. Keep baseline and candidate servers from competing for the same GPUs.

## 9. Compare and report

For valid measurements with a positive baseline:

```text
change_pct = 100 * (candidate_tok_s - baseline_tok_s) / baseline_tok_s
```

Compare matching scenarios separately. If an apparent change is close to run-to-run variation, collect additional comparable pairs and report the spread and aggregation method across retained second runs.

The report must include:

- the five-row applicability assessment and execution blockers;
- container/image identity and baseline/candidate revisions;
- model, concurrency, ISL/OSL, verified MoE mode, total requests, and effective `FAKE_EPLB` for every server purpose;
- CI source commit/files, frozen accuracy criteria, measured values, and pass/fail/not-evaluated status;
- both raw performance runs, clearly marking run 1 as warmup and run 2 as retained;
- baseline and candidate run-2 total token throughput and percentage change;
- configuration differences and measurement limitations;
- paths to commands, logs, structured results, and the final server state.

Distinguish accuracy failure from execution failure, and observed throughput change from a repeatable improvement.
