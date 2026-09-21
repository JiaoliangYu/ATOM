# Trace workflow

Use this procedure only when the user requests a trace. It does not replace the 1k/1k performance benchmark.

## Launch

Stop the current server, then start the matching scenario server with tracing enabled. Leave `FAKE_EPLB` unset so the script default of `1` applies:

```bash
TRACE_DIR=/app/traces/<run-name> TRACE=1 \
  bash <matching-server-script> <model-path>
```

Use a clean, run-specific `TRACE_DIR`. The supported scripts pass `--torch-profiler-dir` and `--mark-trace`; output is organized by rank.

Before launching the client, verify `/app/ATOM/scripts/run_benchmark.sh` contains:

```bash
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"
# ...
--num-warmups=$((CONCURRENCY * 0)) \
```

Do not change other benchmark settings for the trace.

## Drive one short profiling window

After all ranks are ready, run exactly one prompt per concurrency slot and enable client profiling:

```bash
bash /app/ATOM/scripts/run_benchmark.sh \
  <model-path> 8000 1024 50 2048 1 1
```

Here `50` is OSL, `2048` is concurrency, the first `1` is the prompt multiplier, and the final `1` enables profiling.

Wait for asynchronous trace export to finish before stopping the server. Record the server command, effective `FAKE_EPLB`, model path, `TRACE_DIR`, client command, logs, and generated files.

Treat this as trace collection only. Never compare its throughput with the supported 1024/1024 benchmark rows.
