# Benchmarking Guide

This guide explains how to benchmark Nunchaku scripts for performance testing.

## Prerequisites

- Python environment with Nunchaku installed
- GPU access (for inference scripts)
- Test scripts you want to benchmark (e.g., from `examples/` folder)

## Available Benchmark Tools

### 1. Sequential Benchmark (`benchmark.py`)

Runs a script repeatedly in sequence to measure single-execution performance.

**Usage:**
```bash
python benchmark.py <script_path> [options]
```

**Options:**
- `--duration SECONDS` - How long to run the benchmark (default: 600s)
- `--output PATH` - Output file for results (default: `/tmp/nunchaku-benchmark.txt`)

**Example:**
```bash
# Benchmark for 10 minutes
python benchmark.py examples/custom/qwen-image-edit-2509.py --duration 600

# Custom output location
python benchmark.py examples/custom/qwen-image-edit-2509.py \
    --duration 300 \
    --output /tmp/my-benchmark-results.txt
```

**Output Metrics:**
- Total duration
- Number of executions
- Average duration per execution
- P99 latency
- Min/Max execution times

---

### 2. Parallel Benchmark (`benchmark_parallel.py`)

Runs multiple instances of a script in parallel to measure throughput and GPU utilization.

**Usage:**
```bash
python benchmark_parallel.py <script_path> [options]
```

**Options:**
- `--workers N` - Number of parallel workers (default: 4)
- `--duration SECONDS` - How long to run the benchmark (default: 600s)
- `--output PATH` - Output file for results (default: `/tmp/nunchaku-benchmark-parallel.txt`)

**Example:**
```bash
# Benchmark with 8 parallel workers
python benchmark_parallel.py examples/custom/qwen-image-edit-2509.py --workers 8

# Short stress test with 16 workers
python benchmark_parallel.py examples/custom/qwen-image-edit-2509.py \
    --workers 16 \
    --duration 120

# High-VRAM GPU: maximize throughput
python benchmark_parallel.py examples/v1/qwen-image-edit-2509-blackwell-maxperf.py \
    --workers 4 \
    --duration 600 \
    --output /tmp/blackwell-parallel-benchmark.txt
```

**Output Metrics:**
- Total duration and worker count
- Successful/failed executions
- **Throughput** (executions per second)
- Average, P99, Min/Max latencies
- Per-worker execution counts and averages

---

## Choosing the Right Tool

| Scenario | Tool | Recommended Settings |
|----------|------|---------------------|
| Measure single-inference latency | `benchmark.py` | Default settings |
| Test maximum GPU throughput | `benchmark_parallel.py` | `--workers 4-8` for consumer GPUs<br>`--workers 8-16` for datacenter GPUs |
| Quick sanity check | Either | `--duration 60` |
| Production load testing | `benchmark_parallel.py` | Match expected concurrent users |

---

## Tips

1. **Warm-up period**: The first execution may be slower due to model loading and CUDA initialization. Consider ignoring the first few results in your analysis.

2. **Worker count tuning**:
   - Start with `--workers 4` and increase gradually
   - Monitor GPU memory usage (`nvidia-smi`)
   - Too many workers = OOM errors; too few = underutilized GPU

3. **Blackwell-optimized scripts**: For RTX 6000 Pro or similar high-VRAM GPUs, use the optimized scripts in `examples/v1/`:
   - `qwen-image-edit-2509-blackwell-maxperf.py` - Single image, max performance
   - `qwen-image-edit-2509-batch-blackwell.py` - Multiple prompts per run

4. **Output files**: Results are written to `/tmp` by default and include:
   - Summary statistics
   - Per-worker breakdowns (parallel only)
   - Timestamps and execution counts

5. **Monitoring**: Run `nvidia-smi -l 1` in a separate terminal to watch GPU utilization, memory, and temperature during benchmarks.

---

## Example Workflow

```bash
# 1. Test baseline performance (sequential)
python benchmark.py examples/custom/qwen-image-edit-2509.py --duration 300

# 2. Find optimal worker count (parallel)
for workers in 2 4 8 16; do
    python benchmark_parallel.py examples/custom/qwen-image-edit-2509.py \
        --workers $workers \
        --duration 120 \
        --output /tmp/benchmark-w${workers}.txt
done

# 3. Compare results
cat /tmp/benchmark-w*.txt | grep "Throughput"

# 4. Run full benchmark with optimal settings
python benchmark_parallel.py examples/custom/qwen-image-edit-2509.py \
    --workers 8 \
    --duration 600 \
    --output /tmp/final-benchmark.txt
```

---

## Troubleshooting

**Out of Memory (OOM)**
- Reduce `--workers` count
- Use scripts with CPU offloading enabled
- Check GPU memory with `nvidia-smi`

**Timeouts**
- Individual executions have a 300s timeout in parallel mode
- Increase timeout in code if your script legitimately takes longer

**No executions completed**
- Check that the script path is correct
- Ensure the script runs successfully standalone: `python <script_path>`
- Check output file permissions for the `/tmp` directory
