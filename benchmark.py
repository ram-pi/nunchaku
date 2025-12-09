import subprocess
import time
import statistics

def benchmark_script(script_path, duration_seconds=60):
    """Run a Python script repeatedly and collect metrics."""
    durations = []
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        run_start = time.time()
        subprocess.run(['python', script_path], check=True, capture_output=True)
        run_duration = time.time() - run_start
        durations.append(run_duration)

    # Calculate metrics
    total_duration = time.time() - start_time
    num_executions = len(durations)
    avg_duration = statistics.mean(durations)
    p99_duration = statistics.quantiles(durations, n=100)[98]

    print(f"Total duration: {total_duration:.4f}s")
    print(f"Executions: {num_executions}")
    print(f"Average duration: {avg_duration:.4f}s")
    print(f"P99 duration: {p99_duration:.4f}s")
    print(f"Min: {min(durations):.4f}s, Max: {max(durations):.4f}s")

# Run it
benchmark_script('./examples/v1/qwen-image-edit-2509.py', duration_seconds=600)
