import argparse
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_single_execution(script_path: str, worker_id: int) -> tuple[int, float, bool]:
    """Run a single execution of the script and return worker_id, duration, and success status."""
    start = time.time()
    try:
        subprocess.run(["python", script_path], check=True, capture_output=True)
        duration = time.time() - start
        return worker_id, duration, True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start
        return worker_id, duration, False


def benchmark_script_parallel(
    script_path: str,
    num_workers: int = 4,
    duration_seconds: int = 60,
    output_path: Path | None = None,
):
    """Run a Python script in parallel with multiple workers and collect metrics."""

    all_durations = []
    worker_durations = {i: [] for i in range(num_workers)}
    failures = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        worker_idx = 0

        # Submit tasks until duration expires, but let all submitted tasks complete
        while time.time() - start_time < duration_seconds:
            # Keep submitting up to num_workers concurrent tasks
            while len(futures) < num_workers and time.time() - start_time < duration_seconds:
                future = executor.submit(run_single_execution, script_path, worker_idx % num_workers)
                futures.append(future)
                worker_idx += 1

            # Check for completed futures without blocking
            if futures:
                done = set()
                for future in futures:
                    if future.done():
                        worker_id, duration, success = future.result()
                        if success:
                            all_durations.append(duration)
                            worker_durations[worker_id].append(duration)
                        else:
                            failures += 1
                        done.add(future)

                # Remove completed futures
                futures = [f for f in futures if f not in done]

            # Small sleep to avoid busy-waiting
            time.sleep(0.1)

        # Wait for all remaining futures to complete (no time limit)
        for future in futures:
            worker_id, duration, success = future.result()
            if success:
                all_durations.append(duration)
                worker_durations[worker_id].append(duration)
            else:
                failures += 1

    total_duration = time.time() - start_time
    num_executions = len(all_durations)

    if not all_durations:
        raise RuntimeError("No executions completed successfully; check the script path or increase duration.")

    avg_duration = statistics.mean(all_durations)
    throughput = num_executions / total_duration

    if len(all_durations) >= 100:
        p99_duration = statistics.quantiles(all_durations, n=100)[98]
    else:
        p99_duration = max(all_durations)

    summary = (
        f"Parallel Benchmark Results (Workers: {num_workers})\n"
        f"{'=' * 50}\n"
        f"Total duration: {total_duration:.4f}s\n"
        f"Successful executions: {num_executions}\n"
        f"Failed executions: {failures}\n"
        f"Throughput: {throughput:.4f} executions/sec\n"
        f"Average duration: {avg_duration:.4f}s\n"
        f"P99 duration: {p99_duration:.4f}s\n"
        f"Min: {min(all_durations):.4f}s, Max: {max(all_durations):.4f}s\n"
        f"\nPer-Worker Stats:\n"
    )

    for worker_id in range(num_workers):
        durations = worker_durations[worker_id]
        if durations:
            summary += (
                f"  Worker {worker_id}: {len(durations)} executions, "
                f"avg {statistics.mean(durations):.4f}s\n"
            )
        else:
            summary += f"  Worker {worker_id}: 0 executions\n"

    print(summary)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary)
        print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a Python script with parallel workers.")
    parser.add_argument("script_path", help="Path to the Python script to benchmark")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Benchmark duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/nunchaku-benchmark-parallel.txt"),
        help="Output file path for results (default: /tmp/nunchaku-benchmark-parallel.txt)",
    )

    args = parser.parse_args()
    benchmark_script_parallel(
        args.script_path,
        num_workers=args.workers,
        duration_seconds=args.duration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
