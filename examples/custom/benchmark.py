import argparse
import statistics
import subprocess
import time
from pathlib import Path


def benchmark_script(script_path: str, duration_seconds: int = 60, output_path: Path | None = None):
    """Run a Python script repeatedly and collect metrics."""

    print(f"\nStarting sequential benchmark")
    print(f"Script: {script_path}")
    print(f"Duration: {duration_seconds}s\n")

    durations: list[float] = []
    start_time = time.time()
    execution_count = 0

    while time.time() - start_time < duration_seconds:
        execution_count += 1
        run_start = time.time()
        try:
            result = subprocess.run(
                ["python", script_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            run_duration = time.time() - run_start
            durations.append(run_duration)
            output = result.stdout + result.stderr
            print(f"Execution {execution_count} completed in {run_duration:.2f}s")
            if output.strip():
                print(f"Output:\n{output}")
        except subprocess.CalledProcessError as e:
            run_duration = time.time() - run_start
            output = e.stdout + e.stderr if hasattr(e, 'stdout') and hasattr(e, 'stderr') else str(e)
            print(f"Execution {execution_count} FAILED after {run_duration:.2f}s")
            print(f"Error output: {output}")
        except subprocess.TimeoutExpired:
            run_duration = time.time() - run_start
            print(f"Execution {execution_count} TIMEOUT after {run_duration:.2f}s")

    total_duration = time.time() - start_time
    num_executions = len(durations)

    if not durations:
        raise RuntimeError("No executions completed; reduce duration_seconds or check the script path.")

    avg_duration = statistics.mean(durations)
    max_single_duration = max(durations)

    if len(durations) >= 100:
        p99_duration = sorted(durations)[int(len(durations) * 0.99)]
    else:
        p99_duration = max(durations)

    throughput = num_executions / total_duration

    summary = (
        f"\n{'=' * 60}\n"
        f"Sequential Benchmark Results\n"
        f"{'=' * 60}\n"
        f"Script: {script_path}\n"
        f"Wall-clock duration: {total_duration:.4f}s ({total_duration/60:.2f} min)\n"
        f"Total executions: {num_executions}\n"
        f"Throughput: {throughput:.4f} executions/sec\n\n"
        f"Execution Timings:\n"
        f"  Average: {avg_duration:.4f}s\n"
        f"  P99: {p99_duration:.4f}s\n"
        f"  Min: {min(durations):.4f}s\n"
        f"  Max: {max_single_duration:.4f}s\n"
        f"{'=' * 60}\n"
    )

    print(summary, end="")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary)
        print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a Python script repeatedly.")
    parser.add_argument("script_path", help="Path to the Python script to benchmark")
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Benchmark duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/nunchaku-benchmark.txt"),
        help="Output file path for results (default: /tmp/nunchaku-benchmark.txt)",
    )

    args = parser.parse_args()
    benchmark_script(args.script_path, duration_seconds=args.duration, output_path=args.output)


if __name__ == "__main__":
    main()
