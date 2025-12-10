import argparse
import statistics
import subprocess
import time
from pathlib import Path


def benchmark_script(script_path: str, duration_seconds: int = 60, output_path: Path | None = None):
    """Run a Python script repeatedly and collect metrics."""

    durations: list[float] = []
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        run_start = time.time()
        subprocess.run(["python", script_path], check=True, capture_output=True)
        run_duration = time.time() - run_start
        durations.append(run_duration)

    total_duration = time.time() - start_time
    num_executions = len(durations)

    if not durations:
        raise RuntimeError("No executions completed; reduce duration_seconds or check the script path.")

    avg_duration = statistics.mean(durations)
    p99_duration = statistics.quantiles(durations, n=100)[98]
    summary = (
        f"Total duration: {total_duration:.4f}s\n"
        f"Executions: {num_executions}\n"
        f"Average duration: {avg_duration:.4f}s\n"
        f"P99 duration: {p99_duration:.4f}s\n"
        f"Min: {min(durations):.4f}s, Max: {max(durations):.4f}s\n"
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
