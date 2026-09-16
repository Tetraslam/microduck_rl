"""Summarize downloaded benchmark runs: uv run scripts/modal/benchmark_summary.py."""

import argparse
import json
from pathlib import Path
from statistics import median

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("logs/modal"))
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    for path in sorted(args.directory.glob("*bench*/run.json")):
        run = json.loads(path.read_text())
        if run["state"] != "completed":
            continue
        events = EventAccumulator(str(path.parent)).Reload()
        fps = events.Scalars("Perf/total_fps")[args.warmup :]
        collection = events.Scalars("Perf/collection_time")[args.warmup :]
        learning = events.Scalars("Perf/learning_time")[args.warmup :]
        print(
            json.dumps(
                {
                    "run": path.parent.name,
                    "gpu": run["gpu"],
                    "num_envs": run["num_envs"],
                    "samples": len(fps),
                    "median_steps_per_second": median(x.value for x in fps),
                    "median_collection_seconds": median(x.value for x in collection),
                    "median_learning_seconds": median(x.value for x in learning),
                }
            )
        )


if __name__ == "__main__":
    main()
