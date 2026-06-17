"""CLI: probe num_envs_per_env_runner and write training_data/env_runner_capacity.json."""

import argparse

import ray

from .compute_resources import compute_resource
from .env_runner_capacity import CAPACITY_FILE, DEFAULT_CANDIDATES, run_diep_capacity_probe


def main():
    parser = argparse.ArgumentParser(description="Probe Diep env-runner vectorization capacity.")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
        help="num_envs_per_env_runner values to try in ascending order",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=0.85,
        help="Stop when system memory use exceeds this fraction (0-1)",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    compute_resources = compute_resource()

    result = run_diep_capacity_probe(
        compute_resources,
        candidates=args.candidates,
        memory_limit=args.memory_limit,
    )
    print(f"Wrote {CAPACITY_FILE}")
    print(f"Recommended num_envs_per_env_runner={result['num_envs_per_env_runner']}")


if __name__ == "__main__":
    main()
