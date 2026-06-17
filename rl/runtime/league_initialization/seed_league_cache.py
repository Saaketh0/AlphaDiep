"""One-time CLI: seed the Redis league from freshly-initialized main weights.

Run once before the first training run (after ``./start_redis.sh``):

    cd diepcustom/rl/runtime
    python -m league_initialization.seed_league_cache

Fills Redis with iterations 0..count-1 for every class (copied from each
``main_class_{X}`` RLModule) and exports the same weights to SSD under
``diepcustom/training_data/redis/``. Training then hydrates from that cache.
"""

from __future__ import annotations

import argparse
import logging

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from DiepModelConfig import DiepCatalog, DiepConfig, DiepPolicy
from model_store import RedisModelStore
from rl.env import DiepCustomParallelEnv

from .bootstrap import seed_league_from_mains
from .constants import GHOST_POLICIES, MAIN_POLICIES, policy_id_for_agent
from .disk_store import export_league_to_disk
from .paths import LEAGUE_EXPORT_DIR
from rewards import training_env_config
from training_metadata import TrainingRunMetadata, write_metadata, league_metadata_path

logger = logging.getLogger(__name__)

ENV_NAME = "diepcustom_headless"


def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return policy_id_for_agent(agent_id)


def _build_seed_algorithm():
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(DiepCustomParallelEnv(**cfg)))
    spec = RLModuleSpec(
        module_class=DiepPolicy,
        model_config=DiepConfig,
        catalog_class=DiepCatalog,
    )
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config={"agents": 20, "max_ticks": 2000})
        .framework("torch")
        .multi_agent(
            policy_mapping_fn=_policy_mapping_fn,
            policies=set(MAIN_POLICIES + GHOST_POLICIES),
            policies_to_train=MAIN_POLICIES,
        )
        # No rollout workers needed just to materialize the RLModules.
        .env_runners(num_env_runners=0)
        .rl_module(rl_module_spec=spec)
    )
    return config.build()


def seed_league_cache(count: int = 50, *, write_provenance: bool = True, env_config: dict | None = None) -> dict:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algorithm = _build_seed_algorithm()
    try:
        store = RedisModelStore()
        written = seed_league_from_mains(algorithm, store, count=count)
        exported = export_league_to_disk(store)
        metadata_path = None
        if write_provenance:
            metadata = TrainingRunMetadata.create(
                env_config=env_config or training_env_config(),
                latest_league_iteration=store.latest_iteration(),
            )
            metadata_path = write_metadata(league_metadata_path(), metadata)
        return {
            "store": store,
            "written": written,
            "exported": exported,
            "metadata_path": metadata_path,
            "next_iteration": store.next_iteration(),
        }
    finally:
        algorithm.stop()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Seed the Redis league from main weights.")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of league iterations to seed per class (default: 50).",
    )
    args = parser.parse_args()

    result = seed_league_cache(count=args.count)
    per_class = {char_class: len(keys) for char_class, keys in result["written"].items()}
    print(f"Seeded Redis keys per class: {per_class}")
    print(f"Exported {len(result['exported'])} safetensors files to {LEAGUE_EXPORT_DIR}")
    if result.get("metadata_path"):
        print(f"Wrote league metadata: {result['metadata_path']}")
    print(f"next_iteration() = {result['next_iteration']}")


if __name__ == "__main__":
    main()
