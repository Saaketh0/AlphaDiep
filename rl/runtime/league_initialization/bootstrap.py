"""Bootstrap Redis league from live main-policy weights."""

from __future__ import annotations

import logging

from model_store import RedisModelStore

from .constants import main_policy_id
from .module_state import get_module_state

logger = logging.getLogger(__name__)


def seed_league_from_mains(
    algorithm,
    store: RedisModelStore,
    *,
    count: int | None = None,
) -> dict[str, list[str]]:
    """Copy each main_class_{X} state dict to Redis iterations 0..count-1."""
    iterations = count if count is not None else store.window_size
    written: dict[str, list[str]] = {}

    for char_class in store.classes:
        state = get_module_state(algorithm, main_policy_id(char_class))
        keys: list[str] = []
        for iteration in range(iterations):
            keys.append(store.save_class(char_class, state, iteration))
        written[char_class] = keys

    logger.info(
        "Seeded Redis league from mains: %d iterations x %d classes",
        iterations,
        len(store.classes),
    )
    return written


def ensure_league_bootstrapped(
    algorithm,
    store: RedisModelStore,
) -> bool:
    """Seed Redis from main weights once if any class has no league keys."""
    if store.has_league_keys():
        logger.debug("Redis league already bootstrapped; skipping seed")
        return False

    seed_league_from_mains(algorithm, store)
    return True


def bootstrap_league_at_training_start(
    algorithm,
    store: RedisModelStore | None = None,
) -> bool:
    """Run once at training start: seed Redis from mains if needed."""
    league_store = store or RedisModelStore()
    return ensure_league_bootstrapped(algorithm, league_store)
