"""Sample ghost policy weights from the Redis league."""

from __future__ import annotations

import logging
import random

from league_codec import decode_league_blob
from league_initialization.constants import CHAR_CLASSES, GHOST_SLOTS, ghost_policy_id
from league_initialization.module_state import set_module_state, sync_module_weights
from model_store import RedisModelStore

logger = logging.getLogger(__name__)


def weighted_recent_sample(keys, k, decay=0.90):
    """Pick up to k distinct keys, favoring more recent iteration numbers."""
    if not keys:
        return []

    latest = max(int(key.rsplit(":", 1)[-1]) for key in keys)
    pool = list(keys)
    picked = []
    for _ in range(min(k, len(pool))):
        weights = [decay ** (latest - int(key.rsplit(":", 1)[-1])) for key in pool]
        choice = random.choices(pool, weights=weights, k=1)[0]
        picked.append(choice)
        pool.remove(choice)
    return picked


def load_random_league(
    algorithm,
    char_class: str | None = None,
    *,
    store: RedisModelStore | None = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    ghost_slots: int = GHOST_SLOTS,
    sync: bool = True,
) -> dict | list[dict]:
    """Load sampled ghost weights from Redis into RLModules.

    When ``char_class`` is omitted, loads every class in one SCAN + one MGET batch.
    """
    league_store = store or RedisModelStore(host=redis_host, port=redis_port)
    if char_class is not None:
        return _load_random_league_class(
            algorithm,
            char_class,
            store=league_store,
            ghost_slots=ghost_slots,
            sync=sync,
        )
    return _load_random_league_all_classes(
        algorithm,
        store=league_store,
        ghost_slots=ghost_slots,
        sync=sync,
    )


def _load_random_league_class(
    algorithm,
    char_class: str,
    *,
    store: RedisModelStore,
    ghost_slots: int,
    sync: bool,
) -> dict:
    """Load sampled ghost weights for one class from Redis into RLModules."""
    all_keys = store.list_class_keys(char_class)

    if not all_keys:
        logger.error(
            "No Redis league keys for class %s; bootstrap should run before ghost load",
            char_class,
        )
        return {
            "char_class": char_class,
            "source": "none",
            "loaded": 0,
            "keys": [],
        }

    sampled_keys = weighted_recent_sample(all_keys, ghost_slots)
    loaded = 0

    for idx, redis_key in enumerate(sampled_keys):
        safetensor_bytes = store.redis.get(redis_key)
        if safetensor_bytes is None:
            logger.warning("Missing Redis weight key %s", redis_key)
            continue

        weights_dict = decode_league_blob(safetensor_bytes)
        set_module_state(algorithm, ghost_policy_id(char_class, idx), weights_dict)
        loaded += 1

    if loaded and sync:
        sync_module_weights(algorithm)

    return {
        "char_class": char_class,
        "source": "redis",
        "loaded": loaded,
        "keys": sampled_keys,
    }


def _load_random_league_all_classes(
    algorithm,
    *,
    store: RedisModelStore,
    ghost_slots: int,
    sync: bool,
) -> list[dict]:
    """Refresh ghosts for every class using one SCAN and one batch MGET."""
    keys_by_class = store.list_keys_by_class()
    sampled_by_class: dict[str, list[str]] = {}
    all_sampled: list[str] = []
    for char_class in CHAR_CLASSES:
        all_keys = keys_by_class.get(char_class, [])
        if not all_keys:
            logger.error(
                "No Redis league keys for class %s; bootstrap should run before ghost load",
                char_class,
            )
            sampled_by_class[char_class] = []
            continue
        sampled = weighted_recent_sample(all_keys, ghost_slots)
        sampled_by_class[char_class] = sampled
        all_sampled.extend(sampled)

    blobs = store.mget_bytes(all_sampled)

    results: list[dict] = []
    any_loaded = False
    for char_class in CHAR_CLASSES:
        sampled_keys = sampled_by_class[char_class]
        loaded = 0
        for idx, redis_key in enumerate(sampled_keys):
            safetensor_bytes = blobs.get(redis_key)
            if safetensor_bytes is None:
                logger.warning("Missing Redis weight key %s", redis_key)
                continue
            weights_dict = decode_league_blob(safetensor_bytes)
            set_module_state(algorithm, ghost_policy_id(char_class, idx), weights_dict)
            loaded += 1
        any_loaded = any_loaded or loaded > 0
        results.append(
            {
                "char_class": char_class,
                "source": "redis" if sampled_keys else "none",
                "loaded": loaded,
                "keys": sampled_keys,
            }
        )

    if any_loaded and sync:
        sync_module_weights(algorithm)

    return results


def refresh_ghosts_for_all_classes(
    algorithm,
    store: RedisModelStore,
    *,
    ghost_slots: int = GHOST_SLOTS,
    sync: bool = True,
) -> list[dict]:
    """Refresh ghosts for every class using one SCAN and one batch MGET."""
    return load_random_league(
        algorithm,
        store=store,
        ghost_slots=ghost_slots,
        sync=sync,
    )
