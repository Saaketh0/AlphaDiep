"""Per-iteration league loop: save mains to Redis+SSD, refresh ghosts."""

from __future__ import annotations

import logging
import os

from model_store import RedisModelStore
from random_pick import refresh_ghosts_for_all_classes

from .constants import CHAR_CLASSES, main_policy_id
from .module_state import get_module_state

logger = logging.getLogger(__name__)

GHOST_REFRESH_INTERVAL_ENV_VAR = "DIEP_GHOST_REFRESH_INTERVAL"
_DEFAULT_GHOST_REFRESH_INTERVAL = 5


def ghost_refresh_interval() -> int:
    """Read DIEP_GHOST_REFRESH_INTERVAL (default 5; minimum 1) on each call."""
    raw = os.environ.get(GHOST_REFRESH_INTERVAL_ENV_VAR)
    if not raw:
        return _DEFAULT_GHOST_REFRESH_INTERVAL
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to default %d",
            GHOST_REFRESH_INTERVAL_ENV_VAR,
            raw,
            _DEFAULT_GHOST_REFRESH_INTERVAL,
        )
        return _DEFAULT_GHOST_REFRESH_INTERVAL


def collect_main_states(algorithm) -> dict[str, dict]:
    """Snapshot each main_class_{X} RLModule state."""
    return {
        char_class: get_module_state(algorithm, main_policy_id(char_class))
        for char_class in CHAR_CLASSES
    }


def save_mains_to_redis(algorithm, store: RedisModelStore) -> int:
    """Save all main weights at the next iteration index. Returns that index.

    ``save_all`` encodes each class once and returns the resulting blobs; we hand
    those same bytes to ``export_class`` so SSD never re-runs the league codec.
    """
    iteration = store.next_iteration()
    states = collect_main_states(algorithm)
    blobs = store.save_all(states, iteration)
    for char_class, blob in blobs.items():
        store.export_class(char_class, iteration, blob=blob)
    return iteration


def refresh_all_ghosts(algorithm, store: RedisModelStore) -> list[dict]:
    """Resample ghost weights for every class, syncing env runners once."""
    # Batched path: one SCAN partitioned by class + one MGET for all sampled keys,
    # then a single sync_module_weights call inside refresh_ghosts_for_all_classes.
    return refresh_ghosts_for_all_classes(algorithm, store, sync=True)


def save_mains_and_refresh_ghosts(
    algorithm,
    store: RedisModelStore,
    *,
    refresh_interval: int | None = None,
) -> dict:
    """Save mains every iteration; refresh ghosts only every DIEP_GHOST_REFRESH_INTERVAL.

    Mains always enter the Redis/SSD pool so the league keeps growing, but the
    expensive ghost reload + sync_module_weights only runs on interval boundaries.

    ``refresh_interval`` lets long-lived callers (e.g. ``LeagueBootstrapCallback``)
    pass a cached value so the env var lookup runs once at init instead of every
    iteration. When omitted, the env var is re-read for backwards compatibility.
    """
    iteration = save_mains_to_redis(algorithm, store)
    interval = refresh_interval if refresh_interval is not None else ghost_refresh_interval()
    interval = max(1, int(interval))
    should_refresh = iteration % interval == 0
    if should_refresh:
        ghost_results = refresh_all_ghosts(algorithm, store)
        logger.info(
            "League iteration %d: saved mains, refreshed ghosts (interval=%d)",
            iteration,
            interval,
        )
    else:
        ghost_results = []
        logger.debug(
            "League iteration %d: saved mains, skipped ghost refresh (interval=%d)",
            iteration,
            interval,
        )
    return {
        "iteration": iteration,
        "ghosts": ghost_results,
        "ghosts_refreshed": should_refresh,
    }
