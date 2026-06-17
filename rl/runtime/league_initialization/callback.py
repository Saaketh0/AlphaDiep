"""RLlib callback: hydrate the league on init, then save mains + refresh ghosts each iteration."""

from __future__ import annotations

import logging
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from model_store import RedisModelStore
from random_pick import load_random_league

from .disk_store import hydrate_redis_from_disk
from .league_loop import ghost_refresh_interval, save_mains_and_refresh_ghosts
from training_metadata import read_metadata, write_metadata, league_metadata_path, ResumeProvenanceError

logger = logging.getLogger(__name__)

_SEED_HINT = "League empty — run: python -m league_initialization.seed_league_cache"


class LeagueBootstrapCallback(DefaultCallbacks):
    """Load ghosts from the pre-seeded Redis/SSD league, then maintain it each iteration."""

    def __init__(self):
        super().__init__()
        self._store: RedisModelStore | None = None
        # Cache the refresh interval once; the env var is fixed for the lifetime of a run,
        # so parsing it every iteration is wasted work in ``save_mains_and_refresh_ghosts``.
        self._ghost_refresh_interval = ghost_refresh_interval()

    def on_algorithm_init(self, *, algorithm, **kwargs):
        # snapshot_every=0 disables the periodic export inside save_class; league_loop
        # exports every iteration from the in-memory state instead.
        self._store = RedisModelStore(snapshot_every=0)

        if not self._store.has_league_keys():
            hydrate_redis_from_disk(self._store)
            if not self._store.has_league_keys():
                raise RuntimeError(_SEED_HINT)

        # Cache the latest iteration so per-iteration save_mains_to_redis skips a SCAN.
        self._store.warm_iteration_cache()

        load_random_league(algorithm, store=self._store, sync=True)

    def on_train_result(self, *, algorithm, result=None, **kwargs):
        if self._store is None:
            self._store = RedisModelStore(snapshot_every=0)
            self._store.warm_iteration_cache()
        league_result = save_mains_and_refresh_ghosts(
            algorithm,
            self._store,
            refresh_interval=self._ghost_refresh_interval,
        )
        if isinstance(result, dict):
            result["league_iteration"] = league_result["iteration"]
            result["league_ghosts_refreshed"] = league_result.get("ghosts_refreshed", True)
        self._update_league_metadata(league_result["iteration"])

    def _update_league_metadata(self, iteration: int) -> None:
        metadata_path = os.environ.get("DIEP_TRAINING_RUN_METADATA")
        if not metadata_path:
            return
        try:
            metadata = read_metadata(metadata_path).with_latest_iteration(iteration)
            write_metadata(metadata_path, metadata)
            write_metadata(league_metadata_path(), metadata)
        except ResumeProvenanceError as exc:
            logger.warning("Could not update training provenance metadata: %s", exc)
