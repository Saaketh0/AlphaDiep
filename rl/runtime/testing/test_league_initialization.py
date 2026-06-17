"""Tests for league_initialization bootstrap."""

from __future__ import annotations

import fnmatch

import torch

from league_initialization.bootstrap import ensure_league_bootstrapped, seed_league_from_mains
from league_initialization.constants import CHAR_CLASSES, ghost_policy_id, main_policy_id
from model_store import RedisModelStore


class FakeRedis:
    def __init__(self):
        self.data: dict[str, bytes] = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)

    unlink = delete

    def keys(self, pattern):
        return [key.encode("utf-8") for key in self.data if fnmatch.fnmatch(key, pattern)]

    def scan_iter(self, match=None):
        pattern = match or "*"
        for key in list(self.data):
            if fnmatch.fnmatch(key, pattern):
                yield key.encode("utf-8")


class FakeModule:
    def __init__(self, state: dict | None = None):
        self._state = state or {"weight": torch.tensor([1.0])}

    def get_state(self):
        return {name: value.clone() for name, value in self._state.items()}

    def set_state(self, state_dict):
        self._state = {name: value.clone() for name, value in state_dict.items()}


class FakeAlgorithm:
    def __init__(self):
        self.modules = {
            main_policy_id(char_class): FakeModule({"weight": torch.tensor([float(index + 1)])})
            for index, char_class in enumerate(CHAR_CLASSES)
        }
        self.synced = False

    def get_module(self, module_id):
        return self.modules[module_id]

    class _Workers:
        def __init__(self, algorithm):
            self.algorithm = algorithm

        def sync_weights(self):
            self.algorithm.synced = True

    @property
    def workers(self):
        return self._Workers(self)


def _store() -> RedisModelStore:
    return RedisModelStore(client=FakeRedis(), snapshot_every=0)


def test_seed_league_from_mains_writes_fifty_identical_entries_per_class():
    algorithm = FakeAlgorithm()
    store = _store()

    written = seed_league_from_mains(algorithm, store, count=50)

    assert set(written) == set(CHAR_CLASSES)
    assert len(written["A"]) == 50
    assert store.latest_by_class() == {char_class: 49 for char_class in CHAR_CLASSES}
    assert store.next_iteration() == 50

    main_blob = store.redis.get("policy:A:0")
    latest_blob = store.redis.get("policy:A:49")
    assert main_blob == latest_blob
    assert store.load_state_dict("A", 0)["weight"].item() == 1.0


def test_ensure_league_bootstrapped_is_idempotent():
    algorithm = FakeAlgorithm()
    store = _store()

    assert ensure_league_bootstrapped(algorithm, store) is True
    assert ensure_league_bootstrapped(algorithm, store) is False
    assert store.latest_by_class()["B"] == 49


def test_seed_uses_live_main_state_per_class():
    algorithm = FakeAlgorithm()
    store = _store()

    seed_league_from_mains(algorithm, store, count=1)

    for index, char_class in enumerate(CHAR_CLASSES):
        assert store.load_state_dict(char_class, 0)["weight"].item() == float(index + 1)
