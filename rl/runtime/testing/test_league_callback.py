"""Tests for LeagueBootstrapCallback.on_algorithm_init batching behavior."""

from __future__ import annotations

import torch

from league_initialization import callback as callback_module
from league_initialization.callback import LeagueBootstrapCallback
from league_initialization.constants import CHAR_CLASSES, ghost_policy_id, main_policy_id
from model_store import RedisModelStore

from testing.test_league_initialization import FakeRedis


class CountingRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.scan_calls = 0
        self.mget_calls = 0
        self.get_calls = 0

    def scan_iter(self, match=None):
        self.scan_calls += 1
        return super().scan_iter(match=match)

    def get(self, key):
        self.get_calls += 1
        return super().get(key)

    def mget(self, keys):
        self.mget_calls += 1
        return [self.data.get(key) for key in keys]


class FakeModule:
    def __init__(self, value: float = 0.0):
        self._state = {"weight": torch.tensor([value])}

    def get_state(self):
        return {name: value.clone() for name, value in self._state.items()}

    def set_state(self, state_dict):
        self._state = {name: value.clone() for name, value in state_dict.items()}


class FakeAlgorithm:
    def __init__(self):
        self.modules = {}
        for index, char_class in enumerate(CHAR_CLASSES):
            self.modules[main_policy_id(char_class)] = FakeModule(float(index + 1))
            for slot in range(4):
                self.modules[ghost_policy_id(char_class, slot)] = FakeModule()
        self.sync_calls = 0

    def get_module(self, module_id):
        return self.modules[module_id]

    class _Workers:
        def __init__(self, algorithm):
            self.algorithm = algorithm

        def sync_weights(self):
            self.algorithm.sync_calls += 1

    @property
    def workers(self):
        return self._Workers(self)


def _seeded_store() -> RedisModelStore:
    """Return a fake Redis-backed store pre-populated with one ghost weight per class."""
    store = RedisModelStore(client=FakeRedis(), snapshot_every=0)
    from safetensors.torch import save

    for char_class in CHAR_CLASSES:
        blob = save({"weight": torch.tensor([5.0])})
        store.redis.set(store.key(char_class, 0), blob)
    return store


def test_on_algorithm_init_syncs_once_for_all_classes(monkeypatch):
    """All four ghost loads should batch into a single sync_weights call."""
    algorithm = FakeAlgorithm()
    store = _seeded_store()

    monkeypatch.setattr(callback_module, "RedisModelStore", lambda *a, **k: store)
    monkeypatch.setattr(
        callback_module,
        "hydrate_redis_from_disk",
        lambda _store: 0,
    )

    cb = LeagueBootstrapCallback()
    cb.on_algorithm_init(algorithm=algorithm)

    assert algorithm.sync_calls == 1


def test_on_algorithm_init_uses_one_scan_and_one_mget(monkeypatch):
    """Init ghost load should batch Redis reads into one SCAN + one MGET."""
    algorithm = FakeAlgorithm()
    client = CountingRedis()
    store = RedisModelStore(client=client, snapshot_every=0)
    from safetensors.torch import save

    for char_class in CHAR_CLASSES:
        for iteration in range(4):
            blob = save({"weight": torch.tensor([5.0])})
            client.set(store.key(char_class, iteration), blob)

    monkeypatch.setattr(callback_module, "RedisModelStore", lambda *a, **k: store)
    monkeypatch.setattr(
        callback_module,
        "hydrate_redis_from_disk",
        lambda _store: 0,
    )
    monkeypatch.setattr(store, "has_league_keys", lambda: True)
    monkeypatch.setattr(store, "warm_iteration_cache", lambda: 0)

    client.scan_calls = 0
    client.mget_calls = 0
    client.get_calls = 0

    cb = LeagueBootstrapCallback()
    cb.on_algorithm_init(algorithm=algorithm)

    assert client.scan_calls == 1
    assert client.mget_calls == 1
    assert client.get_calls == 0
    assert algorithm.sync_calls == 1


def test_on_algorithm_init_loads_ghost_weights_for_each_class(monkeypatch):
    """Each class should have at least one ghost module hydrated from Redis."""
    algorithm = FakeAlgorithm()
    store = _seeded_store()

    monkeypatch.setattr(callback_module, "RedisModelStore", lambda *a, **k: store)
    monkeypatch.setattr(
        callback_module,
        "hydrate_redis_from_disk",
        lambda _store: 0,
    )

    cb = LeagueBootstrapCallback()
    cb.on_algorithm_init(algorithm=algorithm)

    for char_class in CHAR_CLASSES:
        ghost = algorithm.modules[ghost_policy_id(char_class, 0)]
        assert ghost.get_state()["weight"].item() == 5.0
