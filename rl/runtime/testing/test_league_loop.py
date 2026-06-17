"""Tests for league_loop per-iteration helpers and disk_store hydration."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import torch
from safetensors.torch import save_file

from league_initialization.constants import CHAR_CLASSES, ghost_policy_id, main_policy_id
from league_initialization.disk_store import export_league_to_disk, hydrate_redis_from_disk
from league_initialization.league_loop import (
    GHOST_REFRESH_INTERVAL_ENV_VAR,
    ghost_refresh_interval,
    save_mains_and_refresh_ghosts,
    save_mains_to_redis,
)
from model_store import LEAGUE_EXPORT_SUFFIX, RedisModelStore


class FakePipeline:
    """Minimal redis-py-style pipeline: buffer SETs, flush on execute()."""

    def __init__(self, parent):
        self._parent = parent
        self._ops: list[tuple[str, str, bytes]] = []

    def set(self, key, value):
        self._ops.append(("set", key, value))
        return self

    def execute(self):
        for op, key, value in self._ops:
            if op == "set":
                self._parent.set(key, value)
        self._ops.clear()
        return []


class FakeRedis:
    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.pipeline_calls = 0

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

    def pipeline(self, transaction=True):
        self.pipeline_calls += 1
        return FakePipeline(self)

    def mset(self, mapping):
        for key, value in mapping.items():
            self.data[key] = value

    def exists(self, key):
        return 1 if key in self.data else 0


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


def _store(tmp_path) -> RedisModelStore:
    return RedisModelStore(client=FakeRedis(), snapshot_every=0, snapshot_dir=tmp_path)


def test_save_mains_to_redis_writes_and_exports(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    iteration = save_mains_to_redis(algorithm, store)

    assert iteration == 0
    assert store.latest_by_class() == {char_class: 0 for char_class in CHAR_CLASSES}
    for index, char_class in enumerate(CHAR_CLASSES):
        export = tmp_path / char_class / f"iter_0{LEAGUE_EXPORT_SUFFIX}"
        assert export.exists()
        assert store.load_state_dict(char_class, 0)["weight"].item() == float(index + 1)


def test_save_mains_does_not_reread_redis_for_export(tmp_path, monkeypatch):
    """SSD export should reuse the in-memory state, not GET back from Redis."""
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    def _fail_load(*_args, **_kwargs):
        raise AssertionError("export_class must not re-read Redis when state_dict is provided")

    monkeypatch.setattr(store, "load_state_dict", _fail_load)

    save_mains_to_redis(algorithm, store)

    for index, char_class in enumerate(CHAR_CLASSES):
        export = tmp_path / char_class / f"iter_0{LEAGUE_EXPORT_SUFFIX}"
        assert export.exists()


def test_save_all_encodes_once_and_pipelines_redis(tmp_path, monkeypatch):
    """P4a/b: ``save_all`` runs the codec once per class and batches SETs via pipeline()."""
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    encode_calls = {"count": 0}
    import model_store as model_store_module
    real_encode = model_store_module.encode_league_blob

    def counting_encode(state_dict):
        encode_calls["count"] += 1
        return real_encode(state_dict)

    monkeypatch.setattr(model_store_module, "encode_league_blob", counting_encode)

    save_mains_to_redis(algorithm, store)

    # One encode per main class (4), not 8 like the pre-refactor double-encode.
    assert encode_calls["count"] == len(CHAR_CLASSES)
    # ``save_all`` batched the SETs through a single pipeline.
    assert store.redis.pipeline_calls == 1


def test_save_mains_redis_and_ssd_bytes_match(tmp_path):
    """The exported SSD file must be byte-identical to the Redis value."""
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    save_mains_to_redis(algorithm, store)

    for char_class in CHAR_CLASSES:
        redis_bytes = store.redis.get(store.key(char_class, 0))
        ssd_bytes = (tmp_path / char_class / f"iter_0{LEAGUE_EXPORT_SUFFIX}").read_bytes()
        assert redis_bytes is not None
        assert redis_bytes == ssd_bytes


def test_save_all_returns_encoded_blobs(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)
    states = {char_class: algorithm.modules[main_policy_id(char_class)].get_state() for char_class in CHAR_CLASSES}

    blobs = store.save_all(states, iteration=0)

    assert set(blobs) == set(CHAR_CLASSES)
    for char_class in CHAR_CLASSES:
        assert isinstance(blobs[char_class], bytes)
        assert blobs[char_class] == store.redis.get(store.key(char_class, 0))


def test_save_mains_and_refresh_ghosts_syncs_once(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    result = save_mains_and_refresh_ghosts(algorithm, store)

    assert result["iteration"] == 0
    # One sync for the whole ghost refresh, not one per class.
    assert algorithm.sync_calls == 1
    for index, char_class in enumerate(CHAR_CLASSES):
        ghost = algorithm.modules[ghost_policy_id(char_class, 0)]
        assert ghost.get_state()["weight"].item() == float(index + 1)


def test_hydrate_redis_from_disk_repopulates_empty_redis(tmp_path):
    # Write safetensors exports directly to disk, leave Redis empty.
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        save_file({"weight": torch.tensor([3.0])}, class_dir / "iter_0.safetensors")
        save_file({"weight": torch.tensor([4.0])}, class_dir / "iter_1.safetensors")

    store = _store(tmp_path)
    assert store.has_league_keys() is False

    written = hydrate_redis_from_disk(store)

    assert written == len(CHAR_CLASSES) * 2
    assert store.has_league_keys() is True
    assert store.load_state_dict("A", 1)["weight"].item() == 4.0


def test_hydrate_cold_redis_uses_single_mset(tmp_path):
    """P11: cold Redis hydrate must batch all writes through one MSET."""
    from league_codec import encode_league_blob

    class CountingRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.mset_calls = 0
            self.set_calls = 0
            self.get_calls = 0

        def set(self, key, value):
            self.set_calls += 1
            super().set(key, value)

        def get(self, key):
            self.get_calls += 1
            return super().get(key)

        def mset(self, mapping):
            self.mset_calls += 1
            super().mset(mapping)

    payload = encode_league_blob({"weight": torch.tensor([1.0])})
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        (class_dir / "iter_0.safetensors.zst").write_bytes(payload)

    client = CountingRedis()
    store = RedisModelStore(client=client, snapshot_every=0, snapshot_dir=tmp_path)
    written = hydrate_redis_from_disk(store)

    assert written == len(CHAR_CLASSES)
    assert client.mset_calls == 1
    assert client.set_calls == 0  # never fell back to per-key SET on the cold path

    # Re-hydrate after the cache is warm must be a no-op (idempotent path).
    re_written = hydrate_redis_from_disk(store)
    assert re_written == 0


def test_export_league_uses_single_scan(tmp_path):
    """P11: ``export_league_to_disk`` should use one ``list_keys_by_class`` SCAN."""

    class CountingRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.scan_calls = 0

        def scan_iter(self, match=None):
            self.scan_calls += 1
            return super().scan_iter(match=match)

    algorithm = FakeAlgorithm()
    client = CountingRedis()
    store = RedisModelStore(client=client, snapshot_every=0, snapshot_dir=tmp_path)
    save_mains_to_redis(algorithm, store)

    client.scan_calls = 0
    exported = export_league_to_disk(store)

    assert len(exported) == len(CHAR_CLASSES)
    assert client.scan_calls == 1


def test_hydrate_redis_from_disk_picks_up_zst_exports(tmp_path):
    """The hydrate glob must accept the new ``iter_N.safetensors.zst`` exports."""
    from league_codec import encode_league_blob

    payload = {"weight": torch.tensor([4.5])}
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        (class_dir / "iter_0.safetensors.zst").write_bytes(encode_league_blob(payload))

    store = _store(tmp_path)
    written = hydrate_redis_from_disk(store)

    assert written == len(CHAR_CLASSES)
    for char_class in CHAR_CLASSES:
        assert store.load_state_dict(char_class, 0)["weight"].item() == 4.5


def test_hydrate_redis_from_disk_prefers_zst_over_legacy(tmp_path):
    """When both legacy and zst exports exist for one iteration, prefer the zst payload."""
    from league_codec import encode_league_blob

    class_dir = tmp_path / "A"
    class_dir.mkdir(parents=True)
    save_file({"weight": torch.tensor([1.0])}, class_dir / "iter_0.safetensors")
    (class_dir / "iter_0.safetensors.zst").write_bytes(
        encode_league_blob({"weight": torch.tensor([2.0])})
    )

    store = _store(tmp_path)
    hydrate_redis_from_disk(store)

    assert store.load_state_dict("A", 0)["weight"].item() == 2.0


def test_hydrate_is_idempotent(tmp_path):
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        save_file({"weight": torch.tensor([3.0])}, class_dir / "iter_0.safetensors")

    store = _store(tmp_path)
    assert hydrate_redis_from_disk(store) == len(CHAR_CLASSES)
    assert hydrate_redis_from_disk(store) == 0


def test_export_league_to_disk_writes_all_keys(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)
    save_mains_to_redis(algorithm, store)

    exported = export_league_to_disk(store)

    assert len(exported) == len(CHAR_CLASSES)
    assert all(path.exists() for path in exported)


def test_warmed_cache_avoids_scan_on_next_iteration(tmp_path, monkeypatch):
    """After warm_iteration_cache, save_mains_to_redis must not call latest_iteration."""
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    save_mains_to_redis(algorithm, store)
    assert store._cached_latest == 0

    def _fail_latest():
        raise AssertionError("latest_iteration must not be called when cache is warm")

    monkeypatch.setattr(store, "latest_iteration", _fail_latest)

    iteration = save_mains_to_redis(algorithm, store)
    assert iteration == 1
    assert store._cached_latest == 1


def test_ghost_refresh_interval_default_and_env_override(monkeypatch):
    monkeypatch.delenv(GHOST_REFRESH_INTERVAL_ENV_VAR, raising=False)
    assert ghost_refresh_interval() == 5
    monkeypatch.setenv(GHOST_REFRESH_INTERVAL_ENV_VAR, "3")
    assert ghost_refresh_interval() == 3
    monkeypatch.setenv(GHOST_REFRESH_INTERVAL_ENV_VAR, "0")
    assert ghost_refresh_interval() == 1
    monkeypatch.setenv(GHOST_REFRESH_INTERVAL_ENV_VAR, "not-an-int")
    assert ghost_refresh_interval() == 5


def test_save_mains_and_refresh_ghosts_throttles_ghost_refresh(tmp_path, monkeypatch):
    """Mains save every iteration; ghosts only refresh on interval boundaries."""
    monkeypatch.setenv(GHOST_REFRESH_INTERVAL_ENV_VAR, "5")
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    refreshed_iterations: list[int] = []
    for iteration in range(7):
        result = save_mains_and_refresh_ghosts(algorithm, store)
        assert result["iteration"] == iteration
        if result["ghosts_refreshed"]:
            refreshed_iterations.append(iteration)

    # Iterations 0 and 5 sit on the interval boundary; 1-4 and 6 should be skipped.
    assert refreshed_iterations == [0, 5]
    assert algorithm.sync_calls == 2


def test_save_mains_and_refresh_ghosts_uses_cached_interval(tmp_path, monkeypatch):
    """Passing ``refresh_interval`` skips the env var lookup entirely."""
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    monkeypatch.setenv(GHOST_REFRESH_INTERVAL_ENV_VAR, "1")  # would normally refresh every iter
    import league_initialization.league_loop as league_loop_module

    def _fail_env_read():
        raise AssertionError("ghost_refresh_interval must not be called when refresh_interval is cached")

    monkeypatch.setattr(league_loop_module, "ghost_refresh_interval", _fail_env_read)

    refreshed_iterations: list[int] = []
    for _ in range(4):
        result = save_mains_and_refresh_ghosts(algorithm, store, refresh_interval=3)
        if result["ghosts_refreshed"]:
            refreshed_iterations.append(result["iteration"])

    # interval=3 → refresh at iters 0 and 3 only.
    assert refreshed_iterations == [0, 3]


def test_has_league_keys_probes_iter0_without_scan():
    """Seeded leagues with iter-0 keys should probe via EXISTS, not SCAN."""

    class CountingRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.scan_calls = 0
            self.exists_calls = 0

        def scan_iter(self, match=None):
            self.scan_calls += 1
            return super().scan_iter(match=match)

        def exists(self, key):
            self.exists_calls += 1
            return 1 if key in self.data else 0

    client = CountingRedis()
    store = RedisModelStore(client=client, snapshot_every=0)
    for char_class in CHAR_CLASSES:
        store.redis.set(store.key(char_class, 0), b"x")

    client.scan_calls = 0
    client.exists_calls = 0
    assert store.has_league_keys() is True
    assert client.scan_calls == 0
    assert client.exists_calls == len(CHAR_CLASSES)


def test_has_league_keys_falls_back_to_scan_without_iter0():
    """When iter-0 keys are gone, fall back to one SCAN over policy:*:*."""

    class CountingRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.scan_calls = 0

        def scan_iter(self, match=None):
            self.scan_calls += 1
            return super().scan_iter(match=match)

        def exists(self, key):
            return 1 if key in self.data else 0

    client = CountingRedis()
    store = RedisModelStore(client=client, snapshot_every=0)
    for char_class in CHAR_CLASSES:
        store.redis.set(store.key(char_class, 5), b"x")

    client.scan_calls = 0
    assert store.has_league_keys() is True
    assert client.scan_calls == 1


def test_drop_old_removes_ssd_export(tmp_path):
    """When Redis trims an old key, the matching SSD export must also be deleted."""
    store = RedisModelStore(
        client=FakeRedis(),
        snapshot_every=1,
        snapshot_dir=tmp_path,
        window_size=3,
    )
    state = {"weight": torch.tensor([1.0])}

    # Saves iter 0..3; window_size=3 means iter 0 ages out when iter 3 is saved.
    for iteration in range(4):
        store.save_class("A", state, iteration)

    iter0_export = tmp_path / "A" / f"iter_0{LEAGUE_EXPORT_SUFFIX}"
    iter1_export = tmp_path / "A" / f"iter_1{LEAGUE_EXPORT_SUFFIX}"
    assert store.redis.get(store.key("A", 0)) is None
    assert iter0_export.exists() is False
    assert store.redis.get(store.key("A", 1)) is not None
    assert iter1_export.exists()


def test_drop_old_removes_zst_export(tmp_path):
    """The trim path also deletes future ``.safetensors.zst`` exports."""
    store = RedisModelStore(
        client=FakeRedis(),
        snapshot_every=1,
        snapshot_dir=tmp_path,
        window_size=2,
    )
    state = {"weight": torch.tensor([1.0])}

    store.save_class("A", state, 0)
    zst_path = tmp_path / "A" / "iter_0.safetensors.zst"
    zst_path.write_bytes(b"fake-zstd-payload")
    assert zst_path.exists()

    store.save_class("A", state, 1)
    store.save_class("A", state, 2)
    assert zst_path.exists() is False


def test_export_class_skips_mkdir_when_parent_exists(tmp_path, monkeypatch):
    store = _store(tmp_path)
    state = {"weight": torch.tensor([1.0])}
    mkdir_calls = {"count": 0}
    real_mkdir = Path.mkdir

    def counting_mkdir(self, *args, **kwargs):
        mkdir_calls["count"] += 1
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", counting_mkdir)

    store.export_class("A", 0, state_dict=state)
    store.export_class("A", 1, state_dict=state)
    store.export_class("B", 0, state_dict=state)

    assert mkdir_calls["count"] == 2
