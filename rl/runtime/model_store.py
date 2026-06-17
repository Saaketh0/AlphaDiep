"""Tiny Redis + safetensors model weight store for RLlib experiments.

This is intentionally boring: save PyTorch state_dicts to Redis, keep a rolling
history per class, optionally export safetensors files to SSD, and provide a
small RLlib checkpoint helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import redis

from league_codec import decode_league_blob, encode_league_blob
from league_initialization.tensor_utils import cpu_state_dict


DEFAULT_CLASSES = ("A", "B", "C", "D")

# New saves write zstd-wrapped (int8 by default) safetensors; legacy uncompressed
# fp32 ``.safetensors`` files are still loadable via the codec for backward compat.
LEAGUE_EXPORT_SUFFIX = ".safetensors.zst"
LEGACY_EXPORT_SUFFIX = ".safetensors"

# .../diepcustom/rl/runtime/model_store.py -> .../diepcustom/training_data/redis
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "training_data" / "redis"


class MissingWeights(KeyError):
    """Raised when a requested Redis weight key is missing."""


class RedisModelStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        *,
        classes: Iterable[str] = DEFAULT_CLASSES,
        window_size: int = 50,
        snapshot_every: int = 10,
        key_prefix: str = "policy",
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        client=None,
    ):
        self.redis = client or redis.Redis(host=host, port=port)
        self.classes = tuple(classes)
        self.window_size = int(window_size)
        self.snapshot_every = int(snapshot_every)
        self.key_prefix = key_prefix
        self.snapshot_dir = Path(snapshot_dir)
        # Cached max iteration across all classes; populated by warm_iteration_cache
        # and kept fresh by save_class so next_iteration() can skip the full SCAN.
        self._cached_latest: int | None = None

    def key(self, char_class: str, iteration: int) -> str:
        return f"{self.key_prefix}:{char_class}:{int(iteration)}"

    def _export_path(self, char_class: str, iteration: int) -> Path:
        """Canonical SSD path for new league exports (zstd-wrapped safetensors)."""
        return self.snapshot_dir / char_class / f"iter_{int(iteration)}{LEAGUE_EXPORT_SUFFIX}"

    def _legacy_export_path(self, char_class: str, iteration: int) -> Path:
        """Pre-codec SSD path (uncompressed fp32 safetensors). Kept for trim/hydrate paths."""
        return self.snapshot_dir / char_class / f"iter_{int(iteration)}{LEGACY_EXPORT_SUFFIX}"

    def encode_state_dict(self, state_dict: dict) -> bytes:
        """Single entry point: detach to CPU then encode via the league codec.

        All Redis/SSD writes go through this so each weight set is serialized
        exactly once per save, even when the same blob is reused for both Redis
        and SSD export.
        """
        return encode_league_blob(cpu_state_dict(state_dict))

    def save_class_blob(self, char_class: str, blob: bytes, iteration: int) -> str:
        """Write a pre-encoded blob to Redis and bookkeep window + latest cache."""
        key = self.key(char_class, iteration)
        self.redis.set(key, blob)
        self._drop_old(char_class, iteration)
        self._bump_cached_latest(iteration)
        return key

    def export_class_blob(
        self,
        char_class: str,
        iteration: int,
        blob: bytes,
        export_path: str | Path | None = None,
    ) -> Path:
        """Write a pre-encoded blob to SSD; never re-encodes."""
        path = Path(export_path) if export_path else self._export_path(char_class, iteration)
        parent = path.parent
        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return path

    def save_class(self, char_class: str, state_dict: dict, iteration: int) -> str:
        blob = self.encode_state_dict(state_dict)
        key = self.save_class_blob(char_class, blob, iteration)
        if self.snapshot_every > 0 and int(iteration) % self.snapshot_every == 0:
            self.export_class_blob(char_class, iteration, blob)
        return key

    def save_all(self, state_dicts_by_class: dict[str, dict], iteration: int) -> dict[str, bytes]:
        """Encode each class once, batch SETs via ``pipeline()``, return blobs.

        Returns ``{char_class: encoded_bytes}`` so callers can hand the same bytes
        to ``export_class_blob`` for SSD without re-encoding.
        """
        blobs: dict[str, bytes] = {
            char_class: self.encode_state_dict(state_dict)
            for char_class, state_dict in state_dicts_by_class.items()
        }
        if not blobs:
            return blobs

        pipeline = getattr(self.redis, "pipeline", None)
        if callable(pipeline):
            # ``transaction=False`` keeps this an unsynchronized batch of SETs;
            # league writes are independent per key, so atomicity buys nothing.
            try:
                pipe = pipeline(transaction=False)
            except TypeError:
                pipe = pipeline()
            for char_class, blob in blobs.items():
                pipe.set(self.key(char_class, iteration), blob)
            pipe.execute()
        else:
            for char_class, blob in blobs.items():
                self.redis.set(self.key(char_class, iteration), blob)

        for char_class in blobs:
            self._drop_old(char_class, iteration)
        self._bump_cached_latest(iteration)

        if self.snapshot_every > 0 and int(iteration) % self.snapshot_every == 0:
            for char_class, blob in blobs.items():
                self.export_class_blob(char_class, iteration, blob)

        return blobs

    def _bump_cached_latest(self, iteration: int) -> None:
        iteration_int = int(iteration)
        if self._cached_latest is None or iteration_int > self._cached_latest:
            self._cached_latest = iteration_int

    def load_state_dict(self, char_class: str, iteration: int) -> dict:
        key = self.key(char_class, iteration)
        raw = self.redis.get(key)
        if raw is None:
            raise MissingWeights(f"missing weights: {key}")
        return decode_league_blob(raw)

    def load_model(self, char_class: str, iteration: int, model):
        return model.load_state_dict(self.load_state_dict(char_class, iteration))

    def export_class(
        self,
        char_class: str,
        iteration: int,
        export_path: str | Path | None = None,
        *,
        state_dict: dict | None = None,
        blob: bytes | None = None,
    ) -> Path:
        """Export a league blob to SSD.

        Accepts a pre-encoded ``blob`` (preferred; matches ``save_all`` output),
        a fresh ``state_dict`` (encoded once), or falls back to a Redis round-trip
        when neither is given.
        """
        if blob is None:
            weights = state_dict if state_dict is not None else self.load_state_dict(char_class, iteration)
            blob = self.encode_state_dict(weights)
        return self.export_class_blob(char_class, iteration, blob, export_path)

    def export_all(self, iteration: int) -> list[Path]:
        return [
            self.export_class(char_class, iteration)
            for char_class in self.classes
            if self.redis.get(self.key(char_class, iteration)) is not None
        ]

    def latest_by_class(self) -> dict[str, int]:
        latest = {char_class: -1 for char_class in self.classes}
        for raw_key in self._iter_keys(f"{self.key_prefix}:*:*"):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            parts = key.split(":")
            if len(parts) != 3 or parts[0] != self.key_prefix:
                continue
            char_class, value = parts[1], parts[2]
            try:
                iteration = int(value)
            except ValueError:
                continue
            latest[char_class] = max(latest.get(char_class, -1), iteration)
        return latest

    def latest_iteration(self) -> int:
        latest = self.latest_by_class().values()
        return max(latest, default=-1)

    def warm_iteration_cache(self) -> int:
        """Run one SCAN to populate the local iteration cache. Returns the latest value."""
        self._cached_latest = self.latest_iteration()
        return self._cached_latest

    def has_class_keys(self, char_class: str) -> bool:
        # Class-scoped probe: one SCAN over policy:{class}:* with early exit.
        pattern = f"{self.key_prefix}:{char_class}:*"
        for _ in self._iter_keys(pattern):
            return True
        return False

    def has_league_keys(self) -> bool:
        # Fast path: seeded leagues always have iter-0 keys (matches start_redis.sh probe).
        if all(self._key_exists(self.key(char_class, 0)) for char_class in self.classes):
            return True
        # Fall back to one SCAN when iter-0 keys were trimmed or never written.
        latest = self.latest_by_class()
        return all(latest.get(char_class, -1) >= 0 for char_class in self.classes)

    def list_class_keys(self, char_class: str) -> list[str]:
        pattern = f"{self.key_prefix}:{char_class}:*"
        keys: list[str] = []
        for raw_key in self._iter_keys(pattern):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            keys.append(key)
        return sorted(keys, key=lambda value: int(value.rsplit(":", 1)[-1]))

    def list_keys_by_class(self) -> dict[str, list[str]]:
        """One SCAN over policy:*:* partitioned by class; replaces N per-class SCANs."""
        by_class: dict[str, list[str]] = {char_class: [] for char_class in self.classes}
        for raw_key in self._iter_keys(f"{self.key_prefix}:*:*"):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            parts = key.split(":")
            if len(parts) != 3 or parts[0] != self.key_prefix:
                continue
            char_class, value = parts[1], parts[2]
            if char_class not in by_class:
                continue
            try:
                int(value)
            except ValueError:
                continue
            by_class[char_class].append(key)
        for keys in by_class.values():
            keys.sort(key=lambda v: int(v.rsplit(":", 1)[-1]))
        return by_class

    def mget_bytes(self, keys: list[str]) -> dict[str, bytes | None]:
        """Batch-fetch Redis values for the given keys. Returns a {key: bytes|None} map."""
        if not keys:
            return {}
        mget = getattr(self.redis, "mget", None)
        if callable(mget):
            values = mget(keys)
        else:
            values = [self.redis.get(key) for key in keys]
        return dict(zip(keys, values))

    def mset_bytes(self, mapping: dict[str, bytes]) -> int:
        """Batch-write {key: bytes} into Redis in a single MSET. Returns key count."""
        if not mapping:
            return 0
        mset = getattr(self.redis, "mset", None)
        if callable(mset):
            mset(mapping)
        else:
            for key, value in mapping.items():
                self.redis.set(key, value)
        return len(mapping)

    def next_iteration(self) -> int:
        if self._cached_latest is not None:
            return self._cached_latest + 1
        return self.latest_iteration() + 1

    def save_rllib_checkpoint(self, algorithm, iteration: int, checkpoint_dir: str | Path = "rl/runtime/rllib_checkpoints") -> Path:
        path = Path(checkpoint_dir) / f"iter_{int(iteration)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return Path(algorithm.save_to_path(str(path)))

    def _drop_old(self, char_class: str, iteration: int) -> None:
        old_iteration = int(iteration) - self.window_size
        if old_iteration < 0:
            return
        old_key = self.key(char_class, old_iteration)
        unlink = getattr(self.redis, "unlink", None)
        if callable(unlink):
            unlink(old_key)
        else:
            self.redis.delete(old_key)
        # Mirror the Redis rolling window on SSD; covers the current zstd-wrapped
        # export and any legacy uncompressed ``.safetensors`` file still on disk.
        for stale in (
            self._export_path(char_class, old_iteration),
            self._legacy_export_path(char_class, old_iteration),
        ):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    def _key_exists(self, key: str) -> bool:
        exists = getattr(self.redis, "exists", None)
        if callable(exists):
            return bool(exists(key))
        return self.redis.get(key) is not None

    def _iter_keys(self, pattern: str):
        scan_iter = getattr(self.redis, "scan_iter", None)
        if callable(scan_iter):
            yield from scan_iter(match=pattern)
        else:
            yield from self.redis.keys(pattern)
