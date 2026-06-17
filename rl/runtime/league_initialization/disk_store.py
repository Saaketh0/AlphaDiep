"""Persist the Redis league to/from SSD safetensors files.

The export directory is the single source of truth for cold starts: if Redis is
empty (fresh container, flushed memory), ``hydrate_redis_from_disk`` repopulates
it from ``diepcustom/training_data/redis/{class}/iter_{N}.safetensors``.
"""

from __future__ import annotations

import logging
import re

from model_store import RedisModelStore

from .paths import LEAGUE_EXPORT_DIR

logger = logging.getLogger(__name__)

# Match both the legacy uncompressed export (``iter_42.safetensors``) and the
# current zstd-wrapped one (``iter_42.safetensors.zst``). The trailing ``\.zst?``
# captures both forms in a single pass.
_ITER_RE = re.compile(r"iter_(\d+)\.safetensors(?:\.zst)?$")


def _collect_disk_payload(store: RedisModelStore) -> dict[str, bytes]:
    """Walk the SSD export tree once and return ``{redis_key: file_bytes}``.

    Prefers the compressed ``.safetensors.zst`` variant when both forms exist
    for the same iteration; the on-disk bytes are stored verbatim in Redis so
    ``decode_league_blob`` handles both formats transparently on load.
    """
    payload: dict[str, bytes] = {}
    for char_class in store.classes:
        class_dir = store.snapshot_dir / char_class
        if not class_dir.is_dir():
            continue
        candidates: list[tuple[int, int, Path]] = []
        for path in class_dir.glob("iter_*.safetensors*"):
            match = _ITER_RE.search(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            suffix_rank = 0 if path.suffix == ".zst" else 1
            candidates.append((iteration, suffix_rank, path))
        candidates.sort()
        seen_iters: set[int] = set()
        for iteration, _suffix_rank, path in candidates:
            if iteration in seen_iters:
                continue
            seen_iters.add(iteration)
            payload[store.key(char_class, iteration)] = path.read_bytes()
    return payload


def hydrate_redis_from_disk(store: RedisModelStore) -> int:
    """Load safetensors exports from SSD into Redis. Returns keys written.

    Idempotent: keys already present in Redis are left untouched. The on-disk
    safetensors byte format is identical to what Redis stores, so file bytes are
    written directly without a torch round-trip. When Redis is known to be cold
    (no league keys), the whole payload is flushed in one ``MSET``; otherwise we
    fall back to per-key GET-before-SET to stay idempotent.
    """
    payload = _collect_disk_payload(store)
    if not payload:
        return 0

    if not store.has_league_keys():
        written = store.mset_bytes(payload)
    else:
        written = 0
        for key, value in payload.items():
            if store.redis.get(key) is not None:
                continue
            store.redis.set(key, value)
            written += 1

    if written:
        logger.info("Hydrated Redis league from disk: %d keys from %s", written, LEAGUE_EXPORT_DIR)
    return written


def export_league_to_disk(store: RedisModelStore) -> list:
    """Export every league weight currently in Redis to SSD safetensors files.

    Uses a single ``list_keys_by_class`` SCAN (one Redis round-trip) instead of
    four per-class SCANs.
    """
    exported = []
    for char_class, redis_keys in store.list_keys_by_class().items():
        for redis_key in redis_keys:
            iteration = int(redis_key.rsplit(":", 1)[-1])
            exported.append(store.export_class(char_class, iteration))
    logger.info("Exported %d league weights to %s", len(exported), LEAGUE_EXPORT_DIR)
    return exported
