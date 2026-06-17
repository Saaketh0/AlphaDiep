"""Encode/decode league weight blobs for Redis + SSD storage.

Ghost policies are inference-only and reloaded into RLModules every few iterations,
so the per-iteration cost of storing them is dominated by bytes-on-disk and
bytes-on-the-wire. This codec wraps the existing ``safetensors`` payload with two
optimizations:

* **Per-tensor int8 quantization** for float weight tensors with at least
  ``MIN_QUANT_NUMEL`` elements. Each quantized tensor ``{key}`` is stored as
  ``int8`` plus a companion ``{key}.__scale`` fp32 scalar; dequantization
  reconstructs fp32 on load.
* **zstd compression** of the resulting safetensors bytes. Magic-byte detection
  on read keeps legacy uncompressed fp32 blobs loadable unchanged.

The decoded ``state_dict`` is always fp32, ready for ``RLModule.set_state`` or
``load_state_dict``.

Environment overrides (all default to int8 + zstd):

* ``DIEP_LEAGUE_QUANT=fp32`` — skip quantization (still zstd-wraps).
* ``DIEP_LEAGUE_QUANT=fp16`` — half-precision storage instead of int8.
* ``DIEP_LEAGUE_QUANT=int8`` — explicit default; same as unset.
"""

from __future__ import annotations

import logging
import os
from typing import Final

import torch
import zstandard
from safetensors.torch import load as _st_load
from safetensors.torch import save as _st_save

logger = logging.getLogger(__name__)

# Tensors smaller than this stay fp32 — quantization overhead (scale tensor + per-tensor
# rounding error) is not worth the savings on biases, log-std scalars, etc.
MIN_QUANT_NUMEL: Final[int] = 64

# Suffix appended to a quantized tensor's key for its fp32 scale companion.
SCALE_SUFFIX: Final[str] = ".__scale"

# Bytes 0x28 0xB5 0x2F 0xFD start every zstd frame. See RFC 8478, Section 3.1.1.
_ZSTD_MAGIC: Final[bytes] = b"\x28\xB5\x2F\xFD"

_QUANT_ENV_VAR: Final[str] = "DIEP_LEAGUE_QUANT"
_VALID_QUANT_MODES: Final[tuple[str, ...]] = ("int8", "fp16", "fp32")


def _current_quant_mode() -> str:
    """Read ``DIEP_LEAGUE_QUANT`` on each call so tests can monkeypatch the env."""
    raw = os.environ.get(_QUANT_ENV_VAR, "").strip().lower()
    if not raw:
        return "int8"
    if raw not in _VALID_QUANT_MODES:
        logger.warning(
            "Unknown %s=%r; falling back to int8 (valid: %s)",
            _QUANT_ENV_VAR,
            raw,
            ", ".join(_VALID_QUANT_MODES),
        )
        return "int8"
    return raw


def _quantize_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization. Returns (int8_tensor, fp32_scale_scalar)."""
    flat = tensor.detach().to(torch.float32)
    absmax = flat.abs().max()
    if not torch.isfinite(absmax) or absmax.item() == 0.0:
        # All-zero or non-finite weights: store as zeros with scale=1.0 so dequant is a no-op.
        scale = torch.tensor(1.0, dtype=torch.float32)
        quantized = torch.zeros_like(flat, dtype=torch.int8)
        return quantized, scale
    scale = (absmax / 127.0).to(torch.float32)
    quantized = torch.round(flat / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def _dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quantize_int8`. Always returns a contiguous fp32 tensor."""
    return (quantized.to(torch.float32) * scale.to(torch.float32)).contiguous()


def _quantize_state_dict(state_dict: dict, mode: str) -> dict[str, torch.Tensor]:
    """Apply the configured quantization scheme to a CPU state dict."""
    encoded: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            # ``safetensors`` cannot serialize non-tensor values; surface this loudly.
            raise TypeError(f"League state_dict value {name!r} is not a torch.Tensor")
        cpu_tensor = tensor.detach().cpu()
        if mode == "fp32" or cpu_tensor.numel() < MIN_QUANT_NUMEL or not cpu_tensor.is_floating_point():
            encoded[name] = cpu_tensor.to(torch.float32) if cpu_tensor.is_floating_point() else cpu_tensor
            continue
        if mode == "fp16":
            encoded[name] = cpu_tensor.to(torch.float16)
            continue
        # mode == "int8"
        quantized, scale = _quantize_int8(cpu_tensor)
        encoded[name] = quantized
        encoded[name + SCALE_SUFFIX] = scale
    return encoded


def _dequantize_state_dict(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reverse the quantization step, leaving fp32 tensors ready for RLModule loading."""
    decoded: dict[str, torch.Tensor] = {}
    consumed_scales: set[str] = set()
    for name, tensor in raw_state.items():
        if name.endswith(SCALE_SUFFIX):
            continue
        scale_key = name + SCALE_SUFFIX
        scale = raw_state.get(scale_key)
        if scale is not None:
            decoded[name] = _dequantize_int8(tensor, scale)
            consumed_scales.add(scale_key)
            continue
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            # fp16 (and any future low-precision dtype) widens back to fp32 for parity.
            decoded[name] = tensor.to(torch.float32).contiguous()
            continue
        decoded[name] = tensor
    # Surface unexpected dangling scale keys; safer than silently ignoring them.
    leftover_scales = {
        name for name in raw_state if name.endswith(SCALE_SUFFIX) and name not in consumed_scales
    }
    if leftover_scales:
        raise ValueError(
            f"League blob has scale keys without matching tensors: {sorted(leftover_scales)}"
        )
    return decoded


def encode_league_blob(state_dict: dict, *, compression_level: int = 3) -> bytes:
    """Quantize, serialize, and zstd-compress a league weight state dict."""
    encoded = _quantize_state_dict(state_dict, _current_quant_mode())
    safetensor_bytes = _st_save(encoded)
    compressor = zstandard.ZstdCompressor(level=compression_level)
    return compressor.compress(safetensor_bytes)


def decode_league_blob(raw: bytes) -> dict[str, torch.Tensor]:
    """Decompress (if zstd-framed) and dequantize a league blob into an fp32 state dict.

    Legacy uncompressed ``safetensors`` payloads (no zstd magic, no scale keys) round-trip
    unchanged, which keeps pre-codec exports loadable during the SSD trim window.
    """
    if raw.startswith(_ZSTD_MAGIC):
        decompressor = zstandard.ZstdDecompressor()
        safetensor_bytes = decompressor.decompress(raw)
    else:
        safetensor_bytes = raw
    raw_state = _st_load(safetensor_bytes)
    return _dequantize_state_dict(raw_state)


__all__ = [
    "MIN_QUANT_NUMEL",
    "SCALE_SUFFIX",
    "decode_league_blob",
    "encode_league_blob",
]
