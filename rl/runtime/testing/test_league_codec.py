"""Tests for ``league_codec`` round-trip, legacy fp32 support, and size targets."""

from __future__ import annotations

import torch
import zstandard
from safetensors.torch import save

from league_codec import (
    MIN_QUANT_NUMEL,
    SCALE_SUFFIX,
    decode_league_blob,
    encode_league_blob,
)


def _sample_state_dict(seed: int = 0) -> dict:
    """Realistic-shaped state dict mixing large weights with small biases."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "encoder.weight": torch.randn(256, 256, generator=generator) * 0.1,
        "encoder.bias": torch.randn(256, generator=generator) * 0.05,
        "head.weight": torch.randn(64, 256, generator=generator) * 0.1,
        "head.bias": torch.zeros(64),
        "tiny.weight": torch.tensor([0.1, -0.2, 0.3]),
    }


def test_round_trip_recovers_fp32_state_dict_within_tolerance():
    state = _sample_state_dict()
    blob = encode_league_blob(state)

    decoded = decode_league_blob(blob)

    assert set(decoded) == set(state)
    for name, original in state.items():
        recovered = decoded[name]
        assert recovered.dtype == torch.float32
        assert recovered.shape == original.shape
        if original.numel() >= MIN_QUANT_NUMEL:
            # int8 quantization step at scale = absmax / 127.
            tolerance = float(original.detach().abs().max() / 127.0) + 1e-6
            max_err = float((recovered - original.float()).abs().max())
            assert max_err <= tolerance, (name, max_err, tolerance)
        else:
            assert torch.equal(recovered, original.float())


def test_encoded_blob_is_significantly_smaller_than_fp32_safetensors():
    """int8 + zstd should shrink the typical league blob by at least 3x."""
    state = _sample_state_dict()
    fp32_bytes = save({name: tensor.float() for name, tensor in state.items()})
    encoded = encode_league_blob(state)

    assert len(encoded) <= len(fp32_bytes) * 0.35, (len(encoded), len(fp32_bytes))


def test_legacy_uncompressed_fp32_blob_decodes_unchanged():
    """Pre-codec safetensors files round-trip via the same decode path."""
    state = _sample_state_dict(seed=1)
    legacy_blob = save({name: tensor.float() for name, tensor in state.items()})

    decoded = decode_league_blob(legacy_blob)

    for name, original in state.items():
        assert torch.equal(decoded[name], original.float())


def test_quant_mode_fp32_skips_quantization(monkeypatch):
    """``DIEP_LEAGUE_QUANT=fp32`` keeps tensors fp32 (only zstd wrap)."""
    monkeypatch.setenv("DIEP_LEAGUE_QUANT", "fp32")
    state = _sample_state_dict(seed=2)

    encoded = encode_league_blob(state)
    decoded = decode_league_blob(encoded)

    for name, original in state.items():
        assert torch.equal(decoded[name], original.float())


def test_quant_mode_fp16_widens_back_to_fp32(monkeypatch):
    """``DIEP_LEAGUE_QUANT=fp16`` stores half precision and dequantizes to fp32."""
    monkeypatch.setenv("DIEP_LEAGUE_QUANT", "fp16")
    state = _sample_state_dict(seed=3)

    encoded = encode_league_blob(state)
    decoded = decode_league_blob(encoded)

    for name, original in state.items():
        recovered = decoded[name]
        assert recovered.dtype == torch.float32
        if original.numel() >= MIN_QUANT_NUMEL:
            assert torch.allclose(recovered, original.float(), atol=1e-2, rtol=1e-2)
        else:
            assert torch.equal(recovered, original.float())


def test_unknown_quant_mode_falls_back_to_int8_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("DIEP_LEAGUE_QUANT", "uint4")
    state = {"w": torch.randn(128, 128)}

    with caplog.at_level("WARNING", logger="league_codec"):
        blob = encode_league_blob(state)

    assert any("DIEP_LEAGUE_QUANT" in record.message for record in caplog.records)
    decoded = decode_league_blob(blob)
    assert decoded["w"].dtype == torch.float32


def test_zero_weight_tensor_round_trips_safely():
    state = {"w": torch.zeros(128, 128)}

    decoded = decode_league_blob(encode_league_blob(state))

    assert torch.equal(decoded["w"], state["w"])


def test_dangling_scale_key_raises_value_error():
    """Manually-crafted blob with a scale but no companion tensor must fail loudly."""
    blob = save({"w" + SCALE_SUFFIX: torch.tensor(1.0)})
    cctx = zstandard.ZstdCompressor()

    try:
        decode_league_blob(cctx.compress(blob))
    except ValueError as exc:
        assert "scale keys without matching tensors" in str(exc)
    else:
        raise AssertionError("decode_league_blob should reject dangling scale keys")


def test_non_tensor_values_raise_type_error():
    try:
        encode_league_blob({"not_a_tensor": 1.0})
    except TypeError as exc:
        assert "torch.Tensor" in str(exc)
    else:
        raise AssertionError("encode_league_blob should reject non-tensor values")
