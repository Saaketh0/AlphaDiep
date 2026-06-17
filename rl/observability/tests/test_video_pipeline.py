from __future__ import annotations

import numpy as np

from rl.observability.core.observation_schema import COMBAT_GRID_CHANNELS
from rl.observability.video.render_grid_obs import render_grid_composite
from rl.observability.video.render_world import render_world_frame
from rl.observability.video.video_writer import FfmpegVideoWriter


def test_render_grid_composite_shape():
    grid = np.zeros((len(COMBAT_GRID_CHANNELS), 21, 21), dtype=np.float32)
    grid[0, :, :] = 1.0
    grid[3, 10, 10] = 1.0
    frame = render_grid_composite(grid, cell_scale=4)
    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    assert frame[..., 0].max() > 0
    assert frame[..., 2].max() > 0


def test_render_world_frame_follows_agent_one():
    snapshot = {
        "tick": 1,
        "arena": {"leftX": -100, "rightX": 100, "topY": -100, "bottomY": 100},
        "entities": [
            {
                "kind": "agent",
                "id": 0,
                "agentIndex": 0,
                "position": {"x": -40.0, "y": 0.0, "angle": 0.0},
                "physics": {"sides": 1, "size": 10},
                "health": {"health": 10, "maxHealth": 10},
            },
            {
                "kind": "agent",
                "id": 1,
                "agentIndex": 1,
                "position": {"x": 20.0, "y": 0.0, "angle": 0.0},
                "physics": {"sides": 1, "size": 10},
                "health": {"health": 10, "maxHealth": 10},
            },
            {
                "kind": "shape",
                "id": 2,
                "position": {"x": 35.0, "y": 0.0, "angle": 0.0},
                "physics": {"sides": 4, "size": 8},
                "health": {"health": 10, "maxHealth": 10},
            },
        ],
    }
    frame = render_world_frame(snapshot, followed_agent_id=1, frame_size=(128, 128), view_world_size=120)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    # Followed agent is rendered near frame center in white.
    center_patch = frame[56:72, 56:72]
    assert center_patch.max() == 255


def test_video_writer_smoke(tmp_path):
    output = tmp_path / 'eval.mp4'
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    with FfmpegVideoWriter(output, width=64, height=64, fps=10) as writer:
        for _ in range(4):
            assert writer.write(frame) is True
    assert output.exists()
    assert output.stat().st_size > 0


def test_video_writer_drops_frames_when_queue_is_full(tmp_path):
    output = tmp_path / 'drops.mp4'
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with FfmpegVideoWriter(output, width=32, height=32, fps=10, max_queue=1, write_delay_seconds=0.05) as writer:
        results = [writer.write(frame) for _ in range(10)]
        stats = writer.close()
    assert any(result is False for result in results) or stats.dropped_frames > 0
    assert output.exists()

from pathlib import Path

from rl.observability.config import ObservabilityConfig
from rl.observability.video.eval_video import maybe_write_eval_video


class AlwaysFallbackAlgorithm:
    def compute_single_action(self, *args, **kwargs):
        raise RuntimeError("force sampled fallback")


def test_forced_eval_video_interval_smoke_under_wandb_root(tmp_path: Path):
    runs_root = tmp_path / "training_data" / "W&B"
    config = ObservabilityConfig(
        run_id="forced-video",
        runs_root=runs_root,
        video_interval_iterations=1,
        eval_max_steps=1,
        eval_env_config={"agents": 20, "max_ticks": 8, "seed": 3},
    )
    result = maybe_write_eval_video(AlwaysFallbackAlgorithm(), config, iteration=1)
    assert result is not None
    expected_name = config.video_filename(1)
    assert result.path == runs_root / "videos" / expected_name
    assert expected_name.endswith("-iteration-1.mp4")
    assert result.path.exists()
    assert result.path.stat().st_size > 0
    assert result.elapsed_seconds >= 0.0
    assert result.used_policy_fallback is True


def test_manual_testing_video_path_uses_wandb_videos_folder(tmp_path: Path):
    runs_root = tmp_path / "training_data" / "W&B"
    config = ObservabilityConfig(run_id="manual-video", runs_root=runs_root)
    expected_name = config.video_filename(manual_testing=True)
    assert config.video_path(manual_testing=True) == runs_root / "videos" / expected_name
    assert expected_name.endswith("-manual-testing.mp4")
