from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import os

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_ROOT = ROOT / 'training_data'
RUNS_ROOT = TRAINING_DATA_ROOT / 'W&B'
DEFAULT_STATS_LOG_AGENTS = ('agent_0', 'agent_1', 'agent_2', 'agent_3')


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'run-{stamp}-{os.getpid()}'


def _normalize_agents(values: Iterable[str] | None) -> tuple[str, ...]:
    agents = tuple(str(value).strip() for value in (values or DEFAULT_STATS_LOG_AGENTS) if str(value).strip())
    return agents or DEFAULT_STATS_LOG_AGENTS


@dataclass
class ObservabilityConfig:
    run_id: str = field(default_factory=_default_run_id)
    runs_root: Path = RUNS_ROOT
    wandb_enabled: bool = True
    wandb_mode: str = 'offline'
    stats_log_agents: tuple[str, ...] = field(default_factory=lambda: DEFAULT_STATS_LOG_AGENTS)
    learner_agent: str = 'agent_0'
    project_name: str = 'diepcustom-headless-rl'
    wandb_group: str = 'ppo-training'
    wandb_resume: str = 'allow'
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    upload_checkpoints: bool = False
    video_enabled: bool = True
    video_agent: str = 'agent_1'
    video_interval_iterations: int = 500
    video_fps: int = 20
    eval_max_steps: int = 1000
    eval_env_config: dict = field(default_factory=dict)
    # P8: sample on_episode_step every K env steps. K=1 (default) preserves
    # the original per-step accumulation; K>1 trades resolution for callback
    # overhead while keeping episode-end means unbiased (see rllib_callbacks).
    episode_step_sample_interval: int = 1

    def __post_init__(self) -> None:
        self.runs_root = Path(self.runs_root)
        self.stats_log_agents = _normalize_agents(self.stats_log_agents)
        self.wandb_mode = str(self.wandb_mode or 'offline')
        self.wandb_resume = str(self.wandb_resume or 'allow')
        self.wandb_tags = tuple(str(tag) for tag in self.wandb_tags if str(tag))
        self.video_agent = str(self.video_agent or 'agent_1')
        self.video_interval_iterations = int(self.video_interval_iterations)
        self.video_fps = int(self.video_fps)
        self.eval_max_steps = int(self.eval_max_steps)
        self.episode_step_sample_interval = max(1, int(self.episode_step_sample_interval))

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def episodes_jsonl_path(self) -> Path:
        return self.run_dir / 'episodes.jsonl'

    @property
    def config_json_path(self) -> Path:
        return self.run_dir / 'config.json'

    @property
    def benchmarks_path(self) -> Path:
        return self.runs_root / 'benchmarks.md'

    def eval_episode_dir(self, episode_id: str) -> Path:
        return self.run_dir / 'eval' / str(episode_id)

    # Returns the directory used for the periodic training-iteration eval video.
    def eval_iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / 'eval' / str(int(iteration))

    @property
    def videos_dir(self) -> Path:
        return self.runs_root / 'videos'

    def video_filename(self, iteration: int | None = None, *, manual_testing: bool = False) -> str:
        day_month = datetime.now().strftime('%d-%m')
        if manual_testing:
            suffix = 'manual-testing'
        else:
            suffix = f'iteration-{int(iteration or 0)}'
        return f'{day_month}-{suffix}.mp4'

    def video_path(self, iteration: int | None = None, *, manual_testing: bool = False) -> Path:
        return self.videos_dir / self.video_filename(iteration, manual_testing=manual_testing)

    def ensure_directories(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    # Builds observability defaults from environment variables used in training.
    @classmethod
    def from_env(cls, **overrides) -> 'ObservabilityConfig':
        values = {
            'wandb_mode': os.getenv('WANDB_MODE', 'offline'),
            'project_name': os.getenv('WANDB_PROJECT', 'diepcustom-headless-rl'),
            'wandb_group': os.getenv('WANDB_GROUP', 'ppo-training'),
            'wandb_resume': os.getenv('WANDB_RESUME', 'allow'),
            'wandb_tags': tuple(tag.strip() for tag in os.getenv('WANDB_TAGS', '').split(',') if tag.strip()),
            # Eval video is off by default in training because it forces a blocking
            # rollout + per-frame snapshot render. Enable with DIEP_VIDEO_ENABLED=true.
            'video_enabled': os.getenv('DIEP_VIDEO_ENABLED', '').lower() in {'1', 'true', 'yes'},
            'video_agent': os.getenv('DIEP_VIDEO_AGENT', 'agent_1'),
            'video_interval_iterations': int(os.getenv('DIEP_VIDEO_INTERVAL', '500')),
            'video_fps': int(os.getenv('DIEP_VIDEO_FPS', '20')),
            'eval_max_steps': int(os.getenv('DIEP_EVAL_MAX_STEPS', '1000')),
            'upload_checkpoints': os.getenv('WANDB_UPLOAD_CHECKPOINTS', '').lower() in {'1', 'true', 'yes'},
            'episode_step_sample_interval': int(os.getenv('DIEP_OBS_STEP_INTERVAL', '1')),
        }
        values.update(overrides)
        return cls(**values)


__all__ = ['DEFAULT_STATS_LOG_AGENTS', 'ObservabilityConfig', 'ROOT', 'RUNS_ROOT', 'TRAINING_DATA_ROOT']
