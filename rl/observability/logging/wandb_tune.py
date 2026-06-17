"""Ray Tune W&B callback factory for offline-first training runs."""

from __future__ import annotations

from typing import Any

from rl.observability.config import ObservabilityConfig


def wandb_display_name(config: ObservabilityConfig) -> str:
    """Return the human-facing run name while keeping W&B id equal to run_id."""
    from datetime import datetime

    month_day = datetime.now().strftime("%m-%d")
    return f"run-{config.run_id}-{month_day}"


# Returns plain kwargs so tests can validate defaults without importing Ray.
def wandb_logger_kwargs(config: ObservabilityConfig) -> dict[str, Any]:
    tags = tuple(dict.fromkeys((*config.wandb_tags, f"run_id:{config.run_id}")))
    return {
        "project": config.project_name,
        "group": config.wandb_group,
        "mode": config.wandb_mode,
        "name": wandb_display_name(config),
        "id": config.run_id,
        "resume": config.wandb_resume,
        "tags": tags,
        "dir": str(config.runs_root),
        "log_config": True,
        "upload_checkpoints": config.upload_checkpoints,
    }


# Creates Ray Tune's W&B logger callback only when Ray is available at runtime.
def create_wandb_logger_callback(config: ObservabilityConfig) -> Any:
    try:
        from ray.air.integrations.wandb import WandbLoggerCallback
    except ImportError:  # pragma: no cover - import path differs in some Ray versions.
        from ray.tune.integration.wandb import WandbLoggerCallback  # type: ignore

    return WandbLoggerCallback(**wandb_logger_kwargs(config))


__all__ = ["create_wandb_logger_callback", "wandb_display_name", "wandb_logger_kwargs"]
