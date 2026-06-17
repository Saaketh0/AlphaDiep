"""Training-time metrics supported through Ray Tune W&B and RLlib callbacks."""

__all__ = ["DiepRLlibObservabilityCallback", "create_wandb_logger_callback", "wandb_display_name", "wandb_logger_kwargs"]


def __getattr__(name: str):
    if name == "DiepRLlibObservabilityCallback":
        from rl.observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback

        return DiepRLlibObservabilityCallback
    if name in {"create_wandb_logger_callback", "wandb_display_name", "wandb_logger_kwargs"}:
        from rl.observability.logging.wandb_tune import create_wandb_logger_callback, wandb_display_name, wandb_logger_kwargs

        return {
            "create_wandb_logger_callback": create_wandb_logger_callback,
            "wandb_display_name": wandb_display_name,
            "wandb_logger_kwargs": wandb_logger_kwargs,
        }[name]
    raise AttributeError(name)
