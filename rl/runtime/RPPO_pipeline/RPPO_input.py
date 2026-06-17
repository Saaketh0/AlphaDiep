"""Minimal RPPO input combiner for combat observations."""

import torch
import torch.nn as nn

from .CNN import BasicCombatCNN
from .LINEAR import RPPOVectorInput


class RPPOInput(nn.Module):
    """Concatenate CNN grid features with flat vector observations."""

    def __init__(self, observation_space=None, action_space=None, cnn=None, vector_input=None):
        super().__init__()
        self.cnn = cnn or BasicCombatCNN(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.vector_input = vector_input or RPPOVectorInput()

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Return the module's current device, defaulting to CPU before params exist."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _to_tensor(value, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Accept env arrays/lists/scalars or tensors and return a tensor."""
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _ensure_batch_dim(tensor: torch.Tensor, unbatched_dims: int) -> tuple[torch.Tensor, bool]:
        """Return a batched tensor and whether a batch dim was added."""
        if tensor.dim() == unbatched_dims:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _tensor_obs(self, obs):
        """Convert a PettingZoo combat observation dict into batched torch tensors."""
        device = self._module_device(self)
        tensor_obs = {
            "grid_obs": self._to_tensor(obs["grid_obs"], dtype=torch.float32, device=device),
            "self_obs": self._to_tensor(obs["self_obs"], dtype=torch.float32, device=device),
            "prev_action_obs": self._to_tensor(obs["prev_action_obs"], dtype=torch.float32, device=device),
        }
        if "tank_type_obs" in obs:
            tensor_obs["tank_type_obs"] = self._to_tensor(obs["tank_type_obs"], dtype=torch.long, device=device)

        tensor_obs["grid_obs"], squeezed_grid = self._ensure_batch_dim(tensor_obs["grid_obs"], 3)
        tensor_obs["self_obs"], squeezed_self = self._ensure_batch_dim(tensor_obs["self_obs"], 1)
        tensor_obs["prev_action_obs"], squeezed_prev = self._ensure_batch_dim(tensor_obs["prev_action_obs"], 1)
        if "tank_type_obs" in tensor_obs:
            tensor_obs["tank_type_obs"], _ = self._ensure_batch_dim(tensor_obs["tank_type_obs"], 0)

        return tensor_obs, squeezed_grid and squeezed_self and squeezed_prev

    def forward(self, obs):
        """Return grid + vector features, preserving optional recurrent time dim.

        RLlib recurrent batches arrive as ``[B, T, ...]`` observations.  The CNN
        and vector encoders operate on flat batches, so flatten ``B*T`` for the
        encoders and reshape back to ``[B, T, D]`` for the LSTM wrapper.
        """
        obs, squeeze_output = self._tensor_obs(obs)
        sequence_shape = None
        if obs["grid_obs"].dim() == 5:
            batch_size, time_size = obs["grid_obs"].shape[:2]
            sequence_shape = (batch_size, time_size)
            obs = {
                key: value.reshape(batch_size * time_size, *value.shape[2:])
                for key, value in obs.items()
            }

        features = self.cnn.forward_train({"obs": obs})["features"]
        features, _ = self._ensure_batch_dim(features, 1)
        self_features, prev_action_features = self.vector_input(
            obs["self_obs"],
            obs["prev_action_obs"],
        )
        combined = torch.cat([features, self_features, prev_action_features], dim=1)
        if sequence_shape is not None:
            return combined.reshape(*sequence_shape, combined.shape[-1])
        if squeeze_output:
            return combined.squeeze(0)
        return combined
