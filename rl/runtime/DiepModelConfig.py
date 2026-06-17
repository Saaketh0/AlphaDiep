import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (DefaultPPOTorchRLModule)
from RPPO_pipeline import RPPOInput
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT

# Catalog
import functools
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import MLPEncoderConfig



ENCODER_DIM = 312


def model_config_get(model_config, key, default=None):
    if isinstance(model_config, dict):
        return model_config.get(key, default)
    return getattr(model_config, key, default)


class DiepPolicy(DefaultPPOTorchRLModule):
    def setup(self):
        super().setup()
        self.encoder = Wrapper(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=self.model_config,
        )


class Wrapper(nn.Module):
    def __init__(self, observation_space=None, action_space=None, model_config=None) -> None:
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.model_config = model_config

        self.rppo = RPPOInput(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        lstm_size = model_config_get(model_config, "lstm_cell_size", 256)
        self.lstm = nn.LSTM(input_size=ENCODER_DIM, hidden_size=lstm_size, batch_first=False)

    def get_initial_state(self):
        return {
            "h": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
            "c": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
        }

    def forward(self, batch):
        features = self.rppo(batch[Columns.OBS])
        state_in = batch.get(Columns.STATE_IN)
        state_rank = None
        if state_in is not None:
            state_rank = state_in["h"].dim()

        # RLlib connector-v2 uses different recurrent layouts at sampling and
        # learner time. Sampling receives a synthetic single timestep rank
        # ([T=1, B, ...]) and state tensors shaped [1, B, layers, H]; learner
        # batches are zero-padded as [B, T, ...] with state [B, layers, H].
        # Torch LSTM runs time-major internally, but learner outputs must be
        # converted back to [B, T, ...] so RLlib's GAE unpadding sees rows as
        # sequences rather than timesteps.
        return_time_major = True
        if features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(1)  # [T=1, B=1, D]
        elif features.dim() == 2:
            features = features.unsqueeze(0)  # [T=1, B, D]
        elif features.dim() == 3:
            sampling_time_rank = state_rank == 4 and state_in["h"].shape[0] == 1
            return_time_major = sampling_time_rank
            if not sampling_time_rank:
                features = features.transpose(0, 1)  # [B, T, D] -> [T, B, D]

        batch_size = features.shape[1]
        if state_in is None:
            initial = self.get_initial_state()
            h = initial["h"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
            c = initial["c"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
        else:
            h_in = state_in["h"].to(features.device)
            c_in = state_in["c"].to(features.device)
            # Connector-v2 may provide [1, B, layers, H] during sampling after
            # adding a time rank. Stored learner state is [B, layers, H].
            if h_in.dim() == 4 and h_in.shape[0] == 1:
                h_in = h_in.squeeze(0)
            if c_in.dim() == 4 and c_in.shape[0] == 1:
                c_in = c_in.squeeze(0)
            if h_in.shape[0] != batch_size or c_in.shape[0] != batch_size:
                # Learner-side value passes may use minibatches that no longer
                # align with episode-collected state rows. Reinitialize rather
                # than feeding an invalid hidden batch into torch.nn.LSTM.
                initial = self.get_initial_state()
                h = initial["h"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
                c = initial["c"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
            else:
                h = h_in.transpose(0, 1)
                c = c_in.transpose(0, 1)

        lstm_out, (h_out, c_out) = self.lstm(features, (h, c))
        encoder_out = lstm_out if return_time_major else lstm_out.transpose(0, 1)

        return {
            # Sampling keeps [T=1, B, H] so RLlib's module-to-env connector can
            # remove the single timestep rank. Learner/value passes receive
            # [B, T, H] so zero-padding and GAE operate row-by-row over sequences.
            ENCODER_OUT: {ACTOR: encoder_out, CRITIC: encoder_out},
            Columns.STATE_OUT: {
                "h": h_out.transpose(0, 1),
                "c": c_out.transpose(0, 1),
            },
        }


DiepConfig = {
    # Plain custom RLModule config. Ray's DefaultModelConfig is only intended
    # for RLlib's built-in default modules; this module supplies its own
    # encoder and only needs these PPO head/recurrent settings.
    "vf_share_layers": True,
    "max_seq_len": 10,
    "lstm_cell_size": 256,
    "head_fcnet_hiddens": [],
    "head_fcnet_activation": "relu",
}


class DiepCatalog(PPOCatalog):
    def _determine_components_hook(self):
        # Don't call super() — that tries to parse Dict obs and fails.

        cfg = self._model_config_dict
        latent = cfg.get("lstm_cell_size", 312)

        self.latent_dims = [latent]

        # Dummy encoder config — only so PPOCatalog.__init__ can finish.
        # DiepPolicy.setup() replaces the built encoder with your Wrapper.
        self._encoder_config = MLPEncoderConfig(
            input_dims=[latent],
            hidden_layer_dims=[],
            output_layer_dim=latent,
        )

        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space,
            action_space=self.action_space,
        )
