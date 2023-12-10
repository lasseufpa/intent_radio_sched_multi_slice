import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            tmp_obs_space = obs_space.original_space
        else:
            tmp_obs_space = obs_space
        self.internal_model = TorchFC(
            tmp_obs_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]  # type: ignore
        observations = input_dict["obs"]["observations"]  # type: ignore

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": observations})

        expanded_logits = torch.cat((logits, action_mask), 1)

        # Return expanded logits.
        return expanded_logits, state

    def value_function(self):
        return self.internal_model.value_function()
