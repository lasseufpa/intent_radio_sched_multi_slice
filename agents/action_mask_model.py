import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


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

        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
