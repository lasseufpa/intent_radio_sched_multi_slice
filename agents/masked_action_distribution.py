from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class TorchDiagGaussian(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        *,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        super().__init__(inputs, model)
        assert isinstance(
            self.inputs, torch.Tensor
        ), "Inputs must be a torch.Tensor"
        mean, log_std, masks = torch.chunk(self.inputs, 3, dim=1)
        log_std = torch.exp(log_std)
        log_std[masks == 0] = 1e-9
        mean[masks == 0] = -1
        self.dist = torch.distributions.normal.Normal(mean, log_std)
        # Remember to squeeze action samples in case action space is Box(shape)
        self.zero_action_dim = action_space and action_space.shape == ()

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        sample = super().sample()
        if self.zero_action_dim:
            return torch.squeeze(sample, dim=-1)
        return sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self.dist.mean
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return super().logp(actions).sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        assert action_space.shape is not None, "Action space shape must be set"
        output_size = int(np.prod(action_space.shape, dtype=np.int32) * 2)

        return output_size
