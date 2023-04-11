from typing import Optional

import numpy as np
import scipy.io as sio

from sixg_radio_mgmt import Channel


class QuadrigaChannel(Channel):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            max_number_ues, max_number_basestations, num_available_rbs, rng
        )
        self.current_episode_number = -1
        self.channels_path = (
            "../mult_slice_channel_generation/results/freq_channel/"
        )
        self.spectral_efficiencies = np.array([])

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if episode_number != self.current_episode_number:
            self.current_episode_number = episode_number
            self.spectral_efficiencies = sio.loadmat(
                f"{self.channels_path}ep_{episode_number}/spectral_efficiencies_per_rb.mat"
            )
            self.spectral_efficiencies = np.squeeze(
                self.spectral_efficiencies["spectral_efficiencies_per_rb"]
            )

        return np.array([self.spectral_efficiencies[:, :, step_number]])
