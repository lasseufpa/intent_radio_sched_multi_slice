from typing import Optional

import h5py
import numpy as np

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
        self.thermal_noise_power = 10e-14

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if episode_number != self.current_episode_number:
            self.current_episode_number = episode_number
            file = h5py.File(
                f"{self.channels_path}ep_{episode_number}/target_cell_power.mat",
                "r",
            )
            target_cell_power = file.get("target_cell_power")
            file.close()
            intercell_interference = np.zeros_like(target_cell_power)
            spectral_efficiencies_per_rb = np.log2(
                1
                + np.divide(
                    (target_cell_power / self.num_available_rbs[0])
                    * np.power(np.abs(target_cell_power), 2),
                    (
                        np.power(np.abs(intercell_interference), 2)
                        + self.thermal_noise_power
                    ),
                )
            )
            self.spectral_efficiencies = np.squeeze(
                spectral_efficiencies_per_rb
            )

        return np.array([self.spectral_efficiencies[:, :, step_number]])
