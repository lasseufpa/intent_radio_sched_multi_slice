from typing import Optional, Tuple

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
        root_path: str = "",
        scenario_name: str = "mult_slice",
    ) -> None:
        super().__init__(
            max_number_ues,
            max_number_basestations,
            num_available_rbs,
            rng,
            root_path,
            scenario_name,
        )
        self.scenario_name = (
            "mult_slice"  # We always use the mult_slice scenario for channels
        )
        self.current_episode_number = -1
        self.file = None
        self.channels_path = f"{self.root_path}/../mult_slice_channel_generation/results/{self.scenario_name}/freq_channel/"
        self.spectral_efficiencies = np.array([])
        self.transmission_power = 100  # Watts
        self.thermal_noise_power = 10e-14
        self.channel_eps_per_scenario = 100

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        association_to_use, episode_to_use, condition = self.choose_episode(
            episode_number, self.current_episode_number
        )
        if condition:
            self.current_episode_number = episode_number
            if self.file is not None:
                self.file.close()
            self.file = h5py.File(
                f"{self.channels_path}assoc_{association_to_use}/ep_{episode_to_use}/target_cell_power.mat",
                "r",
            )
        if self.file is not None:
            target_cell_power = self.file.get("target_cell_power")
            target_cell_power = np.array(
                target_cell_power[step_number, :, :, :, :]  # type: ignore
            )
            intercell_interference = np.zeros_like(target_cell_power)
            spectral_efficiencies_per_rb = np.log2(
                1
                + np.divide(
                    (self.transmission_power / self.num_available_rbs[0])
                    * target_cell_power,
                    (intercell_interference + self.thermal_noise_power),
                )
            )
            self.spectral_efficiencies = np.squeeze(
                spectral_efficiencies_per_rb.transpose()
            )
        else:
            raise ValueError("File is None")

        return np.array([self.spectral_efficiencies])

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, int, bool]:
        if episode_number != current_episode:
            association_to_use = episode_number
            episode_to_use = 0
            return (association_to_use, episode_to_use, True)
        return (0, 0, False)
