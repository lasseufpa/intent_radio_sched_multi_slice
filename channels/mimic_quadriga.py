from typing import Optional

import numpy as np

from sixg_radio_mgmt import Channel


class MimicQuadriga(Channel):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
        root_path: str = "",
        scenario_name: str = "",
    ) -> None:
        super().__init__(
            max_number_ues,
            max_number_basestations,
            num_available_rbs,
            rng,
            root_path,
            scenario_name,
        )
        self.current_episode_number = -1
        self.ues_mean_se = np.array([])
        self.default_std = 1.5

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if episode_number != self.current_episode_number:
            self.current_episode_number = episode_number
            self.ues_mean_se = np.abs(
                self.rng.normal(10, 7.5, size=(self.max_number_ues,))
            )

        spectral_efficiencies = np.array(
            [
                np.ones((self.max_number_ues, self.num_available_rbs[i]))
                for i in np.arange(self.max_number_basestations)
            ]
        )
        for ue_idx, ue_mean in enumerate(self.ues_mean_se):
            spectral_efficiencies[0, ue_idx, :] = np.abs(
                self.rng.normal(
                    ue_mean,
                    self.default_std,
                    size=(self.num_available_rbs[0],),
                )
            )

        return spectral_efficiencies
