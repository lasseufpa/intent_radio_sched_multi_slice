from typing import Optional

import numpy as np

from sixg_radio_mgmt import Channel


class FixedSE(Channel):
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
        self.fixed_se = 2.0

    def step(
        self,
        step_number: int,
        episode_number: int,
        mobilities: np.ndarray,
        sched_decision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        spectral_efficiencies = [
            self.fixed_se
            * np.ones((self.max_number_ues, self.num_available_rbs[i]))
            for i in np.arange(self.max_number_basestations)
        ]

        return np.array(spectral_efficiencies)
