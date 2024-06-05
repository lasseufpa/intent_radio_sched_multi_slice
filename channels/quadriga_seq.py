from typing import Optional, Tuple

import h5py
import numpy as np

from channels.quadriga import QuadrigaChannel


class QuadrigaChannelSeq(QuadrigaChannel):
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

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, int, bool]:
        if episode_number != current_episode:
            association_to_use = (
                episode_number // self.channel_eps_per_scenario
            )
            episode_to_use = episode_number % self.channel_eps_per_scenario
            return (association_to_use, episode_to_use, True)
        return (0, 0, False)
