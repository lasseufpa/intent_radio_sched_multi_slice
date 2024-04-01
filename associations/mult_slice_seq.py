from typing import Tuple

import numpy as np

from associations.mult_slice import MultSliceAssociation
from sixg_radio_mgmt import UEs


class MultSliceAssociationSeq(MultSliceAssociation):
    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
        root_path: str = ".",
        generator_mode: bool = False,
        slice_req_changed: bool = True,  # When you change slice_type_model after using gen_assoc_mult_slice.py
        scenario_name: str = "mult_slice",
    ) -> None:
        super().__init__(
            ues=ues,
            max_number_ues=max_number_ues,
            max_number_basestations=max_number_basestations,
            max_number_slices=max_number_slices,
            rng=rng,
            root_path=root_path,
            generator_mode=generator_mode,
            slice_req_changed=slice_req_changed,
            scenario_name=scenario_name,
        )
        self.scenario_name = (
            "mult_slice"  # Reading associations from mult_slice folder
        )
        self.channels_per_scenario = 100

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, bool]:
        episode_to_use = episode_number // self.channels_per_scenario
        if episode_to_use != current_episode:
            return (episode_to_use, True)
        return (0, False)
