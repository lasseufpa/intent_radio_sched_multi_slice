from typing import Optional, Tuple

import numpy as np

from sixg_radio_mgmt import Association


class MultSliceAssociation(Association):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:

        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.rng = rng
        self.initial_slices = self.rng.integers(5, 10)
        self.slices_lifetime = np.zeros(max_number_slices)

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:

        if step_number == 1:
            return self.initial_associations()
        else:
            return (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )

    def initial_associations(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        return (np.array([]), np.array([]), np.array([]), {})
