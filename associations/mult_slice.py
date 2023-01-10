from typing import Optional, Tuple

import numpy as np

from sixg_radio_mgmt.sixg_radio_mgmt.association import Association


class MultSliceAssociation(Association):
    def __init__(
        self,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:

        self.max_steps = 2000
        self.min_steps = 500
        self.min_number_ues_slice = 10
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.rng = rng
        self.slices_lifetime = np.zeros(self.max_number_slices)

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:

        if step_number == 0:
            return self.initial_associations(
                basestation_ue_assoc, basestation_slice_assoc, slice_ue_assoc, slice_req
            )
        else:
            return (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )

    def initial_associations(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        initial_slices = self.rng.integers(1, self.max_number_slices)
        ues_per_slices = self.rng.integers(
            self.min_number_ues_slice,
            int(self.max_number_ues / self.max_number_slices),
            initial_slices,
        )
        self.slices_lifetime[
            self.rng.choice(
                np.arange(self.max_number_slices), initial_slices, replace=False
            )
        ] = self.rng.integers(self.min_steps, self.max_steps, initial_slices)

        basestation_slice_assoc[0, self.slices_lifetime != 0] = 1
        active_ues = np.array(
            self.rng.choice(
                np.arange(self.max_number_ues), np.sum(ues_per_slices), replace=False
            )
        )
        used_ues = 0
        used_slices = 0
        for idx in np.arange(self.max_number_slices):
            if basestation_slice_assoc[0, idx] == 1:
                slice_ue_assoc[
                    idx, active_ues[used_ues : used_ues + ues_per_slices[used_slices]]
                ] = 1
                used_ues += ues_per_slices[used_slices]
                used_slices += 1
        basestation_ue_assoc = np.sum(slice_ue_assoc, axis=0)

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )
