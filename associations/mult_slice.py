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
        self.max_number_ues_slice = int(max_number_ues / max_number_slices)
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

        return self.initial_associations(
            basestation_ue_assoc, basestation_slice_assoc, slice_ue_assoc, slice_req
        )

    # def dynamic_associations(
    #     self,
    #     basestation_ue_assoc: np.ndarray,
    #     basestation_slice_assoc: np.ndarray,
    #     slice_ue_assoc: np.ndarray,
    #     slice_req: Optional[dict],
    # ):
    #     if np.sum(basestation_slice_assoc) < self.max_number_slices:
    #         slices_to_create = self.rng.integers(
    #             0,
    #             int(self.max_number_slices - np.sum(basestation_slice_assoc)),
    #             endpoint=True,
    #         )
    #         ues_per_slices = self.rng.integers(
    #             self.min_number_ues_slice,
    #             self.max_number_ues_slice,
    #             slices_to_create,
    #             endpoint=True,
    #         )
    #         self.slices_lifetime[
    #             self.rng.choice(
    #                 np.arange(self.max_number_slices), slices_to_create, replace=False
    #             )
    #         ] = self.rng.integers(
    #             self.min_steps, self.max_steps, slices_to_create, endpoint=True
    #         )

    def initial_associations(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        initial_slices = self.rng.integers(
            1,
            int(self.max_number_slices - np.sum(basestation_slice_assoc[0])),
            endpoint=True,
        )
        ues_per_slices = self.rng.integers(
            self.min_number_ues_slice,
            self.max_number_ues_slice,
            initial_slices,
            endpoint=True,
        )
        self.slices_lifetime[
            self.rng.choice(
                (self.slices_lifetime == 0).nonzero()[0], initial_slices, replace=False
            )
        ] = self.rng.integers(
            self.min_steps, self.max_steps, initial_slices, endpoint=True
        )

        basestation_slice_assoc[0, self.slices_lifetime != 0] = 1
        active_ues = np.array(
            self.rng.choice(
                (basestation_ue_assoc[0] == 0).nonzero()[0],
                np.sum(ues_per_slices),
                replace=False,
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
        basestation_ue_assoc = np.array([np.sum(slice_ue_assoc, axis=0)])

        print(
            basestation_slice_assoc,
            "\n",
            np.sum(basestation_ue_assoc),
            "\n",
            np.sum(ues_per_slices),
            "\n",
            ues_per_slices,
            "\n",
            np.sum(slice_ue_assoc, axis=1),
        )

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )
