from typing import Tuple

import numpy as np

from sixg_radio_mgmt import Association, UEs


class MultSliceAssociationSeq(Association):
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
            ues,
            max_number_ues,
            max_number_basestations,
            max_number_slices,
            rng,
            root_path,
        )
        self.scenario_name = (
            "mult_slice"  # Reading association files from mult_slice
        )
        self.min_number_slices = 3
        self.max_number_slices = 5
        self.channels_per_association = 100
        self.current_episode = -1

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        episode_to_use = episode_number // self.channels_per_association
        if episode_to_use != self.current_episode:
            self.load_episode_data(episode_to_use)  # Update variables
            self.update_ues(
                self.hist_slice_ue_assoc[step_number],
                self.hist_slices_to_use[step_number],
                self.hist_slice_req[step_number],
            )

        return (
            self.hist_basestation_ue_assoc[step_number],
            self.hist_basestation_slice_assoc[step_number],
            self.hist_slice_ue_assoc[step_number],
            self.hist_slice_req[step_number],
        )

    def update_ues(
        self,
        slice_ue_assoc: np.ndarray,
        slices_to_use: np.ndarray,
        slice_req: dict,
    ) -> None:
        def slice_info(
            parameter: str, num_ues: int, slice_req: dict
        ) -> np.ndarray:
            return np.repeat(
                slice_req[f"slice_{slice}"]["ues"][parameter], num_ues
            )

        for slice in slices_to_use:
            slice_ues = (slice_ue_assoc[slice] == 1).nonzero()[0]
            self.ues.update_ues(
                slice_ues,
                slice_info("buffer_latency", len(slice_ues), slice_req),
                slice_info("buffer_size", len(slice_ues), slice_req),
                slice_info("message_size", len(slice_ues), slice_req),
            )

    def load_episode_data(self, episode_number: int):
        self.association_file = np.load(
            f"{self.root_path}/associations/data/{self.scenario_name}/ep_{episode_number}.npz",
            allow_pickle=True,
            mmap_mode=None,
        )
        self.hist_slice_ue_assoc = self.association_file["hist_slice_ue_assoc"]
        self.hist_slices_to_use = self.association_file["hist_slices_to_use"]
        self.hist_slice_req = self.association_file["hist_slice_req"]
        self.hist_basestation_slice_assoc = self.association_file[
            "hist_basestation_slice_assoc"
        ]
        self.hist_basestation_ue_assoc = self.association_file[
            "hist_basestation_ue_assoc"
        ]
        self.hist_slices_lifetime = self.association_file[
            "hist_slices_lifetime"
        ]
        self.current_episode = episode_number
