import operator
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
        self.update_steps = 500
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

        (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        ) = self.remove_finished_slices(
            basestation_ue_assoc, basestation_slice_assoc, slice_ue_assoc, slice_req
        )

        self.slices_lifetime[(self.slices_lifetime != 0).nonzero()[0]] -= 1

        if (step_number % self.update_steps == 0) and (
            np.sum(basestation_slice_assoc) < 10
        ):
            return self.associations(
                basestation_ue_assoc, basestation_slice_assoc, slice_ue_assoc, slice_req
            )
        else:
            return (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )

    def remove_finished_slices(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:

        slices_to_deactivate = (self.slices_lifetime == 1).nonzero()[0]

        if len(slices_to_deactivate) > 0:
            basestation_slice_assoc[0, slices_to_deactivate] = 0
            slice_ue_assoc[slices_to_deactivate, :] = 0
            basestation_ue_assoc = np.array([np.sum(slice_ue_assoc, axis=0)])

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )

    def associations(
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
        slices_to_use = (self.slices_lifetime == 0).nonzero()[0]
        self.slices_lifetime[
            self.rng.choice(slices_to_use, initial_slices, replace=False)
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
        for idx in slices_to_use:
            if basestation_slice_assoc[0, idx] == 1:
                slice_ue_assoc[
                    idx, active_ues[used_ues : used_ues + ues_per_slices[used_slices]]
                ] = 1
                used_ues += ues_per_slices[used_slices]
                used_slices += 1
        basestation_ue_assoc = np.array([np.sum(slice_ue_assoc, axis=0)])

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )

    def slice_generator(self):
        expectation_params = {
            "at_least": operator.ge,
            "at_most": operator.le,
            "exactly": operator.eq,
            "greater": operator.gt,
            "one_of": operator.contains,
            "smaller": operator.lt,
        }
        slice_types = [
            "control_case_2",
            "mobile_robot_case_2",
            "monitoring_case_1",
            "robotic_surgery_case_1",
            "robotic_diagnosis",
            "medical_monitoring",
            "uav_app_case_1",
            "uav_control_non_vlos",
            "vr_gaming",
            "cloud_gaming",
            "video_streaming",
        ]

        slice_type_req = {
            "control_case_2": {
                "target": "slice",
                "description": "test",
            },
        }

        print(f"{expectation_params['at_least'](5,1)}{slice_types}{slice_type_req}")
