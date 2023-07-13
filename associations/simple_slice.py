from typing import Optional, Tuple

import numpy as np

from sixg_radio_mgmt import Association, UEs


class SimpleSliceAssociation(Association):
    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            ues,
            max_number_ues,
            max_number_basestations,
            max_number_slices,
            rng,
        )

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        expectation_params = {
            "at_least": np.greater_equal,
            "at_most": np.less_equal,
            "exactly": np.equal,
            "greater": np.greater,
            "one_of": np.isin,
            "smaller": np.less,
        }

        if step_number == 0:
            slice_req = {
                "slice_0": {
                    "name": "robotic_surgery_case_1",
                    "parameters": {
                        "par1": {
                            "name": "reliability",
                            "value": 99.00,
                            "unit": "rate",
                            "operator": expectation_params["at_least"],
                        },
                        "par2": {
                            "name": "latency",
                            "value": 20,
                            "unit": "ms",
                            "operator": expectation_params["at_most"],
                        },
                        "par3": {
                            "name": "throughput",
                            "value": 1,
                            "unit": "Mbps",
                            "operator": expectation_params["at_least"],
                        },
                    },
                    "ues": {
                        "buffer_size": 10,  # pkts
                        "buffer_latency": 10,  # ms
                        "message_size": 1,  # bits
                        "mobility": 0,  # Km/h
                        "traffic": 2,  # Mbps
                        "min_number_ues": 8,
                        "max_number_ues": 10,
                    },
                },
                "slice_1": {
                    "name": "control_case_2",
                    "parameters": {
                        "par1": {
                            "name": "reliability",
                            "value": 1.0,
                            "unit": "rate",
                            "operator": expectation_params["at_least"],
                        },
                        "par2": {
                            "name": "latency",
                            "value": 20,
                            "unit": "ms",
                            "operator": expectation_params["at_most"],
                        },
                    },
                    "ues": {
                        "buffer_size": 10,  # pkts
                        "buffer_latency": 10,  # ms
                        "message_size": 1,  # bits
                        "mobility": 0,  # Km/h
                        "traffic": 2,  # Mbps
                        "min_number_ues": 8,
                        "max_number_ues": 10,
                    },
                },
            }

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )
