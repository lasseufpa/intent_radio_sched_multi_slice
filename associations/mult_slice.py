from typing import Tuple

import numpy as np

from sixg_radio_mgmt import Association, UEs


class MultSliceAssociation(Association):
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
        self.scenario_name = "mult_slice"  # Always use mult_slice associations
        self.min_number_slices = 3
        self.generator_mode = generator_mode
        self.max_number_slices = 5
        self.maximum_number_scenarios = 100
        self.current_episode = -1
        self.slices_to_use = np.array([])
        self.slice_types = [
            "control_case_2",
            "monitoring_case_1",
            "robotic_surgery_case_1",
            "robotic_diagnosis",
            "medical_monitoring",
            "uav_app_case_1",
            "uav_control_non_vlos",
            "vr_gaming",
            "cloud_gaming",
            "video_streaming_4k",
        ]
        self.expectation_params = {
            "at_least": np.greater_equal,
            "at_most": np.less_equal,
            "exactly": np.equal,
            "greater": np.greater,
            "one_of": np.isin,
            "smaller": np.less,
        }
        self.slices_lifetime = np.zeros(self.max_number_slices, dtype=int)

        self.slice_type_model = {
            "control_case_2": {
                "name": "control_case_2",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.999999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 50,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 100,  # ms
                    "message_size": 1 * 1024 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 5,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "monitoring_case_1": {
                "name": "monitoring_case_1",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "throughput",
                        "value": 10,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts,
                    "buffer_latency": 100,  # ms
                    "message_size": 1 * 1024 * 8,
                    "mobility": 72,  # Km/h
                    "traffic": 10,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "robotic_surgery_case_1": {
                "name": "robotic_surgery_case_1",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.9999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 20,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 30,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 2000 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 30,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "robotic_diagnosis": {
                "name": "robotic_diagnosis",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 20,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 15,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 80 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 30,  # Mbps,
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "medical_monitoring": {
                "name": "medical_monitoring",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.9999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 100,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 10,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 200,  # ms
                    "message_size": 1000 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 10,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "uav_app_case_1": {
                "name": "uav_app_case_1",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "latency",
                        "value": 200,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par2": {
                        "name": "throughput",
                        "value": 100,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 400,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 30,  # Km/h
                    "traffic": 100,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 4,
                },
            },
            "uav_control_non_vlos": {
                "name": "uav_control_non_vlos",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.99,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 140,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 20,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 300,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 30,  # Km/h
                    "traffic": 20,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "vr_gaming": {
                "name": "vr_gaming",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.99,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 10,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 100,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 20,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 100,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 4,
                },
            },
            "cloud_gaming": {
                "name": "cloud_gaming",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "latency",
                        "value": 80,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par2": {
                        "name": "throughput",
                        "value": 50,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 160,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 50,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 5,
                },
            },
            "video_streaming_4k": {
                "name": "video_streaming_4k",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "throughput",
                        "value": 30,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 100,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 30,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 5,
                },
            },
        }
        self.associations = np.array([])

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        if self.generator_mode:
            if step_number == 0:
                number_slices = self.rng.integers(
                    low=self.min_number_slices,
                    high=self.max_number_slices,
                    endpoint=True,
                )
                self.slices_to_use = self.rng.choice(
                    np.arange(self.max_number_slices),
                    number_slices,
                    replace=False,
                )
                basestation_slice_assoc[0, self.slices_to_use] = 1
                slice_req = {
                    f"slice_{id}": {}
                    for id in np.arange(self.max_number_slices)
                }
                slice_req = self.slice_generator(slice_req, self.slices_to_use)
                ues_per_slices = np.array(
                    [
                        self.rng.integers(
                            slice_req[f"slice_{slice_idx}"]["ues"][
                                "min_number_ues"
                            ],
                            slice_req[f"slice_{slice_idx}"]["ues"][
                                "max_number_ues"
                            ],
                            1,
                            endpoint=True,
                        )
                        for slice_idx in self.slices_to_use
                    ]
                ).flatten()
                active_ues = np.array(
                    self.rng.choice(
                        (basestation_ue_assoc[0] == 0).nonzero()[0],
                        int(np.sum(ues_per_slices)),
                        replace=False,
                    )
                )
                used_ues = 0
                used_slices = 0
                for idx in self.slices_to_use:
                    if basestation_slice_assoc[0, idx] == 1:
                        slice_ue_assoc[
                            idx,
                            active_ues[
                                used_ues : used_ues
                                + ues_per_slices[used_slices]
                            ],
                        ] = 1
                        used_ues += ues_per_slices[used_slices]
                        used_slices += 1
                basestation_ue_assoc = np.array(
                    [np.sum(slice_ue_assoc, axis=0)]
                )

                self.update_ues(slice_ue_assoc, self.slices_to_use, slice_req)

            return (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )
        else:
            episode_to_use, condition = self.choose_episode(
                episode_number, self.current_episode
            )
            if condition:
                # print("Network scenario: ", episode_to_use)
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

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, bool]:
        episode_to_use = episode_number % self.maximum_number_scenarios
        if episode_to_use != current_episode:
            return (episode_to_use, True)
        return (0, False)

    def slice_generator(
        self, slice_req: dict, slices_to_use: np.ndarray
    ) -> dict:
        slices_to_create = self.rng.choice(
            len(self.slice_types), len(slices_to_use), replace=False
        )

        for idx, slice in enumerate(slices_to_create):
            slice_req[f"slice_{slices_to_use[idx]}"] = self.slice_type_model[
                self.slice_types[slice]
            ]

        return slice_req

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
