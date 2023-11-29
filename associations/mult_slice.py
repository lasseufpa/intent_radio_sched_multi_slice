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
    ) -> None:
        super().__init__(
            ues,
            max_number_ues,
            max_number_basestations,
            max_number_slices,
            rng,
            root_path,
        )
        self.max_steps = 2000
        self.min_steps = 500
        self.update_steps = 500
        self.min_number_ues_slice = 2
        self.max_number_ues_slice = int(max_number_ues / max_number_slices)
        self.slices_lifetime = np.zeros(self.max_number_slices)
        self.generator_mode = (
            generator_mode  # False for reading from external files
        )
        self.slice_req_changed = slice_req_changed
        self.scenario_name = "mult_slice"
        self.current_episode = -1
        self.slices_to_use = np.array([])
        self.association_file = dict()
        self.hist_slice_ue_assoc = np.array([])
        self.hist_slices_to_use = np.array([])
        self.hist_slice_req = np.array([])
        self.hist_basestation_slice_assoc = np.array([])
        self.hist_basestation_ue_assoc = np.array([])
        self.hist_slices_lifetime = np.array([])
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
                    "traffic": 2,  # Mbps
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
                        "value": 5,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts,
                    "buffer_latency": 100,  # ms
                    "message_size": 1 * 1024 * 8,
                    "mobility": 72,  # Km/h
                    "traffic": 5,  # Mbps
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
                        "value": 16,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 2000 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 16,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 4,
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
                        "value": 16,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 80 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 16,  # Mbps,
                    "min_number_ues": 2,
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
                        "value": 1,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 200,  # ms
                    "message_size": 1000 * 8,
                    "mobility": 200,  # Km/h
                    "traffic": 1,  # Mbps
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
                    "min_number_ues": 1,
                    "max_number_ues": 1,
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
                        "value": 4,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 300,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 30,  # Km/h
                    "traffic": 4,  # Mbps
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
                    "min_number_ues": 1,
                    "max_number_ues": 1,
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
                        "value": 25,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 160,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 25,  # Mbps
                    "min_number_ues": 1,
                    "max_number_ues": 5,
                },
            },
            "video_streaming_4k": {
                "name": "video_streaming_4k",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "throughput",
                        "value": 15,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 100,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 15,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 5,
                },
            },
        }

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
            if slice_req == {}:
                slice_req = {
                    f"slice_{id}": {}
                    for id in np.arange(self.max_number_slices)
                }

            (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            ) = self.remove_finished_slices(
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )

            self.slices_lifetime[(self.slices_lifetime != 0).nonzero()[0]] -= 1

            if (step_number % self.update_steps == 0) and (
                np.sum(basestation_slice_assoc) < 10
            ):
                return self.associations(
                    basestation_ue_assoc,
                    basestation_slice_assoc,
                    slice_ue_assoc,
                    slice_req,
                )
            else:
                return (
                    basestation_ue_assoc,
                    basestation_slice_assoc,
                    slice_ue_assoc,
                    slice_req,
                )
        else:
            if episode_number != self.current_episode:
                self.load_episode_data(episode_number)  # Update variables
            if self.slice_req_changed:
                tmp_slice_req = {}
                for slice_key in self.hist_slice_req[step_number].keys():
                    if self.hist_slice_req[step_number][slice_key] != {}:
                        tmp_slice_req[slice_key] = self.slice_type_model[
                            self.hist_slice_req[step_number][slice_key]["name"]
                        ]
                    else:
                        tmp_slice_req[slice_key] = {}
                self.hist_slice_req[step_number] = tmp_slice_req
            if step_number % self.update_steps == 0:
                self.update_ues(
                    self.hist_slice_ue_assoc[step_number],
                    self.hist_slices_to_use[step_number],
                    self.hist_slice_req[step_number],
                )

            self.slices_lifetime = self.hist_slices_lifetime[step_number]

            return (
                self.hist_basestation_ue_assoc[step_number],
                self.hist_basestation_slice_assoc[step_number],
                self.hist_slice_ue_assoc[step_number],
                self.hist_slice_req[step_number],
            )

    def remove_finished_slices(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        slices_to_deactivate = (self.slices_lifetime == 1).nonzero()[0]

        if len(slices_to_deactivate) > 0:
            basestation_slice_assoc[0, slices_to_deactivate] = 0
            slice_ue_assoc[slices_to_deactivate, :] = 0
            basestation_ue_assoc = np.array([np.sum(slice_ue_assoc, axis=0)])
            for slice in slices_to_deactivate:
                slice_req[f"slice_{slice}"] = {}

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
        slice_req: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        slices_available = (self.slices_lifetime == 0).nonzero()[0]
        initial_slices = self.rng.integers(
            0,
            int(self.max_number_slices - np.sum(basestation_slice_assoc[0])),
            endpoint=True,
        )
        self.slices_to_use = np.array([])
        if initial_slices > 0:
            self.slices_to_use = self.rng.choice(
                slices_available, initial_slices, replace=False
            )
            slice_req = self.slice_generator(slice_req, self.slices_to_use)
            self.slices_lifetime[self.slices_to_use] = self.rng.integers(
                self.min_steps, self.max_steps, initial_slices, endpoint=True
            )
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

            basestation_slice_assoc[0, self.slices_lifetime != 0] = 1
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
                            used_ues : used_ues + ues_per_slices[used_slices]
                        ],
                    ] = 1
                    used_ues += ues_per_slices[used_slices]
                    used_slices += 1
            basestation_ue_assoc = np.array([np.sum(slice_ue_assoc, axis=0)])

            self.update_ues(slice_ue_assoc, self.slices_to_use, slice_req)

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )

    def slice_generator(
        self, slice_req: dict, slices_to_use: np.ndarray
    ) -> dict:
        slices_to_create = self.rng.choice(
            len(self.slice_types), len(slices_to_use)
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
