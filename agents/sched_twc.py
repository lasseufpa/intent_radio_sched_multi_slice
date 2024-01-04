from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces

from agents.common import (
    calculate_slice_ue_obs,
    get_metric_value,
    intent_drift_calc,
    round_robin,
    scores_to_rbs,
)
from associations.mult_slice import MultSliceAssociation
from sixg_radio_mgmt import Agent, MARLCommEnv


class TWC(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        max_obs_memory = 10
        self.max_number_ues_slice = 5
        self.last_unformatted_obs = deque(maxlen=max_obs_memory)
        self.last_formatted_obs = {}
        self.intent_overfulfillment_rate = 0.2
        self.rbs_per_rbg = 1  # 135/rbs_per_rbg RBGs
        assert isinstance(
            self.env.comm_env.associations, MultSliceAssociation
        ), "Associations must be MultSliceAssociation"
        self.max_throughput_slice = np.max(
            [
                slice_type["ues"]["traffic"]
                * slice_type["ues"]["max_number_ues"]
                for slice_type in self.env.comm_env.associations.slice_type_model.values()
            ]
        )

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        # For each slice keep  nine metrics defined
        # for each slice and UE: spectral efficiency, served throughput,
        # effective throughput, buffer occupancy, packet loss rate, re-
        # quested throughput, average buffer latency, long-term served
        # throughput and the fifth-percentile served throughput.
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.last_unformatted_obs.appendleft(obs_space)

        # Inter-slice observation
        formatted_obs_space = {}
        formatted_obs_space["player_0"] = {
            "observations": np.array([]),
            "action_mask": self.last_unformatted_obs[0][
                "basestation_slice_assoc"
            ][0].astype(np.int8),
        }

        dict_metrics = {
            "requirements": np.array([]),
            "spectral_efficiencies": np.array([]),
            "pkt_throughputs": np.array([]),
            "pkt_effective_thrs": np.array([]),
            "buffer_occupancies": np.array([]),
            "buffer_latencies": np.array([]),
            "pkt_loss_rates": np.array([]),
            "requested_thrs": np.array([]),
        }

        # intra-slice observations
        for agent_idx in range(1, obs_space["slice_ue_assoc"].shape[0] + 1):
            slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                agent_idx - 1
            ].nonzero()[0]

            requirements = np.zeros(3)
            if slice_ues.shape[0] != 0:
                for parameter in self.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["parameters"].values():
                    if parameter["name"] == "reliability":
                        requirements[0] = parameter["value"]
                    elif parameter["name"] == "latency":
                        requirements[1] = parameter["value"]
                    elif parameter["name"] == "throughput":
                        requirements[2] = parameter["value"]
            dict_metrics["requirements"] = np.append(
                dict_metrics["requirements"], requirements
            )
            # Pkt size
            pkt_size = (
                self.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["message_size"]
                if slice_ues.shape[0] != 0
                else 0
            )

            # Spectral efficiency
            if slice_ues.shape[0] != 0:
                spectral_eff = np.mean(
                    self.last_unformatted_obs[0]["spectral_efficiencies"][
                        0, slice_ues, :
                    ],
                    axis=1,
                )
                max_spectral_eff = np.max(spectral_eff)
                spectral_eff = (
                    spectral_eff / max_spectral_eff
                    if max_spectral_eff != 0
                    else spectral_eff
                )
            else:
                spectral_eff = np.array([0])
            dict_metrics["spectral_efficiencies"] = np.append(
                dict_metrics["spectral_efficiencies"], np.mean(spectral_eff)
            )

            # Served Throughput
            served_thr = (
                self.last_unformatted_obs[0]["pkt_throughputs"][slice_ues]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_throughputs"] = np.append(
                dict_metrics["pkt_throughputs"],
                np.mean(self.pkts_to_mbps(served_thr, pkt_size)),
            )

            # Effective Throughput
            eff_thr = (
                self.last_unformatted_obs[0]["pkt_effective_thr"][slice_ues]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_effective_thrs"] = np.append(
                dict_metrics["pkt_effective_thrs"],
                np.mean(self.pkts_to_mbps(eff_thr, pkt_size)),
            )

            # Buffer Occ.
            buffer_occ = (
                self.last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_occupancies"] = np.append(
                dict_metrics["buffer_occupancies"], np.mean(buffer_occ)
            )

            # Buffer latencies
            buffer_latencies = (
                self.last_unformatted_obs[0]["buffer_latencies"][slice_ues]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_latencies"] = np.append(
                dict_metrics["buffer_latencies"], np.mean(buffer_latencies)
            )

            # Packet loss rate
            pkt_loss_rate = (
                get_metric_value(
                    "reliability",
                    self.last_unformatted_obs,
                    agent_idx - 1,
                    slice_ues,
                    reliability_pkt_loss=True,
                )
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_loss_rates"] = np.append(
                dict_metrics["pkt_loss_rates"], np.mean(pkt_loss_rate)
            )

            # Requested throughput
            slice_traffic_req = (
                self.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["traffic"]
                if self.last_unformatted_obs[0]["basestation_slice_assoc"][
                    0, agent_idx - 1
                ]
                == 1
                else 0
            )
            dict_metrics["requested_thrs"] = np.append(
                dict_metrics["requested_thrs"],
                slice_traffic_req,
            )
        for key in dict_metrics.keys():
            formatted_obs_space["player_0"]["observations"] = np.concatenate(
                (
                    formatted_obs_space["player_0"]["observations"],
                    dict_metrics[key],
                )
            )
        formatted_obs_space["player_1"] = np.array([0, 1])

        self.last_formatted_obs = formatted_obs_space

        return formatted_obs_space

    def calculate_reward(self, obs_space: dict) -> dict:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        intent_drift = intent_drift_calc(
            self.last_unformatted_obs,
            self.max_number_ues_slice,
            self.intent_overfulfillment_rate,
        )
        valid_intents = np.array([])
        weights = np.array([])
        reward = {}
        for player_idx in np.arange(self.env.comm_env.max_number_slices + 1):
            slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                player_idx - 1
            ].nonzero()[0]
            if player_idx == 0 or slice_ues.shape[0] == 0:
                continue
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                self.max_number_ues_slice,
                intent_drift,
                player_idx - 1,
                slice_ues,
                self.last_unformatted_obs[0]["slice_req"],
            )
            valid_intent = intent_drift_slice[
                np.logical_not(np.isclose(intent_drift_slice, -2))
            ]
            valid_intents = np.append(
                valid_intents,
                valid_intent,
            )
            valid_intents[
                valid_intents > 0
            ] = 0  # It does not consider positive values
            weight_value = (
                2
                if bool(
                    self.last_unformatted_obs[0]["slice_req"][
                        f"slice_{player_idx - 1}"
                    ]["priority"]
                )
                else 1
            )
            weights = np.append(
                weights, weight_value * np.ones_like(valid_intent)
            )
        reward["player_0"] = (
            np.sum(valid_intents * weights / np.sum(weights))
            if np.sum(weights) != 0
            else 0
        )
        reward["player_1"] = 0  # Keeping a second agent to use MARL ray

        return reward

    def action_format(self, action_ori: Union[np.ndarray, dict]) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        allocation_rbs = np.array(
            [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
        )

        if (
            np.sum(
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :]
            )
            != 0
        ):
            action = deepcopy(action_ori)
            action["player_0"][
                np.where(
                    self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ]
                    == 0
                )[0]
            ] = -1

            # Inter-slice scheduling
            rbs_per_slice = (
                scores_to_rbs(
                    action["player_0"],
                    np.floor(
                        self.num_available_rbs[0] / self.rbs_per_rbg
                    ).astype(int),
                    self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ],
                )
                * self.rbs_per_rbg
            )

            # Intra-slice scheduling
            for slice_idx in np.arange(rbs_per_slice.shape[0]):
                slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                    slice_idx, :
                ].nonzero()[0]
                if slice_ues.shape[0] == 0:
                    continue
                allocation_rbs = round_robin(
                    allocation_rbs,
                    slice_idx,
                    rbs_per_slice,
                    slice_ues,
                    self.last_unformatted_obs,
                )
            assert (
                np.sum(allocation_rbs) == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"

        return allocation_rbs

    @staticmethod
    def get_action_space() -> spaces.Dict:
        num_agents = 2
        action_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Box(
                    low=-1, high=1, shape=(5,), dtype=np.float64
                )
                if idx == 0
                else spaces.Discrete(
                    1
                )  # Three algorithms (RR, PF and Maximum Throughput)
                for idx in range(num_agents)
            }
        )

        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Dict:
        num_agents = 2
        obs_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-2, high=np.inf, shape=(50,), dtype=np.float64
                        ),
                        "action_mask": spaces.Box(
                            0.0, 1.0, shape=(5,), dtype=np.int8
                        ),
                    }
                )
                if idx == 0
                else spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
                for idx in range(num_agents)
            }
        )

        return obs_space

    def pkts_to_mbps(self, pkts: np.ndarray, pkt_size: float) -> np.ndarray:
        return pkts * pkt_size / 1e6
