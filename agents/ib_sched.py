from collections import deque
from typing import Optional, Union

import numpy as np
from gymnasium import spaces

from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSched(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        max_obs_memory = 10
        self.last_unformatted_obs = deque(maxlen=max_obs_memory)
        self.last_formatted_obs = {}
        self.intent_overfulfillment_rate = 0.2

    def step(
        self, agent: str, obs_space: Optional[Union[np.ndarray, dict]]
    ) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        self.last_unformatted_obs.appendleft(obs_space)
        intent_drift_slice_ue = self.intent_drift_calc(
            self.last_unformatted_obs
        )
        formatted_obs_space = {}
        for agent_idx in range(obs_space["slice_ue_assoc"].shape[0] + 1):
            formatted_obs_space[f"player_{agent_idx}"] = (
                np.mean(intent_drift_slice_ue, axis=1)
                if agent_idx == 0
                else intent_drift_slice_ue[agent_idx - 1, :]
            )
        self.last_formatted_obs = formatted_obs_space

        return formatted_obs_space

    def intent_drift_calc(
        self, last_unformatted_obs: deque[dict]
    ) -> np.ndarray:
        def get_metric_value(
            metric_name: str,
            last_unformatted_obs: deque,
            slice_idx: int,
            slice_ues: np.ndarray,
        ) -> np.ndarray:
            def calc_metric_interval(
                metric: str, slice_ues: np.ndarray
            ) -> float:
                return np.sum(
                    [
                        last_unformatted_obs[i][metric][slice_ues]
                        for i in range(len(last_unformatted_obs))
                    ],
                    axis=0,
                )

            if metric_name == "throughput":
                metric_value = (
                    last_unformatted_obs[0]["pkt_throughputs"][slice_ues]
                    * last_unformatted_obs[0]["slice_req"][
                        f"slice_{slice_idx}"
                    ]["ues"]["message_size"]
                ) / 1e6  # Mbps
            elif metric_name == "reliability":
                pkts_snt_over_interval = calc_metric_interval(
                    "pkt_effective_thr", slice_ues
                )
                dropped_pkts_over_interval = calc_metric_interval(
                    "dropped_pkts", slice_ues
                )
                buffer_pkts = (
                    last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
                    * last_unformatted_obs[0]["slice_req"][
                        f"slice_{slice_idx}"
                    ]["ues"]["buffer_size"]
                    + dropped_pkts_over_interval
                    + pkts_snt_over_interval
                )
                metric_value = np.divide(
                    dropped_pkts_over_interval,
                    buffer_pkts,
                    where=buffer_pkts != 0,
                    out=np.zeros_like(buffer_pkts),
                )  # Rate [0,1]
            elif metric_name == "latency":
                metric_value = last_unformatted_obs[0]["buffer_latencies"][
                    slice_ues
                ]  # Seconds
            else:
                raise ValueError("Invalid metric name")

            return metric_value

        last_obs_slice_req = last_unformatted_obs[0]["slice_req"]
        observations = np.zeros_like(
            last_unformatted_obs[0]["slice_ue_assoc"], dtype=float
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        for slice in last_obs_slice_req:
            if last_obs_slice_req[slice] == {}:
                continue
            slice_idx = int(slice.split("_")[1])
            slice_ues = last_unformatted_obs[0]["slice_ue_assoc"][
                slice_idx
            ].nonzero()[0]
            for parameter in last_obs_slice_req[slice]["parameters"].values():
                metric_value = get_metric_value(
                    parameter["name"],
                    last_unformatted_obs,
                    slice_idx,
                    slice_ues,
                )
                intent_fulfillment = (
                    parameter["operator"](
                        metric_value, parameter["value"]
                    ).astype(int)
                    if parameter["name"] != "reliability"
                    else parameter["operator"](
                        100 * (1 - metric_value), parameter["value"]
                    ).astype(int)
                )
                intent_unfulfillment = np.logical_not(intent_fulfillment)
                match parameter["name"]:
                    case "throughput":
                        # Intent fulfillment
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value
                                > (
                                    parameter["value"]
                                    * (1 + self.intent_overfulfillment_rate)
                                )
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]
                            overfulfilled_mask = overfulfilled_mask.nonzero()[
                                0
                            ]
                            # Fulfilled intent
                            observations[slice_idx, fulfilled_mask] += (
                                metric_value[fulfilled_mask]
                                - parameter["value"]
                            ) / (
                                parameter["value"]
                                * self.intent_overfulfillment_rate
                            )
                            # Overfulfilled intent
                            observations[slice_idx, overfulfilled_mask] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx, intent_unfulfillment.nonzero()[0]
                            ] -= (
                                parameter["value"]
                                - metric_value[
                                    intent_unfulfillment.nonzero()[0]
                                ]
                            ) / (
                                parameter["value"]
                            )

                    case "reliability":
                        # Intent fulfillment
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value
                                < (
                                    ((100 - parameter["value"]) / 100)
                                    * (1 - self.intent_overfulfillment_rate)
                                )
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]

                            overfulfilled_mask = overfulfilled_mask.nonzero()[
                                0
                            ]
                            # Fulfilled intent
                            observations[slice_idx, fulfilled_mask] += (
                                (100 - parameter["value"]) / 100
                                - metric_value[fulfilled_mask]
                            ) / (
                                ((100 - parameter["value"]) / 100)
                                * self.intent_overfulfillment_rate
                            )
                            # Overfulfilled intent
                            observations[slice_idx, overfulfilled_mask] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx, intent_unfulfillment.nonzero()[0]
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - ((100 - parameter["value"]) / 100)
                            ) / (
                                parameter["value"] / 100
                            )

                    case "latency":
                        max_latency_per_ue = (
                            self.env.comm_env.ues.max_buffer_latencies[
                                slice_ues
                            ]
                        )
                        # Intent fulfillment
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value
                                < (
                                    parameter["value"]
                                    * (1 - self.intent_overfulfillment_rate)
                                )
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]
                            overfulfilled_mask = (
                                intent_fulfillment * overfulfilled_mask
                            ).nonzero()[0]
                            # Fulfilled intent
                            observations[slice_idx, fulfilled_mask] += (
                                parameter["value"]
                                - metric_value[fulfilled_mask]
                            ) / (
                                parameter["value"]
                                * self.intent_overfulfillment_rate
                            )
                            # Overfulfilled intent
                            observations[slice_idx, overfulfilled_mask] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx, intent_unfulfillment.nonzero()[0]
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - parameter["value"]
                            ) / (
                                max_latency_per_ue[
                                    intent_unfulfillment.nonzero()[0]
                                ]
                                - parameter["value"]
                            )

                    case _:
                        raise ValueError("Invalid parameter name")

            observations[slice_idx, :] = observations[slice_idx, :] / len(
                last_obs_slice_req[slice]["parameters"]
            )

        return observations

    def calculate_reward(self, obs_space: dict) -> dict:
        reward = {}
        for agent_obs in self.last_formatted_obs.items():
            reward[agent_obs[0]] = np.mean(agent_obs[1])

        return reward

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        allocation_rbs = np.array(
            [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
        )

        # Inter-slice scheduling
        rbs_per_slice = (
            np.round(
                self.num_available_rbs[0]
                * (action["player_0"] + 1)
                / np.sum(action["player_0"] + 1)
            )
            if np.sum(action["player_0"] + 1) != 0
            else np.floor(
                self.num_available_rbs[0] / action["player_0"].shape[0]
            )
            * np.ones_like(action["player_0"], dtype=int)
        )
        assert (
            np.sum(rbs_per_slice) <= self.num_available_rbs[0]
        ), "Allocated RBs are bigger than available RBs"
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"

        # Intra-slice scheduling
        for slice_idx in np.arange(rbs_per_slice.shape[0]):
            slice_ues = self.env.comm_env.slices.ue_assoc[
                slice_idx, :
            ].nonzero()[0]
            if slice_ues.shape[0] == 0:
                continue
            match action[f"player_{slice_idx+1}"]:
                case 0:
                    allocation_rbs = self.round_robin(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case 1:
                    allocation_rbs = self.proportional_fairness(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case 2:
                    allocation_rbs = self.max_throughput(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case _:
                    raise ValueError("Invalid intra-slice scheduling action")

        return allocation_rbs

    def round_robin(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        rbs_per_ue = np.ones_like(slice_ues, dtype=float) * np.floor(
            rbs_per_slice[slice_idx] / slice_ues.shape[0]
        )
        remaining_rbs = int(rbs_per_slice[slice_idx] % slice_ues.shape[0])
        rbs_per_ue[0:remaining_rbs] += 1
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "RR: Number of allocated RBs is different than available RBs"
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def proportional_fairness(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        spectral_eff = np.mean(
            self.last_unformatted_obs[0]["spectral_efficiencies"][
                0, slice_ues, :
            ],
            axis=1,
        )
        snt_thoughput = self.last_unformatted_obs[0]["pkt_effective_thr"][
            slice_ues
        ]
        buffer_occ = self.last_unformatted_obs[0]["buffer_occupancies"][
            slice_ues
        ]
        throughput_available = np.min(
            [
                spectral_eff * self.env.comm_env.bandwidths[0],
                buffer_occ
                * self.env.comm_env.ues.max_buffer_pkts[slice_ues]
                * self.env.comm_env.ues.pkt_sizes[slice_ues],
            ],
            axis=0,
        )
        weights = np.divide(
            throughput_available,
            snt_thoughput,
            where=snt_thoughput != 0,
            out=0.00001 * np.ones_like(snt_thoughput),
        )
        rbs_per_ue = (
            np.round(rbs_per_slice[slice_idx] * weights / np.sum(weights))
            if np.sum(weights) != 0
            else np.floor(rbs_per_slice[slice_idx] / weights.shape[0])
            * np.ones_like(weights)
        )
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "PF: Number of allocated RBs is different than available RBs"

        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def max_throughput(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        spectral_eff = np.mean(
            self.last_unformatted_obs[0]["spectral_efficiencies"][
                0, slice_ues, :
            ],
            axis=1,
        )
        buffer_occ = self.last_unformatted_obs[0]["buffer_occupancies"][
            slice_ues
        ]
        throughput_available = np.min(
            [
                spectral_eff * self.env.comm_env.bandwidths[0],
                buffer_occ
                * self.env.comm_env.ues.max_buffer_pkts[slice_ues]
                * self.env.comm_env.ues.pkt_sizes[slice_ues],
            ],
            axis=0,
        )
        rbs_per_ue = np.round(
            rbs_per_slice[slice_idx]
            * throughput_available
            / np.sum(throughput_available)
            if np.sum(throughput_available) != 0
            else np.floor(
                rbs_per_slice[slice_idx] / throughput_available.shape[0]
            )
            * np.ones_like(throughput_available),
        )
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "MT: Number of allocated RBs is different than available RBs"
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def distribute_rbs_ues(
        self,
        rbs_per_ue: np.ndarray,
        allocation_rbs: np.ndarray,
        slice_ues: np.ndarray,
        rbs_per_slice: np.ndarray,
        slice_idx: int,
    ) -> np.ndarray:
        rb_idx = np.sum(rbs_per_slice[:slice_idx], dtype=int)
        for idx, ue_idx in enumerate(slice_ues):
            allocation_rbs[
                0, ue_idx, rb_idx : rb_idx + rbs_per_ue[idx].astype(int)
            ] = 1
            rb_idx += rbs_per_ue[idx].astype(int)

        return allocation_rbs

    @staticmethod
    def get_action_space() -> dict:
        num_agents = 11
        action_space = {
            f"player_{idx}": spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float64
            )
            if idx == 0
            else spaces.Discrete(
                3
            )  # Three algorithms (RR, PF and Maximum Throughput)
            for idx in range(num_agents)
        }

        return action_space

    @staticmethod
    def get_obs_space() -> dict:
        num_agents = 11
        obs_space = {
            f"player_{idx}": spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float64
            )
            for idx in range(num_agents)
        }

        return obs_space
