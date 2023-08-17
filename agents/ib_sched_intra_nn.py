from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from scipy.optimize import minimize

from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSchedIntraNN(Agent):
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
        self.max_number_ues_slice = 10
        self.last_unformatted_obs = deque(maxlen=max_obs_memory)
        self.last_formatted_obs = {}
        self.intent_overfulfillment_rate = 0.2

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        self.last_unformatted_obs.appendleft(obs_space)
        intent_drift_slice_ue = self.intent_drift_calc(
            self.last_unformatted_obs
        )
        formatted_obs_space = {}
        for agent_idx in range(obs_space["slice_ue_assoc"].shape[0] + 1):
            if agent_idx == 0:
                association = self.last_unformatted_obs[0][
                    "basestation_slice_assoc"
                ][0, :]
            else:
                association = np.zeros(self.max_number_ues_slice)
                num_connected_ues = np.sum(
                    self.last_unformatted_obs[0]["slice_ue_assoc"][
                        agent_idx - 1, :
                    ]
                ).astype(int)
                association[0:num_connected_ues] = 1
            formatted_obs_space[f"player_{agent_idx}"] = {
                "observations": np.append(
                    np.mean(intent_drift_slice_ue, axis=1)
                    if agent_idx == 0
                    else intent_drift_slice_ue[agent_idx - 1, :],
                    association,
                ).astype(np.float64),
                "action_mask": association.astype(np.int8),
            }
        self.last_formatted_obs = formatted_obs_space.copy()

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
        observations = np.zeros(
            (
                last_unformatted_obs[0]["slice_ue_assoc"].shape[0],
                self.max_number_ues_slice,
            ),
            dtype=float,
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
        for player_idx, agent_obs in enumerate(
            self.last_formatted_obs.items()
        ):
            elements_idx = agent_obs[1]["action_mask"].nonzero()[0]
            active_observations = (
                agent_obs[1]["observations"][elements_idx]
                if elements_idx.shape[0] > 0
                else np.array([1])
            )
            if np.sum(active_observations < 0) == 0:
                reward[agent_obs[0]] = np.mean(active_observations)
            else:
                negative_obs_idx = (active_observations < 0).nonzero()[0]
                reward[agent_obs[0]] = np.mean(
                    active_observations[negative_obs_idx]
                )

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

        if (
            np.sum(
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :]
            )
            != 0
        ):
            assert (  # verify if the inactive slices are receiving -1 value
                np.prod(
                    action["player_0"][
                        self.last_unformatted_obs[0][
                            "basestation_slice_assoc"
                        ][0, :]
                        == 0
                    ]
                    == -1
                )
                == 1
            ), "Invalid action"
            assert isinstance(
                self.env, MARLCommEnv
            ), "Environment must be MARLCommEnv"

            # Inter-slice scheduling
            rbs_per_slice = self.scores_to_rbs(
                action["player_0"],
                self.num_available_rbs[0],
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :],
            )

            # Intra-slice scheduling
            for slice_idx in np.arange(rbs_per_slice.shape[0]):
                slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                    slice_idx, :
                ].nonzero()[0]
                slice_ue_assoc = np.zeros(self.max_number_ues_slice)
                num_connected_ues = np.sum(
                    self.last_unformatted_obs[0]["slice_ue_assoc"][
                        slice_idx, :
                    ]
                ).astype(int)
                slice_ue_assoc[0:num_connected_ues] = 1
                if slice_ues.shape[0] == 0:
                    continue

                rbs_per_ue = self.scores_to_rbs(
                    action[f"player_{slice_idx+1}"],
                    int(rbs_per_slice[slice_idx]),
                    slice_ue_assoc,
                )
                allocation_rbs = self.distribute_rbs_ues(
                    rbs_per_ue,
                    allocation_rbs,
                    slice_ues,
                    rbs_per_slice,
                    slice_idx,
                )
            assert (
                np.sum(allocation_rbs) == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"

        return allocation_rbs

    def scores_to_rbs(
        self, action: np.ndarray, total_rbs: int, association: np.ndarray
    ) -> np.ndarray:
        rbs_per_unit = (
            self.round_int_equal_sum(
                total_rbs * (action + 1) / np.sum(action + 1),
                total_rbs,
            )
            if np.sum(action + 1) != 0
            else self.round_int_equal_sum(
                (total_rbs / np.sum(association)) * association,
                total_rbs,
            )
        )
        assert np.sum(rbs_per_unit < 0) == 0, "Negative RBs"
        assert (
            np.sum(rbs_per_unit * association).astype(int) == total_rbs
        ), "Allocated RBs are different from available RBs"

        return rbs_per_unit

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

    def round_int_equal_sum(
        self, float_array: np.ndarray, target_sum: int
    ) -> np.ndarray:
        non_zero_indices = np.where(float_array != 0)[0]
        non_zero_values = float_array[non_zero_indices]

        # Proportional distribution to get as close as possible to the target sum
        proportional_integers = np.floor(
            target_sum * non_zero_values / np.sum(non_zero_values)
        ).astype(int)

        # Calculate the remaining adjustment
        adjustment = target_sum - np.sum(proportional_integers)

        # Distribute the remaining adjustment among the highest values
        sorted_indices = np.argsort(non_zero_values)[::-1]
        for i in range(adjustment):
            index = sorted_indices[i % len(sorted_indices)]
            proportional_integers[index] += 1

        # Reconstruct the rounded result
        rounded_integers = np.zeros_like(float_array, dtype=int)
        rounded_integers[non_zero_indices] = proportional_integers

        return rounded_integers

    @staticmethod
    def get_action_space() -> spaces.Dict:
        num_agents = 11
        action_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Box(
                    low=-1, high=1, shape=(10,), dtype=np.float64
                )
                for idx in range(num_agents)
            }
        )

        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Dict:
        num_agents = 11
        obs_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-1, high=1, shape=(20,), dtype=np.float64
                        ),
                        "action_mask": spaces.Box(
                            0.0, 1.0, shape=(10,), dtype=np.int8
                        ),
                    }
                )
                for idx in range(num_agents)
            }
        )

        return obs_space
