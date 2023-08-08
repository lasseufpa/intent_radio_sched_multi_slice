from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from iteround import saferound

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
            formatted_obs_space[f"player_{agent_idx}"] = (
                {
                    "observations": np.append(
                        np.mean(intent_drift_slice_ue, axis=1),
                        self.last_unformatted_obs[0][
                            "basestation_slice_assoc"
                        ][0, :],
                    ),
                    "action_mask": self.last_unformatted_obs[0][
                        "basestation_slice_assoc"
                    ][0].astype(np.int8),
                }
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
        for agent_obs in self.last_formatted_obs.items():
            reward[agent_obs[0]] = (
                np.mean(agent_obs[1])
                if agent_obs[0] != "player_0"
                else np.mean(
                    agent_obs[1]["observations"][
                        0 : self.last_unformatted_obs[0][
                            "basestation_slice_assoc"
                        ][0, :].shape[0]
                    ]
                )
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

            # Inter-slice scheduling
            rbs_per_slice = (
                self.round_int_equal_sum(
                    self.num_available_rbs[0]
                    * (action["player_0"] + 1)
                    / np.sum(action["player_0"] + 1),
                    self.num_available_rbs[0],
                )
                if np.sum(action["player_0"] + 1) != 0
                else self.round_int_equal_sum(
                    np.floor(
                        self.num_available_rbs[0]
                        / np.sum(
                            self.last_unformatted_obs[0][
                                "basestation_slice_assoc"
                            ][0, :]
                        )
                    )
                    * self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ],
                    self.num_available_rbs[0],
                )
            )
            assert (
                np.sum(rbs_per_slice < 0) == 0
            ), "Negative RBs on rbs_per_slice"
            assert (
                np.sum(
                    rbs_per_slice
                    * self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ]
                )
                == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"
            assert isinstance(
                self.env, MARLCommEnv
            ), "Environment must be MARLCommEnv"

            # Intra-slice scheduling
            for slice_idx in np.arange(rbs_per_slice.shape[0]):
                slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
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
                        raise ValueError(
                            "Invalid intra-slice scheduling action"
                        )
            assert (
                np.sum(allocation_rbs) == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"

        return allocation_rbs

    def round_robin(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
        distribute_rbs: bool = True,
    ) -> np.ndarray:
        rbs_per_ue = np.ones_like(slice_ues, dtype=float) * np.floor(
            rbs_per_slice[slice_idx] / slice_ues.shape[0]
        )
        remaining_rbs = int(rbs_per_slice[slice_idx] % slice_ues.shape[0])
        rbs_per_ue[0:remaining_rbs] += 1
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "RR: Number of allocated RBs is different than available RBs"
        assert (
            np.sum(rbs_per_ue < 0) == 0
        ), "Negative RBs on rbs_per_ue are not allowed"

        if distribute_rbs:
            allocation_rbs = self.distribute_rbs_ues(
                rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
            )
            assert (
                np.sum(allocation_rbs[0, slice_ues, :])
                == rbs_per_slice[slice_idx]
            ), "Distribute RBs is different from RR distribution"
            assert np.sum(allocation_rbs) == np.sum(
                rbs_per_slice[0 : slice_idx + 1]
            ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

            return allocation_rbs
        else:
            return rbs_per_ue

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
        snt_thoughput = (
            self.last_unformatted_obs[0]["pkt_effective_thr"][slice_ues]
            * self.env.comm_env.ues.pkt_sizes[slice_ues]
        )
        buffer_occ = self.last_unformatted_obs[0]["buffer_occupancies"][
            slice_ues
        ]
        throughput_available = np.min(
            [
                spectral_eff
                * (
                    rbs_per_slice[slice_idx]
                    * self.env.comm_env.bandwidths[0]
                    / self.num_available_rbs[0]
                )
                / slice_ues.shape[0],
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
            np.array(
                saferound(
                    (
                        rbs_per_slice[slice_idx] * weights / np.sum(weights)
                    ).astype(float),
                    places=0,
                    topline=rbs_per_slice[slice_idx],
                )
            )
            if np.sum(weights) != 0
            else self.round_robin(
                allocation_rbs=np.array([]),
                slice_idx=slice_idx,
                rbs_per_slice=rbs_per_slice,
                slice_ues=slice_ues,
                distribute_rbs=False,
            )
        )
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        assert (
            np.sum(rbs_per_ue < 0) == 0
        ), "Negative RBs on rbs_per_ue are not allowed"
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "PF: Number of allocated RBs is different than available RBs"
        assert (
            np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
        ), "Distribute RBs is different from RR distribution"
        assert np.sum(allocation_rbs) == np.sum(
            rbs_per_slice[0 : slice_idx + 1]
        ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

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
                spectral_eff
                * (
                    rbs_per_slice[slice_idx]
                    * self.env.comm_env.bandwidths[0]
                    / self.num_available_rbs[0]
                )
                / slice_ues.shape[0],
                buffer_occ
                * self.env.comm_env.ues.max_buffer_pkts[slice_ues]
                * self.env.comm_env.ues.pkt_sizes[slice_ues],
            ],
            axis=0,
        )
        rbs_per_ue = (
            np.array(
                saferound(
                    (
                        rbs_per_slice[slice_idx]
                        * throughput_available
                        / np.sum(throughput_available)
                    ).astype(float),
                    places=0,
                    topline=rbs_per_slice[slice_idx],
                )
            )
            if np.sum(throughput_available) != 0
            else self.round_robin(
                allocation_rbs=np.array([]),
                slice_idx=slice_idx,
                rbs_per_slice=rbs_per_slice,
                slice_ues=slice_ues,
                distribute_rbs=False,
            )
        )
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        assert (
            np.sum(rbs_per_ue < 0) == 0
        ), "Negative RBs on rbs_per_ue are not allowed"
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "MT: Number of allocated RBs is different than available RBs"
        assert (
            np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
        ), "Distribute RBs is different from RR distribution"

        assert np.sum(allocation_rbs) == np.sum(
            rbs_per_slice[0 : slice_idx + 1]
        ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

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

    def round_int_equal_sum(
        self, float_array: np.ndarray, target_sum: int
    ) -> np.ndarray:
        # Create a mask for zero and non-zero elements
        zero_mask = float_array == 0
        non_zero_mask = ~zero_mask

        # Calculate the sum of the non-zero elements
        non_zero_sum = np.sum(float_array[non_zero_mask])

        # Round the non-zero elements proportionally
        rounded_integers = np.zeros_like(float_array)
        if non_zero_sum != 0:
            proportion = target_sum / non_zero_sum
            rounded_integers[non_zero_mask] = np.floor(
                float_array[non_zero_mask] * proportion
            ).astype(int)

        # Calculate the current sum of rounded_integers
        current_sum = np.sum(rounded_integers)

        # Calculate the rounding adjustment needed to reach the target sum
        adjustment = target_sum - current_sum

        if adjustment > 0:
            # If the adjustment is positive, distribute it among the non-zero elements
            for i in range(
                int(adjustment)
            ):  # Convert adjustment to an integer here
                non_zero_indices = np.nonzero(rounded_integers != 0)[0]
                index = non_zero_indices[i % len(non_zero_indices)]
                rounded_integers[index] += 1
        elif adjustment < 0:
            # If the adjustment is negative, proportionally adjust the rounded integers
            if np.any(rounded_integers[non_zero_mask] == 0):
                # If any non-zero element becomes zero after proportionally adjusting, revert to the previous approach
                rounded_integers = np.zeros_like(float_array)
                if non_zero_sum != 0:
                    proportion = target_sum / non_zero_sum
                    rounded_integers[non_zero_mask] = np.floor(
                        float_array[non_zero_mask] * proportion
                    ).astype(int)
            else:
                # Proportionally adjust the non-zero elements
                proportion = target_sum / np.sum(
                    rounded_integers[non_zero_mask]
                )
                rounded_integers[non_zero_mask] = np.floor(
                    rounded_integers[non_zero_mask] * proportion
                ).astype(int)

        return rounded_integers

    @staticmethod
    def get_action_space() -> spaces.Dict:
        num_agents = 11
        action_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Box(
                    low=-1, high=1, shape=(10,), dtype=np.float64
                )
                if idx == 0
                else spaces.Discrete(
                    3
                )  # Three algorithms (RR, PF and Maximum Throughput)
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
                if idx == 0
                else spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float64)
                for idx in range(num_agents)
            }
        )

        return obs_space
