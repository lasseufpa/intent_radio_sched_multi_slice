from collections import deque
from typing import Tuple, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from sixg_radio_mgmt import MARLCommEnv


def get_metric_value(
    metric_name: str,
    last_unformatted_obs: deque,
    slice_idx: int,
    slice_ues: np.ndarray,
    reliability_pkt_loss: bool = False,
) -> np.ndarray:
    def calc_metric_interval(metric: str, slice_ues: np.ndarray) -> float:
        return np.sum(
            [
                last_unformatted_obs[i][metric][slice_ues]
                for i in range(len(last_unformatted_obs))
            ],
            axis=0,
        )

    if metric_name == "throughput":
        metric_value = (
            last_unformatted_obs[0]["pkt_effective_thr"][slice_ues]
            * last_unformatted_obs[0]["slice_req"][f"slice_{slice_idx}"][
                "ues"
            ]["message_size"]
        ) / 1e6  # Mbps
    elif metric_name == "reliability":
        if reliability_pkt_loss:
            pkts_snt_over_interval = calc_metric_interval(
                "pkt_effective_thr", slice_ues
            )
            dropped_pkts_over_interval = calc_metric_interval(
                "dropped_pkts", slice_ues
            )
            buffer_pkts = (
                last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
                * last_unformatted_obs[0]["slice_req"][f"slice_{slice_idx}"][
                    "ues"
                ]["buffer_size"]
                + dropped_pkts_over_interval
                + pkts_snt_over_interval
            )
            metric_value = np.divide(
                dropped_pkts_over_interval,
                buffer_pkts,
                where=buffer_pkts != 0,
                out=np.zeros_like(buffer_pkts),
            )  # Rate [0,1]
        else:
            metric_value = last_unformatted_obs[0]["buffer_occupancies"][
                slice_ues
            ]  # Buffer occupancy rate [0,1]
    elif metric_name == "latency":
        metric_value = last_unformatted_obs[0]["buffer_latencies"][
            slice_ues
        ]  # Seconds
    else:
        raise ValueError("Invalid metric name")

    return metric_value


def intent_drift_calc(
    last_unformatted_obs: deque[dict],
    max_number_ues_slice: int,
    intent_overfulfillment_rate: float,
    reliability_pkt_loss: bool = True,
) -> np.ndarray:
    last_obs_slice_req = last_unformatted_obs[0]["slice_req"]
    metrics = {"throughput": 0, "reliability": 1, "latency": 2}
    observations = np.zeros(
        (
            last_unformatted_obs[0]["slice_ue_assoc"].shape[0],
            max_number_ues_slice,
            len(metrics),
        ),
        dtype=float,
    )

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
                reliability_pkt_loss,
            )
            if parameter["name"] == "throughput":
                # In case the throughput is below the minimum required but the
                # buffer occupancy is zero, we consider the intent as overfulfilled
                # because the UE is not requesting the target throughput at this time
                buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][
                    slice_ues
                ]
                zero_mask = np.isclose(buffer_occ, np.zeros_like(buffer_occ))
                if len(last_unformatted_obs) > 1:
                    previous_buffer_occ = last_unformatted_obs[1][
                        "buffer_occupancies"
                    ][slice_ues]
                    previous_zero_mask = np.isclose(
                        previous_buffer_occ,
                        np.zeros_like(previous_buffer_occ),
                    )
                    zero_mask = np.logical_or(zero_mask, previous_zero_mask)
                metric_value[zero_mask.nonzero()[0]] = parameter["value"] * (
                    1.1 + intent_overfulfillment_rate
                )
            buffer_occupancy_threshold = 0.6

            if parameter["name"] == "reliability":
                if reliability_pkt_loss:
                    intent_fulfillment = parameter["operator"](
                        100 * (1 - metric_value), parameter["value"]
                    ).astype(int)
                else:
                    intent_fulfillment = parameter["operator"](
                        1 - metric_value, (1 - buffer_occupancy_threshold)
                    ).astype(int)
            else:
                intent_fulfillment = parameter["operator"](
                    metric_value, parameter["value"]
                ).astype(int)
            intent_unfulfillment = np.logical_not(intent_fulfillment)

            match parameter["name"]:
                case "throughput":
                    # Intent fulfillment
                    if np.sum(intent_fulfillment) > 0:
                        overfulfilled_mask = intent_fulfillment * (
                            metric_value
                            > (
                                parameter["value"]
                                * (1 + intent_overfulfillment_rate)
                            )
                        )
                        fulfilled_mask = (
                            intent_fulfillment
                            * np.logical_not(overfulfilled_mask)
                        ).nonzero()[0]
                        overfulfilled_mask = overfulfilled_mask.nonzero()[0]
                        # Fulfilled intent
                        observations[
                            slice_idx,
                            fulfilled_mask,
                            metrics[parameter["name"]],
                        ] += (
                            metric_value[fulfilled_mask] - parameter["value"]
                        ) / (
                            parameter["value"] * intent_overfulfillment_rate
                        )
                        # Overfulfilled intent
                        observations[
                            slice_idx,
                            overfulfilled_mask,
                            metrics[parameter["name"]],
                        ] += 1

                    # Intent unfulfillment
                    if np.sum(intent_unfulfillment) > 0:
                        observations[
                            slice_idx,
                            intent_unfulfillment.nonzero()[0],
                            metrics[parameter["name"]],
                        ] -= (
                            parameter["value"]
                            - metric_value[intent_unfulfillment.nonzero()[0]]
                        ) / (
                            parameter["value"]
                        )

                case "reliability":
                    if reliability_pkt_loss:
                        # Using pkt loss
                        # Intent fulfillment
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value
                                < (
                                    ((100 - parameter["value"]) / 100)
                                    * (1 - intent_overfulfillment_rate)
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
                            observations[
                                slice_idx,
                                fulfilled_mask,
                                metrics[parameter["name"]],
                            ] += (
                                (100 - parameter["value"]) / 100
                                - metric_value[fulfilled_mask]
                            ) / (
                                ((100 - parameter["value"]) / 100)
                                * intent_overfulfillment_rate
                            )
                            # Overfulfilled intent
                            observations[
                                slice_idx,
                                overfulfilled_mask,
                                metrics[parameter["name"]],
                            ] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx,
                                intent_unfulfillment.nonzero()[0],
                                metrics[parameter["name"]],
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - ((100 - parameter["value"]) / 100)
                            ) / (
                                parameter["value"] / 100
                            )

                    else:
                        # Using buffer occupancy instead of packet loss
                        # Intent fulfillment
                        buffer_occupancy_over_threshold = 0.2
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value <= buffer_occupancy_over_threshold
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]

                            overfulfilled_mask = overfulfilled_mask.nonzero()[
                                0
                            ]
                            # Fulfilled intent
                            observations[
                                slice_idx,
                                fulfilled_mask,
                                metrics[parameter["name"]],
                            ] += (
                                buffer_occupancy_threshold
                                - metric_value[fulfilled_mask]
                            ) / (
                                buffer_occupancy_threshold
                                - buffer_occupancy_over_threshold
                            )
                            # Overfulfilled intent
                            observations[
                                slice_idx,
                                overfulfilled_mask,
                                metrics[parameter["name"]],
                            ] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx,
                                intent_unfulfillment.nonzero()[0],
                                metrics[parameter["name"]],
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - buffer_occupancy_threshold
                            ) / (
                                1 - buffer_occupancy_threshold
                            )

                case "latency":
                    max_latency_per_ue = (
                        np.ones_like(slice_ues)
                        * last_obs_slice_req[slice]["ues"]["buffer_latency"]
                    )
                    # Intent fulfillment
                    if np.sum(intent_fulfillment) > 0:
                        overfulfilled_mask = intent_fulfillment * (
                            metric_value
                            < (
                                parameter["value"]
                                * (1 - intent_overfulfillment_rate)
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
                        observations[
                            slice_idx,
                            fulfilled_mask,
                            metrics[parameter["name"]],
                        ] += (
                            parameter["value"] - metric_value[fulfilled_mask]
                        ) / (
                            parameter["value"] * intent_overfulfillment_rate
                        )
                        # Overfulfilled intent
                        observations[
                            slice_idx,
                            overfulfilled_mask,
                            metrics[parameter["name"]],
                        ] += 1

                    # Intent unfulfillment
                    if np.sum(intent_unfulfillment) > 0:
                        observations[
                            slice_idx,
                            intent_unfulfillment.nonzero()[0],
                            metrics[parameter["name"]],
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

    return observations


def calculate_slice_ue_obs(
    max_number_ues_slice: int,
    intent_drift: np.ndarray,
    slice_idx: int,
    slice_ues: np.ndarray,
    slice_req: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    metrics = {"throughput": 0, "reliability": 1, "latency": 2}
    if slice_ues.shape[0] > 0:
        metrics_idx = (
            np.array(
                [
                    metrics[parameter["name"]]
                    for parameter in slice_req[f"slice_{slice_idx}"][
                        "parameters"
                    ].values()
                ]
            )
            if slice_req[f"slice_{slice_idx}"] != {}
            else np.array([])
        )
        intent_ue_values = np.ones(max_number_ues_slice)
        intent_ue_values = -2 * np.ones((max_number_ues_slice, len(metrics)))
        intent_slice_values = -2 * np.ones(len(metrics))
        for metric_idx in metrics_idx:
            intent_ue_values[
                0 : slice_ues.shape[0], metric_idx
            ] = intent_drift[slice_idx, 0 : slice_ues.shape[0], metric_idx]
            intent_slice_values[metric_idx] = np.mean(
                intent_drift[slice_idx, 0 : slice_ues.shape[0], metric_idx]
            )
    else:
        intent_ue_values = np.ones(max_number_ues_slice) * -2
        intent_slice_values = np.ones(len(metrics)) * -2

    return (intent_ue_values, intent_slice_values)


def calculate_reward_mask(obs_space: dict, last_formatted_obs: dict) -> dict:
    reward = {}
    for player_idx, agent_obs in enumerate(last_formatted_obs.items()):
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


def calculate_reward_no_mask(
    obs_space: dict,
    last_formatted_obs: dict,
    last_unformatted_obs: deque,
    var_obs_per_slice: int,
) -> dict:
    reward = {}
    for player_idx, agent_obs in enumerate(last_formatted_obs.items()):
        if player_idx == 0:
            elements_idx = last_unformatted_obs[0]["basestation_slice_assoc"][
                0, :
            ].nonzero()[0]
            active_observations = np.zeros(len(last_formatted_obs) - 1)
            slice_priorities = np.zeros(len(last_formatted_obs) - 1)
            for element_idx in elements_idx:
                slice_priorities[element_idx] = last_unformatted_obs[0][
                    "slice_req"
                ][f"slice_{element_idx}"]["priority"]
                metrics = agent_obs[1]["observations"][
                    (element_idx * var_obs_per_slice) : (
                        element_idx * var_obs_per_slice
                    )
                    + 3
                ]
                metrics = metrics[np.logical_not(np.isclose(metrics, -2))]
                metrics = np.min(metrics) if metrics.shape[0] > 0 else 1
                active_observations[element_idx] = metrics
            if np.isclose(np.sum(active_observations < 0), 0):
                reward[agent_obs[0]] = 0  # np.mean(active_observations)
            elif not np.isclose(
                np.sum((slice_priorities * active_observations) < 0), 0
            ):
                negative_obs_idx = (
                    active_observations * slice_priorities < 0
                ).nonzero()[0]
                reward[agent_obs[0]] = (
                    np.mean(active_observations[negative_obs_idx]) - 1
                )
            else:
                negative_obs_idx = (active_observations < 0).nonzero()[0]
                reward[agent_obs[0]] = np.mean(
                    active_observations[negative_obs_idx]
                )
        else:
            reward[agent_obs[0]] = 0
            elements_idx = last_unformatted_obs[0]["slice_ue_assoc"][
                player_idx - 1, :
            ].nonzero()[0]
            if elements_idx.shape[0] > 0:
                buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][
                    elements_idx
                ]
                reward[agent_obs[0]] = -(
                    np.mean(buffer_occ) + np.std(buffer_occ)
                )
    return reward


def scores_to_rbs(
    action: np.ndarray, total_rbs: int, association: np.ndarray
) -> np.ndarray:
    rbs_per_unit = (
        round_int_equal_sum(
            total_rbs * (action + 1) / np.sum(action + 1),
            total_rbs,
        )
        if np.sum(action + 1) != 0
        else round_int_equal_sum(
            (total_rbs / np.sum(association)) * association,
            total_rbs,
        )
    )
    assert np.sum(rbs_per_unit < 0) == 0, "Negative RBs"
    assert (
        np.sum(rbs_per_unit * association).astype(int) == total_rbs
    ), f"Allocated RBs {np.sum(rbs_per_unit * association)} are different from available RBs {total_rbs}\n{action}\n{rbs_per_unit}\n{association}"

    return rbs_per_unit


def distribute_rbs_ues(
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
    float_array: np.ndarray, target_sum: int
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


def round_robin(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    last_unformatted_obs: deque,
    distribute_rbs: bool = True,
    account_buffer: bool = True,
) -> np.ndarray:
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    slice_ues_buffer = slice_ues
    if account_buffer:
        slice_ues_buffer = slice_ues[
            np.logical_not(np.isclose(buffer_occ, np.zeros_like(buffer_occ)))
        ]  # Consider only UEs that have packets available in the buffer
        if slice_ues_buffer.shape[0] == 0:
            slice_ues_buffer = slice_ues
    rbs_per_ue = np.ones_like(slice_ues_buffer, dtype=float) * np.floor(
        rbs_per_slice[slice_idx] / slice_ues_buffer.shape[0]
    )
    remaining_rbs = int(rbs_per_slice[slice_idx] % slice_ues_buffer.shape[0])
    rbs_per_ue[0:remaining_rbs] += 1
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "RR: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"

    if distribute_rbs:
        allocation_rbs = distribute_rbs_ues(
            rbs_per_ue,
            allocation_rbs,
            slice_ues_buffer,
            rbs_per_slice,
            slice_idx,
        )
        assert (
            np.sum(allocation_rbs[0, slice_ues_buffer, :])
            == rbs_per_slice[slice_idx]
        ), "Distribute RBs is different from RR distribution"
        assert np.sum(allocation_rbs) == np.sum(
            rbs_per_slice[0 : slice_idx + 1]
        ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

        return allocation_rbs
    else:
        return rbs_per_ue


def proportional_fairness(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: MARLCommEnv,
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            rbs_per_slice[slice_idx]
            * env.comm_env.bandwidths[0]
            / num_available_rbs[0]
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.ues.pkt_sizes[slice_ues],
    )
    pkt_snt_throughput = np.mean(
        [
            last_unformatted_obs[idx]["pkt_effective_thr"][slice_ues]
            for idx in range(len(last_unformatted_obs))
        ],
        axis=0,
    )
    snt_throughput = pkt_snt_throughput * env.comm_env.ues.pkt_sizes[slice_ues]
    snt_throughput[
        np.isclose(throughput_available, np.zeros_like(throughput_available))
    ] = 1
    weights = np.divide(
        throughput_available,
        snt_throughput,
        where=np.logical_not(
            np.isclose(snt_throughput, np.zeros_like(snt_throughput))
        ),
        out=2 * np.max(throughput_available) * np.ones_like(snt_throughput),
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx] * weights / np.sum(weights),
            rbs_per_slice[slice_idx],
        )
        if np.sum(weights) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
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
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: MARLCommEnv,
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            rbs_per_slice[slice_idx]
            * env.comm_env.bandwidths[0]
            / num_available_rbs[0]
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.ues.pkt_sizes[slice_ues],
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx]
            * throughput_available
            / np.sum(throughput_available),
            rbs_per_slice[slice_idx],
        )
        if np.sum(throughput_available) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
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
