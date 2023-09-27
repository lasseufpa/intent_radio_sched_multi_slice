from collections import deque

import numpy as np

from sixg_radio_mgmt import Agent, MARLCommEnv


def intent_drift_calc(
    last_unformatted_obs: deque[dict],
    max_number_ues_slice: int,
    intent_overfulfillment_rate: float,
    env: MARLCommEnv,
) -> np.ndarray:
    def get_metric_value(
        metric_name: str,
        last_unformatted_obs: deque,
        slice_idx: int,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        if metric_name == "throughput":
            metric_value = (
                last_unformatted_obs[0]["pkt_effective_thr"][slice_ues]
                * last_unformatted_obs[0]["slice_req"][f"slice_{slice_idx}"][
                    "ues"
                ]["message_size"]
            ) / 1e6  # Mbps
        elif metric_name == "reliability":
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
            intent_fulfillment = (
                parameter["operator"](metric_value, parameter["value"]).astype(
                    int
                )
                if parameter["name"] != "reliability"
                else parameter["operator"](
                    1 - metric_value, (1 - buffer_occupancy_threshold)
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

                case "reliability":  # Using buffer occupancy instead of packet loss
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

                        overfulfilled_mask = overfulfilled_mask.nonzero()[0]
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
                    max_latency_per_ue = env.comm_env.ues.max_buffer_latencies[
                        slice_ues
                    ]
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

        observations[slice_idx, :, :] = observations[slice_idx, :, :] / len(
            last_obs_slice_req[slice]["parameters"]
        )

    return observations


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
) -> dict:
    reward = {}
    for player_idx, agent_obs in enumerate(last_formatted_obs.items()):
        if player_idx == 0:
            elements_idx = last_unformatted_obs[0]["basestation_slice_assoc"][
                0, :
            ].nonzero()[0]
            active_observations = (
                agent_obs[1]["observations"][elements_idx]
                if elements_idx.shape[0] > 0
                else np.array([1])
            )
        else:
            number_ues_slice = np.sum(
                last_unformatted_obs[0]["slice_ue_assoc"][player_idx - 1, :]
            )
            elements_idx = np.arange(number_ues_slice, dtype=int)
            active_observations = (
                agent_obs[1][elements_idx]
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
        allocation_rbs = distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )
        assert (
            np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
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
            distribute_rbs=False,
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
            distribute_rbs=False,
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
