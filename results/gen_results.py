import os
import sys
from collections import deque
from typing import Tuple

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np

# Import intent_drift_calc function
sys.path.append(os.path.abspath("agents/"))
sys.path.append(os.path.abspath("sixg_radio_mgmt/"))
from common import (  # type: ignore # noqa: E402
    calculate_slice_ue_obs,
    intent_drift_calc,
)

max_number_ues_slice = 5
intent_overfulfillment_rate = 0.2


def gen_results(
    scenario_names: list[str],
    agent_names: list[str],
    episodes: np.ndarray,
    metrics: list,
    slices: np.ndarray,
):
    xlabel = ylabel = ""
    for scenario in scenario_names:
        for episode in episodes:
            for metric in metrics:
                w, h = matfig.figaspect(0.6)
                plt.figure(figsize=(w, h))
                for agent in agent_names:
                    (xlabel, ylabel) = plot_graph(
                        metric, slices, agent, scenario, episode
                    )
                plt.grid()
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=14)
                plt.xticks(fontsize=12)
                plt.legend(fontsize=12)
                os.makedirs(
                    f"./results/{scenario}/ep_{episode}"
                    if len(agent_names) > 1
                    else f"./results/{scenario}/ep_{episode}/{agent_names[0]}/",
                    exist_ok=True,
                )
                plt.savefig(
                    f"./results/{scenario}/ep_{episode}/{metric}.pdf"
                    if len(agent_names) > 1
                    else f"./results/{scenario}/ep_{episode}/{agent_names[0]}/{metric}.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                    format="pdf",
                    dpi=1000,
                )
                plt.close()


def plot_graph(
    metric: str,
    slices: np.ndarray,
    agent: str,
    scenario: str,
    episode: int,
) -> Tuple[str, str]:
    xlabel = ylabel = ""
    data = np.load(
        f"hist/{scenario}/{agent}/ep_{episode}.npz",
        allow_pickle=True,
    )
    data_metrics = {
        "pkt_incoming": data["pkt_incoming"],
        "pkt_throughputs": data["pkt_throughputs"],
        "pkt_effective_thr": data["pkt_effective_thr"],
        "buffer_occupancies": data["buffer_occupancies"],
        "buffer_latencies": data["buffer_latencies"],
        "dropped_pkts": data["dropped_pkts"],
        "mobility": data["mobility"],
        "spectral_efficiencies": data["spectral_efficiencies"],
        "basestation_ue_assoc": data["basestation_ue_assoc"],
        "basestation_slice_assoc": data["basestation_slice_assoc"],
        "slice_ue_assoc": data["slice_ue_assoc"],
        "sched_decision": data["sched_decision"],
        "reward": data["reward"],
        "slice_req": data["slice_req"],
        "obs": data["obs"],
    }
    for slice in slices:
        match metric:
            case (
                "pkt_incoming"
                | "pkt_effective_thr"
                | "pkt_throughputs"
                | "dropped_pkts"
            ):
                slice_throughput = calc_throughput_slice(
                    data_metrics, metric, slice
                )
                plt.plot(slice_throughput, label=f"{agent}, slice {slice}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
            case ("buffer_latencies" | "buffer_occupancies"):
                avg_spectral_efficiency = calc_slice_average(
                    data_metrics, metric, slice
                )
                plt.plot(
                    avg_spectral_efficiency,
                    label=f"{agent}, slice {slice}",
                )
                xlabel = "Step (n)"
                match metric:
                    case "buffer_latencies":
                        ylabel = "Average buffer latency (ms)"
                    case "buffer_occupancies":
                        ylabel = "Buffer occupancy rate"
            case ("basestation_ue_assoc" | "basestation_slice_assoc"):
                number_elements = np.sum(
                    np.sum(data_metrics[metric], axis=2), axis=1
                )
                plt.plot(number_elements, label=f"{agent}")
                xlabel = "Step (n)"
                match metric:
                    case "basestation_ue_assoc":
                        ylabel = "Number of UEs"
                    case "basestation_slice_assoc":
                        ylabel = "Number of slices"
                break
            case "slice_ue_assoc":
                number_uers_per_slice = np.sum(
                    data_metrics[metric][:, slice, :], axis=1
                )
                plt.plot(
                    number_uers_per_slice, label=f"{agent}, slice {slice}"
                )
                xlabel = "Step (n)"
                ylabel = "Number of UEs"
            case "reward":
                if agent != "sb3_ib_sched_intra_rr":
                    reward = [
                        data_metrics["reward"][idx]["player_0"]
                        for idx in range(data_metrics["reward"].shape[0])
                    ]
                else:
                    reward = [
                        data_metrics["reward"][idx]
                        for idx in range(data_metrics["reward"].shape[0])
                    ]
                plt.plot(reward, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Reward (inter-slice agent)"
                break
            case "reward_cumsum":
                if agent != "sb3_ib_sched_intra_rr":
                    reward = [
                        data_metrics["reward"][idx]["player_0"]
                        for idx in range(data_metrics["reward"].shape[0])
                    ]
                else:
                    reward = [
                        data_metrics["reward"][idx]
                        for idx in range(data_metrics["reward"].shape[0])
                    ]
                plt.plot(
                    np.cumsum(reward),
                    label=f"{agent}",
                )
                xlabel = "Step (n)"
                ylabel = "Cumulative reward  (inter-slice agent)"
                break
            case "total_network_throughput":
                total_throughput = calc_total_throughput(
                    data_metrics, "pkt_throughputs", slices
                )
                plt.plot(total_throughput, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
                break
            case "total_network_requested_throughput":
                total_req_throughput = calc_total_throughput(
                    data_metrics, "pkt_incoming", slices
                )
                plt.plot(total_req_throughput, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
                break
            case "spectral_efficiencies":
                if slice not in []:
                    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
                    num = (
                        np.sum(
                            np.mean(np.squeeze(data_metrics[metric]), axis=2)
                            * slice_ues,
                            axis=1,
                        )
                        * 100
                    )
                    den = (
                        np.sum(slice_ues, axis=1)
                        * data_metrics["spectral_efficiencies"].shape[3]
                    )
                    spectral_eff = np.zeros_like(num)
                    spectral_eff = np.divide(
                        num,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                        out=spectral_eff,
                    )
                    plt.plot(spectral_eff, label=f"{agent}, slice {slice}")
                    xlabel = "Step (n)"
                    ylabel = "Thoughput capacity per RB (Mbps)"
            case "distance_fulfill":
                distance = calc_intent_distance(data_metrics)
                plt.plot(distance, label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "distance_fulfill_cumsum":
                distance = calc_intent_distance(data_metrics)
                plt.plot(np.cumsum(distance), label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "distance_fulfill_metrics":
                distance = calc_intent_distance(data_metrics)
                plt.plot(distance, label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "violations":
                violations, _ = calc_slice_violations(data_metrics)
                plt.plot(violations, label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "violations_cumsum":
                violations, _ = calc_slice_violations(data_metrics)
                plt.plot(
                    np.cumsum(violations),
                    label=f"{agent}, total",
                )
                xlabel = "Step (n)"
                ylabel = "Cumulative # violations"
                break
            case "violations_per_slice_type":
                _, violations_per_slice_type = calc_slice_violations(
                    data_metrics
                )
                slice_violations = list(violations_per_slice_type.values())
                slice_names = list(violations_per_slice_type.keys())

                plt.bar(
                    np.arange(len(violations_per_slice_type.keys())),
                    slice_violations,
                    tick_label=slice_names,
                )
                plt.xticks(rotation=65)
                ylabel = "# violations"
                break
            case "sched_decision":
                slice_rbs = np.sum(
                    np.sum(
                        data_metrics["sched_decision"][:, 0, :, :],
                        axis=2,
                    )
                    * data_metrics["slice_ue_assoc"][:, slice, :],
                    axis=1,
                )
                plt.plot(slice_rbs, label=f"{agent}, slice {slice}")
                plt.xlim([0, 500])
                xlabel = "Step (n)"
                ylabel = "# allocated RBs"
            case _:
                raise Exception("Metric not found")

    return (xlabel, ylabel)


def calc_throughput_slice(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    message_sizes = calc_message_sizes(data_metrics, metric, slice)
    den = np.sum(data_metrics["slice_ue_assoc"][:, slice, :], axis=1)
    slice_throughput = np.divide(
        np.sum(
            (
                data_metrics[metric]
                * data_metrics["slice_ue_assoc"][:, slice, :]
            ),
            axis=1,
        )
        * message_sizes,
        (1e6 * den),
        where=np.logical_not(np.isclose(den, np.zeros_like(den))),
    )

    return slice_throughput


def calc_total_throughput(
    data_metrics: dict, metric: str, slices: np.ndarray
) -> np.ndarray:
    total_network_throughput = np.zeros(data_metrics[metric].shape[0])
    for slice in slices:
        message_sizes = calc_message_sizes(data_metrics, metric, slice)
        total_network_throughput += (
            np.sum(
                (
                    data_metrics[metric]
                    * data_metrics["slice_ue_assoc"][:, slice, :]
                ),
                axis=1,
            )
            * message_sizes
            / 1e6
        )

    return total_network_throughput


def calc_message_sizes(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    return np.array(
        [
            (
                data_metrics["slice_req"][step][f"slice_{slice}"]["ues"][
                    "message_size"
                ]
                if data_metrics["slice_req"][step][f"slice_{slice}"] != {}
                else 0
            )
            for step in np.arange(data_metrics[metric].shape[0])
        ]
    )


def calc_slice_average(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
    num_slice_ues = np.sum(slice_ues, axis=1)
    result_values = np.divide(
        np.sum(data_metrics[metric] * slice_ues, axis=1),
        num_slice_ues,
        where=np.logical_not(
            np.isclose(num_slice_ues, np.zeros_like(num_slice_ues))
        ),
    )

    return result_values


def get_intent_drift(data_metrics) -> np.ndarray:
    last_unformatted_obs = deque(maxlen=10)
    intent_drift = np.zeros((data_metrics["obs"].shape[0], 5, 5, 3))
    for step_idx in np.arange(data_metrics["obs"].shape[0]):
        dict_info = {
            "pkt_effective_thr": data_metrics["pkt_effective_thr"][step_idx],
            "slice_req": data_metrics["slice_req"][step_idx],
            "buffer_occupancies": data_metrics["buffer_occupancies"][step_idx],
            "buffer_latencies": data_metrics["buffer_latencies"][step_idx],
            "slice_ue_assoc": data_metrics["slice_ue_assoc"][step_idx],
            "dropped_pkts": data_metrics["dropped_pkts"][step_idx],
        }
        last_unformatted_obs.appendleft(dict_info)
        intent_drift[step_idx, :, :, :] = intent_drift_calc(
            last_unformatted_obs,
            max_number_ues_slice,
            intent_overfulfillment_rate,
            True,
        )

    return intent_drift


def calc_slice_violations(data_metrics) -> Tuple[np.ndarray, dict]:
    intent_drift = get_intent_drift(data_metrics)
    violations = np.zeros(data_metrics["obs"].shape[0])
    violations_per_slice_type = {}
    for step_idx in np.arange(data_metrics["obs"].shape[0]):
        for slice_idx in range(
            0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][
                slice_idx
            ].nonzero()[0]
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                max_number_ues_slice,
                intent_drift[step_idx],
                slice_idx,
                slice_ues,
                data_metrics["slice_req"][step_idx],
            )
            intent_drift_slice[intent_drift_slice == -2] = 1
            intent_drift_slice = np.min(intent_drift_slice)
            slice_violation = int(
                intent_drift_slice < 0
                and not np.isclose(intent_drift_slice, -2)
            )
            violations[step_idx] += slice_violation
            if bool(slice_violation):
                slice_name = data_metrics["slice_req"][step_idx][
                    f"slice_{slice_idx}"
                ]["name"]
                if slice_name in violations_per_slice_type.keys():
                    violations_per_slice_type[slice_name] += 1
                else:
                    violations_per_slice_type[slice_name] = 1
    return violations, violations_per_slice_type


def calc_intent_distance(data_metrics) -> np.ndarray:
    intent_drift = get_intent_drift(data_metrics)
    distance_slice = np.zeros(data_metrics["obs"].shape[0])
    for step_idx in np.arange(data_metrics["obs"].shape[0]):
        for slice_idx in range(
            0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][
                slice_idx
            ].nonzero()[0]
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                max_number_ues_slice,
                intent_drift[step_idx],
                slice_idx,
                slice_ues,
                data_metrics["slice_req"][step_idx],
            )
            intent_drift_slice = np.delete(
                intent_drift_slice,
                np.logical_or(
                    np.isclose(intent_drift_slice, -2), intent_drift_slice >= 0
                ),
            )
            intent_drift_slice = (
                np.min(intent_drift_slice)
                if intent_drift_slice.shape[0] > 0
                else 0
            )
            distance_slice[step_idx] += (
                intent_drift_slice if intent_drift_slice < 0 else 0
            )
    return distance_slice


scenario_names = ["mult_slice"]
# agent_names = ["ib_sched", "round_robin", "random", "ib_sched_no_mask", "ib_sched_intra_nn"]
agent_names = [
    "random",
    "round_robin",
    # "ib_sched_intra_nn",
    # "ib_sched",
    # "ib_sched_no_mask",
    "ib_sched_intra_rr",
    "ib_sched_mask_intra_rr",
    # "sb3_ib_sched_intra_rr",
    # "ib_sched_inter_rr",
]
episodes = np.array([0], dtype=int)
slices = np.arange(5)

metrics = [
    #     "pkt_incoming",
    #     "pkt_effective_thr",
    #     "pkt_throughputs",
    #     "dropped_pkts",
    #     "buffer_occupancies",
    #     "buffer_latencies",
    #     "basestation_ue_assoc",
    #     "basestation_slice_assoc",
    #     "slice_ue_assoc",
    #     "reward",
    #     "reward_cumsum",
    # "total_network_throughput",
    # "total_network_requested_throughput",
    #     "spectral_efficiencies",
    #     "sched_decision",
]

# One graph per agent
metrics = [
    # "sched_decision",
    # "basestation_slice_assoc",
    # "reward",
    # "total_network_throughput",
    # "total_network_requested_throughput",
    # "violations_per_slice_type",
]
for agent in agent_names:
    gen_results(scenario_names, [agent], episodes, metrics, slices)

# One graph for all agents
metrics = [
    "reward",
    "reward_cumsum",
    "violations",
    "violations_cumsum",
    # "sched_decision",
    # "basestation_slice_assoc",
    "distance_fulfill",
    "distance_fulfill_cumsum",
]
gen_results(scenario_names, agent_names, episodes, metrics, slices)
