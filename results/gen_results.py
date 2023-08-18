import os
from typing import Tuple

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np


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
                plt.figure()
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
                reward = [
                    data_metrics["reward"][idx]["player_0"]
                    for idx in range(data_metrics["reward"].shape[0])
                ]
                plt.plot(reward, label=f"{agent}")
                xlabel = "Step (n)"
                ylabel = "Reward (inter-slice agent)"
                break
            case "reward_cumsum":
                plt.plot(
                    np.cumsum(
                        [
                            data_metrics["reward"][idx]["player_0"]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                    ),
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
            case "violations":
                violations = calc_slice_violations(data_metrics)
                plt.plot(violations, label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "violations_cumsum":
                violations = calc_slice_violations(data_metrics)
                plt.plot(
                    np.cumsum(violations),
                    label=f"{agent}, total",
                )
                xlabel = "Step (n)"
                ylabel = "Cumulative # violations"
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
                plt.xlim([5000, 5200])
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


def calc_slice_violations(data_metrics) -> np.ndarray:
    violations = (
        np.array(
            [
                np.sum(step_obs["player_0"]["observations"][0:10] < 0).astype(
                    int
                )
                for step_obs in data_metrics["obs"]
            ]
        )
        if "observations" in data_metrics["obs"][0]["player_0"]
        else np.array(
            [
                np.sum(step_obs["player_0"][0:10] < 0).astype(int)
                for step_obs in data_metrics["obs"]
            ]
        )
    )

    return violations


scenario_names = ["mult_slice"]
# agent_names = ["ib_sched", "round_robin", "random", "ib_sched_no_mask", "ib_sched_intra_nn"]
agent_names = [
    "round_robin",
    "random",
    "ib_sched_intra_nn",
    "ib_sched",
    "ib_sched_no_mask",
    "ib_sched_intra_rr",
]
episodes = np.array([0], dtype=int)
slices = np.arange(10)

# metrics = [
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
#     "total_network_throughput",
#     "total_network_requested_throughput",
#     "spectral_efficiencies",
#     "sched_decision",
# ]

# One graph per agent
# metrics = [
#     "sched_decision",
#     "basestation_slice_assoc",
#     "reward",
# ]
# for agent in agent_names:
# gen_results(scenario_names, [agent], episodes, metrics, slices)

# One graph for all agents
metrics = [
    # "reward_cumsum",
    # "violations",
    "violations_cumsum",
]
gen_results(scenario_names, agent_names, episodes, metrics, slices)
