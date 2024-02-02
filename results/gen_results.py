import os
import sys
from collections import deque
from typing import Tuple

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np
from cv2 import line
from matplotlib import lines

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
    global_dict = {}
    xlabel = ylabel = ""
    for scenario in scenario_names:
        for episode in episodes:
            for metric in metrics:
                w, h = matfig.figaspect(0.6)
                plt.figure(figsize=(w, h))
                for agent in agent_names:
                    (xlabel, ylabel) = plot_graph(
                        metric,
                        slices,
                        agent,
                        scenario,
                        episode,
                        agent_names,
                        global_dict,
                    )
                plt.grid()
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=14)
                plt.xticks(fontsize=12)
                plt.legend(
                    fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left"
                )
                os.makedirs(
                    (
                        f"./results/{scenario}/ep_{episode}"
                        if len(agent_names) > 1
                        else f"./results/{scenario}/ep_{episode}/{agent_names[0]}/"
                    ),
                    exist_ok=True,
                )
                plt.savefig(
                    (
                        f"./results/{scenario}/ep_{episode}/{metric}.pdf"
                        if len(agent_names) > 1
                        else f"./results/{scenario}/ep_{episode}/{agent_names[0]}/{metric}.pdf"
                    ),
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
    agents: list[str] = [],
    global_dict: dict = {},
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
        "agent_action": data["agent_action"],
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
            case "buffer_latencies" | "buffer_occupancies":
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
            case "basestation_ue_assoc" | "basestation_slice_assoc":
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
                if agent not in ["sb3_ib_sched", "round_robin"]:
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
            case "reward_comparison":
                if len(agents) > 2:
                    raise Exception(
                        "Reward comparison support only two agents"
                    )
                xlabel = "Step (n)"
                ylabel = "Reward (inter-slice agent) (Abs. Difference)"
                if agent == agents[0]:
                    global_dict["agent_1_reward"] = (
                        [
                            data_metrics["reward"][idx]["player_0"]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                        if agent not in ["sb3_ib_sched", "round_robin"]
                        else [
                            data_metrics["reward"][idx]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                    )
                elif agent == agents[1]:
                    global_dict["agent_2_reward"] = (
                        [
                            data_metrics["reward"][idx]["player_0"]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                        if agent not in ["sb3_ib_sched", "round_robin"]
                        else [
                            data_metrics["reward"][idx]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                    )
                    plt.plot(
                        np.abs(
                            np.subtract(
                                global_dict["agent_1_reward"],
                                global_dict["agent_2_reward"],
                            )
                        ),
                        label=f"abs({agents[0]} - {agents[1]})",
                    )
                    break
            case "reward_cumsum":
                if agent not in ["sb3_ib_sched", "round_robin"]:
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
            case "total_network_eff_throughput":
                total_throughput = calc_total_throughput(
                    data_metrics, "pkt_effective_thr", slices
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
            case "ues_spectral_efficiencies":
                avg_spec_eff = np.mean(
                    np.squeeze(data_metrics["spectral_efficiencies"]), axis=2
                )
                min_spec_eff = np.min(
                    np.squeeze(data_metrics["spectral_efficiencies"]), axis=2
                )
                max_spec_eff = np.max(
                    np.squeeze(data_metrics["spectral_efficiencies"]), axis=2
                )

                for ue_idx in np.arange(avg_spec_eff.shape[1]):
                    plt.plot(avg_spec_eff[:, ue_idx], label=f"UE {ue_idx}")
                    plt.fill_between(
                        np.arange(avg_spec_eff.shape[0]),
                        min_spec_eff[:, ue_idx],
                        max_spec_eff[:, ue_idx],
                        alpha=0.3,
                    )
                break
            case "throughput_per_rb":
                if slice not in []:
                    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
                    num = (
                        np.sum(
                            np.mean(
                                np.squeeze(
                                    data_metrics["spectral_efficiencies"]
                                ),
                                axis=2,
                            )
                            * slice_ues,
                            axis=1,
                        )
                        * 100  # 100e6/1e6 (MHz/Mb)
                    )
                    num_min = (
                        np.sum(
                            np.min(
                                np.squeeze(
                                    data_metrics["spectral_efficiencies"]
                                ),
                                axis=2,
                            )
                            * slice_ues,
                            axis=1,
                        )
                        * 100
                    )
                    num_max = (
                        np.sum(
                            np.max(
                                np.squeeze(
                                    data_metrics["spectral_efficiencies"]
                                ),
                                axis=2,
                            )
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
                    spectral_eff_min = np.zeros_like(num)
                    spectral_eff_max = np.zeros_like(num)
                    spectral_eff = np.divide(
                        num,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                        out=spectral_eff,
                    )
                    spectral_eff_min = np.divide(
                        num_min,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                        out=spectral_eff_min,
                    )
                    spectral_eff_max = np.divide(
                        num_max,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                        out=spectral_eff_max,
                    )
                    plt.plot(spectral_eff, label=f"{agent}, slice {slice}")
                    plt.fill_between(
                        np.arange(spectral_eff.shape[0]),
                        spectral_eff_min,
                        spectral_eff_max,
                        alpha=0.3,
                    )
                    xlabel = "Step (n)"
                    ylabel = "Thoughput capacity per RB (Mbps)"
            case "rbs_needed_slice" | "rbs_needed_total":
                slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
                den = np.sum(slice_ues, axis=1)
                avg_num = np.sum(
                    np.mean(
                        np.squeeze(data_metrics["spectral_efficiencies"]),
                        axis=2,
                    )
                    * slice_ues,
                    axis=1,
                )
                min_num = np.sum(
                    np.min(
                        np.squeeze(data_metrics["spectral_efficiencies"]),
                        axis=2,
                    )
                    * slice_ues,
                    axis=1,
                )
                max_num = np.sum(
                    np.max(
                        np.squeeze(data_metrics["spectral_efficiencies"]),
                        axis=2,
                    )
                    * slice_ues,
                    axis=1,
                )
                avg_spectral_eff = np.zeros_like(avg_num)
                min_spectral_eff = np.zeros_like(avg_num)
                max_spectral_eff = np.zeros_like(avg_num)
                avg_spectral_eff = np.divide(
                    avg_num,
                    den,
                    where=np.logical_not(np.isclose(den, np.zeros_like(den))),
                    out=avg_spectral_eff,
                )
                min_spectral_eff = np.divide(
                    min_num,
                    den,
                    where=np.logical_not(np.isclose(den, np.zeros_like(den))),
                    out=min_spectral_eff,
                )
                max_spectral_eff = np.divide(
                    max_num,
                    den,
                    where=np.logical_not(np.isclose(den, np.zeros_like(den))),
                    out=max_spectral_eff,
                )
                requested_thr = np.array(
                    [
                        (
                            data_metrics["slice_req"][step][f"slice_{slice}"][
                                "ues"
                            ]["traffic"]
                            if "ues"
                            in data_metrics["slice_req"][step][
                                f"slice_{slice}"
                            ]
                            else 0
                        )
                        for step in np.arange(
                            data_metrics["slice_req"].shape[0]
                        )
                    ]
                )
                avg_needed_rbs = np.zeros(requested_thr.shape[0])
                min_needed_rbs = np.zeros(requested_thr.shape[0])
                max_needed_rbs = np.zeros(requested_thr.shape[0])
                avg_needed_rbs = np.divide(
                    requested_thr * np.sum(slice_ues, axis=1),
                    ((100 / 135) * avg_spectral_eff),
                    where=avg_spectral_eff > 0,
                    out=avg_needed_rbs,
                )
                min_needed_rbs = np.divide(
                    requested_thr * np.sum(slice_ues, axis=1),
                    ((100 / 135) * max_spectral_eff),
                    where=max_spectral_eff > 0,
                    out=min_needed_rbs,
                )
                max_needed_rbs = np.divide(
                    requested_thr * np.sum(slice_ues, axis=1),
                    ((100 / 135) * min_spectral_eff),
                    where=min_spectral_eff > 0,
                    out=max_needed_rbs,
                )
                max_needed_rbs[max_needed_rbs > 135] = 135
                if slice == 0:
                    global_dict["avg_needed_rbs"] = avg_needed_rbs
                    global_dict["min_needed_rbs"] = min_needed_rbs
                    global_dict["max_needed_rbs"] = max_needed_rbs
                else:
                    global_dict["avg_needed_rbs"] = (
                        global_dict["avg_needed_rbs"] + avg_needed_rbs
                    )
                    global_dict["min_needed_rbs"] = (
                        global_dict["min_needed_rbs"] + min_needed_rbs
                    )
                    global_dict["max_needed_rbs"] = (
                        global_dict["max_needed_rbs"] + max_needed_rbs
                    )
                if metric == "rbs_needed_slice":
                    plt.plot(avg_needed_rbs, label=f"{agent}, slice {slice}")
                    plt.fill_between(
                        np.arange(avg_needed_rbs.shape[0]),
                        min_needed_rbs,
                        max_needed_rbs,
                        alpha=0.3,
                    )
                elif metric == "rbs_needed_total" and slice == slices[-1]:
                    plt.plot(
                        global_dict["avg_needed_rbs"],
                        label=f"avg total",
                        linestyle="--",
                    )
                    plt.plot(
                        global_dict["min_needed_rbs"],
                        linestyle="--",
                        label="min total",
                    )
                    plt.plot(
                        global_dict["max_needed_rbs"],
                        linestyle="--",
                        label="max total",
                    )
                xlabel = "Step (n)"
                ylabel = "# RBs"
            case "distance_fulfill":
                distance = calc_intent_distance(data_metrics)
                plt.plot(distance, label=f"{agent}, total")
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "distance_fulfill_cumsum":
                distance = calc_intent_distance(data_metrics)
                plt.plot(np.cumsum(distance), label=f"{agent}, total")
                distance = calc_intent_distance(data_metrics, priority=True)
                plt.plot(
                    np.cumsum(distance),
                    label=f"{agent}, prioritary",
                    color=plt.gca().lines[-1].get_color(),
                    linestyle="--",
                )
                xlabel = "Step (n)"
                ylabel = "Distance to fulfill"
                break
            case "violations":
                violations, _, _, _ = calc_slice_violations(data_metrics)
                plt.plot(violations, label=f"{agent}, total")
                violations, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                plt.plot(
                    violations,
                    label=f"{agent}, prioritary",
                    color=plt.gca().lines[-1].get_color(),
                    linestyle="--",
                )
                xlabel = "Step (n)"
                ylabel = "# Violations"
                break
            case "violations_cumsum":
                violations, _, _, _ = calc_slice_violations(data_metrics)
                plt.plot(
                    np.cumsum(violations),
                    label=f"{agent}, total",
                )
                violations, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                plt.plot(
                    np.cumsum(violations),
                    label=f"{agent}, prioritary",
                    color=plt.gca().lines[-1].get_color(),
                    linestyle="--",
                )
                xlabel = "Step (n)"
                ylabel = "Cumulative # violations"
                break
            case "violations_per_slice_type":
                _, violations_per_slice_type, _, _ = calc_slice_violations(
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

            case "violations_per_slice_type_metric":
                (
                    _,
                    _,
                    _,
                    violations_slice_metric,
                ) = calc_slice_violations(data_metrics, slice_per_metric=True)
                metric_idxs = {
                    "throughput": 0,
                    "reliability": 1,
                    "latency": 2,
                }
                for metric_idx in metric_idxs.keys():
                    metric_slice_values = np.zeros(
                        len(violations_slice_metric)
                    )
                    for idx, slice_name in enumerate(
                        violations_slice_metric.keys()
                    ):
                        if metric_idx in violations_slice_metric[slice_name]:
                            metric_slice_values[idx] = violations_slice_metric[
                                slice_name
                            ][metric_idx]

                    plt.bar(
                        np.arange(
                            metric_idxs[metric_idx],
                            metric_slice_values.shape[0] * len(metric_idxs),
                            len(metric_idxs),
                        ),
                        metric_slice_values,
                        tick_label=(
                            list(violations_slice_metric.keys())
                            if metric_idx == "reliability"
                            else None
                        ),
                        label=metric_idx,
                    )
                    plt.xticks(rotation=90)
                    ylabel = "# violations"
                break
            case "intent_slice_metric":
                _, _, intent_slice_metric, _ = calc_slice_violations(
                    data_metrics
                )
                metrics_slice = {
                    "throughput": 0,
                    "reliability": 1,
                    "latency": 2,
                }
                for metric_slice in metrics_slice.keys():
                    plt.scatter(
                        np.arange(intent_slice_metric.shape[0]),
                        intent_slice_metric[
                            :, slice, metrics_slice[metric_slice]
                        ],
                        label=f"{agent}, slice {slice}, {metric}",
                    )
                xlabel = "Step (n)"
                ylabel = "Intent-drift metric"
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
                plt.scatter(
                    np.arange(slice_rbs.shape[0]),
                    slice_rbs,
                    label=f"{agent}, slice {slice}",
                )
                xlabel = "Step (n)"
                ylabel = "# allocated RBs"
            case "sched_decision_comparison":
                if len(agents) > 2:
                    raise Exception(
                        "Sched decision comparison support only two agents"
                    )
                if agent == agents[0]:
                    global_dict["agent_1_slice_rbs"] = np.sum(
                        np.sum(
                            data_metrics["sched_decision"][:, 0, :, :],
                            axis=2,
                        )
                        * data_metrics["slice_ue_assoc"][:, slice, :],
                        axis=1,
                    )
                elif agent == agents[1]:
                    global_dict["agent_2_slice_rbs"] = np.sum(
                        np.sum(
                            data_metrics["sched_decision"][:, 0, :, :],
                            axis=2,
                        )
                        * data_metrics["slice_ue_assoc"][:, slice, :],
                        axis=1,
                    )
                    plt.scatter(
                        np.arange(global_dict["agent_2_slice_rbs"].shape[0]),
                        np.abs(
                            global_dict["agent_1_slice_rbs"]
                            - global_dict["agent_2_slice_rbs"]
                        ),
                        label=f"abs({agents[0]} - {agents[1]}), slice {slice}",
                    )
                xlabel = "Step (n)"
                ylabel = "# allocated RBs (Abs. Difference)"
            case "agent_action":
                actions = (
                    data_metrics["agent_action"]["agent_0"][:, slice]
                    if agent not in ["sb3_ib_sched"]
                    else data_metrics["agent_action"][:, slice]
                )
                plt.scatter(
                    np.arange(actions.shape[0]),
                    actions,
                    label=f"{agent}, slice {slice}",
                )
                xlabel = "Step (n)"
                ylabel = "action factor"
            case (
                "observation_intent"
                | "observation_priority"
                | "observation_slice_traffic"
                | "observation_spectral_eff"
                | "observation_buffer_occ"
                | "observation_buffer_lat"
            ):
                number_slices = data_metrics["slice_ue_assoc"].shape[1]
                metrics_per_slice = int(
                    data_metrics["obs"].shape[1] / number_slices
                )
                slice_obs = data_metrics["obs"][
                    :,
                    metrics_per_slice
                    * slice : metrics_per_slice
                    * (slice + 1),
                ]
                metrics_slice = {
                    "throughput": 0,
                    "reliability": 1,
                    "latency": 2,
                    "active_throughput": 3,
                    "active_reliability": 4,
                    "active_latency": 5,
                    "slice_priority": 6,
                    "total_slice_traffic": 7,
                    "slice_ues": 8,
                    "spectral_eff": 9,
                    "slice_buffer_occ": 10,
                    "slice_buffer_lat": 11,
                }
                if metric == "observation_intent":
                    for metric_slice in list(metrics_slice.keys())[0:3]:
                        plt.scatter(
                            np.arange(slice_obs.shape[0]),
                            slice_obs[:, metrics_slice[metric_slice]],
                            label=f"{agent}, slice {slice}, {metric_slice}",
                        )
                    ylabel = "Intent-drift value"
                elif metric == "observation_priority":
                    plt.scatter(
                        np.arange(slice_obs.shape[0]),
                        slice_obs[:, metrics_slice["slice_priority"]],
                        label=f"{agent}, slice {slice}",
                    )
                    ylabel = "Priority"
                elif metric == "observation_slice_traffic":
                    plt.scatter(
                        np.arange(slice_obs.shape[0]),
                        slice_obs[:, metrics_slice["total_slice_traffic"]],
                        label=f"{agent}, slice {slice}",
                    )
                    ylabel = "Total traffic"
                elif metric == "observation_spectral_eff":
                    plt.scatter(
                        np.arange(slice_obs.shape[0]),
                        slice_obs[:, metrics_slice["spectral_eff"]],
                        label=f"{agent}, slice {slice}",
                    )
                    ylabel = "Spectral efficiency (bit/step/Hz)"
                elif metric == "observation_buffer_occ":
                    plt.scatter(
                        np.arange(slice_obs.shape[0]),
                        slice_obs[:, metrics_slice["slice_buffer_occ"]],
                        label=f"{agent}, slice {slice}",
                    )
                    ylabel = "Buffer occupancy"
                elif metric == "observation_buffer_lat":
                    plt.scatter(
                        np.arange(slice_obs.shape[0]),
                        slice_obs[:, metrics_slice["slice_buffer_lat"]],
                        label=f"{agent}, slice {slice}",
                    )
                    ylabel = "Buffer latency"
                xlabel = "Step (n)"
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
    number_slices = data_metrics["slice_ue_assoc"].shape[1]
    number_ues_slice = int(
        data_metrics["slice_ue_assoc"].shape[2] / number_slices
    )
    intent_drift = np.zeros(
        (data_metrics["obs"].shape[0], number_slices, number_ues_slice, 3)
    )
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


def calc_slice_violations(
    data_metrics, priority=False, slice_per_metric=False
) -> Tuple[np.ndarray, dict, np.ndarray, dict]:
    intent_drift = get_intent_drift(data_metrics)
    violations = np.zeros(data_metrics["obs"].shape[0])
    violations_per_slice_type = {}
    number_intent_metrics = 3
    metric_idxs = {
        "throughput": 0,
        "reliability": 1,
        "latency": 2,
    }
    intent_slice_metric = -2 * np.ones(
        (
            data_metrics["obs"].shape[0],
            data_metrics["slice_ue_assoc"][0].shape[0],
            number_intent_metrics,
        )
    )
    violations_slice_metric = {}
    for step_idx in np.arange(data_metrics["obs"].shape[0]):
        for slice_idx in range(
            0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][
                slice_idx
            ].nonzero()[0]
            if (
                data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx]
                == 0
            ):
                continue
            if (
                priority
                and data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"][
                    "priority"
                ]
                == 0
            ):
                continue
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
            intent_slice_metric[step_idx, slice_idx, :] = intent_drift_slice
            intent_drift_slice[intent_drift_slice == -2] = 1

            if slice_per_metric and np.sum(
                intent_drift_slice < 0
            ):  # Accounts slice violation per metric
                for metric_idx in metric_idxs.keys():
                    slice_name = data_metrics["slice_req"][step_idx][
                        f"slice_{slice_idx}"
                    ]["name"]
                    if intent_drift_slice[metric_idxs[metric_idx]] < 0:
                        if slice_name in violations_slice_metric.keys():
                            if (
                                metric_idx
                                in violations_slice_metric[slice_name].keys()
                            ):
                                violations_slice_metric[slice_name][
                                    metric_idx
                                ] += 1
                            else:
                                violations_slice_metric[slice_name][
                                    metric_idx
                                ] = 1
                        else:
                            violations_slice_metric[slice_name] = {}
                            violations_slice_metric[slice_name][metric_idx] = 1

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
    return (
        violations,
        violations_per_slice_type,
        intent_slice_metric,
        violations_slice_metric,
    )


def calc_intent_distance(data_metrics, priority=False) -> np.ndarray:
    intent_drift = get_intent_drift(data_metrics)
    distance_slice = np.zeros(data_metrics["obs"].shape[0])
    for step_idx in np.arange(data_metrics["obs"].shape[0]):
        intent_array = np.array([])
        for slice_idx in range(
            0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            if (
                data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx]
                == 0
            ):
                continue
            if (
                priority
                and data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"][
                    "priority"
                ]
                == 0
            ):
                continue
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
            min_intent = (
                np.min(intent_drift_slice)
                if intent_drift_slice.shape[0] > 0
                else 0
            )
            intent_array = np.append(intent_array, min_intent)
        distance_slice[step_idx] += (
            np.sum(intent_array) if intent_array.shape[0] > 0 else 0
        )
    return distance_slice


scenario_names = ["mult_slice"]
agent_names = [
    # "random",
    "round_robin",
    # "ib_sched",
    # "ib_sched_old",
    # "ib_sched_deepmind",
    # "ib_sched_mask",
    # "ib_sched_mask_deepmind",
    # "ib_sched_lstm",
    # "sched_twc",
    "sb3_ib_sched",
]
episodes = np.arange(200, 201, dtype=int)
slices = np.arange(5)

# One graph per agent
metrics = [
    # "agent_action",
    "sched_decision",
    # "intent_slice_metric",
    "observation_intent",
    "observation_slice_traffic",
    "observation_priority",
    "observation_spectral_eff",
    "observation_buffer_occ",
    "observation_buffer_lat",
    # "basestation_slice_assoc",
    "reward",
    # "total_network_throughput",
    # "total_network_eff_throughput",
    # "total_network_requested_throughput",
    # "violations_per_slice_type",
    # "violations_per_slice_type_metric",
    # "throughput_per_rb",
    # "ues_spectral_efficiencies",
    "rbs_needed_slice",
    "rbs_needed_total",
    "reward_cumsum",
]
for agent in agent_names:
    gen_results(scenario_names, [agent], episodes, metrics, slices)
# One graph for all agents
metrics = [
    "reward",
    "reward_comparison",
    "reward_cumsum",
    # "violations",
    "violations_cumsum",
    # "sched_decision",
    # "basestation_slice_assoc",
    # "distance_fulfill",
    "distance_fulfill_cumsum",
    # "intent_slice_metric",
    # "sched_decision_comparison",
]
gen_results(scenario_names, agent_names, episodes, metrics, slices)
