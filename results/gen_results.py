import os
import sys
from collections import deque
from typing import Tuple

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                if agent not in ["sb3_sched", "marr"]:
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
                        if agent not in ["sb3_sched", "marr"]
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
                        if agent not in ["sb3_sched", "marr"]
                        else [
                            data_metrics["reward"][idx]
                            for idx in range(data_metrics["reward"].shape[0])
                        ]
                    )
                    plt.plot(
                        np.subtract(
                            global_dict["agent_1_reward"],
                            global_dict["agent_2_reward"],
                        ),
                        label=f"{agents[0]} - {agents[1]}",
                    )
                    break
            case "reward_cumsum":
                if agent not in [
                    "sb3_sched",
                    "marr",
                    "finetune_sb3_sched",
                    "base_sb3_sched",
                ]:
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
                std_spec_eff = np.std(
                    np.squeeze(data_metrics["spectral_efficiencies"]), axis=2
                )

                for ue_idx in np.arange(avg_spec_eff.shape[1]):
                    plt.plot(avg_spec_eff[:, ue_idx], label=f"UE {ue_idx}")
                    plt.fill_between(
                        np.arange(avg_spec_eff.shape[0]),
                        avg_spec_eff[:, ue_idx] - std_spec_eff[:, ue_idx],
                        avg_spec_eff[:, ue_idx] - std_spec_eff[:, ue_idx],
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
                std_spec_eff = np.std(
                    np.squeeze(data_metrics["spectral_efficiencies"]),
                    axis=2,
                )
                min_num = np.sum(
                    (
                        np.mean(
                            np.squeeze(data_metrics["spectral_efficiencies"]),
                            axis=2,
                        )
                        - std_spec_eff
                    )
                    * slice_ues,
                    axis=1,
                )
                max_num = np.sum(
                    (
                        np.mean(
                            np.squeeze(data_metrics["spectral_efficiencies"]),
                            axis=2,
                        )
                        + std_spec_eff
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
                        label="avg total",
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
                    if agent not in ["sb3_sched"]
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


def get_metric_episodes(
    metric, scenario, agent, episodes
) -> Tuple[np.ndarray, np.ndarray]:
    y_values = np.array([])
    y2_values = np.array([])
    for episode in episodes:
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
        match metric:
            case "reward_per_episode" | "reward_per_episode_cumsum":
                reward = (
                    [
                        data_metrics["reward"][idx]["player_0"]
                        for idx in range(data_metrics["reward"].shape[0])
                    ]
                    if "ib_sched" in agent
                    else data_metrics["reward"]
                )
                y_values = np.append(y_values, np.sum(reward))
                y2_values = np.array([])
            case "violations_per_episode" | "violations_per_episode_cumsum":
                violations, _, _, _ = calc_slice_violations(data_metrics)
                violations_pri, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                y_values = np.append(y_values, np.sum(violations))
                y2_values = np.append(y2_values, np.sum(violations_pri))
            case (
                "normalized_violations_per_episode"
                | "normalized_violations_per_episode_cumsum"
            ):
                violations, _, _, _ = calc_slice_violations(data_metrics)
                active_slices_episode = (
                    np.sum(data_metrics["basestation_slice_assoc"][0])
                    * violations.shape[0]
                )
                y_values = np.append(
                    y_values, np.sum(violations) / active_slices_episode
                )
                active_slices_episode_pri = (
                    np.sum(
                        [
                            data_metrics["slice_req"][0][slice]["priority"]
                            for slice in data_metrics["slice_req"][0]
                            if data_metrics["slice_req"][0][slice] != {}
                        ]
                    )
                    * violations.shape[0]
                )
                violations_pri, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                y2_values = np.append(
                    y2_values,
                    (
                        np.sum(violations_pri) / active_slices_episode_pri
                        if active_slices_episode_pri > 0
                        else 0
                    ),
                )
            case "distance_fulfill" | "distance_fulfill_cumsum":
                distance = calc_intent_distance(data_metrics)
                y_values = np.append(y_values, np.sum(distance))
                distance_pri = calc_intent_distance(
                    data_metrics, priority=True
                )
                y2_values = np.append(y2_values, np.sum(distance_pri))
            case (
                "normalized_distance_fulfill"
                | "normalized_distance_fulfill_cumsum"
            ):
                distance = calc_intent_distance(data_metrics)
                active_slices_episode = (
                    np.sum(data_metrics["basestation_slice_assoc"][0])
                    * distance.shape[0]
                )
                y_values = np.append(
                    y_values, np.sum(distance) / active_slices_episode
                )
                active_slices_episode_pri = (
                    np.sum(
                        [
                            data_metrics["slice_req"][0][slice]["priority"]
                            for slice in data_metrics["slice_req"][0]
                            if data_metrics["slice_req"][0][slice] != {}
                        ]
                    )
                    * distance.shape[0]
                )
                distance_pri = calc_intent_distance(
                    data_metrics, priority=True
                )
                y2_values = np.append(
                    y2_values,
                    (
                        np.sum(distance_pri) / active_slices_episode_pri
                        if active_slices_episode_pri > 0
                        else 0
                    ),
                )
    return (y_values, y2_values)


def plot_total_agent(
    metric, x_values, y_values, y2_values, agent
) -> Tuple[str, str]:
    xlabel = "Episode number"
    ylabel = ""
    match metric:
        case "reward_per_episode_cumsum":
            plt.plot(x_values, np.cumsum(y_values), label=f"{agent}")
            ylabel = "Cumulative reward (inter-slice agent)"
        case "reward_per_episode":
            plt.scatter(x_values, y_values, label=f"{agent}")
            ylabel = "Reward (inter-slice agent)"
        case (
            "violations_per_episode_cumsum"
            | "normalized_violations_per_episode_cumsum"
        ):
            plt.plot(x_values, np.cumsum(y_values), label=f"{agent}")
            plt.plot(
                x_values,
                np.cumsum(y2_values),
                label=f"{agent}, prioritary",
                color=plt.gca().lines[-1].get_color(),
                linestyle="--",
            )
            ylabel = "Cumulative # Violations"
        case "violations_per_episode" | "normalized_violations_per_episode":
            plt.scatter(x_values, y_values, label=f"{agent}")
            ylabel = "# Violations"
        case "distance_fulfill" | "normalized_distance_fulfill":
            plt.scatter(x_values, y_values, label=f"{agent}")
            plt.scatter(x_values, y2_values, label=f"{agent}, prioritary")
            ylabel = (
                "Distance to fulfill"
                if metric == "distance_fulfill"
                else "Normalized distance to fulfill"
            )
        case "distance_fulfill_cumsum" | "normalized_distance_fulfill_cumsum":
            plt.plot(x_values, np.cumsum(y_values), label=f"{agent}")
            plt.plot(
                x_values,
                np.cumsum(y2_values),
                label=f"{agent}, prioritary",
                color=plt.gca().lines[-1].get_color(),
                linestyle="--",
            )
            ylabel = (
                "Cumulative distance to fulfill"
                if metric == "distance_fulfill_cumsum"
                else "Cumulative normalized distance to fulfill"
            )

    return (xlabel, ylabel)


def plot_total_episodes(
    metric, scenario, agent, episodes
) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.arange(episodes.shape[0], dtype=int)
    y_values = np.array([])
    y2_values = np.array([])

    y_values, y2_values = get_metric_episodes(
        metric, scenario, agent, episodes
    )

    x_label, y_label = plot_total_agent(
        metric, x_values, y_values, y2_values, agent
    )

    return (x_label, y_label, x_values, y_values, y2_values)


def get_metric_values_scenarios(metric, scenario, agent, num_agent_scenarios):
    x_values = (
        np.arange(20 * num_agent_scenarios.shape[0], dtype=int)
        if scenario == "mult_slice_seq"
        else np.arange(10, dtype=int)
    )
    y_values = np.array([])
    y2_values = np.array([])
    for num_agent_scenario in num_agent_scenarios:
        if scenario == "mult_slice_seq":
            episodes_to_use = np.arange(
                (100 * num_agent_scenario),
                20 + (100 * num_agent_scenario),
                dtype=int,
            )
        elif scenario == "mult_slice":
            episodes_to_use = np.arange(10, dtype=int)
        else:
            raise Exception("Scenario not found")
        tmp_y_values, tmp_y2_values = get_metric_episodes(
            metric,
            scenario,
            f"{agent}_{num_agent_scenario}",
            episodes_to_use,
        )
        y_values = np.append(y_values, tmp_y_values)
        y2_values = np.append(y2_values, tmp_y2_values)

    return (x_values, y_values, y2_values)


def plot_rbs_needed_network_scenarios(
    scenario, agent, slices, number_network_scenarios
):
    scenario_results = {
        f"{scenario}": {} for scenario in number_network_scenarios
    }
    for number_scenario in number_network_scenarios:
        global_dict = {}
        episode = 100 * number_scenario
        data = np.load(
            f"hist/{scenario}/{agent}_{number_scenario}/ep_{episode}.npz",
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
            std_spec_eff = np.std(
                np.squeeze(data_metrics["spectral_efficiencies"]),
                axis=2,
            )
            min_num = np.sum(
                (
                    np.mean(
                        np.squeeze(data_metrics["spectral_efficiencies"]),
                        axis=2,
                    )
                    - std_spec_eff
                )
                * slice_ues,
                axis=1,
            )
            max_num = np.sum(
                (
                    np.mean(
                        np.squeeze(data_metrics["spectral_efficiencies"]),
                        axis=2,
                    )
                    + std_spec_eff
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
                        in data_metrics["slice_req"][step][f"slice_{slice}"]
                        else 0
                    )
                    for step in np.arange(data_metrics["slice_req"].shape[0])
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

        if slice == slices[-1]:
            scenario_results[f"{number_scenario}"] = global_dict
            scenario_results[f"{number_scenario}"][
                "total_avg_needed_rbs"
            ] = np.mean(
                scenario_results[f"{number_scenario}"]["avg_needed_rbs"]
            )
    summary_results = [
        scenario_results[f"{number_scenario}"]["total_avg_needed_rbs"]
        for number_scenario in number_network_scenarios
    ]
    max_scenario = np.argmax(summary_results)
    min_scenario = np.argmin(summary_results)
    median_scenario = np.argsort(summary_results)[len(summary_results) // 2]
    summary_dict = {
        "max_scenario": {
            "scenario_number": max_scenario,
            "values": scenario_results[f"{max_scenario}"],
        },
        "median_scenario": {
            "scenario_number": median_scenario,
            "values": scenario_results[f"{median_scenario}"],
        },
        "min_scenario": {
            "scenario_number": min_scenario,
            "values": scenario_results[f"{min_scenario}"],
        },
    }
    data_plot = pd.DataFrame()
    data_plot["x"] = np.arange(
        summary_dict["min_scenario"]["values"]["max_needed_rbs"].shape[0]
    )
    for scenario_key in summary_dict.keys():
        data_plot[f"{scenario_key}_max"] = summary_dict[scenario_key][
            "values"
        ]["max_needed_rbs"]
        data_plot[f"{scenario_key}_avg"] = summary_dict[scenario_key][
            "values"
        ]["avg_needed_rbs"]
        data_plot[f"{scenario_key}_min"] = summary_dict[scenario_key][
            "values"
        ]["min_needed_rbs"]
        plt.plot(
            summary_dict[scenario_key]["values"]["max_needed_rbs"],
            label=f"Scenario {summary_dict[scenario_key]['scenario_number']}, max",
            linestyle="dashed",
        )
        plt.plot(
            summary_dict[scenario_key]["values"]["avg_needed_rbs"],
            label=f"Scenario {summary_dict[scenario_key]['scenario_number']}, avg",
            linestyle="solid",
            color=plt.gca().lines[-1].get_color(),
        )
        plt.plot(
            summary_dict[scenario_key]["values"]["min_needed_rbs"],
            label=f"Scenario {summary_dict[scenario_key]['scenario_number']}, min",
            color=plt.gca().lines[-1].get_color(),
            linestyle="dotted",
        )
    data_plot.to_csv(
        f"./results/{scenario}/rbs_needed_network_scenarios.csv", index=False
    )


def plot_total_scenarios(
    metric, scenario, agents, num_agent_scenarios, slices, name_postfix=""
):
    # Create dir for results
    os.makedirs(
        (
            f"./results/{scenario}/"
            if len(agent_names) > 1
            else f"./results/{scenario}/"
        ),
        exist_ok=True,
    )
    xlabel = "# of episodes"
    if metric == "normalized_violations_per_episode_cumsum":
        ylabel = "Normalized # of violations (cumulative)"
    elif metric == "normalized_distance_fulfill_cumsum":
        ylabel = "Normalized distance to fulfill (cumulative)"
    elif metric == "rbs_needed_network_scenarios":
        ylabel = "# of RBs"
        xlabel = "Step (n)"
    else:
        raise Exception("Metric not found")
    x_values = np.array([])
    w, h = matfig.figaspect(0.6)
    plt.figure(figsize=(w, h))
    if metric in [
        "normalized_violations_per_episode_cumsum",
        "normalized_distance_fulfill_cumsum",
    ]:
        data_plot = pd.DataFrame()
        for agent in agents:
            x_values, y_values, y2_values = get_metric_values_scenarios(
                metric,
                scenario,
                agent,
                num_agent_scenarios,
            )
            data_plot[agent + "_total"] = np.cumsum(y_values)
            data_plot[agent + "_pri"] = np.cumsum(y2_values)
            plot_total_agent(metric, x_values, y_values, y2_values, agent)
        data_plot["x"] = x_values
        data_plot.to_csv(
            f"./results/{scenario}/{metric}{name_postfix}.csv", index=False
        )
    elif metric == "rbs_needed_network_scenarios":
        plot_rbs_needed_network_scenarios(
            scenario, "marr", slices, num_agent_scenarios
        )
    plt.grid()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if metric == "rbs_needed_network_scenarios":
        plt.xticks(fontsize=12)
    else:
        if scenario == "mult_slice_seq":
            plt.xticks(np.arange(0, x_values.shape[0], 20), fontsize=12)
        elif scenario == "mult_slice":
            plt.xticks(np.arange(0, x_values.shape[0]), fontsize=12)
        else:
            raise Exception("Scenario not found")
    plt.legend(fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left")
    os.makedirs(
        f"./results/{scenario}/",
        exist_ok=True,
    )
    plt.savefig(
        f"./results/{scenario}/{metric}{name_postfix}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()


def gen_results_total(
    scenario_names: list[str],
    agent_names: list[str],
    episodes: np.ndarray,
    metrics: list,
    slices: np.ndarray,
    scenario_number: int = 0,
):
    xlabel = ylabel = ""
    for scenario in scenario_names:
        for metric in metrics:
            data_plot = pd.DataFrame()
            w, h = matfig.figaspect(0.6)
            plt.figure(figsize=(w, h))
            for agent in agent_names:
                (
                    xlabel,
                    ylabel,
                    x_values,
                    y_values,
                    y2_values,
                ) = plot_total_episodes(metric, scenario, agent, episodes)
                data_plot[agent + "_total"] = np.cumsum(y_values)
                data_plot[agent + "_pri"] = np.cumsum(y2_values)
            data_plot["x"] = x_values
            data_plot.to_csv(f"./results/{scenario}/{metric}.csv", index=False)
            plt.grid()
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.xticks(fontsize=12)
            plt.legend(fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left")
            os.makedirs(
                f"./results/{scenario}/",
                exist_ok=True,
            )
            plt.savefig(
                f"./results/{scenario}/{metric}{scenario_number}.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
                dpi=1000,
            )
            plt.close()


def fair_comparison_check(
    agent_names: list[str], episodes: np.ndarray, scenarios: list[str]
):
    # Check if agents are compared in episodes with the same characteristics
    # (e.g., same number of UEs, same traffic, same number of slices etc.).
    base_agent = agent_names[0]
    for scenario in scenarios:
        for agent in agent_names[1:]:
            for episode in episodes:
                data = np.load(
                    f"hist/{scenario}/{agent}/ep_{episode}.npz",
                    allow_pickle=True,
                )
                data_metrics = {
                    "pkt_incoming": data["pkt_incoming"],
                    "mobility": data["mobility"],
                    "spectral_efficiencies": data["spectral_efficiencies"],
                    "basestation_ue_assoc": data["basestation_ue_assoc"],
                    "basestation_slice_assoc": data["basestation_slice_assoc"],
                    "slice_ue_assoc": data["slice_ue_assoc"],
                    "slice_req": data["slice_req"],
                }
                data_base = np.load(
                    f"hist/{scenario}/{base_agent}/ep_{episode}.npz",
                    allow_pickle=True,
                )
                data_metrics_base = {
                    "pkt_incoming": data_base["pkt_incoming"],
                    "mobility": data_base["mobility"],
                    "spectral_efficiencies": data_base[
                        "spectral_efficiencies"
                    ],
                    "basestation_ue_assoc": data_base["basestation_ue_assoc"],
                    "basestation_slice_assoc": data_base[
                        "basestation_slice_assoc"
                    ],
                    "slice_ue_assoc": data_base["slice_ue_assoc"],
                    "slice_req": data_base["slice_req"],
                }

                for metric in data_metrics.keys():
                    if not np.array_equal(
                        data_metrics[metric], data_metrics_base[metric]
                    ):
                        raise Exception(
                            f"Scenario {scenario}: Agents {base_agent} and {agent} are not compared in the same episode {episode} characteristics due to {metric} differences"
                        )

    return True


def get_scenario_metrics(
    episodes: np.ndarray,
    slices: np.ndarray,
    scenario: str,
    number_metrics: int,
    sort_thr: bool = False,
):
    metrics = np.zeros((len(episodes), len(slices), number_metrics))
    for idx, episode in enumerate(episodes):
        data = np.load(
            f"associations/data/{scenario}/ep_{episode}.npz",
            allow_pickle=True,
        )
        data_metrics = {
            "hist_basestation_slice_assoc": data[
                "hist_basestation_slice_assoc"
            ],
            "hist_slice_ue_assoc": data["hist_slice_ue_assoc"],
            "hist_slice_req": data["hist_slice_req"],
        }
        # Array metrics for each slice: reliability, latency, throughput,
        # number_ues, mobility, buffer_size, message_size, max_buffer_lat, traffic
        for slice in slices:
            if data_metrics["hist_basestation_slice_assoc"][0][0][slice] == 1:
                slice_req = data_metrics["hist_slice_req"][0][f"slice_{slice}"]
                reliability, latency, throughput = 0, 0, 0
                for par in slice_req["parameters"].values():
                    if par["name"] == "reliability":
                        reliability = par["value"]
                    elif par["name"] == "latency":
                        latency = par["value"]
                    elif par["name"] == "throughput":
                        throughput = par["value"]
                metrics[idx, slice, :] = np.array(
                    [
                        reliability,
                        latency,
                        throughput,
                        np.sum(data_metrics["hist_slice_ue_assoc"][0][slice]),
                        slice_req["ues"]["mobility"],
                        slice_req["ues"]["buffer_size"],
                        slice_req["ues"]["message_size"],
                        slice_req["ues"]["buffer_latency"],
                        slice_req["ues"]["traffic"],
                    ]
                )
    if sort_thr:  # Sort slices by throughput as done in the observation space
        for episode in np.arange(episodes.shape[0]):
            metrics[episode, :, :] = metrics[
                episode, np.argsort(metrics[episode, :, 2]), :
            ]
    return metrics


def plot_scenario_analysis(
    scenario_names: list[str],
    episodes: np.ndarray,
    slices: np.ndarray,
    sort_thr: bool = False,
):
    metric_names = [
        "req_reliability",
        "req_latency",
        "req_throughput",
        "number_ues",
        "mobility",
        "buffer_size",
        "message_size",
        "max_buffer_lat",
        "traffic",
    ]
    number_metrics = len(metric_names)
    for scenario in scenario_names:
        metrics = get_scenario_metrics(
            episodes, slices, scenario, number_metrics, sort_thr
        )
        w, h = matfig.figaspect(0.6)
        fig, axs = plt.subplots(
            ncols=3, nrows=3, figsize=(w, h), layout="constrained"
        )
        metric_idx = 0
        for row in range(3):
            for col in range(3):
                for slice in slices:
                    y_values = metrics[
                        :,
                        slice,
                        metric_idx,
                    ]
                    y_values = y_values[y_values != 0]
                    axs[row, col].boxplot(  # type: ignore
                        y_values,
                        positions=[slice],
                    )
                    axs[row, col].grid()  # type: ignore
                    axs[row, col].set_xlabel("Slice number")  # type: ignore
                    axs[row, col].set_ylabel(metric_names[metric_idx])  # type: ignore
                metric_idx += 1
        sort_str = "sorted" if sort_thr else "unsorted"
        os.makedirs(
            f"./results/{scenario}/",
            exist_ok=True,
        )
        fig.savefig(
            f"./results/{scenario}/scenario_analysis_{episodes[0]}_{episodes[-1]}_{sort_str}.pdf"
        )
        plt.close(fig)


def scenario_diff_train_test(
    scenario: str,
    train_episodes: np.ndarray,
    test_episodes: np.ndarray,
    slices: np.ndarray,
):
    metric_names = [
        "req_reliability",
        "req_latency",
        "req_throughput",
        "number_ues",
        "mobility",
        "buffer_size",
        "message_size",
        "max_buffer_lat",
        "traffic",
    ]
    number_metrics = len(metric_names)
    train_metrics = get_scenario_metrics(
        train_episodes, slices, scenario, number_metrics, False
    )
    test_metrics = get_scenario_metrics(
        test_episodes, slices, scenario, number_metrics, False
    )
    max_metrics = np.max(
        np.max(np.concatenate((train_metrics, test_metrics)), axis=0), axis=0
    )
    max_metrics = np.repeat([max_metrics], len(slices), axis=0)

    diff_test_per_ep = 9999999 * np.ones(test_metrics.shape[0])
    for test_ep in np.arange(test_metrics.shape[0]):
        for train_ep in np.arange(train_metrics.shape[0]):
            diff_mag = np.sum(
                np.abs(
                    (
                        test_metrics[test_ep, :, :]
                        - train_metrics[train_ep, :, :]
                    )
                    / max_metrics
                )
            )
            diff_test_per_ep[test_ep] = min(
                diff_test_per_ep[test_ep], diff_mag
            )
    w, h = matfig.figaspect(0.6)
    plt.figure(figsize=(w, h))
    plt.plot(test_episodes, diff_test_per_ep)
    plt.grid()
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Normalized Min Absolute Error", fontsize=14)
    plt.xticks(fontsize=12)
    os.makedirs(
        f"./results/{scenario}/",
        exist_ok=True,
    )
    plt.savefig(
        f"./results/{scenario}/min_abs_diff_train_{train_episodes[0]}-{train_episodes[-1]}_test_{test_episodes[0]}-{test_episodes[-1]}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()


def print_scenarios(scenario_numbers: np.ndarray):
    for scenario_number in scenario_numbers:
        scenario = np.load(
            f"associations/data/mult_slice/ep_{scenario_number}.npz",
            allow_pickle=True,
        )
        print(f"Association {scenario_number}")
        for idx, slice in enumerate(scenario["hist_slice_req"][0]):
            name = ""
            if "name" in scenario["hist_slice_req"][0][slice].keys():
                name = scenario["hist_slice_req"][0][slice]["name"]
            print(f"Slice {idx}: {name}")
        print("\n")


# scenarios = ["mult_slice_seq", "mult_slice", "finetune_mult_slice_seq"]
scenarios = ["mult_slice_seq"]

for scenario in scenarios:
    if scenario == "mult_slice_seq":
        scenario_numbers = np.arange(10)
        agent_names = [
            "ray_ib_sched_default",
            "sched_twc",
            "sched_coloran",
            "mapf",
            "marr",
        ]

        # Check if agents are compared in episodes with the same characteristics
        for scenario_number in scenario_numbers:
            episodes = np.arange(
                (100 * scenario_number),
                20 + (100 * scenario_number),
                dtype=int,
            )
            if fair_comparison_check(
                [
                    agent_name + f"_{scenario_number}"
                    for agent_name in agent_names
                ],
                episodes,
                [scenario],
            ):
                print(
                    f"Scenario {scenario_number}: Agents are compared in episodes with the same characteristics"
                )

        # One graph for all agents considering all episodes (one graph for all episodes)
        metrics = [
            "normalized_distance_fulfill_cumsum",
            "normalized_violations_per_episode_cumsum",
            "rbs_needed_network_scenarios",
        ]
        slices = np.arange(5)
        for metric in metrics:
            plot_total_scenarios(
                metric, scenario, agent_names, scenario_numbers, slices
            )
            if metric != "rbs_needed_network_scenarios":
                plot_total_scenarios(
                    metric,
                    scenario,
                    agent_names,
                    np.array([6, 4, 2]),
                    slices,
                    "_selected_scenarios",
                )
    elif scenario == "mult_slice":
        agent_names = [
            "ray_ib_sched_default",
            "sched_twc",
            "sched_coloran",
            "mapf",
            "marr",
        ]
        episodes = np.arange(10, dtype=int)
        slices = np.arange(5)
        scenario_numbers = np.array([0])

        # Check if agents are compared in episodes with the same characteristics
        if fair_comparison_check(
            [agent_name + "_0" for agent_name in agent_names],
            episodes,
            [scenario],
        ):
            print(
                "Agents are compared in episodes with the same characteristics"
            )

        # One graph for all agents considering all episodes (one graph for all episodes)
        metrics = [
            "normalized_distance_fulfill_cumsum",
            "normalized_violations_per_episode_cumsum",
        ]
        for metric in metrics:
            plot_total_scenarios(
                metric, scenario, agent_names, scenario_numbers, slices
            )
