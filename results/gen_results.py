import os
from typing import Tuple

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np


def gen_results(
    scenario_name: str, episodes: np.ndarray, metrics: list, slices: np.ndarray
):
    for episode in episodes:
        data = np.load(
            f"hist/{scenario_name}/ep_{episode}.npz", allow_pickle=True
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
        }

        for metric in metrics:
            plt.figure()
            w, h = matfig.figaspect(0.6)
            plt.figure(figsize=(w, h))
            (xlabel, ylabel) = plot_graph(data_metrics, metric, slices)
            plt.grid()
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.xticks(fontsize=12)
            plt.legend(fontsize=12)
            os.makedirs(
                f"./results/{scenario_name}/ep_{episode}", exist_ok=True
            )
            plt.savefig(
                "./results/{}/ep_{}/{}.pdf".format(
                    scenario_name, episode, metric
                ),
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
                dpi=1000,
            )
            plt.close()


def plot_graph(
    data_metrics: dict, metric: str, slices: np.ndarray
) -> Tuple[str, str]:
    xlabel = ylabel = ""
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
                plt.plot(slice_throughput, label=f"Slice {slice}")
                xlabel = "Step (n)"
                ylabel = "Throughput (Mbps)"
            case ("buffer_latencies" | "buffer_occupancies"):
                avg_spectral_efficiency = calc_slice_average(
                    data_metrics, metric, slice
                )
                plt.plot(avg_spectral_efficiency, label=f"Slice {slice}")
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
                plt.plot(number_elements)
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
                plt.plot(number_uers_per_slice, label=f"Slice {slice}")
                xlabel = "Step (n)"
                ylabel = "Number of UEs"
            case "reward":
                plt.plot(data_metrics[metric])
                xlabel = "Step (n)"
                ylabel = "Reward"
                break
            case "total_network_throughput":
                if slice == slices[0]:
                    total_throughput = np.zeros(
                        data_metrics["pkt_throughputs"].shape[0], dtype=float
                    )
                total_throughput += calc_throughput_slice(  # type: ignore
                    data_metrics, "pkt_throughputs", slice
                )

                if slice == slices[-1]:
                    plt.plot(total_throughput)
                    xlabel = "Step (n)"
                    ylabel = "Throughput (Mbps)"
            case "spectral_efficiencies":
                if slice not in [11]:
                    slice_ues = data_metrics["slice_ue_assoc"][:, slice, :]
                    num = (
                        np.sum(
                            np.sum(np.squeeze(data_metrics[metric]), axis=2)
                            * slice_ues,
                            axis=1,
                        )
                        * 100
                        / 135
                    )
                    den = np.sum(slice_ues, axis=1)
                    spectral_eff = np.divide(
                        num,
                        den,
                        where=np.logical_not(
                            np.isclose(den, np.zeros_like(den))
                        ),
                    )
                    plt.plot(spectral_eff, label=f"Slice {slice}")
                    xlabel = "Step (n)"
                    ylabel = "Thoughput capacity per RB (Mbps)"
            case _:
                raise Exception("Metric not found")

    return (xlabel, ylabel)


def calc_throughput_slice(
    data_metrics: dict, metric: str, slice: int
) -> np.ndarray:
    message_sizes = np.array(
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


scenario_name = "mult_slice"
metrics = [
    "pkt_incoming",
    "pkt_effective_thr",
    "pkt_throughputs",
    "dropped_pkts",
    "buffer_occupancies",
    "buffer_latencies",
    "basestation_ue_assoc",
    "basestation_slice_assoc",
    "slice_ue_assoc",
    "reward",
    "total_network_throughput",
    "spectral_efficiencies",
]
# metrics = ["spectral_efficiencies"]
episodes = np.array([0], dtype=int)
slices = np.arange(10)

gen_results(scenario_name, episodes, metrics, slices)
