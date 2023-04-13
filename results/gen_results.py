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
            case "pkt_incoming":
                message_sizes = np.array(
                    [
                        (
                            data_metrics["slice_req"][step][f"slice_{slice}"][
                                "ues"
                            ]["message_size"]
                            if data_metrics["slice_req"][step][
                                f"slice_{slice}"
                            ]
                            != {}
                            else 0
                        )
                        for step in np.arange(
                            data_metrics["pkt_incoming"].shape[0]
                        )
                    ]
                )
                slice_throughput = (
                    np.sum(
                        (
                            data_metrics["pkt_incoming"]
                            * data_metrics["slice_ue_assoc"][:, slice, :]
                        ),
                        axis=1,
                    )
                    * message_sizes
                    / (
                        1e6
                        * np.sum(
                            data_metrics["slice_ue_assoc"][:, slice, :], axis=1
                        )
                    )
                )
                plt.plot(slice_throughput, label=f"Slice {slice}")
                xlabel = "Time (s)"
                ylabel = "Throughput (Mbps)"
            case _:
                raise Exception("Metric not found")

    return (xlabel, ylabel)


scenario_name = "mult_slice"
metrics = ["pkt_incoming"]
episodes = np.array([0])
slices = np.array([1, 2])

gen_results(scenario_name, episodes, metrics, slices)
