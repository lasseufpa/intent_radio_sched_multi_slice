import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np

episode = 0
slice_idxs = np.arange(0, 10)
scenario = "mult_slice"
agent_names = [
    "round_robin",
    # "random",
    # "ib_sched_intra_nn",
    # "ib_sched",
    # "ib_sched_no_mask",
    "ib_sched_intra_rr",
    # "ib_sched_inter_rr",
]
metrics = ["throughput", "reliability", "latency"]


def violations_per_slice():
    for agent in agent_names:
        file_violations = np.load(
            f"hist/{scenario}/{agent}/violations_ep_{episode}.npz"
        )
        violations = file_violations["violations"]
        for slice_idx in slice_idxs:
            w, h = matfig.figaspect(0.6)
            plt.figure(figsize=(w, h))
            for idx_metric, metric in enumerate(metrics):
                number_violations = np.sum(
                    violations[:, slice_idx, :, idx_metric] < 0, axis=1
                )
                plt.plot(
                    number_violations,
                    label=f"{metric}",
                )
            plt.grid()
            plt.xlabel("Step (n)", fontsize=14)
            plt.ylabel("# Violations", fontsize=14)
            plt.xticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.savefig(
                f"./results/{scenario}/ep_{episode}/{agent_names[0]}/violation_analysis_slice_{slice_idx}.pdf",
                bbox_inches="tight",
                pad_inches=0,
                format="pdf",
                dpi=1000,
            )
            plt.close()


def plot_bar_violations():
    barWidth = 0.25
    fig, axs = plt.subplots(len(agent_names))
    number_violations = []
    x_axis = []
    for agent_idx, agent in enumerate(agent_names):
        if agent == "round_robin":
            continue
        file_violations = np.load(
            f"hist/{scenario}/{agent}/violations_ep_{episode}.npz"
        )
        violations = file_violations["violations"]
        for idx_metric, metric in enumerate(metrics):
            number_violations = np.sum(
                np.sum(violations[:, :, :, idx_metric] < 0, axis=2), axis=0
            )
            x_axis = (
                np.arange(len(number_violations))
                if idx_metric == 0
                else [x + barWidth for x in x_axis]
            )
            axs[agent_idx].bar(
                x_axis,
                number_violations,
                label=f"{metric}",
                width=barWidth,
            )
        axs[agent_idx].set_title(f"Agent {agent}")
        axs[agent_idx].set_ylabel("# Violations")
        axs[agent_idx].set_xlabel("Slices")
    plt.xticks(
        [r + barWidth for r in range(len(number_violations))],
        np.arange(1, len(number_violations) + 1),
    )
    plt.legend()
    plt.show()
    # plt.savefig(
    #     f"./results/{scenario}/ep_{episode}/{agent_names[0]}/violation_analysis_slice_{slice_idx}.pdf",
    #     bbox_inches="tight",
    #     pad_inches=0,
    #     format="pdf",
    #     dpi=1000,
    # )
    # plt.close()


plot_bar_violations()
