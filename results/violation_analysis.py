import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np


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
    w, h = matfig.figaspect(0.6)
    fig, axs = plt.subplots(len(agent_names), sharey=True, figsize=(w, h))
    number_violations = []
    x_axis = []
    number_violations = np.zeros(
        (len(agent_names), len(metrics), len(slice_idxs))
    )
    for agent_idx, agent in enumerate(agent_names):
        file_violations = np.load(
            f"hist/{scenario}/{agent}/violations_ep_{episode}.npz"
        )
        violations = file_violations["violations"]
        for idx_metric, metric in enumerate(metrics):
            number_violations[agent_idx, idx_metric, :] = np.sum(
                np.sum(violations[:, :, :, idx_metric] < 0, axis=2), axis=0
            )
            x_axis = (
                np.arange(number_violations.shape[2])
                if idx_metric == 0
                else [x + barWidth for x in x_axis]
            )
            axs[agent_idx].bar(
                x_axis,
                number_violations[agent_idx, idx_metric, :],
                label=f"{metric}",
                width=barWidth,
            )
        axs[agent_idx].set_title(f"Agent {agent}")
        axs[agent_idx].set_ylabel("# Violations")
        axs[agent_idx].set_xticks(
            [r + barWidth for r in range(number_violations.shape[2])],
            np.arange(1, number_violations.shape[2] + 1),
        )
        axs[agent_idx].grid()
        axs[agent_idx].legend()
        if agent_idx == len(agent_names) - 1:
            axs[agent_idx].set_xlabel("Slices")
    plt.savefig(
        f"./results/{scenario}/ep_{episode}/violation_analysis.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close(fig)

    # Plot figure 2 - difference between agents
    plt.figure(figsize=(w, h))
    for idx_metric, metric in enumerate(metrics):
        x_axis = (
            np.arange(number_violations.shape[2])
            if idx_metric == 0
            else [x + barWidth for x in x_axis]
        )
        plt.bar(
            x_axis,
            number_violations[0, idx_metric, :]
            - number_violations[1, idx_metric, :],
            label=f"{metric}",
            width=barWidth,
        )
    plt.title(f"Agent {agent_names[0]} - Agent {agent_names[1]}")
    plt.ylabel("# Violations")
    plt.xlabel("Slices")
    plt.xticks(
        [r + barWidth for r in range(number_violations.shape[2])],
        np.arange(1, number_violations.shape[2] + 1),
    )
    plt.grid()
    plt.legend()
    plt.savefig(
        f"./results/{scenario}/ep_{episode}/violation_diff.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()


def plot_drift_per_metric(agent_name: str, slice_idx: int, metrics: list):
    file_violations = np.load(
        f"hist/{scenario}/{agent_name}/violations_ep_{episode}.npz"
    )
    file_hist = np.load(
        f"hist/{scenario}/{agent_name}/ep_{episode}.npz", allow_pickle=True
    )
    violations = file_violations["violations"]
    slice_ue_assoc = np.sum(
        file_hist["slice_ue_assoc"][:, slice_idx, :], axis=1
    ).astype(int)
    slice_req = file_hist["slice_req"]
    slice_req_count = np.array(
        [
            len(slice_req[step][f"slice_{slice_idx}"]["parameters"])
            if slice_req[step][f"slice_{slice_idx}"] != {}
            else 0
            for step in np.arange(slice_req.shape[0])
        ]
    )
    w, h = matfig.figaspect(0.6)
    plt.figure(figsize=(w, h))
    for idx_metric, metric in enumerate(metrics):
        if metric in ["throughput", "latency"]:
            continue
        drift_metric = np.array(
            [
                np.mean(
                    violations[
                        step, slice_idx, 0 : slice_ue_assoc[step], idx_metric
                    ]
                )
                * slice_req_count[step]
                if slice_ue_assoc[step] != 0
                else -2
                for step in range(violations.shape[0])
            ]
        )
        plt.scatter(
            np.arange(drift_metric.shape[0]),
            drift_metric,
            label=f"{metric}",
        )
    plt.grid()
    plt.xlabel("Step (n)", fontsize=14)
    plt.ylabel("Intent drift", fontsize=14)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(
        f"./results/{scenario}/ep_{episode}/{agent_name}/drift_slice_{slice_idx}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()


episode = 0
slice_idxs = np.arange(0, 5)
scenario = "mult_slice"
agent_names = [
    "round_robin",
    "ib_sched_intra_rr",
]  # Maximum 2 agents
metrics = ["throughput", "reliability", "latency"]

# plot_drift_per_metric(
#     "ib_sched_intra_rr", 0, ["throughput", "latency", "pkt_loss"]
# )
plot_bar_violations()
