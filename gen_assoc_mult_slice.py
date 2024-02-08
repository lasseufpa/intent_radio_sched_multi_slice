import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import savemat

from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_fixed import MultSliceAssociationFixed
from sixg_radio_mgmt import UEs
from traffics.mult_slice import MultSliceTraffic

config_file = "mult_slice_fixed"
seed = 10
initial_episode = 0
number_episodes = 1000
generate_quadriga = True
class_association = MultSliceAssociationFixed

with open(f"./env_config/{config_file}.yml") as file:
    data = yaml.safe_load(file)

max_number_ues = data["ues"]["max_number_ues"]  # Total in the system
max_number_slices = data["slices"]["max_number_slices"]
max_number_basestations = data["basestations"]["max_number_basestations"]
scenario_name = data["simulation"]["simu_name"]
association_file_path = f"associations/data/{scenario_name}/"
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
number_steps = data["simulation"]["max_number_steps"]


def generate_quadriga_files(
    episode_number,
    number_steps,
    max_number_ues,
    max_number_slices,
    hist_slice_ue_assoc,
    hist_slice_req,
    basestation_ue_assoc,
):
    print("Generating Quadriga files (UEs velocities)\n\n")
    ues_velocities_per_step = np.zeros((number_steps, max_number_ues))
    for step_number in np.arange(number_steps):
        for slice_number in np.arange(max_number_slices):
            slice = hist_slice_req[step_number][f"slice_{slice_number}"]
            if slice != {}:
                slice_ues_indexes = (
                    hist_slice_ue_assoc[step_number, slice_number, :]
                ).nonzero()[0]
                ues_velocities_per_step[step_number, slice_ues_indexes] = (
                    slice["ues"]["mobility"] / 3.6
                )  # Convert to m/s

    ues_velocities = np.empty((0, max_number_ues))
    speed_change_steps = np.array([])
    for step_number in np.arange(number_steps):
        if step_number == 0:
            reshape_ues_velocities = np.reshape(
                ues_velocities_per_step[step_number],
                (1, ues_velocities.shape[1]),
            )
            ues_velocities = np.append(
                ues_velocities, reshape_ues_velocities, axis=0
            )
        else:
            if not np.array_equal(
                ues_velocities_per_step[step_number],
                ues_velocities_per_step[step_number - 1],
            ):
                reshape_ues_velocities = np.reshape(
                    ues_velocities_per_step[step_number],
                    (1, ues_velocities.shape[1]),
                )
                speed_change_steps = np.append(speed_change_steps, step_number)
                ues_velocities = np.append(
                    ues_velocities,
                    reshape_ues_velocities,
                    axis=0,
                )

    savemat(
        f"{association_file_path}ep_{episode_number}.mat",
        {
            "ues_velocities": ues_velocities,
            "speed_change_steps": speed_change_steps,
            "basestation_ue_assoc": basestation_ue_assoc,
        },
    )


for episode in np.arange(initial_episode, number_episodes):
    ues = UEs(
        max_number_ues,
        np.repeat(100, max_number_ues),
        np.repeat(1024, max_number_ues),
        np.repeat(100, max_number_ues),
    )
    mult_slice_assoc = class_association(
        ues,
        max_number_ues,
        max_number_basestations,
        max_number_slices,
        rng,
        generator_mode=True,
    )
    mult_slice_assoc.generator_mode = True
    mult_slice_traffic = MultSliceTraffic(max_number_ues, rng)

    # Init hist variables
    hist_basestation_ue_assoc = np.empty(
        (number_steps, max_number_basestations, max_number_ues)
    )
    hist_basestation_slice_assoc = np.empty(
        (number_steps, max_number_basestations, max_number_slices)
    )
    hist_slice_ue_assoc = np.empty(
        (number_steps, max_number_slices, max_number_ues)
    )
    hist_slice_req = np.empty(number_steps, dtype=dict)
    hist_slices_to_use = []

    basestation_ue_assoc = np.zeros((max_number_basestations, max_number_ues))
    basestation_slice_assoc = np.zeros(
        (max_number_basestations, max_number_slices)
    )
    slice_ue_assoc = np.zeros((max_number_slices, max_number_ues))
    slice_req = {}

    ues_per_slice = np.empty((max_number_slices, 0))
    hist_slices_lifetime = np.empty((number_steps, max_number_slices))
    ues_basestation = np.array([])
    traffic_slice_watch = 0
    traffic_hist = np.array([])
    traffic_type_hist = [("Initial", 0)]
    hist_total_throughput = np.array([])

    for step in np.arange(number_steps):
        (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        ) = mult_slice_assoc.step(
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
            step,
            episode,
        )

        # Assign hist variables
        hist_basestation_ue_assoc[step] = basestation_ue_assoc
        hist_basestation_slice_assoc[step] = basestation_slice_assoc
        hist_slice_ue_assoc[step] = slice_ue_assoc
        hist_slice_req[step] = slice_req.copy()
        hist_slices_to_use.append(mult_slice_assoc.slices_to_use.copy())
        traffics = mult_slice_traffic.step(slice_ue_assoc, slice_req, step, 0)
        hist_total_throughput = np.append(
            hist_total_throughput, np.sum(traffics)
        )

        traffic_hist = (
            np.append(
                traffic_hist,
                np.sum(traffics * slice_ue_assoc[traffic_slice_watch, :])
                / np.sum(slice_ue_assoc[traffic_slice_watch, :]),
            )
            if np.sum(slice_ue_assoc[traffic_slice_watch, :]) != 0.0
            else np.append(traffic_hist, 0.0)
        )
        slice_watch_type = (
            slice_req[f"slice_{traffic_slice_watch}"]["name"]
            if slice_req[f"slice_{traffic_slice_watch}"] != {}
            else traffic_type_hist[-1][0]
        )
        if slice_watch_type != traffic_type_hist[-1][0]:
            traffic_type_hist.append((slice_watch_type, step))

        print(f"Episode {episode}, Step {step}")
        ues_per_slice = np.append(
            ues_per_slice,
            np.reshape(
                np.sum(slice_ue_assoc, axis=1),
                (max_number_slices, 1),
            ),
            axis=1,
        )
        hist_slices_lifetime[step] = mult_slice_assoc.slices_lifetime
        ues_basestation = np.append(
            ues_basestation, np.sum(basestation_ue_assoc)
        )

        if np.sum(np.sum(slice_ue_assoc, axis=0) > 1) > 0:
            raise Exception("Error: UE associated with more than one slice")

        if np.sum(slice_ue_assoc) != np.sum(basestation_ue_assoc):
            raise Exception(
                "Error: Different number of UEs in slices and the basestation"
            )

        for slice in slice_req:
            if (
                slice_req[slice] == {}
                and basestation_slice_assoc[0, int(slice[6])] != 0
            ) or (
                slice_req[slice] != {}
                and basestation_slice_assoc[0, int(slice[6])] == 0
            ):
                raise Exception(
                    "Error: Slice requirements and slice association doesn't match"
                )

            if slice_req[slice] != {}:
                ue_sample = (slice_ue_assoc[int(slice[6]), :] == 1).nonzero()[
                    0
                ][0]
                if (
                    slice_req[slice]["ues"]["buffer_latency"]
                    != ues.buffers[ue_sample].max_packets_age
                ):
                    raise Exception(
                        "Slice characteristics are different than the ones implemented on UEs"
                    )

    # Create folder and save associations for external file
    if not os.path.exists(association_file_path):
        os.makedirs(association_file_path)
    np.savez_compressed(
        f"{association_file_path}ep_{episode}.npz",
        hist_basestation_ue_assoc=hist_basestation_ue_assoc,
        hist_basestation_slice_assoc=hist_basestation_slice_assoc,
        hist_slice_ue_assoc=hist_slice_ue_assoc,
        hist_slice_req=hist_slice_req,
        hist_slices_lifetime=hist_slices_lifetime,
        hist_slices_to_use=np.array(hist_slices_to_use, dtype=object),
    )

    # Generate files for QuaDRiGa
    if generate_quadriga:
        generate_quadriga_files(
            episode,
            number_steps,
            max_number_ues,
            max_number_slices,
            hist_slice_ue_assoc,
            hist_slice_req,
            hist_basestation_ue_assoc[0, :, :],
        )

    # Create result folder for episode
    resuts_common_path = f"results/{scenario_name}/ep_{episode}/associations/"
    if not os.path.exists(resuts_common_path):
        os.makedirs(resuts_common_path)

    # Total number of UEs in the system
    plt.figure()
    plt.plot(np.arange(len(ues_basestation)), ues_basestation)
    plt.title("UEs connected to the base station")
    plt.ylabel("Number of UEs")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.savefig(
        f"{resuts_common_path}ues_per_basestation.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()

    # Number of UEs per slice
    plt.figure()
    for idx in np.arange(max_number_slices):
        plt.plot(
            np.arange(ues_per_slice.shape[1]),
            ues_per_slice[idx].T,
            label=f"Slice {idx+1}",
        )
    plt.title("Number of UEs per slice")
    plt.ylabel("Number of UEs")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.legend()
    plt.savefig(
        f"{resuts_common_path}ues_per_slice.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()

    # Slice life-time
    plt.figure()
    for idx in np.arange(max_number_slices):
        plt.plot(
            np.arange(hist_slices_lifetime.shape[0]),
            hist_slices_lifetime[:, idx].T,
            label=f"Slice {idx+1}",
        )
    plt.title("Slice lifetime (remaining steps)")
    plt.grid()
    plt.legend()
    plt.ylabel("Remaining steps")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.savefig(
        f"{resuts_common_path}slices_lifetime.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()

    # Number of slices
    plt.figure()
    plt.plot(
        np.arange(hist_slices_lifetime.shape[0]),
        np.sum(hist_slices_lifetime > 0, axis=1),
    )
    plt.title("Slices in the system")
    plt.grid()
    plt.ylabel("Number of slices")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.savefig(
        f"{resuts_common_path}number_slices_per_step.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()

    # Slice traffic for specific slice defined by variable traffic_slice_watch
    plt.figure()
    plt.plot(np.arange(traffic_hist.shape[0]), traffic_hist / 1e6)
    plt.title(f"Slice {traffic_slice_watch} traffic")
    plt.grid()
    plt.ylabel("Throughput (Mbps)")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.savefig(
        f"{resuts_common_path}test_slice_traffic.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()

    # Slice traffic for specific slice defined by variable traffic_slice_watch
    plt.figure()
    plt.plot(
        np.arange(hist_total_throughput.shape[0]), hist_total_throughput / 1e6
    )
    plt.grid()
    plt.ylabel("Throughput (Mbps)")
    plt.xlabel("Simulation step (n)")
    plt.grid()
    plt.savefig(
        f"{resuts_common_path}total_req_throughput.pdf",
        bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    plt.close()
