import matplotlib.pyplot as plt
import numpy as np

from associations.mult_slice import MultSliceAssociation
from sixg_radio_mgmt import UEs
from traffics.mult_slice import MultSliceTraffic

max_number_ues = 1000
max_number_slices = 10
max_number_basestations = 1
seed = 10
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
ues = UEs(
    1000,
    np.repeat(100, max_number_ues),
    np.repeat(1024, max_number_ues),
    np.repeat(100, max_number_ues),
)
mult_slice_assoc = MultSliceAssociation(
    ues, max_number_ues, max_number_basestations, max_number_slices, rng
)
mult_slice_traffic = MultSliceTraffic(max_number_ues, rng)

number_steps = 10000
basestation_ue_assoc = np.zeros((max_number_basestations, max_number_ues))
basestation_slice_assoc = np.zeros(
    (max_number_basestations, max_number_slices)
)
slice_ue_assoc = np.zeros((max_number_slices, max_number_ues))
slice_req = {}

# Hist
ues_per_slice = np.empty((10, 0))
slices_lifetime = np.empty((10, 0))
ues_basestation = np.array([])
traffic_slice_watch = 5
traffic_hist = np.array([])
traffic_type_hist = [("Initial", 0)]

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
        0,
    )
    traffic_hist = (
        np.append(
            traffic_hist,
            np.sum(
                mult_slice_traffic.step(slice_ue_assoc, slice_req, step, 0)
                * slice_ue_assoc[traffic_slice_watch, :]
            )
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

    print(f"Step {step}")
    ues_per_slice = np.append(
        ues_per_slice,
        np.reshape(np.sum(slice_ue_assoc, axis=1), (10, 1)),
        axis=1,
    )
    slices_lifetime = np.append(
        slices_lifetime,
        np.reshape(mult_slice_assoc.slices_lifetime, (10, 1)),
        axis=1,
    )
    ues_basestation = np.append(ues_basestation, np.sum(basestation_ue_assoc))

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
            ue_sample = (slice_ue_assoc[int(slice[6]), :] == 1).nonzero()[0][0]
            if (
                slice_req[slice]["ues"]["buffer_latency"]
                != ues.buffers[ue_sample].max_packets_age
            ):
                raise Exception(
                    "Slice characteristics are different than the ones implemented on UEs"
                )

# Total number of UEs in the system
plt.figure()
plt.plot(np.arange(len(ues_basestation)), ues_basestation)
plt.title("UEs connected to the base station")
plt.ylabel("Number of UEs")
plt.xlabel("Simulation step (n)")
plt.grid()
plt.show()

# Number of UEs per slice
plt.figure()
for idx in np.arange(3):
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
plt.show()

# Slice life-time
plt.figure()
for idx in np.arange(5):
    plt.plot(
        np.arange(slices_lifetime.shape[1]),
        slices_lifetime[idx].T,
        label=f"Slice {idx+1}",
    )
plt.title("Slice lifetime (remaining steps)")
plt.grid()
plt.legend()
plt.ylabel("Remaining steps")
plt.xlabel("Simulation step (n)")
plt.grid()
plt.show()

# Number of slices
plt.figure()
plt.plot(
    np.arange(slices_lifetime.shape[1]), np.sum(slices_lifetime > 0, axis=0)
)
plt.title("Slices in the system")
plt.grid()
plt.ylabel("Number of slices")
plt.xlabel("Simulation step (n)")
plt.grid()
plt.show()

# Slice traffic for specific slice defined by variable traffic_slice_watch
plt.figure()
plt.plot(np.arange(traffic_hist.shape[0]), traffic_hist / 1e9)
plt.title(f"Slice {traffic_slice_watch} traffic")
plt.grid()
plt.ylabel("Throughput (Mbps)")
plt.xlabel("Simulation step (n)")
plt.grid()
plt.show()

print(f"\nSlice {traffic_slice_watch} types: {traffic_type_hist}")
