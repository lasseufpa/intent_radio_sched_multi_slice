import matplotlib.pyplot as plt
import numpy as np

from associations.mult_slice import MultSliceAssociation

max_number_ues = 1000
max_number_slices = 10
max_number_basestations = 1
seed = 10
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
mult_slice_assoc = MultSliceAssociation(
    max_number_ues, max_number_basestations, max_number_slices, rng
)

number_steps = 10000
basestation_ue_assoc = np.zeros((max_number_basestations, max_number_ues))
basestation_slice_assoc = np.zeros((max_number_basestations, max_number_slices))
slice_ue_assoc = np.zeros((max_number_slices, max_number_ues))
slice_req = {}

# Hist
ues_per_slice = np.empty((10, 0))
slices_lifetime = np.empty((10, 0))
ues_basestation = np.array([])

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
    print(f"Step {step}")
    ues_per_slice = np.append(
        ues_per_slice, np.reshape(np.sum(slice_ue_assoc, axis=1), (10, 1)), axis=1
    )
    slices_lifetime = np.append(
        slices_lifetime, np.reshape(mult_slice_assoc.slices_lifetime, (10, 1)), axis=1
    )
    ues_basestation = np.append(ues_basestation, np.sum(basestation_ue_assoc))

    if np.sum(np.sum(slice_ue_assoc, axis=0) > 1) > 0:
        raise Exception("Error: UE associated with more than one slice")

    if np.sum(slice_ue_assoc) != np.sum(basestation_ue_assoc):
        raise Exception("Error: Different number of UEs in slices and the basestation")

    for slice in slice_req:
        if (
            slice_req[slice] == {} and basestation_slice_assoc[0, int(slice[6])] != 0
        ) or (
            slice_req[slice] != {} and basestation_slice_assoc[0, int(slice[6])] == 0
        ):
            raise Exception(
                "Error: Slice requirements and slice association doesn't match"
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
        np.arange(ues_per_slice.shape[1]), ues_per_slice[idx].T, label=f"Slice {idx+1}"
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
plt.plot(np.arange(slices_lifetime.shape[1]), np.sum(slices_lifetime > 0, axis=0))
plt.title("Slices in the system")
plt.grid()
plt.ylabel("Number of slices")
plt.xlabel("Simulation step (n)")
plt.grid()
plt.show()
