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

number_steps = 10
basestation_ue_assoc = np.zeros((max_number_basestations, max_number_ues))
basestation_slice_assoc = np.zeros((max_number_basestations, max_number_slices))
slice_ue_assoc = np.zeros((max_number_slices, max_number_ues))
slice_req = {}

for step in np.arange(number_steps):
    (
        basestation_ue_assoc,
        basestation_slice_assoc,
        slice_ue_assoc,
        slice_req,
    ) = mult_slice_assoc.step(
        basestation_ue_assoc, basestation_slice_assoc, slice_ue_assoc, {}, step, 0
    )
