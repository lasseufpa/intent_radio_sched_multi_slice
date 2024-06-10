import matplotlib.pyplot as plt
import numpy as np

from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_seq import MultSliceAssociationSeq
from channels.quadriga import QuadrigaChannel
from channels.quadriga_seq import QuadrigaChannelSeq
from sixg_radio_mgmt import UEs

max_number_ues = 25
max_number_basestations = 1
max_number_slices = 5
num_available_rbs = np.array([135])
steps_per_episode = 1000
scenarios = {
    "mult_slice": {
        "channel": QuadrigaChannel,
        "association": MultSliceAssociation,
        "num_associations": 200,
        "num_episodes": 1,
    },
    "mult_slice_seq": {
        "channel": QuadrigaChannelSeq,
        "association": MultSliceAssociationSeq,
        "num_associations": 10,
        "num_episodes": 100,
    },
}
ues = UEs(
    max_number_ues,
    np.repeat(100, max_number_ues),
    np.repeat(1024, max_number_ues),
    np.repeat(100, max_number_ues),
)
for scenario_name, scenario in scenarios.items():
    channel_gen = scenario["channel"](
        max_number_ues,
        max_number_basestations,
        num_available_rbs,
        root_path=".",
    )
    association_gen = scenario["association"](
        ues,
        max_number_ues,
        max_number_basestations,
        max_number_slices,
        root_path=".",
    )
    episode = 0
    tmp_slice_req = {}
    for association_number in np.arange(scenario["num_associations"]):
        basestation_ue_assoc = np.zeros(
            (max_number_basestations, max_number_ues)
        )
        basestation_slice_assoc = np.zeros(
            (max_number_basestations, max_number_slices)
        )
        slice_ue_assoc = np.zeros((max_number_slices, max_number_ues))
        slice_req = {}
        (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        ) = association_gen.step(
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
            0,
            episode,
        )
        assert (
            slice_req != tmp_slice_req
        ), f"Slice request is the same on association {association_number} in scenario {scenario_name}"
        tmp_slice_req = slice_req.copy()
        initial_slice_ue_assoc = slice_ue_assoc.copy()
        for episode_number in np.arange(scenario["num_episodes"]):
            spectral_eff = channel_gen.step(0, episode, np.array([1]))
            (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            ) = association_gen.step(
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
                0,
                episode,
            )
            assoc_spectral_eff = np.logical_not(
                np.isclose(np.sum(spectral_eff, 2), 0)
            )
            assert np.array_equal(
                assoc_spectral_eff, basestation_ue_assoc
            ), f"Channels and associations are different on episode {episode_number} and association {association_number} in scenario {scenario_name}"
            assert np.array_equal(
                initial_slice_ue_assoc, slice_ue_assoc
            ), f"Slice association changed on episode {episode_number} and association {association_number} in scenario {scenario_name}"
            episode += 1
print("All scenarios passed!")
