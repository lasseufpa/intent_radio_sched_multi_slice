import numpy as np
import yaml

number_basestations = 1
number_slices = 10
number_ues = 1000

config = {
    "basestations": {
        "max_number_basestations": number_basestations,
        "bandwidths": np.repeat(100e6, number_basestations).tolist(),
        "carrier_frequencies": np.repeat(28e9, number_basestations).tolist(),
        "num_available_rbs": np.repeat(8, number_basestations).tolist(),
        "basestation_ue_assoc": np.zeros((number_basestations, number_ues)).tolist(),
        "basestation_slice_assoc": np.zeros(
            (number_basestations, number_slices)
        ).tolist(),
    },
    "slices": {
        "max_number_slices": number_slices,
        "slice_ue_assoc": np.zeros((number_slices, number_ues)).tolist(),
        "slice_req": {},
    },
    "ues": {
        "max_number_ues": number_ues,
        "max_buffer_latencies": np.repeat(1e3, number_ues).tolist(),
        "max_buffer_pkts": np.repeat(1024, number_ues).tolist(),
        "pkt_sizes": np.repeat(8192 * 8, number_ues).tolist(),
    },
    "simulation": {
        "simu_name": "mult_bs",
        "max_number_steps": 10,
        "max_number_episodes": 1,
        "hist_root_path": "./",
    },
}

with open("./env_config/mult_slice.yml", "w") as f:
    yaml.dump(config, f, default_flow_style=None)
