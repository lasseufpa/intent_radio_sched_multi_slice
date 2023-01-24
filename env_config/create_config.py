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
    },
    "slices": {
        "max_number_slices": number_slices,
    },
    "ues": {
        "max_number_ues": number_ues,
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
