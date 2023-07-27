from os import getcwd

import numpy as np
from pettingzoo.test import api_test, seed_test
from tqdm import tqdm

from agents.marr import MARR
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

env_config = {
    "seed": 10,
    "agent_class": MARR,
    "channel_class": QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociation,
    "scenario": "mult_slice",
    "agent": "round_robin",
    "root_path": str(getcwd()),
    "number_agents": 11,
}

marl_comm_env = MARLCommEnv(
    env_config["channel_class"],
    env_config["traffic_class"],
    env_config["mobility_class"],
    env_config["association_class"],
    env_config["scenario"],
    env_config["agent"],
    env_config["seed"],
    obs_space=env_config["agent_class"].get_obs_space,
    action_space=env_config["agent_class"].get_action_space,
    root_path=env_config["root_path"],
    number_agents=env_config["number_agents"],
)
marr_agent = env_config["agent_class"](
    marl_comm_env,
    marl_comm_env.comm_env.max_number_ues,
    marl_comm_env.comm_env.max_number_basestations,
    marl_comm_env.comm_env.num_available_rbs,
)
marl_comm_env.comm_env.set_agent_functions(
    marr_agent.obs_space_format,
    marr_agent.action_format,
    marr_agent.calculate_reward,
)


# testing
seed = 10
total_test_steps = 10000
obs, _ = marl_comm_env.reset(seed=seed)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = marr_agent.step(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)
