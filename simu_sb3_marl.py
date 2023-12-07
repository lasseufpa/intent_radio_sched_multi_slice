from os import getcwd

import numpy as np
from tqdm import tqdm

from agents.action_mask_model import TorchActionMaskModel
from agents.masked_action_distribution import TorchDiagGaussian
from agents.sb3_ib_sched import IBSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

read_checkpoint = "./ray_results/"
training_flag = False  # False for reading from checkpoint
debug_mode = (
    True  # When true executes in a local mode where GPU cannot be used
)


def env_creator(env_config):
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
    agent = env_config["agent_class"](
        marl_comm_env,
        marl_comm_env.comm_env.max_number_ues,
        marl_comm_env.comm_env.max_number_basestations,
        marl_comm_env.comm_env.num_available_rbs,
    )
    marl_comm_env.comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
    )

    return marl_comm_env


env_config = {
    "seed": 10,
    "agent_class": IBSched,
    "channel_class": QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociation,
    "scenario": "mult_slice",
    "agent": "sb3_ib_sched",
    "root_path": str(getcwd()),
    "number_agents": 2,
}

max_number_ues = 25
max_number_basestations = 1
num_available_rbs = np.array([135])
marl_comm_env = env_creator(env_config)
sb3_agent = IBSched(
    marl_comm_env, max_number_ues, max_number_basestations, num_available_rbs
)
marl_comm_env.comm_env.set_agent_functions(
    sb3_agent.obs_space_format,
    sb3_agent.action_format,
    sb3_agent.calculate_reward,
)

# Training
number_episodes = 1
steps_per_episode = 10000
total_time_steps = number_episodes * steps_per_episode

sb3_agent.train(total_time_steps)
# sb3_agent.agent.load("./agents/models/final_ssr_protect.zip", marl_comm_env)

# Testing
seed = 10
total_test_steps = 10000
obs, _ = marl_comm_env.reset(seed=seed)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = sb3_agent.step(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
