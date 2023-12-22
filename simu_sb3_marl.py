from os import getcwd

import numpy as np
from tqdm import tqdm

from agents.sb3_ib_sched import IBSchedSB3
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

read_checkpoint = "./ray_results/"
training_flag = True  # False for reading from checkpoint
agent_type = "sac"  # "ppo" or "sac"
env_config = {
    "seed": 10,
    "agent_class": IBSchedSB3,
    "channel_class": QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociation,
    "scenario": "mult_slice",
    "agent": "sb3_ib_sched",
    "root_path": str(getcwd()),
    "number_agents": 6,
    "training_epochs": 10,
    "test_episodes": 5,
}


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
        False,
        agent_type,
    )
    marl_comm_env.comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
    )

    return marl_comm_env, agent


marl_comm_env, sb3_agent = env_creator(env_config)

# Training
number_episodes = marl_comm_env.comm_env.max_number_episodes
steps_per_episode = marl_comm_env.comm_env.max_number_steps
total_time_steps = (
    number_episodes * steps_per_episode * env_config["training_epochs"]
)

if training_flag:
    sb3_agent.train(total_time_steps)

sb3_agent.agent.load(
    "./agents/models/best_sb3_ib_sched/best_model.zip", marl_comm_env
)

# Testing
total_test_steps = env_config["test_episodes"] * steps_per_episode

marl_comm_env.comm_env.max_number_episodes = (
    number_episodes + env_config["test_episodes"]
)
obs, _ = marl_comm_env.reset(
    seed=env_config["seed"], options={"initial_episode": number_episodes}
)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = sb3_agent.step(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
    if terminated:
        obs, _ = marl_comm_env.reset(seed=env_config["seed"])
