from os import getcwd

import numpy as np
from tqdm import tqdm

from agents.sb3_ib_sched import IBSchedSB3
from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_fixed import MultSliceAssociationFixed
from channels.fixed_se import FixedSE
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

training_flag = False  # False for reading from checkpoint
agent_type = "sac"  # "ppo" or "sac"
env_config = {
    "seed": 10,
    "seed_test": 15,
    "agent_class": IBSchedSB3,
    "channel_class": MimicQuadriga,  # QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociationFixed,
    "scenario": "mult_slice_fixed",
    "agent": "finetune_sb3_ib_sched",
    "root_path": str(getcwd()),
    "training_epochs": 1,
    "initial_training_episode": 1000,
    "max_training_episodes": 80,
    "test_episodes": 20,
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
        root_path=env_config["root_path"],
        initial_episode_number=env_config["initial_training_episode"],
    )
    marl_comm_env.comm_env.max_number_episodes = (
        env_config["initial_training_episode"]
        + env_config["max_training_episodes"]
    )
    agent = env_config["agent_class"](
        marl_comm_env,
        marl_comm_env.comm_env.max_number_ues,
        marl_comm_env.comm_env.max_number_slices,
        marl_comm_env.comm_env.max_number_basestations,
        marl_comm_env.comm_env.num_available_rbs,
        None,
        agent_type,
        seed=env_config["seed"],
        agent_name=env_config["agent"],
    )
    marl_comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )
    agent.init_agent()

    return marl_comm_env, agent


marl_comm_env, sb3_agent = env_creator(env_config)

# Training
number_episodes = (
    marl_comm_env.comm_env.max_number_episodes
    - env_config["initial_training_episode"]
)
steps_per_episode = marl_comm_env.comm_env.max_number_steps
total_time_steps = (
    number_episodes * steps_per_episode * env_config["training_epochs"]
)

if training_flag:
    sb3_agent.load(
        f"./agents/models/{env_config['scenario']}/final_base_sb3_ib_sched.zip"
    )
    sb3_agent.train(total_time_steps)

# Testing
sb3_agent.load(
    f"./agents/models/{env_config['scenario']}/final_finetune_sb3_ib_sched.zip"
)
total_test_steps = env_config["test_episodes"] * steps_per_episode

test_initial_episode = (
    env_config["initial_training_episode"]
    + env_config["max_training_episodes"]
)
marl_comm_env.comm_env.max_number_episodes = (
    test_initial_episode + env_config["test_episodes"]
)
obs, _ = marl_comm_env.reset(
    seed=env_config["seed_test"],
    options={"initial_episode": test_initial_episode},
)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = sb3_agent.step(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
    if terminated:
        obs, _ = marl_comm_env.reset()