from os import getcwd

import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.util import inspect_serializability
from tqdm import tqdm

from agents.ib_sched import IBSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic


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
        config_file_root_path=env_config["config_file_root_path"],
        number_agents=env_config["number_agents"],
    )
    marl_test_agent = env_config["agent_class"](
        marl_comm_env,
        marl_comm_env.comm_env.max_number_ues,
        marl_comm_env.comm_env.max_number_basestations,
        marl_comm_env.comm_env.num_available_rbs,
    )
    marl_comm_env.comm_env.set_agent_functions(
        marl_test_agent.obs_space_format,
        marl_test_agent.action_format,
        marl_test_agent.calculate_reward,
    )

    return marl_comm_env


# Ray RLlib
register_env("marl_comm_env", lambda config: env_creator(config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    agent_idx = int(agent_id.partition("_")[2])

    return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"


env_config = {
    "seed": 10,
    "agent_class": IBSched,
    "channel_class": QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociation,
    "scenario": "mult_slice",
    "agent": "ib_sched",
    "config_file_root_path": str(getcwd()),
    "number_agents": 11,
}

algo_config = (
    PPOConfig()
    .environment(env="marl_comm_env", env_config=env_config)
    .multi_agent(
        policies={"inter_slice_sched", "intra_slice_sched"},
        policy_mapping_fn=policy_mapping_fn,
        count_steps_by="env_steps",
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=0,
        enable_connectors=False,
        batch_mode="complete_episodes",
    )
)

# Training
stop = {
    "training_iteration": 2,
}
results = tune.Tuner(
    "PPO",
    param_space=algo_config.to_dict(),
    run_config=air.RunConfig(
        storage_path="./ray_results/", stop=stop, verbose=2
    ),
).fit()

# Testing
algo = algo_config.build()
marl_comm_env = env_creator(env_config)
seed = 10
total_test_steps = 10000
obs, _ = marl_comm_env.reset(seed=seed)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = {}
    assert isinstance(obs, dict), "Observation must be a dict"
    for agent_id, agent_obs in obs.items():
        policy_id = policy_mapping_fn(agent_id)
        action[agent_id] = algo.compute_single_action(
            agent_obs, policy_id=policy_id
        )
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)
