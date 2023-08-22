from os import getcwd

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.util import inspect_serializability
from tqdm import tqdm

from agents.ib_sched_no_mask import IBSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

read_checkpoint = "./ray_results/"
training_flag = True  # False for reading from checkpoint
debug_mode = (
    True  # When true executes in a local mode where GPU cannot be used
)


ray.init(local_mode=debug_mode)


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
    "agent": "ib_sched_no_mask",
    "root_path": str(getcwd()),
    "number_agents": 11,
}

# Training
if training_flag:
    algo_config = (
        PPOConfig()
        .environment(
            env="marl_comm_env",
            env_config=env_config,
            is_atari=False,
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "inter_slice_sched": PolicySpec(),
                "intra_slice_sched": PolicySpec(),
            },
            policy_mapping_fn=policy_mapping_fn,
            count_steps_by="env_steps",
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
            enable_connectors=False,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=1, num_gpus_per_worker=1, num_gpus_per_learner_worker=1
        )
        .training(
            _enable_learner_api=False
        )  # TODO Remove after migrating from ModelV2 to RL Module
        .rl_module(_enable_rl_module_api=False)
    )
    stop = {
        "episodes_total": 10,
    }
    results = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=air.RunConfig(
            storage_path=f"./ray_results/{env_config['agent']}/",
            stop=stop,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
        ),
    ).fit()

# Testing
analysis = tune.ExperimentAnalysis(read_checkpoint + env_config["agent"] + "/")
assert analysis.trials is not None, "Analysis trial is None"
best_checkpoint = analysis.get_best_checkpoint(
    analysis.trials[0], "episode_reward_mean", "max"
)
assert best_checkpoint is not None, "Best checkpoint is None"
algo = Algorithm.from_checkpoint(best_checkpoint)
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
            agent_obs,
            policy_id=policy_id,
            explore=False,
        )
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)

ray.shutdown()