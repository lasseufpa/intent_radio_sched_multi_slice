import numpy as np
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
    seed = 10
    marl_comm_env = MARLCommEnv(
        QuadrigaChannel,
        MultSliceTraffic,
        SimpleMobility,
        MultSliceAssociation,
        "mult_slice",
        "ib_sched",
        seed,
        obs_space=IBSched.get_obs_space,
        action_space=IBSched.get_action_space,
        number_agents=11,
    )
    marl_test_agent = IBSched(
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


config = {
    "multiagent": {
        "policies": {
            "inter_slice_sched",
            "intra_slice_sched",
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
}
algo_config = (
    PPOConfig()
    .environment("marl_comm_env")
    .multi_agent(
        policies=config["multiagent"]["policies"],
        policy_mapping_fn=config["multiagent"]["policy_mapping_fn"],
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0, enable_connectors=False)
)
algo = algo_config.build()

# Training
total_train_steps = 1
for _ in range(total_train_steps):
    result = algo.train()
    print(pretty_print(result))

# Testing
marl_comm_env = env_creator({})
seed = 10
total_test_steps = 10000
obs, _ = marl_comm_env.reset(seed=seed)
for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
    action = {}
    assert isinstance(obs, dict), "Observation must be a dict"
    for agent_id, agent_obs in obs.items():
        policy_id = config["multiagent"]["policy_mapping_fn"](agent_id)
        action[agent_id] = algo.compute_single_action(
            agent_obs, policy_id=policy_id
        )
    obs, reward, terminated, truncated, info = marl_comm_env.step(action)
