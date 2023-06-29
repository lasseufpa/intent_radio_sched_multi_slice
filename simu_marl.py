import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from agents.ib_sched import IBSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

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

register_env("marl_comm_env", lambda config: PettingZooEnv(marl_comm_env))

config = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment("marl_comm_env")
    .framework("torch")
)

algo = config.build()

total_train_steps = 1
for _ in range(total_train_steps):
    result = algo.train()
    print(pretty_print(result))

marl_comm_env.reset(seed=seed)
for agent in marl_comm_env.agent_iter():
    obs, reward, termination, truncation, info = marl_comm_env.last()
    if termination:
        break
    sched_decision = np.array(algo.compute_single_action(obs))
    marl_comm_env.step(sched_decision)
