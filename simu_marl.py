import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from agents.ib_sched import IBSched
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

# Remove after
# from agents.marl_test import MARLTest
# from associations.simple import SimpleAssociation
# from channels.simple import SimpleChannel
# from mobilities.simple import SimpleMobility
# from sixg_radio_mgmt import MARLCommEnv
# from traffics.simple import SimpleTraffic

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

# Ray RLlib
register_env("marl_comm_env", lambda config: PettingZooEnv(marl_comm_env))


# def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#     agent_idx = int(agent_id[-1])

#     return "inte_slice_sched" if agent_idx == 0 else "intra_slice_sched"


# config = {
#     "multiagent": {
#         "policies": {
#             "inte_slice_sched": PolicySpec(),
#             "intra_slice_sched": PolicySpec(),
#         },
#         "policy_mapping_fn": policy_mapping_fn,
#     },
# }
# algo_config = (
#     PPOConfig()
#     .environment("marl_comm_env")
#     .multi_agent(
#         policies=config["multiagent"]["policies"],
#         policy_mapping_fn=config["multiagent"]["policy_mapping_fn"],
#     )
# )
# algo = algo_config.build()

# # Training
# total_train_steps = 1
# for _ in range(total_train_steps):
#     result = algo.train()
#     print(pretty_print(result))

# # Testing
# marl_comm_env.reset(seed=seed)
# for agent in marl_comm_env.agent_iter():
#     obs, reward, termination, truncation, info = marl_comm_env.last()
#     if termination:
#         break
#     sched_decision = np.array(algo.compute_single_action(obs))
#     marl_comm_env.step(sched_decision)
