import numpy as np

from agents.ib_sched import IBSched
from agents.round_robin_slice import RoundRobin
from associations.simple_slice import SimpleSliceAssociation
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.simple import SimpleTraffic

seed = 10
steps_number = 10
number_agents = 3

marl_comm_env = MARLCommEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    SimpleSliceAssociation,
    "simple_slice",
    "test_agent_ib",
    seed,
    number_agents=number_agents,
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

sched_decisions = [
    {
        "player_0": np.array([-1, -1]),
        "player_1": int(2),
        "player_2": int(2),
    },
    {
        "player_0": np.array([0, 0]),
        "player_1": int(0),
        "player_2": int(0),
    },
]
reward_hist = {
    "player_0": [],
    "player_1": [],
    "player_2": [],
}
marl_comm_env.reset(seed=seed)
for agent in marl_comm_env.agent_iter():
    obs, reward, termination, truncation, info = marl_comm_env.last()
    reward_hist[agent].append(reward)
    if termination:
        break
    sched_decision = sched_decisions[0][agent]
    marl_comm_env.step(sched_decision)
print(reward_hist)
