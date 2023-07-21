import numpy as np
from pettingzoo.test import api_test, seed_test
from tqdm import tqdm

from agents.marr_test import MARRTest
from associations.simple import SimpleAssociation
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.simple import SimpleTraffic

seed = 10
number_steps = 10

marl_comm_env = MARLCommEnv(
    SimpleChannel,
    SimpleTraffic,
    SimpleMobility,
    SimpleAssociation,
    "simple",
    "agent_marr_test",
    seed,
    obs_space=MARRTest.get_obs_space,
    action_space=MARRTest.get_action_space,
    number_agents=2,
)
marr_test_agent = MARRTest(
    marl_comm_env,
    marl_comm_env.comm_env.max_number_ues,
    marl_comm_env.comm_env.max_number_basestations,
    marl_comm_env.comm_env.num_available_rbs,
)
marl_comm_env.comm_env.set_agent_functions(
    marr_test_agent.obs_space_format,
    marr_test_agent.action_format,
    marr_test_agent.calculate_reward,
)

obs, _ = marl_comm_env.reset(seed=seed)
for step in np.arange(number_steps):
    sched_decision = marr_test_agent.step(obs)
    obs, reward, terminated, truncated, info = marl_comm_env.step(
        sched_decision
    )
    if terminated["__all__"]:
        break
