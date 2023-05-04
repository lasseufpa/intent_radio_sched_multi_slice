import numpy as np
from tqdm import tqdm

from agents.round_robin import RoundRobin
from associations.mult_slice import MultSliceAssociation
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.mult_slice import MultSliceTraffic

seed = 10

comm_env = CommunicationEnv(
    QuadrigaChannel,
    MultSliceTraffic,
    SimpleMobility,
    MultSliceAssociation,
    "mult_slice",
    "round_robin",
)

round_robin = RoundRobin(
    comm_env,
    comm_env.max_number_ues,
    comm_env.max_number_basestations,
    comm_env.num_available_rbs,
)
comm_env.set_agent_functions(
    round_robin.obs_space_format,
    round_robin.action_format,
    round_robin.calculate_reward,
)

obs = comm_env.reset(seed=seed)[0]
for step_number in tqdm(
    np.arange(comm_env.max_number_steps * comm_env.max_number_episodes)
):
    sched_decision = round_robin.step(obs)
    obs, _, end_ep, _, _ = comm_env.step(sched_decision)
    if end_ep:
        comm_env.reset()
