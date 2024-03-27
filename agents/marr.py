from collections import deque
from typing import Optional, Union

import numpy as np
from gymnasium import spaces

from agents.ib_sched import IBSched
from sixg_radio_mgmt import Agent, MARLCommEnv


class MARR(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        seed: int = 0,
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.fake_agent = IBSched(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
        )

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        slice_ue_assoc = self.fake_agent.last_unformatted_obs[0][
            "slice_ue_assoc"
        ]
        action = (np.sum(slice_ue_assoc, axis=1) > 0).astype(int)
        action[action == 0] = -1

        return action

    def init_agent(self) -> None:
        pass  # No need to initialize the agent

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        obs = self.fake_agent.obs_space_format(obs_space)

        return obs["player_0"]["observations"]

    def calculate_reward(self, obs_space: Union[np.ndarray, dict]) -> float:
        obs_dict = {"player_0": obs_space}
        reward = self.fake_agent.calculate_reward(obs_dict)
        return reward["player_0"]

    def action_format(self, action_ori: Union[np.ndarray, dict]) -> np.ndarray:
        action = {
            "player_0": action_ori,
        }
        allocation_rbs = self.fake_agent.action_format(
            action, fixed_intra="rr"
        )

        return allocation_rbs

    @staticmethod
    def get_action_space() -> spaces.Box:
        action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float64)

        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Box:
        obs_space = spaces.Box(low=-2, high=1, shape=(30,), dtype=np.float64)

        return obs_space
