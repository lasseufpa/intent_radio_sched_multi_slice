from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from sixg_radio_mgmt import Agent, CommunicationEnv, MARLCommEnv


class MARRTest(Agent):
    def __init__(
        self,
        env: Union[CommunicationEnv, MARLCommEnv],
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> dict:
        return {
            "player_0": np.array([[1, 0], [0, 1]]),
            "player_1": np.array([[1, 0], [0, 1]]),
        }

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        return {
            "player_0": np.zeros(4),
            "player_1": np.zeros(4),
        }

    def calculate_reward(self, obs_space: dict) -> float:
        return 0

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        assert isinstance(action, dict), "Action must be a dictionary"
        return np.array([action["player_0"], action["player_1"]])

    @staticmethod
    def get_action_space() -> dict:
        return {
            "player_0": spaces.Box(low=-1, high=1, shape=(2 * 2 * 2,)),
            "player_1": spaces.Box(low=-1, high=1, shape=(2 * 2 * 2,)),
        }

    @staticmethod
    def get_obs_space() -> dict:
        return {
            "player_0": spaces.Box(
                low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64
            ),
            "player_1": spaces.Box(
                low=0, high=np.inf, shape=(2 * 2,), dtype=np.float64
            ),
        }
