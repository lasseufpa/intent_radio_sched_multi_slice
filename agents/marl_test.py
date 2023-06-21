from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from sixg_radio_mgmt import Agent, CommunicationEnv


class MARLTest(Agent):
    def __init__(
        self,
        env: Union[CommunicationEnv, AECEnv],
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )

    def step(
        self, agent: str, obs_space: Optional[Union[np.ndarray, dict]]
    ) -> np.ndarray:
        if agent == "player_0":  # Basestation 1
            return np.array([1, 0, 0, 1])
        else:  # Basestation 2
            return np.array([1, 0, 0, 1])

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        formatted_obs_space = np.array([])
        hist_labels = [
            # "pkt_incoming",
            "dropped_pkts",
            # "pkt_effective_thr",
            "buffer_occupancies",
            # "spectral_efficiencies",
        ]
        for hist_label in hist_labels:
            if hist_label == "spectral_efficiencies":
                formatted_obs_space = np.append(
                    formatted_obs_space,
                    np.squeeze(np.sum(obs_space[hist_label], axis=2)),
                    axis=0,
                )
            else:
                formatted_obs_space = np.append(
                    formatted_obs_space, obs_space[hist_label], axis=0
                )

        return {
            "player_0": formatted_obs_space,
            "player_1": formatted_obs_space,
        }

    def calculate_reward(self, obs_space: dict) -> float:
        reward = -np.sum(obs_space["dropped_pkts"], dtype=float)
        return reward

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        assert isinstance(action, dict), "Action must be a dictionary"

        def select_ue(action_bs):
            action_bs_writtable = (
                action_bs.copy()
            )  # Ray puts actions in read-only mode
            if action_bs_writtable[0, 0] >= action_bs_writtable[1, 0]:
                action_bs_writtable[0, 0] = 1
                action_bs_writtable[1, 0] = 0
            else:
                action_bs_writtable[0, 0] = 0
                action_bs_writtable[1, 0] = 1

            if action_bs_writtable[0, 1] >= action_bs_writtable[1, 1]:
                action_bs_writtable[0, 1] = 1
                action_bs_writtable[1, 1] = 0
            else:
                action_bs_writtable[0, 1] = 0
                action_bs_writtable[1, 1] = 1

            return action_bs_writtable

        # print(action["player_0"].shape)
        basestation_1 = select_ue(np.reshape(action["player_0"], (2, 2)))
        basestation_2 = select_ue(np.reshape(action["player_1"], (2, 2)))

        return np.array(
            [
                basestation_1,
                basestation_2,
            ]
        )

    @staticmethod
    def get_action_space() -> dict:
        return {
            "player_0": spaces.Box(low=-1, high=1, shape=(2 * 2,)),
            "player_1": spaces.Box(low=-1, high=1, shape=(2 * 2,)),
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
