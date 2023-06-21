from typing import Union

import numpy as np

from sixg_radio_mgmt import Agent, CommunicationEnv


class RoundRobin(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )
        self.current_ues = np.array([])
        self.rbs_per_ue = np.array([])
        self.allocation_rbs = []

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        idx_active_ues = np.array([])

        if not np.array_equal(self.current_ues, obs_space[0]):
            self.allocation_rbs = [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
            self.current_ues = obs_space[0]
            idx_active_ues = obs_space[0].nonzero()[0]
            num_active_ues = int(np.sum(obs_space[0]))
            num_rbs_per_ue = int(
                (
                    np.floor(self.num_available_rbs[0] / num_active_ues)
                    if num_active_ues > 0
                    else 0
                )
            )
            remaining_rbs = (
                self.num_available_rbs[0] - num_rbs_per_ue * num_active_ues
            )
            self.rbs_per_ue = np.ones(num_active_ues) * num_rbs_per_ue
            self.rbs_per_ue[:remaining_rbs] += 1

        initial_rb = 0
        for idx, ue_idx in enumerate(idx_active_ues):
            self.allocation_rbs[0][
                ue_idx, initial_rb : initial_rb + int(self.rbs_per_ue[idx])
            ] = 1
            initial_rb += int(self.rbs_per_ue[idx])

        self.rbs_per_ue = np.roll(self.rbs_per_ue, 1)
        # print(
        #     f"Number Active UEs: {int(np.sum(obs_space[0]))}\nAllocated RBs: {np.sum(self.allocation_rbs)}\n\n"
        # )
        return np.array(self.allocation_rbs)

    def obs_space_format(self, obs_space: dict) -> np.ndarray:
        return np.array(obs_space["basestation_ue_assoc"])

    def calculate_reward(self, obs_space: dict) -> float:
        return 0

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        return np.array(action)
