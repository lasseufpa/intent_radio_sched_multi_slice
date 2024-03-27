from collections import deque
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from matplotlib.pylab import f

from agents.ib_sched import IBSched
from sixg_radio_mgmt import Agent, MARLCommEnv


class MAPF(Agent):
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
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        action = {}
        slices_thr_sent = np.zeros(self.max_number_slices)
        slices_buffer_occ = np.zeros(self.max_number_slices)
        weights = -1 * np.ones(self.max_number_slices)

        active_slice_idx = self.fake_agent.last_unformatted_obs[0][
            "basestation_slice_assoc"
        ][0, :].nonzero()[0]
        for slice_idx in active_slice_idx:
            slice_ues = self.fake_agent.last_unformatted_obs[0][
                "slice_ue_assoc"
            ][slice_idx].nonzero()[0]
            slice_pkt_size = self.fake_agent.last_unformatted_obs[0][
                "slice_req"
            ][f"slice_{slice_idx}"]["ues"]["message_size"]
            slice_buffer_max_size = self.fake_agent.last_unformatted_obs[0][
                "slice_req"
            ][f"slice_{slice_idx}"]["ues"]["buffer_size"]
            slices_buffer_occ[slice_idx] = (
                (
                    np.mean(
                        self.fake_agent.last_unformatted_obs[0][
                            "buffer_occupancies"
                        ][slice_ues]
                    )
                    * slice_buffer_max_size
                )
                * slice_pkt_size
                / 1e6
            )  # In Mbps
            slices_thr_sent[slice_idx] = (
                np.mean(
                    np.mean(
                        [
                            self.fake_agent.last_unformatted_obs[idx][
                                "pkt_effective_thr"
                            ][slice_ues]
                            for idx in range(
                                len(self.fake_agent.last_unformatted_obs)
                            )
                        ],
                        axis=0,
                    )
                )
                * slice_pkt_size
            ) / 1e6  # In Mbps
        weights = np.divide(
            slices_buffer_occ,
            slices_thr_sent,
            where=np.logical_not(
                np.isclose(slices_thr_sent, np.zeros_like(slices_thr_sent))
            ),
            out=2 * np.max(slices_buffer_occ) * np.ones_like(slices_thr_sent),
        )
        inactive_slice_idxs = np.logical_not(
            self.fake_agent.last_unformatted_obs[0]["basestation_slice_assoc"][
                0, :
            ]
        )
        weights[inactive_slice_idxs] = 0
        action = (
            weights / np.sum(weights)
            if np.sum(weights) > 0
            else 2 * np.ones_like(weights)
        ) - 1

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
            action, fixed_intra="pf"
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
