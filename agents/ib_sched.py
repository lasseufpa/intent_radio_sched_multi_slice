from typing import Optional, Union

import numpy as np
from gymnasium import spaces

from sixg_radio_mgmt import Agent, CommunicationEnv, MARLCommEnv


class IBSched(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.last_unformatted_obs = {}

    def step(
        self, agent: str, obs_space: Optional[Union[np.ndarray, dict]]
    ) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        self.last_unformatted_obs = obs_space
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
        allocation_rbs = np.array(
            [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
        )

        # Inter-slice scheduling
        rbs_per_slice = np.round(
            self.num_available_rbs[0]
            * (action["player_0"] + 1)
            / np.sum(action["player_0"] + 1)
        )
        assert (
            np.sum(rbs_per_slice) > self.num_available_rbs[0]
        ), "Allocated RBs are bigger than available RBs"
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"

        # Intra-slice scheduling
        for slice_idx in np.arange(rbs_per_slice.shape[0]):
            slice_ues = self.env.comm_env.slices.ue_assoc[
                slice_idx, :
            ].nonzero()[0]
            if slice_ues.shape[0] == 0:
                continue
            match action[f"player_{slice_idx+1}"]:
                case 0:
                    allocation_rbs = self.round_robin(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case 1:
                    allocation_rbs = self.proportional_fairness(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case 2:
                    allocation_rbs = self.max_throughput(
                        allocation_rbs, slice_idx, rbs_per_slice, slice_ues
                    )
                case _:
                    raise ValueError("Invalid intra-slice scheduling action")

        return allocation_rbs

    def round_robin(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        rbs_per_ue = np.floor(rbs_per_slice[slice_idx] / slice_ues.shape[0])
        remaining_rbs = rbs_per_slice[slice_idx] % slice_ues.shape[0]
        rbs_per_ue[0:remaining_rbs] += 1
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "RR: Number of allocated RBs is different than available RBs"
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def proportional_fairness(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        spectral_eff = np.mean(
            self.last_unformatted_obs["spectral_efficiencies"][
                0, slice_ues, :
            ],
            axis=1,
        )
        snt_thoughput = self.last_unformatted_obs["pkt_effective_thr"][
            slice_ues
        ]
        buffer_occ = self.last_unformatted_obs["buffer_occupancies"][slice_ues]
        throughput_available = np.min(
            [
                spectral_eff * self.env.comm_env.bandwidths[0],
                buffer_occ
                * self.env.comm_env.ues.max_buffer_pkts[slice_ues]
                * self.env.comm_env.ues.pkt_sizes[slice_ues],
            ],
            axis=0,
        )
        weights = np.divide(
            throughput_available,
            snt_thoughput,
            where=snt_thoughput != 0,
            out=0.00001 * np.ones_like(snt_thoughput),
        )
        rbs_per_ue = np.round(
            rbs_per_slice[slice_idx] * weights / np.sum(weights)
        )
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "PF: Number of allocated RBs is different than available RBs"

        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def max_throughput(
        self,
        allocation_rbs: np.ndarray,
        slice_idx: int,
        rbs_per_slice: np.ndarray,
        slice_ues: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        spectral_eff = np.mean(
            self.last_unformatted_obs["spectral_efficiencies"][
                0, slice_ues, :
            ],
            axis=1,
        )
        buffer_occ = self.last_unformatted_obs["buffer_occupancies"][slice_ues]
        throughput_available = np.min(
            [
                spectral_eff * self.env.comm_env.bandwidths[0],
                buffer_occ
                * self.env.comm_env.ues.max_buffer_pkts[slice_ues]
                * self.env.comm_env.ues.pkt_sizes[slice_ues],
            ],
            axis=0,
        )
        rbs_per_ue = np.round(
            rbs_per_slice[slice_idx]
            * throughput_available
            / np.sum(throughput_available)
        )
        assert (
            np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
        ), "MT: Number of allocated RBs is different than available RBs"
        allocation_rbs = self.distribute_rbs_ues(
            rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
        )

        return allocation_rbs

    def distribute_rbs_ues(
        self,
        rbs_per_ue: np.ndarray,
        allocation_rbs: np.ndarray,
        slice_ues: np.ndarray,
        rbs_per_slice: np.ndarray,
        slice_idx: int,
    ) -> np.ndarray:
        rb_idx = np.sum(rbs_per_slice[:slice_idx], dtype=int)
        for ue_idx in slice_ues:
            allocation_rbs[0, ue_idx, rb_idx : rb_idx + rbs_per_ue] = 1
            rb_idx += rbs_per_ue

        return allocation_rbs

    @staticmethod
    def get_action_space() -> dict:
        num_agents = 11
        action_space = {
            f"player_{idx}": spaces.Box(low=-1, high=1, shape=(10,))
            if idx == 0
            else spaces.Discrete(
                3
            )  # Three algorithms (RR, PF and Waterfilling)
            for idx in range(num_agents)
        }

        return action_space

    @staticmethod
    def get_obs_space() -> dict:
        num_agents = 11
        obs_space = {
            f"player_{idx}": spaces.Box(low=-1, high=1, shape=(10,))
            for idx in range(num_agents)
        }

        return obs_space
