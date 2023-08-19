from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from scipy.optimize import minimize

from agents.common import (
    calculate_reward_mask,
    distribute_rbs_ues,
    intent_drift_calc,
    scores_to_rbs,
)
from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSchedIntraNN(Agent):
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
        max_obs_memory = 10
        self.max_number_ues_slice = 10
        self.last_unformatted_obs = deque(maxlen=max_obs_memory)
        self.last_formatted_obs = {}
        self.intent_overfulfillment_rate = 0.2

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def calculate_reward(self, obs_space: dict) -> float | dict:
        return calculate_reward_mask(obs_space, self.last_formatted_obs)

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        self.last_unformatted_obs.appendleft(obs_space)
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        intent_drift_slice_ue = intent_drift_calc(
            self.last_unformatted_obs,
            self.max_number_ues_slice,
            self.intent_overfulfillment_rate,
            self.env,
        )
        formatted_obs_space = {}

        # Inter-slice observation
        formatted_obs_space["player_0"] = {
            "observations": np.append(
                np.zeros(obs_space["slice_ue_assoc"].shape[0]),
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :],
            ),
            "action_mask": self.last_unformatted_obs[0][
                "basestation_slice_assoc"
            ][0].astype(np.int8),
        }

        # intra-slice observations
        for agent_idx in range(1, obs_space["slice_ue_assoc"].shape[0] + 1):
            assert isinstance(
                self.env, MARLCommEnv
            ), "Environment must be MARLCommEnv"
            slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                agent_idx - 1
            ].nonzero()[0]
            intent_ue_values = intent_drift_slice_ue[agent_idx - 1, :]
            formatted_obs_space["player_0"]["observations"][agent_idx - 1] = (
                np.mean(intent_ue_values[0 : slice_ues.shape[0]])
                if slice_ues.shape[0] != 0
                else 1
            )
            association = np.zeros(self.max_number_ues_slice)
            association[0 : slice_ues.shape[0]] = 1
            formatted_obs_space[f"player_{agent_idx}"] = {
                "observations": np.append(
                    intent_ue_values, association
                ).astype(np.float64),
                "action_mask": association.astype(np.int8),
            }
        self.last_formatted_obs = formatted_obs_space.copy()

        return formatted_obs_space

    def action_format(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        allocation_rbs = np.array(
            [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
        )

        if (
            np.sum(
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :]
            )
            != 0
        ):
            assert (  # verify if the inactive slices are receiving -1 value
                np.prod(
                    action["player_0"][
                        self.last_unformatted_obs[0][
                            "basestation_slice_assoc"
                        ][0, :]
                        == 0
                    ]
                    == -1
                )
                == 1
            ), "Invalid action"
            assert isinstance(
                self.env, MARLCommEnv
            ), "Environment must be MARLCommEnv"

            # Inter-slice scheduling
            rbs_per_slice = scores_to_rbs(
                action["player_0"],
                self.num_available_rbs[0],
                self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :],
            )

            # Intra-slice scheduling
            for slice_idx in np.arange(rbs_per_slice.shape[0]):
                slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                    slice_idx, :
                ].nonzero()[0]
                slice_ue_assoc = np.zeros(self.max_number_ues_slice)
                num_connected_ues = np.sum(
                    self.last_unformatted_obs[0]["slice_ue_assoc"][
                        slice_idx, :
                    ]
                ).astype(int)
                slice_ue_assoc[0:num_connected_ues] = 1
                if slice_ues.shape[0] == 0:
                    continue

                rbs_per_ue = scores_to_rbs(
                    action[f"player_{slice_idx+1}"],
                    int(rbs_per_slice[slice_idx]),
                    slice_ue_assoc,
                )
                allocation_rbs = distribute_rbs_ues(
                    rbs_per_ue,
                    allocation_rbs,
                    slice_ues,
                    rbs_per_slice,
                    slice_idx,
                )
            assert (
                np.sum(allocation_rbs) == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"

        return allocation_rbs

    @staticmethod
    def get_action_space() -> spaces.Dict:
        num_agents = 11
        action_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Box(
                    low=-1, high=1, shape=(10,), dtype=np.float64
                )
                for idx in range(num_agents)
            }
        )

        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Dict:
        num_agents = 11
        obs_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-1, high=1, shape=(20,), dtype=np.float64
                        ),
                        "action_mask": spaces.Box(
                            0.0, 1.0, shape=(10,), dtype=np.int8
                        ),
                    }
                )
                for idx in range(num_agents)
            }
        )

        return obs_space
