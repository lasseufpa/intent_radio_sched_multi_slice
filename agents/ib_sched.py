from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces

from agents.common import (
    calculate_reward_no_mask,
    calculate_slice_ue_obs,
    intent_drift_calc,
    max_throughput,
    proportional_fairness,
    round_robin,
    scores_to_rbs,
)
from associations.mult_slice import MultSliceAssociation
from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSched(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        debug_violations: bool = False,
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
        max_obs_memory = 10
        self.max_number_ues_slice = 5
        self.last_unformatted_obs = deque(maxlen=max_obs_memory)
        self.last_formatted_obs = {}
        self.intent_overfulfillment_rate = 0.2
        self.var_obs_inter_slice = 6
        self.var_obs_intra_ue = 2
        self.rbs_per_rbg = 9  # 135/rbs_per_rbg RBGs
        assert isinstance(
            self.env.comm_env.associations, MultSliceAssociation
        ), "Associations must be MultSliceAssociation"
        self.max_throughput_slice = np.max(
            [
                slice_type["ues"]["traffic"]
                * slice_type["ues"]["max_number_ues"]
                for slice_type in self.env.comm_env.associations.slice_type_model.values()
            ]
        )
        self.debug_violations = debug_violations
        if self.debug_violations:
            self.number_metrics = 3
            self.violations = np.zeros(
                (
                    self.env.comm_env.max_number_steps,
                    self.env.comm_env.max_number_slices,
                    self.max_number_ues_slice,
                    self.number_metrics,
                ),
                dtype=float,
            )

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        raise NotImplementedError("IBSched does not implement step()")

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        self.last_unformatted_obs.appendleft(obs_space)
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        intent_drift = intent_drift_calc(
            self.last_unformatted_obs,
            self.max_number_ues_slice,
            self.intent_overfulfillment_rate,
        )
        if self.debug_violations:
            self.violations[
                self.env.comm_env.step_number - 1,
                :,
                :,
                :,
            ] = intent_drift
            if (
                self.env.comm_env.step_number
                == self.env.comm_env.max_number_steps
            ):
                np.savez_compressed(
                    "violations_ep_0.npz", violations=self.violations
                )
                self.violations = np.zeros(
                    (
                        self.env.comm_env.max_number_steps,
                        self.env.comm_env.max_number_slices,
                        self.max_number_ues_slice,
                        self.number_metrics,
                    ),
                    dtype=float,
                )
        formatted_obs_space = {}

        # Inter-slice observation
        formatted_obs_space["player_0"] = {
            "observations": np.array([]),
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

            (
                intent_drift_ue_values,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                self.max_number_ues_slice,
                intent_drift,
                agent_idx - 1,
                slice_ues,
                self.last_unformatted_obs[0]["slice_req"],
            )

            spectral_eff = np.pad(
                np.mean(
                    self.last_unformatted_obs[0]["spectral_efficiencies"][
                        0, slice_ues, :
                    ],
                    axis=1,
                ),
                (0, self.max_number_ues_slice - slice_ues.shape[0]),
                "constant",
            )
            max_spectral_eff = np.max(spectral_eff)
            spectral_eff = (
                spectral_eff / max_spectral_eff
                if max_spectral_eff != 0
                else spectral_eff
            )
            buffer_occ = np.pad(
                self.last_unformatted_obs[0]["buffer_occupancies"][slice_ues],
                (0, self.max_number_ues_slice - slice_ues.shape[0]),
                "constant",
            )
            slice_traffic_req = (
                self.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["traffic"]
                if self.last_unformatted_obs[0]["basestation_slice_assoc"][
                    0, agent_idx - 1
                ]
                == 1
                else 0
            )
            slice_priority = (
                self.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["priority"]
                if slice_ues.shape[0] != 0
                else 0
            )

            # Inter-slice scheduling
            formatted_obs_space["player_0"]["observations"] = (
                np.concatenate(
                    (
                        formatted_obs_space["player_0"]["observations"],
                        intent_drift_slice,
                        np.array([slice_priority]),
                        np.array(
                            [
                                slice_traffic_req
                                * slice_ues.shape[0]
                                / self.max_throughput_slice
                            ]
                        ),
                        np.array(
                            [
                                np.mean(
                                    spectral_eff[0 : slice_ues.shape[0]]
                                    * max_spectral_eff
                                )
                                / 20
                            ]
                        ),
                    )
                )
                if self.last_unformatted_obs[0]["basestation_slice_assoc"][
                    0, agent_idx - 1
                ]
                == 1
                else np.append(
                    formatted_obs_space["player_0"]["observations"],
                    np.concatenate((intent_drift_slice, np.array([0, 0, 0]))),
                )
            )

            # Intra-slice scheduling
            if agent_idx < len(self.env.agents):
                association_slice_ue = np.zeros(
                    self.max_number_ues_slice, dtype=np.int8
                )
                association_slice_ue[0 : slice_ues.shape[0]] = 1
                formatted_obs_space[f"player_{agent_idx}"] = {
                    "observations": np.concatenate(
                        (
                            np.array(
                                [
                                    slice_traffic_req
                                    * slice_ues.shape[0]
                                    / self.max_throughput_slice
                                ]
                            ),
                            buffer_occ,
                            spectral_eff,
                        )
                    ),
                    "action_mask": association_slice_ue,
                }

        self.last_formatted_obs = formatted_obs_space

        return formatted_obs_space

    def calculate_reward(self, obs_space: dict) -> dict:
        return calculate_reward_no_mask(
            obs_space,
            self.last_formatted_obs,
            self.last_unformatted_obs,
            self.var_obs_inter_slice,
        )

    def action_format(
        self, action_ori: Union[np.ndarray, dict], intra_rr: bool = False
    ) -> np.ndarray:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
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
            action = deepcopy(action_ori)
            action["player_0"][
                np.where(
                    self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ]
                    == 0
                )[0]
            ] = -1

            # Inter-slice scheduling
            rbs_per_slice = (
                scores_to_rbs(
                    action["player_0"],
                    np.floor(
                        self.num_available_rbs[0] / self.rbs_per_rbg
                    ).astype(int),
                    self.last_unformatted_obs[0]["basestation_slice_assoc"][
                        0, :
                    ],
                )
                * self.rbs_per_rbg
            )

            # Intra-slice scheduling
            for slice_idx in np.arange(rbs_per_slice.shape[0]):
                slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                    slice_idx, :
                ].nonzero()[0]
                if slice_ues.shape[0] == 0:
                    continue
                if intra_rr:
                    allocation_rbs = round_robin(
                        allocation_rbs,
                        slice_idx,
                        rbs_per_slice,
                        slice_ues,
                        self.last_unformatted_obs,
                    )
                else:
                    match action[f"player_{slice_idx+1}"]:
                        case 0:
                            allocation_rbs = round_robin(
                                allocation_rbs,
                                slice_idx,
                                rbs_per_slice,
                                slice_ues,
                                self.last_unformatted_obs,
                            )
                        case 1:
                            allocation_rbs = proportional_fairness(
                                allocation_rbs,
                                slice_idx,
                                rbs_per_slice,
                                slice_ues,
                                self.env,
                                self.last_unformatted_obs,
                                self.num_available_rbs,
                            )
                        case 2:
                            allocation_rbs = max_throughput(
                                allocation_rbs,
                                slice_idx,
                                rbs_per_slice,
                                slice_ues,
                                self.env,
                                self.last_unformatted_obs,
                                self.num_available_rbs,
                            )
                        case _:
                            raise ValueError(
                                "Invalid intra-slice scheduling action"
                            )
            assert (
                np.sum(allocation_rbs) == self.num_available_rbs[0]
            ), "Allocated RBs are different from available RBs"

        return allocation_rbs

    def get_action_space(self) -> spaces.Dict:
        action_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.max_number_slices,),
                    dtype=np.float64,
                )
                if idx == 0
                else spaces.Discrete(
                    3
                )  # Three algorithms (RR, PF and Maximum Throughput)
                for idx in range(self.max_number_slices + 1)
            }
        )

        return action_space

    def get_obs_space(self) -> spaces.Dict:
        obs_space = spaces.Dict(
            {
                f"player_{idx}": spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-2,
                            high=1,
                            shape=(
                                self.max_number_slices
                                * self.var_obs_inter_slice,
                            ),
                            dtype=np.float64,
                        ),
                        "action_mask": spaces.Box(
                            0.0,
                            1.0,
                            shape=(self.max_number_slices,),
                            dtype=np.int8,
                        ),
                    }
                )
                if idx == 0
                else spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-2,
                            high=1,
                            shape=(
                                int(
                                    self.max_number_ues
                                    / self.max_number_slices
                                )
                                * self.var_obs_intra_ue
                                + 1,
                            ),
                            dtype=np.float64,
                        ),
                        "action_mask": spaces.Box(
                            0.0,
                            1.0,
                            shape=(
                                int(
                                    self.max_number_ues
                                    / self.max_number_slices
                                ),
                            ),
                            dtype=np.int8,
                        ),
                    }
                )
                for idx in range(self.max_number_slices + 1)
            }
        )

        return obs_space
