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
        eval_env: Optional[MARLCommEnv] = None,
        agent_type: str = "None",
        seed: int = np.random.randint(1000),
        agent_name: str = "ib_sched",
        episode_evaluation_freq: Optional[int] = None,
        number_evaluation_episodes: Optional[int] = None,
        checkpoint_episode_freq: Optional[int] = None,
        eval_initial_env_episode: Optional[int] = None,
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
        self.var_obs_inter_slice = 10
        self.var_obs_intra_ue = 2
        self.rbs_per_rbg = 5  # 135/rbs_per_rbg RBGs
        self.sorted_slices = np.arange(self.max_number_slices)

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
        formatted_obs_space = {}

        # Inter-slice observation
        formatted_obs_space["player_0"] = {
            "observations": np.array([]),
            "action_mask": self.last_unformatted_obs[0][
                "basestation_slice_assoc"
            ][0].astype(np.int8),
        }

        self.sorted_slices = self.sort_slices(
            self.last_unformatted_obs[0]["slice_req"],
            self.last_unformatted_obs[0]["slice_ue_assoc"],
            self.max_number_slices,
        )

        # intra-slice observations
        for agent_idx in self.sorted_slices + 1:
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
            active_metrics = np.logical_not(
                np.isclose(intent_drift_slice, -2)
            ).astype(int)
            intent_drift_slice[np.isclose(intent_drift_slice, -2)] = 0
            spectral_eff_slice = (
                np.mean(
                    np.mean(
                        self.last_unformatted_obs[0]["spectral_efficiencies"][
                            0, slice_ues, :
                        ],
                        axis=1,
                    )
                )
                if slice_ues.shape[0] > 0
                else 0
            )

            # Inter-slice scheduling
            formatted_obs_space["player_0"]["observations"] = np.concatenate(
                (
                    formatted_obs_space["player_0"]["observations"],
                    intent_drift_slice,
                    active_metrics,
                    np.array([slice_priority]),
                    np.array([slice_traffic_req]) / 120.0,
                    np.array([slice_ues.shape[0]]) / 5.0,
                    np.array([spectral_eff_slice])
                    / 40.0,  # TODO needs a proper normalization
                    # np.array([slice_buffer_occ]),
                    # np.array([slice_buffer_latency]),
                )
            )

            # Intra-slice scheduling
            slice_rbs_allocated = np.sum(
                np.sum(self.last_unformatted_obs[0]["sched_decision"], axis=2)[
                    0
                ]
                * self.last_unformatted_obs[0]["slice_ue_assoc"][agent_idx - 1]
            )
            association_slice_ue = np.zeros(
                self.max_number_ues_slice, dtype=np.int8
            )
            association_slice_ue[0 : slice_ues.shape[0]] = 1
            formatted_obs_space[f"player_{agent_idx}"] = {
                "observations": np.concatenate(
                    (
                        intent_drift_slice,
                        active_metrics,
                        np.array([slice_rbs_allocated])
                        / self.num_available_rbs[0],
                        np.array([slice_traffic_req]) / 120.0,
                        np.array([slice_ues.shape[0]]) / 5.0,
                        buffer_occ,
                        spectral_eff / 40.0,
                    )
                ),
                "action_mask": association_slice_ue,
            }

        self.last_formatted_obs = formatted_obs_space

        return formatted_obs_space

    def calculate_reward(self, obs_space: dict) -> dict:
        self.tmp_last_formatted_obs = deepcopy(self.last_formatted_obs)
        self.tmp_last_formatted_obs["player_0"][
            "observations"
        ] = self.unsort_slices(
            self.tmp_last_formatted_obs["player_0"]["observations"],
            self.sorted_slices,
            self.max_number_slices,
            10,
        )
        return calculate_reward_no_mask(
            obs_space,
            self.tmp_last_formatted_obs,
            self.last_unformatted_obs,
            self.var_obs_inter_slice,
        )

    def action_format(
        self,
        action_ori: Union[np.ndarray, dict],
        fixed_intra: Optional[str] = None,
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
            action["player_0"] = action["player_0"][self.sorted_slices]
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
                if fixed_intra is not None:
                    if fixed_intra == "rr":
                        allocation_rbs = round_robin(
                            allocation_rbs,
                            slice_idx,
                            rbs_per_slice,
                            slice_ues,
                            self.last_unformatted_obs,
                        )
                    elif fixed_intra == "pf":
                        allocation_rbs = proportional_fairness(
                            allocation_rbs,
                            slice_idx,
                            rbs_per_slice,
                            slice_ues,
                            self.env,
                            self.last_unformatted_obs,
                            self.num_available_rbs,
                        )
                    elif fixed_intra == "mt":
                        allocation_rbs = max_throughput(
                            allocation_rbs,
                            slice_idx,
                            rbs_per_slice,
                            slice_ues,
                            self.env,
                            self.last_unformatted_obs,
                            self.num_available_rbs,
                        )
                    else:
                        raise ValueError(
                            "Invalid intra-slice scheduling action"
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

    def sort_slices(
        self,
        slice_req: dict,
        slice_ue_assoc: np.ndarray,
        max_number_slices: int,
    ) -> np.ndarray:
        ues_per_slice = np.sum(slice_ue_assoc, axis=1)
        slice_req_traffic = np.array(
            [
                (
                    slice_req[f"slice_{idx}"]["ues"]["traffic"]
                    if slice_req[f"slice_{idx}"] != {}
                    else 0.0
                )
                for idx in np.arange(max_number_slices)
            ]
        )
        total_slice_traffic = ues_per_slice * slice_req_traffic

        return np.argsort(total_slice_traffic)

    def unsort_slices(
        self,
        obs_space: np.ndarray,
        sorted_slices: np.ndarray,
        max_number_slices: int,
        var_obs_inter_slice: int,
    ) -> np.ndarray:
        unsorted_obs_space = np.ones_like(obs_space) * -100
        for idx, sorted_idx in enumerate(sorted_slices):
            unsorted_obs_space[
                sorted_idx
                * var_obs_inter_slice : (sorted_idx + 1)
                * var_obs_inter_slice
            ] = obs_space[
                idx * var_obs_inter_slice : (idx + 1) * var_obs_inter_slice
            ]
        assert (  # TODO Remove this check
            np.sum(unsorted_obs_space == -100) == 0
        ), "Unsorted obs_space has missing values"

        return unsorted_obs_space

    def get_action_space(self) -> spaces.Dict:
        action_space = spaces.Dict(
            {
                f"player_{idx}": (
                    spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.max_number_slices,),
                        dtype=np.float64,
                    )
                    if idx == 0
                    else spaces.Discrete(3)
                )  # Three algorithms (RR, PF and Maximum Throughput)
                for idx in range(self.max_number_slices + 1)
            }
        )

        return action_space

    def get_obs_space(self) -> spaces.Dict:
        obs_space = spaces.Dict(
            {
                f"player_{idx}": (
                    spaces.Dict(
                        {
                            "observations": spaces.Box(
                                low=-1,
                                high=np.inf,
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
                                low=-1,
                                high=np.inf,
                                shape=(
                                    int(
                                        self.max_number_ues
                                        / self.max_number_slices
                                    )
                                    * self.var_obs_intra_ue
                                    + 9,
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
                )
                for idx in range(self.max_number_slices + 1)
            }
        )

        return obs_space

    def init_agent(self) -> None:
        pass
