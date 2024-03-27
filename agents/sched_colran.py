from collections import deque
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from agents.common import (
    calculate_slice_ue_obs,
    get_metric_value,
    intent_drift_calc,
)
from agents.ib_sched import IBSched
from agents.sb3_callbacks import CustomEvalCallback as EvalCallback
from sixg_radio_mgmt import Agent, MARLCommEnv


class SchedColORAN(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        eval_env: Optional[MARLCommEnv] = None,
        agent_type: str = "ppo",
        seed: int = np.random.randint(1000),
        agent_name: str = "sched_coloran",
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
            seed=seed,
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.episode_evaluation_freq = 80
        self.number_evaluation_episodes = 20
        checkpoint_episode_freq = 10
        eval_initial_env_episode = 1080
        eval_maximum_env_episode = (
            eval_initial_env_episode + self.number_evaluation_episodes
        )
        self.checkpoint_frequency = (
            self.env.comm_env.max_number_steps * checkpoint_episode_freq
        )
        self.eval_env = eval_env
        if self.eval_env is not None:
            self.eval_env.comm_env.initial_episode_number = (
                eval_initial_env_episode
            )
            self.eval_env.comm_env.max_number_episodes = (
                eval_maximum_env_episode
            )
        self.fake_agent = IBSched(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
        )
        self.agent = None

    def init_agent(self) -> None:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        if self.eval_env is not None:
            self.callback_evaluation = EvalCallback(
                eval_env=self.eval_env,
                log_path=f"./evaluations/{self.env.comm_env.simu_name}/{self.agent_name}",
                best_model_save_path=f"./agents/models/{self.env.comm_env.simu_name}/best_{self.agent_name}/",
                n_eval_episodes=self.number_evaluation_episodes,
                eval_freq=self.env.comm_env.max_number_steps
                * self.episode_evaluation_freq,
                verbose=False,
                warn=False,
                seed=self.eval_env.comm_env.seed,
            )
        else:
            self.callback_evaluation = None
        self.callback_checkpoint = CheckpointCallback(
            save_freq=self.checkpoint_frequency,
            save_path=f"./agents/models/{self.env.comm_env.simu_name}/{self.agent_name}/",
            name_prefix=self.agent_name,
        )
        if self.agent_type == "ppo":
            self.agent = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log=f"tensorboard-logs/{self.env.comm_env.simu_name}/{self.agent_name}/",
                seed=self.seed,
            )
        elif self.agent_type == "sac":
            self.agent = SAC(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log=f"tensorboard-logs/{self.env.comm_env.simu_name}/{self.agent_name}/",
                seed=self.seed,
                # policy_kwargs=dict(net_arch=[2048, 2048]),
            )
        else:
            raise ValueError("Invalid agent type")

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        assert self.agent is not None, "Agent must be created first"
        return self.agent.predict(np.asarray(obs_space), deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        assert self.agent is not None, "Agent must be created first"
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        callbacks = [
            self.callback_checkpoint,
            self.callback_evaluation,
        ]
        callbacks = [cb for cb in callbacks if cb is not None]
        self.agent.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=callbacks,
            log_interval=1,  # Number of episodes
        )
        self.agent.save(
            f"./agents/models/{self.env.comm_env.simu_name}/final_{self.agent_name}"
        )

    def load(self, path: str) -> None:
        assert self.agent is not None, "Agent must be created first"
        if self.agent_type == "ppo":
            self.agent = PPO.load(path, self.env)
        elif self.agent_type == "sac":
            self.agent = SAC.load(path, self.env)

    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        # For each slice keep  nine metrics defined
        # for each slice and UE: spectral efficiency, served throughput,
        # effective throughput, buffer occupancy, packet loss rate, re-
        # quested throughput, average buffer latency, long-term served
        # throughput and the fifth-percentile served throughput.
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.fake_agent.obs_space_format(
            obs_space
        )  # updating internal fake_agent variables
        self.fake_agent.last_unformatted_obs.appendleft(obs_space)

        # Inter-slice observation
        formatted_obs_space = {}
        formatted_obs_space["player_0"] = {
            "observations": np.array([]),
            "action_mask": self.fake_agent.last_unformatted_obs[0][
                "basestation_slice_assoc"
            ][0].astype(np.int8),
        }

        ordered_metrics = [
            "requirements",
            "spectral_efficiencies",
            "pkt_throughputs",
            "pkt_effective_thrs",
            "buffer_occupancies",
            "buffer_latencies",
            "pkt_loss_rates",
            "requested_thrs",
        ]
        dict_metrics = {
            "requirements": np.array([]),
            "spectral_efficiencies": np.array([]),
            "pkt_throughputs": np.array([]),
            "pkt_effective_thrs": np.array([]),
            "buffer_occupancies": np.array([]),
            "buffer_latencies": np.array([]),
            "pkt_loss_rates": np.array([]),
            "requested_thrs": np.array([]),
        }

        # intra-slice observations
        for agent_idx in range(1, obs_space["slice_ue_assoc"].shape[0] + 1):
            slice_ues = self.fake_agent.last_unformatted_obs[0][
                "slice_ue_assoc"
            ][agent_idx - 1].nonzero()[0]

            requirements = np.zeros(3)
            if slice_ues.shape[0] != 0:
                for parameter in self.fake_agent.last_unformatted_obs[0][
                    "slice_req"
                ][f"slice_{agent_idx-1}"]["parameters"].values():
                    if parameter["name"] == "reliability":
                        requirements[0] = parameter["value"]
                    elif parameter["name"] == "latency":
                        requirements[1] = parameter["value"]
                    elif parameter["name"] == "throughput":
                        requirements[2] = parameter["value"]
            dict_metrics["requirements"] = np.append(
                dict_metrics["requirements"], requirements
            )
            # Pkt size
            pkt_size = (
                self.fake_agent.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["message_size"]
                if slice_ues.shape[0] != 0
                else 0
            )

            # Spectral efficiency
            if slice_ues.shape[0] != 0:
                spectral_eff = np.mean(
                    self.fake_agent.last_unformatted_obs[0][
                        "spectral_efficiencies"
                    ][0, slice_ues, :],
                    axis=1,
                )
            else:
                spectral_eff = np.array([0])
            dict_metrics["spectral_efficiencies"] = np.append(
                dict_metrics["spectral_efficiencies"], np.mean(spectral_eff)
            )

            # Served Throughput
            served_thr = (
                self.fake_agent.last_unformatted_obs[0]["pkt_throughputs"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_throughputs"] = np.append(
                dict_metrics["pkt_throughputs"],
                np.mean(self.pkts_to_mbps(served_thr, pkt_size)),
            )

            # Effective Throughput
            eff_thr = (
                self.fake_agent.last_unformatted_obs[0]["pkt_effective_thr"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_effective_thrs"] = np.append(
                dict_metrics["pkt_effective_thrs"],
                np.mean(self.pkts_to_mbps(eff_thr, pkt_size)),
            )

            # Buffer Occ.
            buffer_occ = (
                self.fake_agent.last_unformatted_obs[0]["buffer_occupancies"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_occupancies"] = np.append(
                dict_metrics["buffer_occupancies"], np.mean(buffer_occ)
            )

            # Buffer latencies
            buffer_latencies = (
                self.fake_agent.last_unformatted_obs[0]["buffer_latencies"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_latencies"] = np.append(
                dict_metrics["buffer_latencies"], np.mean(buffer_latencies)
            )

            # Packet loss rate
            pkt_loss_rate = (
                get_metric_value(
                    "reliability",
                    self.fake_agent.last_unformatted_obs,
                    agent_idx - 1,
                    slice_ues,
                    reliability_pkt_loss=True,
                )
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_loss_rates"] = np.append(
                dict_metrics["pkt_loss_rates"], np.mean(pkt_loss_rate)
            )

            # Requested throughput
            slice_traffic_req = (
                self.fake_agent.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["traffic"]
                if np.isclose(
                    self.fake_agent.last_unformatted_obs[0][
                        "basestation_slice_assoc"
                    ][0, agent_idx - 1],
                    1,
                )
                else 0
            )
            dict_metrics["requested_thrs"] = np.append(
                dict_metrics["requested_thrs"],
                slice_traffic_req,
            )
        for key in ordered_metrics:
            formatted_obs_space["player_0"]["observations"] = np.concatenate(
                (
                    formatted_obs_space["player_0"]["observations"],
                    dict_metrics[key],
                )
            )

        self.last_formatted_obs = formatted_obs_space

        return formatted_obs_space["player_0"]["observations"]

    def calculate_reward(self, obs_space: Union[np.ndarray, dict]) -> float:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        # We classify any slice type with a latency requirement smaller
        # than 20ms as urllc. We also classify any slice with only throughput
        # requirements or with throughput requirements bigger than 20 Mbps as eMBB.
        # A slice can be classified as both eMBB and urllc at the same time too.
        # In case the slice type does not fit none of the above assumptions, we assume eMBB.
        maximum_throughput = 200  # in Mbps (only for normalization purpose)
        maximum_buffer_size = 2000  # in Mbps (only for normalization purpose)
        slice_type_to_use_case = {
            "control_case_2": ["urllc"],
            "monitoring_case_1": ["embb"],
            "robotic_surgery_case_1": ["embb", "urllc"],
            "robotic_diagnosis": ["urllc"],
            "medical_monitoring": ["embb"],
            "uav_app_case_1": ["embb"],
            "uav_control_non_vlos": ["embb"],
            "vr_gaming": ["embb", "urllc"],
            "cloud_gaming": ["embb"],
            "video_streaming_4k": ["embb"],
        }
        reward = 0
        active_slice_idx = self.fake_agent.last_unformatted_obs[0][
            "basestation_slice_assoc"
        ][0, :].nonzero()[0]
        for slice_idx in active_slice_idx:
            slice_ues = self.fake_agent.last_unformatted_obs[0][
                "slice_ue_assoc"
            ][slice_idx].nonzero()[0]
            if slice_ues.shape[0] == 0:
                continue
            slice_name = self.fake_agent.last_unformatted_obs[0]["slice_req"][
                f"slice_{slice_idx}"
            ]["name"]

            if "embb" in slice_type_to_use_case[slice_name]:
                slice_pkt_size = self.fake_agent.last_unformatted_obs[0][
                    "slice_req"
                ][f"slice_{slice_idx}"]["ues"]["message_size"]
                slice_throughput = (
                    np.mean(
                        self.fake_agent.last_unformatted_obs[0][
                            "pkt_throughputs"
                        ][slice_ues]
                    )
                    * slice_pkt_size
                ) / 1e6  # in Mbps
                reward += slice_throughput / maximum_throughput
            if "urllc" in slice_type_to_use_case[slice_name]:
                slice_pkt_size = self.fake_agent.last_unformatted_obs[0][
                    "slice_req"
                ][f"slice_{slice_idx}"]["ues"]["message_size"]
                slice_buffer_max_size = self.fake_agent.last_unformatted_obs[
                    0
                ]["slice_req"][f"slice_{slice_idx}"]["ues"]["buffer_size"]
                slice_buffer_mbps = (
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
                )  # in Mbps
                reward -= slice_buffer_mbps / maximum_buffer_size

        return reward

    def action_format(self, action_ori: Union[np.ndarray, dict]) -> np.ndarray:
        action = {
            "player_0": action_ori,
        }
        allocation_rbs = self.fake_agent.action_format(
            action, fixed_intra="rr"
        )

        return allocation_rbs

    def get_action_space(self) -> spaces.Space:
        action_space = self.fake_agent.get_action_space()

        return action_space["player_0"]

    def get_obs_space(self) -> spaces.Space:
        obs_space = self.fake_agent.get_obs_space()["player_0"]["observations"]  # type: ignore

        return obs_space

    def pkts_to_mbps(self, pkts: np.ndarray, pkt_size: float) -> np.ndarray:
        return pkts * pkt_size / 1e6