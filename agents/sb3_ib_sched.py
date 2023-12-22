from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from agents.ib_sched import IBSched
from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSchedSB3(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        debug_violations: bool = False,
        agent_type: str = "ppo",
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        episode_evaluation_freq = 4
        number_evaluation_episodes = 1
        checkpoint_episode_freq = 4
        checkpoint_frequency = (
            self.env.comm_env.max_number_steps * checkpoint_episode_freq
        )
        self.callback_checkpoint = CheckpointCallback(
            save_freq=checkpoint_frequency,
            save_path="./agents/models/sb3_ib_sched/",
            name_prefix="sb3_ib_sched",
        )
        self.callback_evaluation = EvalCallback(
            eval_env=self.env,
            log_path="./evaluations/sb3_ib_sched",
            best_model_save_path="./agents/models/best_sb3_ib_sched/",
            n_eval_episodes=number_evaluation_episodes,
            eval_freq=self.env.comm_env.max_number_steps
            * episode_evaluation_freq,
            verbose=False,
            warn=False,
        )
        self.fake_agent = IBSched(
            env,
            max_number_ues,
            max_number_basestations,
            num_available_rbs,
        )
        if agent_type == "ppo":
            self.agent = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log="tensorboard-logs/",
                seed=self.seed,
            )
        elif agent_type == "sac":
            self.agent = SAC(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log="tensorboard-logs/",
                seed=self.seed,
            )
        else:
            raise ValueError("Invalid agent type")

    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:
        return self.agent.predict(np.asarray(obs_space), deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=[self.callback_checkpoint, self.callback_evaluation],
        )
        self.agent.save("./agents/models/final_sb3_ib_sched")

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
        allocation_rbs = self.fake_agent.action_format(action, intra_rr=True)

        return allocation_rbs

    @staticmethod
    def get_action_space() -> spaces.Box:
        action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float64)

        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Box:
        obs_space = spaces.Box(low=-2, high=1, shape=(30,), dtype=np.float64)

        return obs_space
