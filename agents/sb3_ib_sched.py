from collections import deque
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from agents.ib_sched import IBSched
from agents.sb3_callbacks import CustomEvalCallback as EvalCallback
from sixg_radio_mgmt import Agent, MARLCommEnv


class IBSchedSB3(Agent):
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
        agent_name: str = "sb3_ib_sched",
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
        self.episode_evaluation_freq = 10
        self.number_evaluation_episodes = 10
        checkpoint_episode_freq = 10
        eval_initial_env_episode = self.env.comm_env.max_number_episodes
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
                seed=self.seed,
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
                tensorboard_log=f"tensorboard-logs/{self.env.comm_env.simu_name}/",
                seed=self.seed,
            )
        elif self.agent_type == "sac":
            self.agent = SAC(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log=f"tensorboard-logs/{self.env.comm_env.simu_name}/",
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

    def get_action_space(self) -> spaces.Space:
        action_space = self.fake_agent.get_action_space()

        return action_space["player_0"]

    def get_obs_space(self) -> spaces.Space:
        obs_space = self.fake_agent.get_obs_space()["player_0"]["observations"]  # type: ignore

        return obs_space
