from pathlib import Path
from typing import Callable

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from tqdm import tqdm

from agents.action_mask_model import TorchActionMaskModel
from agents.masked_action_distribution import TorchDiagGaussian


class RayAgent:
    # It does not implement any of the MARL logic. It is a wrapper to use Ray
    # with the general simulation script simu.py
    def __init__(
        self,
        env_creator: Callable,
        env_config: dict,
        debug_mode: bool = False,
        enable_masks: bool = True,
    ):
        ray.init(local_mode=debug_mode)
        register_env("marl_comm_env", lambda config: env_creator(config, True))
        ModelCatalog.register_custom_action_dist(
            "masked_gaussian", TorchDiagGaussian
        )

        self.env_config = env_config
        self.agent = None
        self.enable_masks = enable_masks
        self.read_checkpoint = str(Path("./ray_results/").resolve())
        self.algo = None
        self.train_batch_size = 2048
        self.steps_per_episode = 1000
        self.eps_per_iteration = np.rint(
            self.train_batch_size // self.steps_per_episode
        ).astype(int)

    def train(self, total_timesteps: int):
        # Total timesteps is not used in this implementation
        # it is just a placeholder to keep the same interface as SB3
        algo_config = self.gen_config(self.env_config)
        stop = {
            "episodes_total": int(
                (
                    self.env_config["max_training_episodes"]
                    - self.env_config["initial_training_episode"]
                )
                * self.env_config["training_epochs"]
            ),
        }
        results = tune.Tuner(
            "PPO",
            param_space=algo_config.to_dict(),
            run_config=air.RunConfig(
                storage_path=f"{self.read_checkpoint}/{self.env_config['scenario']}/",
                name=self.env_config["agent"],
                stop=stop,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=np.rint(
                        self.env_config["checkpoint_episode_freq"]
                        / self.eps_per_iteration
                    ).astype(int),
                    checkpoint_at_end=True,
                ),
            ),
        ).fit()
        print(results)

    def gen_config(self, env_config):
        algo_config = (
            PPOConfig()
            .environment(
                env="marl_comm_env",
                env_config=env_config,
                is_atari=False,
                disable_env_checking=True,
                # clip_rewards=False,
                # normalize_actions=True,
                # clip_actions=False,
            )
            .multi_agent(
                policies={
                    "inter_slice_sched": (
                        self.action_mask_policy()
                        if self.enable_masks
                        else PolicySpec()
                    ),
                    "intra_slice_sched": PolicySpec(),
                },
                policy_mapping_fn=self.policy_mapping_fn,
                count_steps_by="env_steps",
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=0,
                enable_connectors=False,
                num_envs_per_worker=1,
            )
            .resources(
                num_gpus=1,
                num_gpus_per_worker=1,
                num_gpus_per_learner_worker=1,
            )
            .training(
                lr=0.0003,  # SB3 LR
                train_batch_size=self.train_batch_size,  # SB3 n_steps
                sgd_minibatch_size=64,  # type: ignore SB3 batch_size
                num_sgd_iter=10,  # type: ignore SB3 n_epochs
                gamma=0.99,  # SB3 gamma
                lambda_=0.95,  # type: ignore # SB3 gae_lambda
                clip_param=0.2,  # type: ignore SB3 clip_range,
                vf_clip_param=np.inf,  # type: ignore SB3 equivalent to clip_range_vf=None
                use_gae=True,  # type: ignore SB3 normalize_advantage
                entropy_coeff=0.01,  # type: ignore SB3 ent_coef
                vf_loss_coeff=0.5,  # type: ignore SB3 vf_coef
                grad_clip=0.5,  # SB3 max_grad_norm TODO
                # kl_target=0.00001,  # SB3 target_kl
            )
            .experimental(
                _enable_new_api_stack=False
            )  # TODO Remove after migrating from ModelV2 to RL Module
            .debugging(
                seed=env_config["seed"],
            )
            .reporting(metrics_num_episodes_for_smoothing=1)
        )
        if self.env_config["enable_evaluation"]:
            algo_config.evaluation(
                evaluation_interval=np.rint(
                    env_config["episode_evaluation_freq"]
                    / self.eps_per_iteration
                ).astype(
                    int
                ),  # Convert to iterations
                evaluation_duration=env_config["number_evaluation_episodes"],
                evaluation_duration_unit="episodes",
                evaluation_config={
                    "explore": False,
                    "env_config": dict(
                        env_config,
                        initial_episode_number=env_config[
                            "eval_initial_env_episode"
                        ],
                        max_episode_number=(
                            env_config["eval_initial_env_episode"]
                            + env_config["number_evaluation_episodes"]
                        ),
                    ),
                },
                always_attach_evaluation_results=True,
            )
        algo_config["model"]["fcnet_hiddens"] = [
            64,
            64,
        ]  # Set neural network size

        return algo_config

    def action_mask_policy(self):
        config = PPOConfig.overrides(
            model={
                "custom_model": TorchActionMaskModel,
                "custom_action_dist": "masked_gaussian",
            },
        )
        return PolicySpec(config=config)

    def policy_mapping_fn(self, agent_id, episode=None, worker=None, **kwargs):
        agent_idx = int(agent_id.partition("_")[2])

        return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"

    def load(self, agent_name, scenario, method="last") -> None:
        analysis = tune.ExperimentAnalysis(
            f"{self.read_checkpoint}/{scenario}/{agent_name}/"
        )
        assert analysis.trials is not None, "Analysis trial is None"
        if method == "last":
            checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
        elif method == "best":
            checkpoint = analysis.get_best_checkpoint(
                analysis.trials[0], "episode_reward_mean", "max"
            )
        elif isinstance(method, int):
            raise NotImplementedError(
                "Checkpoint by iteration not implemented"
            )
        else:
            raise ValueError(f"Invalid method {method} for finetune load")
        assert checkpoint is not None, "Ray checkpoint is None"
        self.algo = Algorithm.from_checkpoint(checkpoint)

    def step(self, obs):
        action = {}
        assert isinstance(obs, dict), "Observations must be a dictionary."
        assert isinstance(
            self.algo, Algorithm
        ), "Algorithm must be an instance of Algorithm."
        for agent_id, agent_obs in obs.items():
            policy_id = self.policy_mapping_fn(agent_id)
            action[agent_id] = self.algo.compute_single_action(
                agent_obs,
                policy_id=policy_id,
                explore=False,
            )
        return action

    def train_alternative(self, total_timesteps: int):
        # Total timesteps is not used in this implementation
        # it is just a placeholder to keep the same interface as SB3
        algo_config = self.gen_config(self.env_config)
        self.algo = algo_config.build()
        total_episodes_train = (
            self.env_config["max_training_episodes"]
            - self.env_config["initial_training_episode"]
        ) * self.env_config["training_epochs"]
        train_iterations = np.ceil(
            total_episodes_train / self.eps_per_iteration
        ).astype(int)
        checkpoint_iter_freq = np.rint(
            self.env_config["checkpoint_episode_freq"] / self.eps_per_iteration
        ).astype(int)
        for it_idx in tqdm(np.arange(train_iterations), desc="Training..."):
            result = self.algo.train()
            print(
                f"\nIteration {it_idx + 1}/{train_iterations}: time {result['time_this_iter_s']:.2f}s"
            )
            print(
                pretty_print(result["sampler_results"]["policy_reward_mean"])
            )

            if it_idx % checkpoint_iter_freq == 0:
                checkpoint = self.algo.save(
                    f"{self.read_checkpoint}/{self.env_config['scenario']}/{self.env_config['agent']}"
                )
                print(f"Checkpoint saved at {checkpoint}")
        # Save final checkpoint
        checkpoint = self.algo.save(
            f"{self.read_checkpoint}/{self.env_config['scenario']}/{self.env_config['agent']}"
        )
