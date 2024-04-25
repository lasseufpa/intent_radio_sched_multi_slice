from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2

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
        restore: bool = False,
        param_config_mode: str = "default",
        param_config_scenario: Optional[str] = None,
        param_config_agent: Optional[str] = None,
        stochastic_policy: bool = False,
    ):
        ray.init(local_mode=debug_mode)
        register_env("marl_comm_env", lambda config: env_creator(config, True))
        ModelCatalog.register_custom_action_dist(
            "masked_gaussian", TorchDiagGaussian
        )

        self.stochastic_policy = stochastic_policy
        self.restore = restore
        self.env_config = env_config
        self.agent = None
        self.enable_masks = enable_masks
        self.read_checkpoint = str(Path("./ray_results/").resolve())
        self.algo = None
        self.steps_per_episode = 1000

        if "hyperparam_opt" in env_config["scenario"]:
            self.hyperparam_bounds = {
                # hyperparameter bounds based on SB-Zoo https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
                "lr": [0.0001, 0.1],
                "sgd_minibatch_size": [8, 512],
                "train_batch_size": [128, 2048],
                "gamma": [0.5, 0.9999],
                "num_sgd_iter": [1, 20],
            }
            self.initial_hyperparam = {
                "lr": tune.uniform(
                    self.hyperparam_bounds["lr"][0],
                    self.hyperparam_bounds["lr"][1],
                ),
                "sgd_minibatch_size": tune.randint(
                    self.hyperparam_bounds["sgd_minibatch_size"][0],
                    self.hyperparam_bounds["sgd_minibatch_size"][1],
                ),
                "train_batch_size": tune.sample_from(
                    lambda config: np.random.randint(
                        config["sgd_minibatch_size"],  # type: ignore
                        self.hyperparam_bounds["train_batch_size"][1],
                    )
                ),
                "gamma": tune.uniform(
                    self.hyperparam_bounds["gamma"][0],
                    self.hyperparam_bounds["gamma"][1],
                ),
                "num_sgd_iter": tune.randint(
                    self.hyperparam_bounds["num_sgd_iter"][0],
                    self.hyperparam_bounds["num_sgd_iter"][1],
                ),
            }
            self.pertubation_interval = 2
            self.num_samples = 50
        else:
            self.initial_hyperparam = None
            self.hyperparam_bounds = {}
            self.pertubation_interval = 0
            self.num_samples = 0

        if param_config_mode == "default":
            self.param_config = {
                "lr": 0.0003,
                "train_batch_size": 2048,
                "sgd_minibatch_size": 64,
                "num_sgd_iter": 10,
                "gamma": 0.99,
            }
        elif param_config_mode == "checkpoint":
            self.param_config = self.load_config(
                param_config_agent, param_config_scenario
            )
        elif param_config_mode == "pre_computed":
            # Computed using hyperparam_opt_mult_slice scenario
            self.param_config = {
                "lr": 0.019999999552965164,
                "train_batch_size": 10240,
                "sgd_minibatch_size": 8,
                "num_sgd_iter": 30,
                "gamma": 0.5,
            }
        else:
            raise ValueError(
                f"Invalid param_config_mode: {param_config_mode}."
            )

        self.eps_per_iteration = np.rint(
            self.param_config["train_batch_size"] // self.steps_per_episode
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
        tune_config = None
        if "hyperparam_opt" in self.env_config["scenario"]:

            def explore(config):
                # ensure we collect enough timesteps to do sgd
                if config["train_batch_size"] < config["sgd_minibatch_size"]:
                    config["train_batch_size"] = config["sgd_minibatch_size"]
                # ensure we run at least one sgd iter
                if config["num_sgd_iter"] < 1:
                    config["num_sgd_iter"] = 1
                return config

            pb2 = PB2(
                time_attr="training_iteration",
                perturbation_interval=self.pertubation_interval,
                hyperparam_bounds=self.hyperparam_bounds,
                custom_explore_fn=explore,
            )
            tune_config = tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pb2,
                num_samples=self.num_samples,
            )

        if self.restore:
            tuner = tune.Tuner.restore(
                f"{self.read_checkpoint}/{self.env_config['scenario']}/{self.env_config['agent']}/",
                trainable="PPO",
                param_space=algo_config.to_dict(),
            )
            results = tuner.fit()
            print(results)
        else:
            results = tune.Tuner(
                "PPO",
                param_space=algo_config.to_dict(),
                tune_config=tune_config,
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
            .training(
                lr=(
                    self.param_config["lr"]
                    if self.initial_hyperparam is None
                    else self.initial_hyperparam["lr"]
                ),  # SB3 LR
                train_batch_size=(
                    self.param_config["train_batch_size"]
                    if self.initial_hyperparam is None
                    else self.initial_hyperparam["train_batch_size"]
                ),  # SB3 n_steps
                sgd_minibatch_size=(  # type: ignore SB3 batch_size
                    self.param_config["sgd_minibatch_size"]
                    if self.initial_hyperparam is None
                    else self.initial_hyperparam["sgd_minibatch_size"]
                ),
                num_sgd_iter=(  # type: ignore SB3 n_epochs
                    self.param_config["num_sgd_iter"]
                    if self.initial_hyperparam is None
                    else self.initial_hyperparam["num_sgd_iter"]
                ),
                gamma=(
                    self.param_config["gamma"]
                    if self.initial_hyperparam is None
                    else self.initial_hyperparam["gamma"]
                ),  # SB3 gamma
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
            .callbacks(UpdatePolicyCallback)
        )

        if not ("hyperparam" in self.env_config["scenario"]):
            # Uses the GPU in case not doing hyperparameter optimization
            algo_config.resources(
                num_gpus=1,
                num_gpus_per_worker=1,
                num_gpus_per_learner_worker=1,
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
                    "explore": self.stochastic_policy,
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

    def load(
        self, agent_name, scenario, method="last", finetune=False
    ) -> None:
        if not finetune:
            analysis = tune.ExperimentAnalysis(
                f"{self.read_checkpoint}/{scenario}/{agent_name}/"
            )
            assert analysis.trials is not None, "Analysis trial is None"
            if method == "last":
                checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
            elif method == "best":
                checkpoint = analysis.get_best_checkpoint(
                    analysis.trials[0], "evaluation/episode_reward_mean", "max"
                )
            elif isinstance(method, int):  # TODO check if correct
                raise NotImplementedError(
                    "Checkpoint by iteration not implemented"
                )
            elif isinstance(method, int):
                raise NotImplementedError(
                    "Checkpoint by iteration not implemented"
                )
            else:
                raise ValueError(f"Invalid method {method} for finetune load")
            assert checkpoint is not None, "Ray checkpoint is None"
            self.algo = Algorithm.from_checkpoint(checkpoint)

    def load_config(self, agent_name, scenario) -> dict:
        hyperparameters = [
            "lr",
            "sgd_minibatch_size",
            "train_batch_size",
            "gamma",
            "num_sgd_iter",
        ]
        analysis = tune.ExperimentAnalysis(
            f"{self.read_checkpoint}/{scenario}/{agent_name}/"
        )
        assert analysis.trials is not None, "Analysis trial is None"
        config = analysis.get_best_config(
            metric="episode_reward_mean", mode="max"
        )
        assert isinstance(config, dict), "Config is not a dictionary"
        selected_config = {key: config[key] for key in hyperparameters}

        return selected_config

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
                explore=self.stochastic_policy,
            )
        return action


class UpdatePolicyCallback(DefaultCallbacks):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """
        Loading previous Policy weights from a checkpoint to fine-tune the model.
        """
        agent = algorithm.config.env_config["agent"]
        if "finetune" in agent:
            root_path = (
                algorithm.config.env_config["root_path"] + "/ray_results"
            )
            base_agent = algorithm.config.env_config["base_agent"]
            base_scenario = algorithm.config.env_config["base_scenario"]
            load_method = algorithm.config.env_config["load_method"]
            policies = self.load(
                root_path, base_agent, base_scenario, load_method
            )
            inter_slice_weights = policies["inter_slice_sched"].get_weights()
            intra_slice_weights = policies["intra_slice_sched"].get_weights()
            loaded_policy_weights = {
                "inter_slice_sched": inter_slice_weights,
                "intra_slice_sched": intra_slice_weights,
            }
            algorithm.set_weights(loaded_policy_weights)

    def load(self, root_path, agent_name, scenario, method="last") -> dict:
        analysis = tune.ExperimentAnalysis(
            f"{root_path}/{scenario}/{agent_name}/"
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
        policy = Policy.from_checkpoint(checkpoint)
        assert isinstance(policy, dict), "Policy is not a dictionary"
        return policy
