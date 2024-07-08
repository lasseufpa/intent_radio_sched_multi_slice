from pathlib import Path
from random import choice
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
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler

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
        hyper_opt_algo: Optional[str] = None,
        hyper_opt_enable: bool = False,
        shared_policies: bool = True,
        number_rollout_workers: int = 0,
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
        self.number_rollout_workers = number_rollout_workers
        self.min_eps_iteration_checkpoint = 2
        self.hyper_opt_algo = hyper_opt_algo
        self.hyper_opt_enable = hyper_opt_enable
        self.shared_policies = shared_policies
        self.maximum_number_slices = 5
        self.net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
            "large": [256, 256, 256],
            "verybig": [512, 512, 512],
        }

        # Hyperparameter optimizer configuration
        if self.hyper_opt_algo == "asha":
            self.num_samples = 200
            self.max_t = 800
            self.time_attr = "episodes_total"
            self.grace_period = 50
            self.reduction_factor = 3
            self.brackets = 1
            train_batch_size_options = np.array(
                [
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                ]
            )
            self.initial_hyperparam = {
                "lr": tune.loguniform(
                    1e-5,
                    0.01,
                ),
                "sgd_minibatch_size": tune.choice(
                    [8, 16, 32, 64, 128, 256, 512]
                ),
                "train_batch_size": tune.sample_from(
                    lambda config: np.random.choice(
                        [
                            config["sgd_minibatch_size"] * 2**i
                            for i in range(
                                train_batch_size_options.shape[0]
                                - (
                                    train_batch_size_options
                                    == config["sgd_minibatch_size"]
                                ).nonzero()[0][0],
                            )
                        ],
                    )
                ),
                "gamma": tune.choice(
                    [
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        0.95,
                        0.98,
                        0.99,
                        0.995,
                        0.999,
                        0.9999,
                    ]
                ),
                "num_sgd_iter": tune.choice([1, 5, 10, 20]),
                "lambda": tune.choice([0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
                "net_arch": tune.sample_from(
                    lambda config: choice(
                        [
                            self.net_arch["small"],
                            self.net_arch["medium"],
                            self.net_arch["big"],
                            self.net_arch["large"],
                            self.net_arch["verybig"],
                        ]
                    ),
                ),
            }
        elif self.hyper_opt_algo is None:
            self.initial_hyperparam = None
        else:
            raise ValueError(f"Invalid hyper_opt_algo: {self.hyper_opt_algo}.")

        # Initial hyperparameter loading in case not using hyperparameter optimization
        if param_config_mode == "default":
            self.param_config = {
                "lr": 0.0003,
                "train_batch_size": 2048,
                "sgd_minibatch_size": 64,
                "num_sgd_iter": 10,
                "gamma": 0.99,
                "lambda": 0.95,
                "net_arch": self.net_arch["small"],
            }
        elif param_config_mode in [
            "checkpoint",
            "checkpoint_avg",
            "checkpoint_avg_peaks",
        ]:
            self.param_config = self.load_config(
                param_config_mode, param_config_agent, param_config_scenario
            )
        elif param_config_mode == "pre_computed":
            # Computed using hyperparam_opt_mult_slice scenario
            self.param_config = {
                "lr": 0.0066760717960586274,
                "sgd_minibatch_size": 32,
                "train_batch_size": 128,
                "gamma": 0.98,
                "num_sgd_iter": 5,
                "lambda": 0.95,
                "net_arch": self.net_arch["small"],
            }
        else:
            raise ValueError(
                f"Invalid param_config_mode: {param_config_mode}."
            )

        self.eps_per_iteration = (
            self.param_config["train_batch_size"] // self.steps_per_episode
            if self.param_config["train_batch_size"] >= self.steps_per_episode
            else self.param_config["train_batch_size"] / self.steps_per_episode
        )

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

        # Whether to use a hyperparameter opt algo
        if self.hyper_opt_enable:
            if self.hyper_opt_algo == "asha":
                asha = AsyncHyperBandScheduler(
                    time_attr=self.time_attr,
                    grace_period=self.grace_period,
                    max_t=self.max_t,
                    reduction_factor=self.reduction_factor,
                    brackets=self.brackets,
                    stop_last_trials=True,
                )
                tune_config = tune.TuneConfig(
                    metric="evaluation/episode_reward_mean",
                    mode="max",
                    scheduler=asha,
                    num_samples=self.num_samples,
                )
            elif self.hyper_opt_algo is None:
                tune_config = None
            else:
                raise ValueError(
                    f"Invalid hyper_opt_algo: {self.hyper_opt_algo}."
                )
        else:
            tune_config = None

        # Whether to restore from previous experiment
        if self.restore and tune.Tuner.can_restore(
            f"{self.read_checkpoint}/{self.env_config['scenario']}/{self.env_config['agent']}/"
        ):
            tuner = tune.Tuner.restore(
                f"{self.read_checkpoint}/{self.env_config['scenario']}/{self.env_config['agent']}/",
                trainable="PPO",
                param_space=algo_config.to_dict(),
                restart_errored=False,
                resume_errored=True,
                resume_unfinished=True,
            )
        else:
            tuner = tune.Tuner(
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
            )
        results = tuner.fit()
        print(results)

    def gen_config(self, env_config):
        algo_config = (
            PPOConfig()
            .environment(
                env="marl_comm_env",
                env_config=env_config,
                is_atari=False,
                disable_env_checking=True,
            )
            .multi_agent(
                policies=self.generate_policies(
                    self.shared_policies, self.maximum_number_slices
                ),
                policy_mapping_fn=(
                    self.policy_mapping_fn_shared
                    if self.shared_policies
                    else self.policy_mapping_fn_non_shared
                ),
                count_steps_by="env_steps",
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=self.number_rollout_workers,
                enable_connectors=False,
                num_envs_per_worker=1,
            )
            .training(
                lr=(
                    self.param_config["lr"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["lr"]
                ),  # SB3 LR
                train_batch_size=(
                    self.param_config["train_batch_size"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["train_batch_size"]
                ),  # SB3 n_steps
                sgd_minibatch_size=(  # type: ignore SB3 batch_size
                    self.param_config["sgd_minibatch_size"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["sgd_minibatch_size"]
                ),
                num_sgd_iter=(  # type: ignore SB3 n_epochs
                    self.param_config["num_sgd_iter"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["num_sgd_iter"]
                ),
                gamma=(
                    self.param_config["gamma"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["gamma"]
                ),  # SB3 gamma
                lambda_=(  # type: ignore # SB3 gae_lambda
                    self.param_config["lambda"]
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else self.initial_hyperparam["lambda"]
                ),
                model=(
                    {
                        "fcnet_hiddens": self.param_config["net_arch"],
                    }
                    if not self.hyper_opt_enable
                    or self.initial_hyperparam is None
                    else {"fcnet_hiddens": self.initial_hyperparam["net_arch"]}
                ),
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

        return algo_config

    def action_mask_policy(self):
        config = PPOConfig.overrides(
            model={
                "custom_model": TorchActionMaskModel,
                "custom_action_dist": "masked_gaussian",
            },
        )
        return PolicySpec(config=config)

    def generate_policies(
        self, shared_policies: bool, max_number_slices: int
    ) -> Dict[str, PolicySpec]:
        if shared_policies:
            return {
                "inter_slice_sched": (
                    self.action_mask_policy()
                    if self.enable_masks
                    else PolicySpec()
                ),
                "intra_slice_sched": PolicySpec(),
            }
        else:
            policies = {
                "inter_slice_sched": (
                    self.action_mask_policy()
                    if self.enable_masks
                    else PolicySpec()
                ),
            }
            for i in range(max_number_slices):
                policies[f"intra_slice_sched_{i}"] = PolicySpec()
            return policies

    def policy_mapping_fn_shared(
        self, agent_id, episode=None, worker=None, **kwargs
    ):
        agent_idx = int(agent_id.partition("_")[2])

        return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"

    def policy_mapping_fn_non_shared(
        self, agent_id, episode=None, worker=None, **kwargs
    ):
        agent_idx = int(agent_id.partition("_")[2])

        return (
            "inter_slice_sched"
            if agent_idx == 0
            else f"intra_slice_sched_{agent_idx-1}"
        )

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
                    analysis.trials[0],
                    "evaluation/policy_reward_mean/inter_slice_sched",
                    "max",
                )
            elif method == "best_train":
                checkpoint = analysis.get_best_checkpoint(
                    analysis.trials[0],
                    "policy_reward_mean/inter_slice_sched",
                    "max",
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

    def load_config(self, mode, agent_name, scenario) -> dict:
        metric = "evaluation/policy_reward_mean/inter_slice_sched"
        assert isinstance(
            self.initial_hyperparam, dict
        ), "Initial hyperparam is not a dictionary"
        hyperparameters = list(self.initial_hyperparam.keys())
        hyperparameters.remove("net_arch")
        analysis = tune.ExperimentAnalysis(
            f"{self.read_checkpoint}/{scenario}/{agent_name}/"
        )
        assert analysis.trials is not None, "Analysis trial is None"
        if mode == "checkpoint":
            config = analysis.get_best_config(metric=metric, mode="max")
        elif mode == "checkpoint_avg":
            trial_dfs = analysis.trial_dataframes
            trials_avg = {}
            for trial_name in trial_dfs.keys():
                if metric in trial_dfs[trial_name].columns:
                    trial_df = (
                        trial_dfs[trial_name][metric].dropna().to_numpy()
                    )
                    if trial_df.shape[0] >= 10:
                        trials_avg[trial_name] = np.mean(trial_df)
            best_trial_name = max(trials_avg, key=lambda key: trials_avg[key])
            config = analysis.get_all_configs()[best_trial_name]
        elif mode == "checkpoint_avg_peaks":
            peaks_number = 10
            trial_dfs = analysis.trial_dataframes
            trials_avg = {}
            for trial_name in trial_dfs.keys():
                if metric in trial_dfs[trial_name].columns:
                    trial_df = (
                        trial_dfs[trial_name][metric].dropna().to_numpy()
                    )
                    if trial_df.shape[0] >= peaks_number:
                        trial_df = np.sort(np.unique(trial_df))[-peaks_number:]
                        trials_avg[trial_name] = np.mean(trial_df)
            best_trial_name = max(trials_avg, key=lambda key: trials_avg[key])
            config = analysis.get_all_configs()[best_trial_name]
        else:
            raise ValueError(f"Invalid mode {mode} for load_config")
        assert isinstance(config, dict), "Config is not a dictionary"
        selected_config = {key: config[key] for key in hyperparameters}
        selected_config["net_arch"] = config["model"]["fcnet_hiddens"]

        return selected_config

    def step(self, obs):
        action = {}
        assert isinstance(obs, dict), "Observations must be a dictionary."
        assert isinstance(
            self.algo, Algorithm
        ), "Algorithm must be an instance of Algorithm."
        for agent_id, agent_obs in obs.items():
            policy_id = (
                self.policy_mapping_fn_shared(agent_id)
                if self.shared_policies
                else self.policy_mapping_fn_non_shared(agent_id)
            )
            action[agent_id] = self.algo.compute_single_action(
                agent_obs,
                policy_id=policy_id,
                explore=self.stochastic_policy,
            )
        return action

    @staticmethod
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"]:
            config["train_batch_size"] = config["sgd_minibatch_size"]
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config


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
        enable_finetune = algorithm.config.env_config.get(
            "enable_finetune", False
        )
        if enable_finetune:
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
                analysis.trials[0],
                "evaluation/policy_reward_mean/inter_slice_sched",
                "max",
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
