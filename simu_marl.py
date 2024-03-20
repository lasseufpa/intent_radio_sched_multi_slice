from os import getcwd
from pathlib import Path

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from tqdm import tqdm

from agents.action_mask_model import TorchActionMaskModel
from agents.ib_sched import IBSched
from agents.masked_action_distribution import TorchDiagGaussian
from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_seq import MultSliceAssociationSeq
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

read_checkpoint = str(Path("./ray_results/").resolve())
training_flag = True  # False for reading from checkpoint
debug_mode = (
    True  # When true executes in a local mode where GPU cannot be used
)
agents_name = [
    "ib_sched",
    # "ib_sched_mask",
    # "ib_sched_deepmind",
    # "ib_sched_mask_deepmind",
]
env_config = {
    "seed": 10,
    "seed_test": 15,
    "agent_class": IBSched,
    "channel_class": MimicQuadriga,  # QuadrigaChannel,
    "traffic_class": MultSliceTraffic,
    "mobility_class": SimpleMobility,
    "association_class": MultSliceAssociationSeq,  # TODO MultSliceAssociation,
    "scenario": "mult_slice_seq",
    "root_path": str(getcwd()),
    "training_episodes": 70,  # TODO 100 * 100,
    "training_epochs": 10,
    "testing_episodes": 30,  # TODO 1000,
    "evaluation_interval": 10000000,  # TODO large value to avoid evaluation  # based on number of training iterations (batch size)
    "evaluation_duration": 1,  # Unit defined by evaluation_duration_unit
    "evaluation_duration_unit": "episodes",
    "evaluation_initial_episode": 100 * 100,
    "max_episode_number": 70,
}

ray.init(local_mode=debug_mode)


def env_creator(env_config):
    marl_comm_env = MARLCommEnv(
        env_config["channel_class"],
        env_config["traffic_class"],
        env_config["mobility_class"],
        env_config["association_class"],
        env_config["scenario"],
        env_config["agent"],
        env_config["seed"],
        root_path=env_config["root_path"],
        initial_episode_number=(
            env_config["initial_episode_number"]
            if "initial_episode_number" in env_config.keys()
            else 0
        ),
        max_episode_number=(
            env_config["max_episode_number"]
            if "max_episode_number" in env_config.keys()
            else None
        ),
    )
    agent = env_config["agent_class"](
        marl_comm_env,
        marl_comm_env.comm_env.max_number_ues,
        marl_comm_env.comm_env.max_number_slices,
        marl_comm_env.comm_env.max_number_basestations,
        marl_comm_env.comm_env.num_available_rbs,
    )
    marl_comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )

    return marl_comm_env


ModelCatalog.register_custom_action_dist("masked_gaussian", TorchDiagGaussian)


def action_mask_policy():
    config = SACConfig.overrides(
        model={
            "custom_model": TorchActionMaskModel,
            "custom_action_dist": "masked_gaussian",
        },
    )
    return PolicySpec(config=config)


# Ray RLlib
register_env("marl_comm_env", lambda config: env_creator(config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    agent_idx = int(agent_id.partition("_")[2])

    return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"


for agent in agents_name:
    env_config["agent"] = agent
    using_mask = "mask" in agent
    using_deepmind = "deepmind" in agent

    # Training
    if training_flag:
        algo_config = (
            SACConfig()  # TODO
            .environment(
                env="marl_comm_env",
                env_config=env_config,
                is_atari=False,
                disable_env_checking=True,  # Avoid stepping for test and skipping first ep
                # clip_rewards=False,
                # normalize_actions=True,
                # clip_actions=False,
            )
            .multi_agent(
                policies={
                    "inter_slice_sched": (
                        action_mask_policy() if using_mask else PolicySpec()
                    ),
                    "intra_slice_sched": PolicySpec(),
                },
                policy_mapping_fn=policy_mapping_fn,
                count_steps_by="env_steps",
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=0,
                enable_connectors=False,
                num_envs_per_worker=1,
                preprocessor_pref="deepmind" if using_deepmind else None,
            )
            .resources(
                num_gpus=1,
                num_gpus_per_worker=1,
                num_gpus_per_learner_worker=1,
            )
            .training(
                # lr=0.0003,  # SB3 LR
                # gamma=0.99,  # SB3 gamma
                # tau=5e-3,  # type: ignore # SB3 tau
                # target_network_update_freq=1,  # type: ignore # SB3 target_network_update_freq,
                train_batch_size=256,  # type: ignore # SB3 train_batch_size
                training_intensity=256 / 10,  # 256 * 4,  # type: ignore
            )
            .experimental(
                _enable_new_api_stack=False
            )  # TODO Remove after migrating from ModelV2 to RL Module
            .evaluation(
                evaluation_interval=env_config["evaluation_interval"],
                evaluation_duration=env_config["evaluation_duration"],
                evaluation_duration_unit=env_config[
                    "evaluation_duration_unit"
                ],
                evaluation_config={
                    "explore": False,
                    "env_config": dict(
                        env_config,
                        initial_episode_number=env_config[
                            "evaluation_initial_episode"
                        ],
                        max_episode_number=env_config[
                            "evaluation_initial_episode"
                        ]
                        + env_config["evaluation_duration"],
                    ),
                },
                always_attach_evaluation_results=True,
            )
            .debugging(
                seed=env_config["seed"],
            )
        )
        algo_config["min_sample_timesteps_per_iteration"] = 2000
        stop = {
            "episodes_total": env_config["training_episodes"]
            * env_config["training_epochs"],
        }
        results = tune.Tuner(
            "SAC",  # TODO
            param_space=algo_config.to_dict(),
            run_config=air.RunConfig(
                storage_path=f"{read_checkpoint}/{env_config['scenario']}/",
                name=env_config["agent"],
                stop=stop,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=3,  # TODO SAC
                    checkpoint_at_end=True,
                ),
            ),
        ).fit()

    # Testing
    analysis = tune.ExperimentAnalysis(
        f"{read_checkpoint}/{env_config['scenario']}/{env_config['agent']}/"
    )
    assert analysis.trials is not None, "Analysis trial is None"
    best_checkpoint = analysis.get_best_checkpoint(
        analysis.trials[0], "episode_reward_mean", "max"
    )
    assert best_checkpoint is not None, "Best checkpoint is None"
    last_checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
    assert last_checkpoint is not None, "Last checkpoint is None"
    algo = Algorithm.from_checkpoint(last_checkpoint)
    marl_comm_env = env_creator(env_config)
    marl_comm_env.comm_env.max_number_episodes = (
        env_config["testing_episodes"] + env_config["training_episodes"]
    )
    obs, _ = marl_comm_env.reset(
        seed=env_config["seed_test"],
        options={"initial_episode": env_config["training_episodes"]},
    )
    for step in tqdm(
        np.arange(
            marl_comm_env.comm_env.max_number_steps
            * env_config["testing_episodes"]
        ),
        desc="Testing...",
    ):
        action = {}
        assert isinstance(obs, dict), "Observation must be a dict"
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            action[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id=policy_id,
                explore=False,
            )
        obs, reward, terminated, truncated, info = marl_comm_env.step(action)
        assert isinstance(terminated, dict), "Termination must be a dict"
        if terminated["__all__"]:
            obs, _ = marl_comm_env.reset()

ray.shutdown()
