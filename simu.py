from os import getcwd

import numpy as np
import ray
from tqdm import tqdm

from agents.ib_sched import IBSched
from agents.mapf import MAPF
from agents.marr import MARR
from agents.ray_agent import RayAgent
from agents.sb3_sched import IBSchedSB3
from agents.sched_colran import SchedColORAN
from agents.sched_twc import SchedTWC
from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_seq import MultSliceAssociationSeq
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannel
from channels.quadriga_seq import QuadrigaChannelSeq
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

scenarios = {
    # "hyperparam_opt_mult_slice": MultSliceAssociation,
    "mult_slice_seq": MultSliceAssociationSeq,
    # "mult_slice": MultSliceAssociation,
    # "finetune_mult_slice_seq": MultSliceAssociationSeq,
}
agents = {
    "sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": True,
        "load_method": "best",
    },
    "ray_ib_sched": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "stochastic_policy": False,
        "hyper_opt_algo": "asha",
        "param_config_mode": "pre_computed",
        "param_config_scenario": "hyperparam_opt_mult_slice",
        "param_config_agent": "ray_ib_sched_hyper_asha_0",
    },
    "ray_ib_sched_non_shared": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "stochastic_policy": False,
        "hyper_opt_algo": "asha",
        "param_config_mode": "pre_computed",
        "param_config_scenario": "hyperparam_opt_mult_slice",
        "param_config_agent": "ray_ib_sched_hyper_asha",
        "shared_policies": False,
    },
    "sched_twc": {
        "class": SchedTWC,
        "rl": True,
        "train": True,
        "load_method": "best",
    },
    "sched_coloran": {
        "class": SchedColORAN,
        "rl": True,
        "train": True,
        "load_method": "best",
    },
    "mapf": {"class": MAPF, "rl": False, "train": False},
    "marr": {"class": MARR, "rl": False, "train": False},
    "ray_ib_sched_hyper_asha": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "hyper_opt_enable": True,
        "hyper_opt_algo": "asha",
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "test": False,
        "restore": True,
    },
    "finetune_sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": True,
        "enable_finetune": True,
        "base_agent": "sb3_sched",
        "base_scenario": "mult_slice",
        "load_method": "best",
    },
    "finetune_sched_twc": {
        "class": SchedTWC,
        "rl": True,
        "train": True,
        "enable_finetune": True,
        "base_agent": "sched_twc",
        "base_scenario": "mult_slice",
        "load_method": "best",
    },
    "scratch_sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": True,
        "load_method": "best",
    },
    "finetune_sched_colran": {
        "class": SchedColORAN,
        "rl": True,
        "train": True,
        "enable_finetune": True,
        "base_agent": "sched_coloran",
        "base_scenario": "mult_slice",
        "load_method": "best",
    },
    "finetune_ray_ib_sched": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "enable_finetune": True,
        "base_agent": "ray_ib_sched",
        "base_scenario": "mult_slice",
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
    },
    "scratch_ray_ib_sched": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
    },
    "base_ray_ib_sched": {
        "class": IBSched,
        "rl": True,
        "train": False,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "base_agent": "ray_ib_sched",
        "base_scenario": "mult_slice",
    },
    "base_ray_ib_sched_non_shared": {
        "class": IBSched,
        "rl": True,
        "train": False,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "base_agent": "ray_ib_sched_non_shared",
        "base_scenario": "mult_slice",
        "shared_policies": False,
    },
    "scratch_ray_ib_sched_non_shared": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "load_method": "best",
        "enable_masks": True,
        "debug_mode": False,
        "shared_policies": False,
    },
    "finetune_ray_ib_sched_non_shared": {
        "class": IBSched,
        "rl": True,
        "train": True,
        "enable_finetune": True,
        "base_agent": "ray_ib_sched_non_shared",
        "base_scenario": "mult_slice",
        "load_method": "best",  # Could be "best", "last" or a int number
        "enable_masks": True,
        "debug_mode": False,
    },
}
env_config_scenarios = {
    "mult_slice_seq": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": QuadrigaChannelSeq,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 10,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 60,
        "initial_testing_episode": 80,
        "test_episodes": 20,
        "episode_evaluation_freq": 10,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 60,
        "save_hist": False,
        "agents": [
            "ray_ib_sched",
            "ray_ib_sched_non_shared",
            "sb3_sched",
            "sched_twc",
            "sched_coloran",
            "mapf",
            "marr",
        ],
        "number_scenarios": 10,
    },
    "mult_slice": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": QuadrigaChannel,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 10,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 160,
        "initial_testing_episode": 180,
        "test_episodes": 20,
        "episode_evaluation_freq": 10,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 160,
        "save_hist": False,
        "agents": [
            "ray_ib_sched",
            "ray_ib_sched_non_shared",
            "sb3_sched",
            "sched_twc",
            "sched_coloran",
            "mapf",
            "marr",
        ],
    },
    "hyperparam_opt_mult_slice": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": QuadrigaChannel,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 10,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 160,
        "initial_testing_episode": 180,
        "test_episodes": 20,
        "episode_evaluation_freq": 10,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 160,
        "save_hist": False,
        "agents": ["ray_ib_sched_hyper_asha"],
    },
    "finetune_mult_slice_seq": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": QuadrigaChannelSeq,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 5,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 60,  # 80 different channels from the same scenario
        "initial_testing_episode": 80,
        "test_episodes": 20,  # Testing on 20 channels from the same scenario
        "episode_evaluation_freq": 10,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 60,
        "save_hist": False,
        "agents": [
            "base_ray_ib_sched",
            "finetune_ray_ib_sched",
            "scratch_ray_ib_sched",
            "base_ray_ib_sched_non_shared",
            "finetune_ray_ib_sched_non_shared",
            "scratch_ray_ib_sched_non_shared",
            "finetune_sb3_sched",
            "finetune_sched_twc",
            "finetune_sched_colran",
            "mapf",
            "marr",
        ],
        "number_scenarios": 10,
    },
}


def env_creator(env_config, only_env=True):
    initial_episode_number = env_config.get(
        "initial_episode_number", env_config["initial_training_episode"]
    )
    max_episode_number = env_config.get(
        "max_episode_number", env_config["max_training_episodes"]
    )
    marl_comm_env = MARLCommEnv(
        env_config["channel_class"],
        env_config["traffic_class"],
        env_config["mobility_class"],
        env_config["association_class"],
        "mult_slice",
        env_config["agent"],
        env_config["seed"],
        root_path=env_config["root_path"],
        initial_episode_number=initial_episode_number,
        simu_name=env_config["scenario"],
        save_hist=env_config["save_hist"],
        max_episode_number=max_episode_number,
    )
    eval_env = MARLCommEnv(
        env_config["channel_class"],
        env_config["traffic_class"],
        env_config["mobility_class"],
        env_config["association_class"],
        "mult_slice",
        env_config["agent"],
        env_config["seed"],
        root_path=env_config["root_path"],
        simu_name=env_config["scenario"],
        save_hist=env_config["save_hist"],
    )
    if env_config["rl"]:
        agent = env_config["agent_class"](
            env=marl_comm_env,
            max_number_ues=marl_comm_env.comm_env.max_number_ues,
            max_number_slices=marl_comm_env.comm_env.max_number_slices,
            max_number_basestations=marl_comm_env.comm_env.max_number_basestations,
            num_available_rbs=marl_comm_env.comm_env.num_available_rbs,
            eval_env=eval_env if env_config["enable_evaluation"] else None,
            agent_name=env_config["agent"],
            seed=env_config["seed"],
            episode_evaluation_freq=env_config["episode_evaluation_freq"],
            number_evaluation_episodes=env_config[
                "number_evaluation_episodes"
            ],
            checkpoint_episode_freq=env_config["checkpoint_episode_freq"],
            eval_initial_env_episode=env_config["eval_initial_env_episode"],
        )
    else:
        agent = env_config["agent_class"](
            marl_comm_env,
            marl_comm_env.comm_env.max_number_ues,
            marl_comm_env.comm_env.max_number_slices,
            marl_comm_env.comm_env.max_number_basestations,
            marl_comm_env.comm_env.num_available_rbs,
            seed=env_config["seed"],
        )
    marl_comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )
    eval_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )
    agent.init_agent()

    if only_env:
        return marl_comm_env
    else:
        return marl_comm_env, agent


for scenario in scenarios.keys():
    for agent_name in env_config_scenarios[scenario]["agents"]:
        env_config = env_config_scenarios[scenario].copy()
        env_config["scenario"] = scenario
        env_config["agent_class"] = agents[agent_name]["class"]
        env_config["association_class"] = scenarios[scenario]
        env_config["rl"] = agents[agent_name]["rl"]
        enable_finetune = agents[agent_name].get("enable_finetune", False)
        env_config["enable_finetune"] = enable_finetune
        if enable_finetune:
            env_config["base_agent"] = agents[agent_name]["base_agent"]
            env_config["base_scenario"] = agents[agent_name]["base_scenario"]
            env_config["load_method"] = agents[agent_name]["load_method"]

        number_scenarios = env_config.get("number_scenarios", 1)
        for scenario_number in range(number_scenarios):
            env_config["agent"] = agent_name + f"_{str(scenario_number)}"
            marl_comm_env, agent = env_creator(env_config, False)  # type: ignore
            if "ray" in agent_name:
                param_config_mode = agents[agent_name].get(
                    "param_config_mode", "default"
                )
                param_config_scenario = agents[agent_name].get(
                    "param_config_scenario", None
                )
                param_config_agent = agents[agent_name].get(
                    "param_config_agent", None
                )
                restore = agents[agent_name].get("restore", False)
                stochastic_policy = agents[agent_name].get(
                    "stochastic_policy", False
                )
                hyper_opt_algo = agents[agent_name].get("hyper_opt_algo", None)
                hyper_opt_enable = agents[agent_name].get(
                    "hyper_opt_enable", False
                )
                shared_policies = agents[agent_name].get(
                    "shared_policies", True
                )
                agent = RayAgent(
                    env_creator=env_creator,
                    env_config=env_config,
                    debug_mode=agents[agent_name]["debug_mode"],
                    enable_masks=agents[agent_name]["enable_masks"],
                    param_config_mode=param_config_mode,
                    param_config_scenario=param_config_scenario,
                    param_config_agent=param_config_agent,
                    restore=restore,
                    stochastic_policy=stochastic_policy,
                    hyper_opt_algo=hyper_opt_algo,
                    hyper_opt_enable=hyper_opt_enable,
                    shared_policies=shared_policies,
                )
            number_episodes = (
                marl_comm_env.comm_env.max_number_episodes
                - env_config["initial_training_episode"]
            )
            steps_per_episode = marl_comm_env.comm_env.max_number_steps
            total_time_steps = (
                number_episodes
                * steps_per_episode
                * env_config["training_epochs"]
            )
            if agents[agent_name]["rl"]:
                if agents[agent_name]["train"]:
                    # Training
                    if enable_finetune:
                        print(
                            f"Fine-tuning model from Agent {agents[agent_name]['base_agent']} scenario {agents[agent_name]['base_scenario']} on {scenario} scenario"
                        )
                        agent.load(
                            agent_name=agents[agent_name]["base_agent"],
                            scenario=agents[agent_name]["base_scenario"],
                            method=agents[agents[agent_name]["base_agent"]][
                                "load_method"
                            ],
                            finetune=True,
                        )  # Loading base model
                    print(f"Training {agent_name} on {scenario} scenario")
                    agent.train(total_time_steps)

            enable_test = env_config.get("test", True)
            if enable_test:
                # Testing
                agent_load = (
                    agents[agent_name]["base_agent"]
                    if "base" in agent_name
                    else env_config["agent"]
                )
                scenario_load = (
                    agents[agent_name]["base_scenario"]
                    if "base" in agent_name
                    else scenario
                )
                if agents[agent_name]["rl"]:
                    agent.load(
                        agent_load,
                        scenario_load,
                        agents[agent_name]["load_method"],
                    )
                print(f"Testing {agent_name} on {scenario} scenario")
                total_test_steps = (
                    env_config["test_episodes"] * steps_per_episode
                )
                marl_comm_env.comm_env.max_number_episodes = (
                    env_config["initial_testing_episode"]
                    + env_config["test_episodes"]
                )
                marl_comm_env.comm_env.save_hist = (
                    True  # Save metrics for test
                )
                obs, _ = marl_comm_env.reset(
                    seed=env_config["seed_test"],
                    options={
                        "initial_episode": env_config[
                            "initial_testing_episode"
                        ]
                    },
                )
                for step in tqdm(
                    np.arange(total_test_steps), desc="Testing..."
                ):
                    action = agent.step(obs)
                    obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
                    if isinstance(terminated, dict):
                        terminated = terminated["__all__"]
                    assert isinstance(
                        terminated, bool
                    ), "Terminated must be a boolean"
                    if terminated:
                        obs, _ = marl_comm_env.reset()
            ray.shutdown()

            # Updating values for next scenario
            if "number_scenarios" in env_config.keys():
                scenario_episodes = (
                    (
                        env_config["max_training_episodes"]
                        - env_config["initial_training_episode"]
                    )
                    + env_config["test_episodes"]
                    + env_config["number_evaluation_episodes"]
                )
                env_config["initial_training_episode"] += scenario_episodes
                env_config["max_training_episodes"] += scenario_episodes
                env_config["initial_testing_episode"] += scenario_episodes
                env_config["eval_initial_env_episode"] += scenario_episodes
