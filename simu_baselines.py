from os import getcwd

import numpy as np
from tqdm import tqdm

from agents.mapf import MAPF
from agents.marr import MARR
from agents.sb3_sched import IBSchedSB3
from agents.sched_colran import SchedColORAN
from agents.sched_twc import SchedTWC
from associations.mult_slice import MultSliceAssociation
from associations.mult_slice_seq import MultSliceAssociationSeq
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import MARLCommEnv
from traffics.mult_slice import MultSliceTraffic

scenarios = {
    "mult_slice_seq": MultSliceAssociationSeq,
    "mult_slice": MultSliceAssociation,
    "mult_slice_test_on_trained": MultSliceAssociation,
}
agents = {
    "sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": False,
    },
    "sched_twc": {
        "class": SchedTWC,
        "rl": True,
        "train": True,
    },
    "sched_coloran": {
        "class": SchedColORAN,
        "rl": True,
        "train": True,
    },
    "mapf": {"class": MAPF, "rl": False, "train": False},
    "marr": {"class": MARR, "rl": False, "train": False},
}
env_config_scenarios = {
    "mult_slice_seq": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": MimicQuadriga,  # QuadrigaChannelSeq,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 10,
        "enable_evaluation": False,
        "initial_training_episode": 0,
        "max_training_episodes": 70,
        "initial_testing_episode": 70,
        "test_episodes": 30,
        "episode_evaluation_freq": None,
        "number_evaluation_episodes": None,
        "checkpoint_episode_freq": None,
        "eval_initial_env_episode": None,
    },
    "mult_slice": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": MimicQuadriga,  # QuadrigaChannel,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 20,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 80,  # 80 different scenarios with 1 channel episodes each
        "initial_testing_episode": 80,
        "test_episodes": 20,  # Testing on 20 different unseen scenarios
        "episode_evaluation_freq": 80,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 80,
    },
    "mult_slice_test_on_trained": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": MimicQuadriga,  # QuadrigaChannel,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 20,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 80,  # 80 different scenarios with 1 channel episodes each
        "initial_testing_episode": 0,
        "test_episodes": 80,  # Testing on 80 different seen scenarios
        "episode_evaluation_freq": 80,
        "number_evaluation_episodes": 80,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 0,
    },
}


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
        initial_episode_number=env_config["initial_training_episode"],
    )
    marl_comm_env.comm_env.max_number_episodes = (
        env_config["initial_training_episode"]
        + env_config["max_training_episodes"]
    )
    eval_env = MARLCommEnv(
        env_config["channel_class"],
        env_config["traffic_class"],
        env_config["mobility_class"],
        env_config["association_class"],
        env_config["scenario"],
        env_config["agent"],
        env_config["seed"],
        root_path=env_config["root_path"],
    )
    if env_config["rl"]:
        agent = env_config["agent_class"](
            marl_comm_env,
            marl_comm_env.comm_env.max_number_ues,
            marl_comm_env.comm_env.max_number_slices,
            marl_comm_env.comm_env.max_number_basestations,
            marl_comm_env.comm_env.num_available_rbs,
            eval_env if env_config["enable_evaluation"] else None,
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

    return marl_comm_env, agent


for scenario in scenarios.keys():
    for agent_name in agents.keys():
        env_config = env_config_scenarios[scenario]
        env_config["agent"] = agent_name
        env_config["scenario"] = scenario
        env_config["agent_class"] = agents[agent_name]["class"]
        env_config["association_class"] = scenarios[scenario]
        env_config["rl"] = agents[agent_name]["rl"]

        marl_comm_env, agent = env_creator(env_config)

        # Training
        number_episodes = (
            marl_comm_env.comm_env.max_number_episodes
            - env_config["initial_training_episode"]
        )
        steps_per_episode = marl_comm_env.comm_env.max_number_steps
        total_time_steps = (
            number_episodes * steps_per_episode * env_config["training_epochs"]
        )
        if agents[agent_name]["rl"]:
            if agents[agent_name]["train"]:
                print(f"Training {agent_name} on {scenario} scenario")
                agent.train(total_time_steps)

            agent.load(
                f"./agents/models/{env_config['scenario']}/final_{env_config['agent']}.zip"
            )

        # Testing
        print(f"Testing {agent_name} on {scenario} scenario")
        total_test_steps = env_config["test_episodes"] * steps_per_episode
        marl_comm_env.comm_env.max_number_episodes = (
            env_config["initial_testing_episode"] + env_config["test_episodes"]
        )
        obs, _ = marl_comm_env.reset(
            seed=env_config["seed_test"],
            options={"initial_episode": env_config["initial_testing_episode"]},
        )
        for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
            action = agent.step(obs)
            obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
            assert isinstance(terminated, bool), "Terminated must be a boolean"
            if terminated:
                obs, _ = marl_comm_env.reset()
