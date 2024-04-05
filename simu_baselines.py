from os import getcwd

import numpy as np
from tqdm import tqdm

from agents.ib_sched import IBSched
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
    # "mult_slice_seq": MultSliceAssociationSeq,
    # "mult_slice": MultSliceAssociation,
    # "mult_slice_test_on_trained": MultSliceAssociation,
    "finetune_mult_slice_seq": MultSliceAssociationSeq,
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
        "load_method": "last",
    },
    "mapf": {"class": MAPF, "rl": False, "train": False},
    "marr": {"class": MARR, "rl": False, "train": False},
    "finetune_sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": True,
        "base_agent": "sb3_sched",
        "base_scenario": "mult_slice",
        "load_method": 50000,  # Could be "best", "last" or a int number
    },
    "finetune_sched_twc": {
        "class": SchedTWC,
        "rl": True,
        "train": True,
        "base_agent": "sched_twc",
        "base_scenario": "mult_slice",
        "load_method": "last",  # Could be "best", "last" or a int number
    },
    "scratch_sb3_sched": {
        "class": IBSchedSB3,
        "rl": True,
        "train": True,
        "load_method": 50000,
    },
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
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": None,
        "save_hist": False,
        # "agents": [
        #     agent for agent in list(agents.keys()) if ("finetune" not in agent)
        # ],  # All agents besides fine-tuned ones
        "agents": ["marr"],
    },
    "mult_slice": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": MimicQuadriga,  # QuadrigaChannel,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 10,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 60,
        "initial_testing_episode": 80,
        "test_episodes": 20,
        "episode_evaluation_freq": 60,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 60,
        "save_hist": False,
        # "agents": [
        #     agent for agent in list(agents.keys()) if ("finetune" not in agent)
        # ],  # All agents besides fine-tuned ones
        "agents": ["sched_twc"],  # "sb3_sched"],
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
        "save_hist": False,
        "agents": [
            agent for agent in list(agents.keys()) if ("finetune" not in agent)
        ],  # All agents besides fine-tuned ones
    },
    "finetune_mult_slice_seq": {
        "seed": 10,
        "seed_test": 15,
        "channel_class": MimicQuadriga,  # QuadrigaChannelSeq,
        "traffic_class": MultSliceTraffic,
        "mobility_class": SimpleMobility,
        "root_path": str(getcwd()),
        "training_epochs": 5,
        "enable_evaluation": True,
        "initial_training_episode": 0,
        "max_training_episodes": 80,  # 80 different channels from the same scenario
        "initial_testing_episode": 80,
        "test_episodes": 20,  # Testing on 20 channels from the same scenario
        "episode_evaluation_freq": 10,
        "number_evaluation_episodes": 20,
        "checkpoint_episode_freq": 10,
        "eval_initial_env_episode": 80,
        "save_hist": False,
        # "agents": ["finetune_sched_twc"],  # , "marr"],
        "agents": ["scratch_sb3_sched", "finetune_sb3_sched"],
        "number_scenarios": 1,
    },
}


def env_creator(env_config):
    marl_comm_env = MARLCommEnv(
        env_config["channel_class"],
        env_config["traffic_class"],
        env_config["mobility_class"],
        env_config["association_class"],
        "mult_slice",
        env_config["agent"],
        env_config["seed"],
        root_path=env_config["root_path"],
        initial_episode_number=env_config["initial_training_episode"],
        simu_name=env_config["scenario"],
        save_hist=env_config["save_hist"],
    )
    marl_comm_env.comm_env.max_number_episodes = env_config[
        "max_training_episodes"
    ]
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
            marl_comm_env,
            marl_comm_env.comm_env.max_number_ues,
            marl_comm_env.comm_env.max_number_slices,
            marl_comm_env.comm_env.max_number_basestations,
            marl_comm_env.comm_env.num_available_rbs,
            eval_env if env_config["enable_evaluation"] else None,
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

    return marl_comm_env, agent


def sb3_load_path(agent_name, scenario, method="last"):
    if method == "last":
        return f"./agents/models/{scenario}/final_{agent_name}.zip"
    elif method == "best":
        return f"./agents/models/{scenario}/best_{agent_name}/best_model.zip"
    elif isinstance(method, int):
        return f"./agents/models/{scenario}/{agent_name}/{agent_name}_{method}_steps.zip"
    else:
        raise ValueError(f"Invalid method {method} for finetune load")


for scenario in scenarios.keys():
    for agent_name in env_config_scenarios[scenario]["agents"]:
        env_config = env_config_scenarios[scenario].copy()
        env_config["agent"] = agent_name
        env_config["scenario"] = scenario
        env_config["agent_class"] = agents[agent_name]["class"]
        env_config["association_class"] = scenarios[scenario]
        env_config["rl"] = agents[agent_name]["rl"]

        number_scenarios = env_config.get("number_scenarios", 1)
        for scenario_number in range(number_scenarios):
            marl_comm_env, agent = env_creator(env_config)
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
                    if "finetune" in agent_name:
                        print(
                            f"Fine-tuning model from Agent {agents[agent_name]['base_agent']} scenario {agents[agent_name]['base_scenario']} on {scenario} scenario"
                        )
                        path = sb3_load_path(
                            agents[agent_name]["base_agent"],
                            agents[agent_name]["base_scenario"],
                            agents[agents[agent_name]["base_agent"]][
                                "load_method"
                            ],
                        )
                        agent.load(path)  # Loading base model
                    print(f"Training {agent_name} on {scenario} scenario")
                    agent.train(total_time_steps)

                path_test = sb3_load_path(
                    agent_name, scenario, agents[agent_name]["load_method"]
                )
                agent.load(path_test)

            # Testing
            print(f"Testing {agent_name} on {scenario} scenario")
            total_test_steps = env_config["test_episodes"] * steps_per_episode
            marl_comm_env.comm_env.max_number_episodes = (
                env_config["initial_testing_episode"]
                + env_config["test_episodes"]
            )
            marl_comm_env.comm_env.save_hist = True  # Save metrics for test
            obs, _ = marl_comm_env.reset(
                seed=env_config["seed_test"],
                options={
                    "initial_episode": env_config["initial_testing_episode"]
                },
            )
            for step in tqdm(np.arange(total_test_steps), desc="Testing..."):
                action = agent.step(obs)
                obs, reward, terminated, truncated, info = marl_comm_env.step(action)  # type: ignore
                assert isinstance(
                    terminated, bool
                ), "Terminated must be a boolean"
                if terminated:
                    obs, _ = marl_comm_env.reset()

            # Updating values for next scenario
            if "finetune" in scenario:
                scenario_episodes = (
                    env_config["max_training_episodes"]
                    - env_config["initial_training_episode"]
                ) + env_config["test_episodes"]
                env_config["initial_training_episode"] += scenario_episodes
                env_config["max_training_episodes"] += scenario_episodes
                env_config["initial_testing_episode"] += scenario_episodes
                env_config["eval_initial_env_episode"] += scenario_episodes
