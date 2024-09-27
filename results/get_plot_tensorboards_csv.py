# Based on Subhaditya Mukherjee's code https://gist.github.com/SubhadityaMukherjee/83b4477bbc0cf0786e61a5f4bb895fe1

import os
import re
from pathlib import Path

import matplotlib.figure as matfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from tqdm import tqdm

agent = "ray_ib_sched_hyper_asha_0"
scenario = "hyperparam_opt_mult_slice"
main_path = f"./ray_results/{scenario}/{agent}/"
errored_files = [
    "ray_results/hyperparam_opt_mult_slice/ray_ib_sched_hyper_asha_0/PPO_marl_comm_env_933f4_00194_194_clip_param=0.4000,entropy_coeff=0.0000,gamma=0.5000,grad_clip=2,kl_target=0.0050,lambda=0.9800,l_2024-07-12_14-54-03/events.out.tfevents.1720912104.na1",
    "ray_results/hyperparam_opt_mult_slice/ray_ib_sched_hyper_asha_0/PPO_marl_comm_env_933f4_00194_194_clip_param=0.4000,entropy_coeff=0.0000,gamma=0.5000,grad_clip=2,kl_target=0.0050,lambda=0.9800,l_2024-07-12_14-54-03/events.out.tfevents.1721046754.na1",
]


def get_event_files(main_path):
    """Return a list of event files under the given directory"""
    all_files = []
    for root, _, filenames in os.walk(main_path):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                all_files.append(str(Path(root) / Path(filename)))
    return all_files


def process_event_acc(event_acc, file, df, tags=None):
    """Process the EventAccumulator and return a dictionary of tag values"""
    # all_tags = event_acc.Tags()
    if tags is None:
        tags = {
            "scalars": [
                "ray/tune/evaluation/env_runners/policy_reward_mean/inter_slice_sched",
                "ray/tune/evaluation/env_runners/policy_reward_mean/inter_slice_sched",
                "ray/tune/evaluation/env_runners/episode_return_mean",
            ]
        }
    assert isinstance(tags, dict), "Tags should be a dictionary"
    try:
        for tag in tags.keys():
            if tag == "scalars":
                for subtag in tags[tag]:
                    for scalar in event_acc.Scalars(tag=subtag):
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    [
                                        {
                                            "file": file,
                                            "metric": subtag,
                                            "step": scalar.step,
                                            "value": scalar.value,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
    except Exception as e:
        print(f"Error processing {file} with error {e}")
        pass
    return df


def process_runs(main_path, tags=None):
    """Iterate over all the runs and return the dataframe"""
    all_files = get_event_files(main_path=main_path)
    df = pd.DataFrame(columns=["file", "step", "value"])
    for file in tqdm(all_files, total=len(all_files)):
        event_acc = EventAccumulator(file)
        event_acc.Reload()
        df = process_event_acc(event_acc, file, df, tags)
    return df


def get_substring_between(s, start, end):
    pattern = start + r"(\d+)" + re.escape(end)
    match = re.search(pattern, s)
    return match.group(1) if match else ""


# # Creating CSV
# df_files = process_runs(main_path=main_path)
# print(df_files.head())
# df_files.to_csv("results/hyperparam_opt_results.csv", index=False)

# Plotting
df_files = pd.read_csv("results/hyperparam_opt_results.csv")
metric_inter = (
    "ray/tune/evaluation/env_runners/policy_reward_mean/inter_slice_sched"
)
reward_inter = df_files[df_files["metric"] == metric_inter].sort_values(  # type: ignore
    by="step"
)
reward_inter = reward_inter[~reward_inter["file"].isin(errored_files)]
max_reward_per_file = reward_inter.groupby("file")["value"].max().reset_index()
top_10_avg_reward = max_reward_per_file.nlargest(10, "value")
file_names = top_10_avg_reward["file"].values

w, h = matfig.figaspect(0.6)
plt.figure(figsize=(w, h))
for file in file_names:
    trial_number = get_substring_between(file, "_", "_clip_param")
    reward_inter_file = reward_inter[reward_inter["file"] == file]
    plt.plot(
        reward_inter_file["step"],
        reward_inter_file["value"],
        label=f"Trial {trial_number}",
    )
plt.grid()
plt.xlabel("Steps (n)", fontsize=14)
plt.ylabel("Inter-slice scheduler reward", fontsize=14)
plt.xticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left")
os.makedirs(
    f"./results/{scenario}/{agent}/",
    exist_ok=True,
)
plt.savefig(
    (f"./results/{scenario}/{agent}/hyperparam_opt_reward_inter.pdf"),
    bbox_inches="tight",
    pad_inches=0,
    format="pdf",
    dpi=1000,
)
plt.close()
