# Intent-based Radio resource scheduler using RL for RAN slicing

Code implementation from paper "Intent-based Radio Scheduler for RAN Slicing: Learning to deal with different network scenarios" available in [ArXiv](https://arxiv.org/abs/2501.00950). It utilizes a python implementation to the radio resource scheduler simulation based on [sixg_radio_mgmt ](https://github.com/lasseufpa/sixg_radio_mgmt), channel generation using [QuaDRiGa](https://quadriga-channel-model.de/), the proposed RL implementation using [Ray Rllib](https://docs.ray.io/en/latest/rllib/index.html), and [Stable-baselines3](stable-baselines3.readthedocs.io/) to the baseline agents using RL.

## Requirements
 - python 3.10
 - pipenv

## Cloning
Remember to clone submodules using `git clone --recursive git@github.com:lasseufpa/intent_radio_sched_multi_slice.git`

## Install the virtual environment
Use pipenv to generate a virtual environment with all the dependencies needed to execute the project:
- Install the virtual environment using `pipenv install`.
- Access the environment using `pipenv shell` in case you want to execute commands using the virtual environment.

## <a name="generate_data"></a>Generating pre-training data

Before running the RL simulation, we need to generate two datasets for the network scenarios (defining which slices will be active, their slice types, number of UEs and etc.) and the UE channels. In case you want to save time, just download the same network scenarios (also called associations) and channel data files that were used in the paper by skipping this section and going to the [next section](#precomputed).

### Generating network scenarios
Use the command `pipenv run python gen_assoc_mult_slice.py` to generate the network scenarios (associations) defining the network scenario characteristics for each RL episode. This information will be used in both the radio resource scheduler + RL simulation and also in the channel generation. After the script finishes, the folder `associations/data` should contain the scenario `mult_slice` with network scenario files for each RL episode.

### Generating channel data
We utilize the [QuaDRiGa](https://quadriga-channel-model.de/) channel simulator to generate the UEs channel data based on the network scenarios generated before. Therefore, you need to have Matlab or Octave installed with the QuaDRiGa module enabled. You can have more information about how to install the QuaDRiGa simulator in their [repo](https://github.com/fraunhoferhhi/QuaDRiGa). In this work, we utilized Matlab and QuaDRiGa (version 2.6.1).

Once Matlab/Octave and QuaDRiGa are installed, you can open the folder `mult_slice_channel_generation` in Matlab/Octave and execute the script `simu.m`. It will take a long time to generate all the channels used in the paper. Once the simulation finishes, a folder `results/` should contain the channel data for each UE in each specific network scenario. At this point all the pre-generated datasets were created, and you can run the simulation.

## <a name="precomputed"></a> Download pre-computed data for network scenarios and channels

Follow this section only if you have skipped the section before on [generate pre-computed data](#generate_data).

Download the channel dataset from this [link](https://nextcloud.lasseufpa.org/s/CXKz5fK8LmyDtR6) into the folder `mult_slice_channel_generation/results`. After extracting the downloaded file, the folder structure should contain the folder `mult_slice_channel_generation/results/mult_slice`.

Download the network scenarios (association) dataset from this [link](https://nextcloud.lasseufpa.org/s/Y9CGsR8GHpHPdbX) into the folder `associations/`. After extracting the compressed file, the folder structure should contain the folder `associations/data/mult_slice`.

## Running the radio resource scheduler + RL simulation

Returning to the root directory of this project (`intent_radio_sched_multi_slice`), the script `simu.py` is the main responsible for running each of the simulation scenarios presented in the paper. The scenario where the RL agent is trained and tested for each specific network scenario is called `mult_slice_seq`, the simulation scenario where the RL agent is trained and tested on different network scenarios is called `mult_slice`, while fine-tune simulation scenario is called `finetune_mult_slice_seq`.

In order to execute these three scenarios, execute the command `pipenv run python simu.py`. It should take a long time to finish training all the simulation scenarios. You can decide the simulation scenarios you want by commenting them in the variable `scenarios` in the beginning of the `simu.py` file.

Once the simulation is finished, the network data generated during the simulation training and testing will be available in the folder `hist`, `ray_results`, and `tensorboard-logs`. The folder `hist` contains the test results obtained after the training process. The folder `ray_results` contains the training and evaluation data obtained during Ray training to the proposed agent (including RL agent models). The folder `tensorboard-logs` contains the training and evaluation data to the baseline agents using RL (Stable-baselines3).

## Generating results

Run the command `pipenv run python results/gen_results.py` will generate the results to the three network scenarios evaluated in the paper. All the results will be available into the folder `results/` with one folder for each simulation scenario. It can take a while to generate all the figures.

## Contributing to this repo
- Activate pre-commit hooks to use [black formatter](https://github.com/psf/black), [flake8 lint](https://gitlab.com/pycqa/flake8), [Isort references](https://github.com/timothycrosley/isort) and [Pyright type check](https://github.com/microsoft/pyright). Run `pre-commit install` inside the virtual environment. Now every time you make a commit, black formatter, flake8, isort and pyrights will make tests to verify if your code is following the [patterns](https://realpython.com/python-pep8/) (you can adapt your IDE or text editor to follow this patterns, e.g. [vs code](https://code.visualstudio.com/docs/python/python-tutorial#_next-steps)).

## Cite this project

```
@ARTICLE{11179861,
  author={Nahum, Cleverson V. and D'Oro, Salvatore and Batista, Pedro and Both, Cristiano B. and Cardoso, Kleber V. and Klautau, Aldebaro and Melodia, Tommaso},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Intent-based Radio Scheduler for RAN Slicing: Learning to Deal with Different Network Scenarios}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Throughput;Resource management;6G mobile communication;Ultra reliable low latency communication;Training;Radio access networks;Base stations;Traffic control;Quality of service;Network slicing;Radio resource scheduling;RAN slicing;intent-based scheduler;multi-agent reinforcement learning},
  doi={10.1109/TMC.2025.3614453}}
```
