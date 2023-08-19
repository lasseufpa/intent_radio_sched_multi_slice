#!/bin/bash

pipenv run python3 simu_marl_intra_rr.py
pipenv run python3 simu_marl_intra_nn.py
pipenv run python3 simu_marl.py
pipenv run python3 simu_marl_no_mask.py
pipenv run python3 simu_marr.py
pipenv run python3 simu_random.py