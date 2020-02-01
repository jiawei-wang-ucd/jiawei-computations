#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# get git branch commits info
sh get_git_info.sh

# generate test_cases
sage ./scripts/generate_test_cases.sage

# submit jobs
sbatch -t 01:00:00 ./scripts/SLURM-faster-subadditivity-test.sage

# generate plots

