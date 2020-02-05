#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# get git branch commits info
sh get_git_info.sh

# clear test cases folder
rm -f ./jiawei-computational-results/test_cases_datatable/*

# generate test_cases
sage ./scripts/generate_test_cases.sage

# clear results folder
rm -f ./jiawei-computational-results/results_datatable/*

# submit jobs
sbatch -t 01:00:00 ./scripts/SLURM-faster-subadditivity-test.sage

