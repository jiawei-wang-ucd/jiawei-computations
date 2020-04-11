#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# get git branch commits info
sh get_git_info.sh

# clear test cases folder
if [[ -d ./jiawei-computational-results/test_cases_datatable ]]; then
  rm -f ./jiawei-computational-results/test_cases_datatable/*
else
  mkdir ./jiawei-computational-results/test_cases_datatable/
fi

# clear results folder
if [[ -d ./jiawei-computational-results/results_datatable ]]; then
  rm -f ./jiawei-computational-results/results_datatable/*
else
  mkdir ./jiawei-computational-results/results_datatable/
fi

# submit jobs
# sbatch --array=1-150 -t 01:00:00 -n 4 --mem-per-cpu 8000 ./scripts/SLURM-faster-subadditivity-test.sage

# submit jobs sample
sbatch 
