#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# store and print git branch commits info
sh ./scripts/get_git_info.sh

# store cpu info and print experiment info
sage ./jiawei-computational-results/prerun.sage

# clear results folder
for p in ./jiawei-computational-results/results_datatable/*; do
    if [ -d ${p} ]; then
        find $p -maxdepth 1 -type f -name "*.csv" -delete
    fi
done

# ready to submit jobs, print instructions
echo "It is ready to submit jobs to the cluster. Remeber to specify parameters; the number of jobs and time limit must be provided, other parameters are optional. For example:"
echo "sbatch --array=1-1650 --time 01:00:00 SLURM-computation.sage"
