#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# store and print git branch commits info
sh ./scripts/get_git_info.sh

# store cpu info
sage ./scripts/get_cpuinfo.sage

# clear results folder
for p in ./jiawei-computational-results/results_datatable/*; do
    if [ -d ${p} ]; then
        find $p -maxdepth 1 -type f -name "*.csv" -delete
    fi
done

# submit jobs
# sbatch --array=1-150 -t 01:00:00 -n 4 --mem-per-cpu 8000 ./scripts/SLURM-faster-subadditivity-test.sage
