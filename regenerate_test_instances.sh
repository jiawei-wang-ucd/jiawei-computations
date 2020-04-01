#!/usr/bin/env bash

# get sage path to run sage directly
source ./path.env

# clear test_instances folder
if [[ -d ./test_instances ]]; then
  rm -f ./test_instances/*
else
  mkdir ./test_instances/
fi

mkdir ./test_instances/extreme_functions/
mkdir ./test_instances/minimal_functions/
mkdir ./test_instances/non_subadditive_functions/

# generate test_cases
echo 'The generation usually takes a couple of minutes.'
sage ./scripts/generate_test_cases.sage
