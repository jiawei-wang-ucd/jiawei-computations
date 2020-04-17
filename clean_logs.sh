#!/usr/bin/env bash

# rename slurm*.out files and move them to correspoding directory.
for i in *.out
do
  while IFS= read -r line
    do
      if [[ "$line" == "Output file"* ]]; then
        RM_PREFIX=${line#"Output file:"}
        mv "$i" "${RM_PREFIX%.*}.out"
      fi
  done < "$i"
done
