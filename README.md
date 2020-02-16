# jiawei-computations
Top-level repository for computational paper

Repository for reproducible benchmarking of checking subadditivity of cut generating functions in a High Performance Cluster (HPC) using the slurm batch queue manager.
Contribution and feedback are very welcome!

## Structure

- The scripts of the reproducible benchmarking work are stored in [scripts](https://github.com/mkoeppe/jiawei-computations/tree/master/scripts) folder.
- The computational results are stored in the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results).
- The source code and algorithms on cut generating functions are stored in the submodule [cutgeneratingfunctionology](https://github.com/mkoeppe/cutgeneratingfunctionology).

## Prerequisite package 

- The python-based software [SageMath](https://www.sagemath.org/) needs to be installed in the HPC.

## Reproduce

- edit the file `path.env` to specify the path of the executable sage in the HPC.
- edit the first line of the file `./scripts/SLURM-faster-subadditivity-test.sage` to specify the path of the executable sage in the HPC.
- edit the last line of `run.sh` to specify the options of submitting jobs to the cluster, including number of nodes, priority, cpu memory.
- run the file `run.sh` to generate all test cases and submit jobs to the cluster.
```
sh run.sh
```

## Acknowledgement 
