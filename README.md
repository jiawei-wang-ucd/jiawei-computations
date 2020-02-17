# jiawei-computations
Top-level repository for computational paper

Repository for reproducible benchmarking of checking subadditivity of cut generating functions in a High Performance Cluster (HPC) using the slurm batch queue manager.
Contribution and feedback are very welcome!

## Structure

- The scripts of the reproducible benchmarking work are stored in [scripts](https://github.com/mkoeppe/jiawei-computations/tree/master/scripts) folder.
- The computational results are stored in the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results). The branch (name) of the submodule specifies the (one) algorithm used in the experiment. 
- The source code and algorithms on cut generating functions are stored in the submodule [cutgeneratingfunctionology](https://github.com/mkoeppe/cutgeneratingfunctionology). The branch of the submodule specifies the version of the cut generating function code in the experiment. 

## Prerequisite package 

- The python-based software [SageMath](https://www.sagemath.org/) needs to be installed in the HPC.

## Reproduce

- edit the file `path.env` to specify the path of the executable sage in the HPC.
- edit the first line of the file `./scripts/SLURM-faster-subadditivity-test.sage` to specify the path of the executable sage in the HPC.
- edit the last line of `run.sh` to specify the options of submitting jobs to the cluster, including number of nodes, priority, cpu memory.
- choose one algorithm and one version of the cut generating function code, and checkout the corresponding branches of the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results) and [cutgeneratingfunctionology](https://github.com/mkoeppe/cutgeneratingfunctionology). 
- run the file `run.sh` to generate all test cases and submit jobs to the cluster.
```
sh run.sh
```

## Acknowledgement 
