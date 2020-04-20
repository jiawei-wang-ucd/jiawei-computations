# jiawei-computations
Top-level repository for computational paper

Repository for reproducible benchmarking of studying subadditivity slack (delta pi) of cut generating functions in a High Performance Cluster (HPC) using the slurm batch queue manager.
Contribution and feedback are very welcome!

## Structure

- The scripts of the reproducible benchmarking work are stored in [scripts](https://github.com/mkoeppe/jiawei-computations/tree/master/scripts) folder.
- The test instances are stored in the [test_instances](https://github.com/mkoeppe/jiawei-computations/tree/master/test_instances) folder. The folder contains a file [test_instances_info.csv](https://github.com/mkoeppe/jiawei-computations/tree/master/test_instances/test_instances_info.csv) which explaines how each instance is generated. Note that not all test instances may be used in the experiment. For example, we do not include non subadditive functions in the experiment of generating additive faces. 
- The computational results are stored in the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results). The branch name of the submodule specifies the computational task. So far there are four computational tasks. See how the branch name represents the computational task as below.
* delta_pi_minimum - computation of the minimum of delta pi, including all test instances.
* delta_pi_objective_zero - verification of the delta pi reaching the objective limit 0, excluding non subadditive instances.
* delta_pi_objective_one_percent - verification of the delta pi reaching the objective limit -0.01, including all test instances.
* generate_additive_faces - generation of additive faces, excluding non subadditive faces.
- The source code and algorithms on cut generating functions are stored in the submodule [cutgeneratingfunctionology](https://github.com/mkoeppe/cutgeneratingfunctionology), which is contained in the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results). The branch of the submodule specifies the version of the cut generating function code in the experiment. 
- The folder [system_info](https://github.com/mkoeppe/jiawei-computations/tree/master/system_info) stores the git branch/commit info of current reprositories and CPU info of the HPC used for the experiment.

## Prerequisite package 

- The python-based software [SageMath](https://www.sagemath.org/) needs to be installed in the HPC.
- The experiment will utilize optimization solvers to solve LPs and MIPs. Those solvers (like cbc or cplex) should be made callable by [MixedIntegerLinearProgram](http://doc.sagemath.org/html/en/reference/numerical/sage/numerical/mip) in [SageMath](https://www.sagemath.org/).

## Reproduce

- edit the file `path.env` to specify the path of the executable sage in the HPC.
- (optional) it is possible to regenerate all test instances using file `regenerate_test_instances.sh`.
```
sh regenerate_test_instances.sh
```
- choose one computational task and checkout the corresponding branch of the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results).
- choose the version of submodule [cutgeneratingfunctionology](https://github.com/mkoeppe/cutgeneratingfunctionology) and checkout the branch/commit.
- run the file `prerun.sh`, and it will print out information about the experiment.
```
sh prerun.sh
```
- edit the first line of the file `SLURM-computation.sage` to specify the path of the executable sage in the HPC.
- submit jobs to the cluster using `SLURM-computation.sage`, and specify parameters, including the number of jobs, time limit, memory. For example, the following command will submit 1500 jobs to the cluster, and each job request 4 nodes and 8000MB per CPU for the computation with 1 hour time limit.
```
sbatch --array=1-1500 --time 01:00:00 -n 4 --mem-per-cpu 8000 SLURM-computation.sage
```
- (optional) monitor the status of jobs by using the following command.
```
squeue -u USER_NAME
```
It is possible that some jobs are suspended, and they can be requeued by using the following command.
```
scontrol requeue JOB_ID
```
- clear log files after all computations are done (which could takes several days depending on the expriment scale and computation priority).
```
sh clean_logs.sh
```
- use git to stage and commit results files created in the submodule [jiawei-computational-results](https://github.com/mkoeppe/jiawei-computational-results).

## Acknowledgement 
