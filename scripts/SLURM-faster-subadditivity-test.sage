#! /home/wangjw/sage/sage-9.0/sage
#SBATCH --array=1-3
#
# --array: Specify the range of the array tasks.
# --time: Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds" etc.
import os
task_id = os.getenv("SLURM_ARRAY_TASK_ID")
if task_id:
    task_id = int(task_id)

print("Sage hello from task {}".format(task_id))

import sys
sys.path.append(os.path.join(os.getcwd(), 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

load(os.path.join(os.getcwd(), 'scripts', 'test_functions.sage'))
load(os.path.join(os.getcwd(), 'scripts', 'helper.sage'))

logging.disable(logging.INFO)

test_library_path = './jiawei-computational-results/test_cases_datatable/'
input_file_name = test_library_path + 'function_' + str(task_id) + '.sobj'
print('Input file:' + input_file_name)
sys.stdout.flush()

results_path = './jiawei-computational-results/results_datatable/'
result_file_name = results_path+'results_' + str(task_id) + '.csv'
print('Output file:' + result_file_name)
sys.stdout.flush()

fn = load(input_file_name)
kwargs = branch_parameter_dictionary[git_branch('./jiawei-computational-results')]

report_performance(minimum_of_delta_pi, result_file_name, iterations=30, args=[fn], kwargs = kwargs)



