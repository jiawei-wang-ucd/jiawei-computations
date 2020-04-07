#! /home/wangjw/sage/sage-9.0/sage
#

import os
task_id = os.getenv("SLURM_ARRAY_TASK_ID")
if task_id:
    task_id = int(task_id)

print("Sage hello from task {}".format(task_id))

import sys
sys.path.append(os.path.join(os.getcwd(), 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

load(os.path.join(os.getcwd(), 'scripts', 'test_functions.sage'))
load(os.path.join(os.getcwd(), 'jiawei-computational-results', 'helper.sage'))

logging.disable(logging.INFO)

input_file_name, output_file_path = from_taskid_to_instance_and_algorithm(task_id, total_algorithms, total_instances, all_algorithms_path, all_instances)
output_file_name = output_file_path+'/' + input_file_name.split('/')[-1].split('.')[0] + '.csv'

print('Input file:' + input_file_name)
print('Output file:' + output_file_name)
sys.stdout.flush()

fn = load(input_file_name)
load(output_file_path +'/algorithm_helper.sage')

report_performance(function, output_file_name, iterations=30, args=[fn], kwargs = branch_parameters)



