#! /usr/bin/python3
#

import os
import sys
sys.path.append('./scripts')
import analysis_scripts; from analysis_scripts import *

task_id = os.getenv("SLURM_ARRAY_TASK_ID")
if task_id:
    task_id = int(task_id)

if task_id % 2 == 0:
    dist = "norm"
else:
    dist = "lognorm"
alg_id = task_id//2

path = './computational-results-minimum/results_datatable'
col = 'cputime(s)'
alg = [f for f in os.listdir(path) if not f.startswith('.')][alg_id]
instance_names = [f[:-4] for f in os.listdir(path+"/"+alg) if f.endswith('.csv')]
cov = []
size = []
means = []
for f in instance_names:
    temp_df = pd.read_csv(path + '/' + alg + '/' + f + '.csv')
    if temp_df.shape[0] == 30:
        data = list(temp_df[col])
        true_mean, coverage, sample_size = bootstrap_coverage(data, precision = 0.01, iteration = 10000, pilot_size = 10, alpha = 0.95, distribution = dist)
        cov.append(coverage)
        size.append(sample_size)
        means.append(true_mean)
df = pd.DataFrame()
df['means'] = means
df['size'] = size
df['cov'] = cov
df.to_csv('minimum_' + algo + '.csv')




