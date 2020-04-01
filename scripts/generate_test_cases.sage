import os, sys, gc
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

logging.disable(logging.WARN)

# parameters used for constructing two slope extreme functions
two_slope_fill_in_epsilon = QQ(1/10)

# parameters used for constructing non extreme (minimal) and non subadditive functions
convex_combination_coefficient = QQ(9/10)

# specify the paths of test instances, store in three directories.
test_library_write_path = os.getcwd() + '/test_instances/'
extreme_functions_path = test_library_write_path + '/extreme_functions/'
minimal_functions_path = test_library_write_path + '/minimal_functions/'
non_subadditive_functions_path = test_library_write_path + '/non_subadditive_functions/'

# create a csv file which stores the information of test instances
test_instances_df = pd.DataFrame()

base_function_family = ['chen_4_slope','chen_4_slope','drlm_backward_3_slope','drlm_backward_3_slope','gj_forward_3_slope','gj_forward_3_slope','kzh_3_slope_param_extreme_1','kzh_3_slope_param_extreme_1','kzh_3_slope_param_extreme_2','kzh_3_slope_param_extreme_2','kzh_4_slope_param_extreme_1','kzh_4_slope_param_extreme_1','kzh_5_slope_fulldim_2','kzh_5_slope_fulldim_covers_2','kzh_5_slope_q22_f10_1','kzh_6_slope_1','kzh_6_slope_fulldim_covers_2','kzh_6_slope_fulldim_covers_3','kzh_7_slope_1','kzh_7_slope_2','kzh_7_slope_3','kzh_7_slope_4','kzh_10_slope_1','kzh_28_slope_1','kzh_28_slope_2','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope']

base_function_parameters = ['f:7/10','f:2/3','f:1/10','f:1/12','f:4/5','f:2/3','f:6/19','f:1/3','f:5/9','f:1/2','f:13/18','f:7/10',None,None,None,None,None,None,None,None,None,None,None,None,None,'k:3','k:4','k:5','k:7','k:10']

two_slope_function_family = ['chen_4_slope','chen_4_slope','drlm_backward_3_slope','drlm_backward_3_slope','gj_forward_3_slope','gj_forward_3_slope','kzh_3_slope_param_extreme_1','kzh_3_slope_param_extreme_1','kzh_3_slope_param_extreme_2','kzh_3_slope_param_extreme_2','kzh_4_slope_param_extreme_1','kzh_4_slope_param_extreme_1','kzh_5_slope_fulldim_2','kzh_5_slope_fulldim_covers_2','kzh_6_slope_1','kzh_6_slope_fulldim_covers_2','kzh_6_slope_fulldim_covers_3','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope','bcdsp_arbitrary_slope']

two_slope_function_parameters = ['f:7/10','f:2/3','f:1/10','f:1/12','f:4/5','f:2/3','f:6/19','f:1/3','f:5/9','f:1/2','f:13/18','f:7/10',None,None,None,None,None,'k:3','k:4','k:5']

extreme_function_file_name = ['extreme_'+str(i) for i in range(1,len(base_function_family)+1)]+['extreme_'+str(i)+'_2s' for i in range(len(base_function_family)+1,len(two_slope_function_family)+len(base_function_family)+1)]
minimal_function_file_name = ['minimal_'+str(i) for i in range(1,len(base_function_family)+1)]+['minimal_'+str(i)+'_2s' for i in range(len(base_function_family)+1,len(two_slope_function_family)+len(base_function_family)+1)]
nonsub_function_file_name = ['nonsub_'+str(i) for i in range(1,len(base_function_family)+1)]+['nonsub_'+str(i)+'_2s' for i in range(len(base_function_family)+1,len(two_slope_function_family)+len(base_function_family)+1)]

two_slope_approximation = ['N'] * len(base_function_family) + ['Y'] * len(two_slope_function_family)

test_instances_df['file name'] = extreme_function_file_name + minimal_function_file_name + nonsub_function_file_name
test_instances_df['family'] = (base_function_family + two_slope_function_family) * 3
test_instances_df['parameters'] = (base_function_parameters + two_slope_function_parameters) * 3
test_instances_df['two slope approximation'] = two_slope_approximation* 3
test_instances_df['instance type'] = ['extreme'] * (len(base_function_family) + len(two_slope_function_family)) + ['minimal'] * (len(base_function_family) + len(two_slope_function_family)) + ['non subadditive'] * (len(base_function_family) + len(two_slope_function_family))

test_instances_df.to_csv('./test_instances/test_instances_info.csv', index = False)

# generate test instances files (sage objects).

def non_subadditive_function_example(f):
    '''
    Return a non subadditive function which satisfies symmetry condition. The minimum of delta pi is always -1/10.
    This function is used to generate non_subadditive test instances by convex combination.
    '''
    bkpts = [0, f/3, 2*f/3, f, 1]
    values = [0, 3/10, 7/10, 1, 0]
    return piecewise_function_from_breakpoints_and_values(bkpts,values)

idx = 1

# base functions and their minimal and non subadditive perturbations.
for i in range(len(base_function_family)):
    # extreme functions
    if base_function_parameters[i] is None:
        fn = eval(base_function_family[i])()
    else:
        kwags = {base_function_parameters[i].split(':')[0]: QQ(base_function_parameters[i].split(':')[1])}
        fn = eval(base_function_family[i])(**kwags)
    save(fn, extreme_functions_path + 'extreme_'+str(idx))

    # minimal functions
    minimal_perturbation_function = gmic(f = find_f(fn))

    minimal_fn = convex_combination_coefficient * fn + (1 - convex_combination_coefficient) * minimal_perturbation_function
    save(minimal_fn, minimal_functions_path + 'minimal_'+str(idx))

    # non subadditive functions
    nonsub_perturbation_function = non_subadditive_function_example(f = find_f(fn))

    nonsub_fn = convex_combination_coefficient * fn + (1 - convex_combination_coefficient) * nonsub_perturbation_function
    save(nonsub_fn, non_subadditive_functions_path + 'nonsub_'+str(idx))

    idx+=1
    del fn
    del minimal_fn
    del nonsub_fn
    gc.collect()

# two slope approximation functions and base functions and their minimal and non subadditive perturbations.
for i in range(len(two_slope_function_family)):
    # two slope approximation functions.
    if two_slope_function_parameters[i] is None:
        fn = eval(two_slope_function_family[i])()
    else:
        kwags = {two_slope_function_parameters[i].split(':')[0]: QQ(two_slope_function_parameters[i].split(':')[1])}
        fn = eval(two_slope_function_family[i])(**kwags)
    fn_2slope = two_slope_fill_in_extreme(fn, two_slope_fill_in_epsilon)
    save(fn_2slope, extreme_functions_path + 'extreme_'+str(idx)+'_2s')

    # minimal functions
    minimal_perturbation_function = gmic(f = find_f(fn))

    minimal_fn_2slope = convex_combination_coefficient * fn_2slope + (1 - convex_combination_coefficient) * minimal_perturbation_function
    save(minimal_fn_2slope, minimal_functions_path + 'minimal_'+str(idx)+'_2s')

    # non subadditive functions
    nonsub_perturbation_function = non_subadditive_function_example(f = find_f(fn))

    nonsub_fn_2slope = convex_combination_coefficient * fn_2slope + (1 - convex_combination_coefficient) * nonsub_perturbation_function
    save(nonsub_fn_2slope, non_subadditive_functions_path + 'nonsub_'+str(idx)+'_2s')

    idx+=1
    del fn
    del fn_2slope
    del minimal_fn_2slope
    del nonsub_fn_2slope
    gc.collect()
