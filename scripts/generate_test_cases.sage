import os, sys

sys.path.append(os.path.join(os.getcwd(), 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

test_library_write_path = os.getcwd() + '/jiawei-computational-results/test_cases_datatable/'

# used for constructor non extreme function by convex combination.
non_extreme_function_constructor = gj_2_slope_repeat(f=1/2,m=10,n=10)

# used for constructor non subadditive function by convex combination.
non_subadditive_function_constructor = gj_2_slope_repeat(f=1/2,m=10,n=30)

test_function_dictionary={}

for i in range(1,51):
    test_function_dictionary[str(i)] = 'extreme_' + str(i) + '0_slope_function'
    test_function_dictionary[str(i+50)] = 'non_extreme_' + str(i) + '0_slope_function'
    test_function_dictionary[str(i+100)] = 'non_subadditive_' + str(i) + '0_slope_function'

for i in range(1,51):
    extreme_function = bcdsp_arbitrary_slope (k = int(i)*10)
    non_extreme_function = (extreme_function + non_extreme_function_constructor)/2
    non_subadditive_function = (extreme_function + non_subadditive_function_constructor)/2

    save(extreme_function,test_library_write_path+'function_'+str(i))
    save(non_extreme_function,test_library_write_path+'function_'+str(i+50))
    save(non_subadditive_function,test_library_write_path+'function_'+str(i+100))





