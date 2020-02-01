import os, sys

sys.path.append(os.path.join(os.getcwd(), '..', 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

test_library_write_path = os.path.dirname(__file__) + '/../jiawei-computational-results/test_cases_datatable/'

test_function_dictionary={
"1":'chen_4_slope',
"2":'drlm_backward_3_slope',
"3":'kzh_7_slope_1'
}

for key in test_function_dictionary.keys():
    fn=eval(test_function_dictionary[key])()
    save(fn,test_library_write_path+'function_'+key)




