import gc
import sys

import resource; from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'jiawei-computational-results', 'cutgeneratingfunctionology'))
import cutgeneratingfunctionology.igp; from cutgeneratingfunctionology.igp import *

def measure_time_resource(function, args=tuple(), kwargs={}):
    '''Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by your
    function/callable by summing `user` and `sys`. `real` is just like the wall
    clock.
    Note that `sys` and `user`'s resolutions are limited by the resolution of
    the operating system's software clock (check `man 7 time` for more
    details).
    '''
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    function(*args, **kwargs)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
    res={'real': end_time - start_time,
        'sys': end_resources.ru_stime - start_resources.ru_stime,
        'user': end_resources.ru_utime - start_resources.ru_utime}
    res['cpu']=res['sys']+res['user']
    res['mem']=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
    return res

def report_performance(function, file_name, iterations, args=tuple(), kwargs={}):
    with open(file_name, mode = 'w') as writefile:
        performance_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_table.writerow(['walltime(s)','cputime(s)','systime(s)','usertime(s)','memory(GB)'])
        writefile.flush()
        for i in xrange(iterations):
            res = measure_time_resource(function, args=args, kwargs=kwargs)
            performance_table.writerow([res['real'], res['cpu'], res['sys'], res['user'], res['mem']])
            del res
            gc.collect()
    writefile.close()

def minimum_of_delta_pi(fn, method = 'branch_bound', search_method = 'DFS', lp_size = 0, solver = 'Coin'):
    if method == 'naive':
        return minimum_of_delta_pi_naive(fn)
    elif method == 'branch_bound':
        T = SubadditivityTestTree(fn)
        global_min = T.minimum(search_method = search_method, max_number_of_bkpts = lp_size, solver = solver)
        del T
        gc.collect()
        return global_min
    else:
        raise ValueError

def minimum_of_delta_pi_naive(fn):
    """
    Return the min of delta_pi of fn. (Quatratic complexity)
    """
    global_min=10000
    for x in fn.end_points():
        for y in fn.end_points():
            if y<x:
                continue
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            z=1+z
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    return global_min
